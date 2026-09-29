"""E66: are the long-stride relay heads necessary? With a map at layer a, zero the outputs of the heads that carry strides of two or
more lines (chosen by E37b on held-out programs: for each layer and stride j >= 2, the head with the largest same-chain-minus-other
attention, kept if that difference exceeds a threshold), and compare with zeroing the stride-one heads, and with zeroing the same number
of random heads in the same layers. Accuracy (choice between the two roots) on two-chain programs, interleaved, by length.

usage: python e66_head_ablation.py MODEL --map PATH --a 14 --heads_from e37b_strides_TAG.json --tag TAG
"""
import argparse
import json
import random

import numpy as np
import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS
from mqa_common import Map, attach_map, decoder_layers

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--heads_from", required=True)
ap.add_argument("--thr", type=float, default=0.1)
ap.add_argument("--tag", required=True)
ap.add_argument("--order", default="interleave")
ap.add_argument("--depths", default="2,4,8,12,16")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min_layer", type=int, default=0, help="only heads in layers >= this (e.g. the map's layer)")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
cfg = model.config.get_text_config()
d_model, H = cfg.hidden_size, cfg.num_attention_heads
hd = getattr(cfg, "head_dim", None) or d_model // H
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
M = Map.from_state(torch.load(args.map, map_location="cpu"), d_model).to(args.device)
attach_map(model, M, args.a)

S = json.load(open(args.heads_from))["map"]
diff, best = np.array(S["diff"]), np.array(S["head"])          # [L, J]
long_heads, short_heads = set(), set()
for l in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        if diff[l, j] > args.thr:
            (short_heads if j == 0 else long_heads).add((l, int(best[l, j])))
long_heads -= short_heads
long_heads = {(l, h) for l, h in long_heads if l >= args.min_layer}
short_heads = {(l, h) for l, h in short_heads if l >= args.min_layer}
rng = random.Random(args.seed)


def matched_random(target):
    """the same number of heads in the same layers, avoiding both selected sets"""
    out = set()
    for l in sorted({l for l, _ in target}):
        k = sum(1 for ll, _ in target if ll == l)
        cand = [h for h in range(H) if (l, h) not in long_heads and (l, h) not in short_heads]
        out |= {(l, h) for h in rng.sample(cand, k)}
    return out


rand_heads = matched_random(long_heads)
rand_short = matched_random(short_heads)
print("long-stride heads", sorted(long_heads), "\nstride-one heads", sorted(short_heads), flush=True)

ABL = {"set": set()}


def make_hook(l):
    def pre_hook(mod, inputs):
        x = inputs[0]
        heads = [h for (ll, h) in ABL["set"] if ll == l]
        if not heads:
            return None
        x = x.clone()
        for h in heads:
            x[..., h * hd:(h + 1) * hd] = 0
        return (x,) + tuple(inputs[1:])
    return pre_hook


for l, layer in enumerate(decoder_layers(model)):
    layer.self_attn.o_proj.register_forward_pre_hook(make_hook(l))


@torch.no_grad()
def accuracy(d):
    r = random.Random(6600 + d)
    its = [make_vb(r, d, 2, args.order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters) for _ in range(args.n)]
    ok = 0
    for i in range(0, len(its), 25):
        ch = its[i:i + 25]
        lg = R.last_logits([it.prompt for it in ch], bs=25).float()
        for k, it in enumerate(ch):
            rl = torch.stack([lg[k, vids[x]] for x in it.roots])
            ok += int(it.roots[rl.argmax().item()] == it.answer)
    return ok / len(its)


res = {"model": args.model, "map": args.map, "a": args.a, "min_layer": args.min_layer, "long_heads": sorted(long_heads),
       "short_heads": sorted(short_heads), "rand_heads": sorted(rand_heads), "rand_short": sorted(rand_short), "acc": {}}
for name, hs in (("none", set()), ("long", long_heads), ("random", rand_heads), ("short", short_heads), ("random_short", rand_short)):
    ABL["set"] = hs
    row = {d: accuracy(d) for d in [int(x) for x in args.depths.split(",")]}
    res["acc"][name] = row
    print(f"ablate {name:6s} ({len(hs):2d} heads): " + " ".join(f"d{d}:{v:.2f}" for d, v in row.items()), flush=True)
    save_json(res, f"{RESULTS}/e66_ablate_{args.tag}.json")
