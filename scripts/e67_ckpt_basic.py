"""E67: basic retrieval in early checkpoints. For a model (optionally model@revision) and saved rank-r maps, exact accuracy and
choice among the roots on one- to four-line chains (two chains, upper- and lower-case letters, as in the placement runs), plus a
single one-line assignment with no distractor, frozen and with each map. Answers whether a checkpoint that a map cannot unlock
lacks linking or lacks retrieval itself.

usage: python e67_ckpt_basic.py MODEL[@REV] --tag TAG --maps PATH:LAYER[,PATH:LAYER...]
"""
import argparse
import random

import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS
from mqa_common import Map, attach_map

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--maps", default="")
ap.add_argument("--depths", default="1,2,3,4")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


@torch.no_grad()
def score(items):
    ex = ch = 0
    for i in range(0, len(items), 25):
        chunk = items[i:i + 25]
        lg = R.last_logits([it.prompt for it in chunk], bs=25).float()
        for k, it in enumerate(chunk):
            ex += int(lg[k].argmax().item() == vids[it.answer])
            rl = torch.stack([lg[k, vids[x]] for x in it.roots])
            ch += int(it.roots[rl.argmax().item()] == it.answer)
    return ex / len(items), ch / len(items)


def suite():
    out = {}
    rng = random.Random(6700)
    out["single"] = score([make_vb(rng, 1, 1, "forward", header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
                           for _ in range(args.n)])
    for d in [int(x) for x in args.depths.split(",")]:
        rng = random.Random(6700 + d)
        out[f"d{d}"] = score([make_vb(rng, d, 2, "forward", header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
                              for _ in range(args.n)])
    return out


res = {"model": args.model, "frozen": suite()}
print("frozen", {k: f"{e:.2f}/{c:.2f}" for k, (e, c) in res["frozen"].items()}, flush=True)
for spec in [s for s in args.maps.split(",") if s]:
    path, a = spec.rsplit(":", 1)
    M = Map.from_state(torch.load(path, map_location="cpu"), d_model).to(args.device)
    h = attach_map(model, M, int(a))
    res[spec] = suite()
    h.remove()
    print(spec.split("/")[-1], {k: f"{e:.2f}/{c:.2f}" for k, (e, c) in res[spec].items()}, flush=True)
save_json(res, f"{RESULTS}/e67_basic_{args.tag}.json")
