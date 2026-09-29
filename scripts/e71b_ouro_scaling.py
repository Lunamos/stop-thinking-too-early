"""E71b: with a map switched on, does a looped model's reach keep growing with the number of loops at inference? Ouro-1.4B with a map
trained by e71_ouro_map.py (T=4, chains of up to 20 lines), evaluated on two-chain programs of 8 to 96 lines (two-letter names when
single letters run out) for T = 1..16 loops, with and without the map. Exact accuracy and choice between the two roots.

Runs in loopdyn/.venv-loop.
usage: python e71b_ouro_scaling.py --map PATH --a 6 --tag TAG [--apply all|first]
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--apply", default="all")
ap.add_argument("--tag", required=True)
ap.add_argument("--Ts", default="1,2,3,4,5,6,8,10,12,16")
ap.add_argument("--depths", default="8,16,24,32,48,64,96")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--frozen", type=int, default=1)
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--pool", default="auto", help="auto: single letters while they suffice, else two-letter names; both: single letters plus two-letter names always")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
inner = model.model
L = len(inner.layers)
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
one = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
       if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
two = [x + y for x in LETTERS for y in LETTERS
       if len(tok.encode(" " + x + y, add_special_tokens=False)) == 1 and len(tok.encode(x + y, add_special_tokens=False)) == 1]
print("single-letter names", len(one), "two-letter names", len(two), flush=True)


class Map(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


sd = torch.load(args.map, map_location="cpu")
M = Map(model.config.hidden_size, sd["A.weight"].shape[0]).to(args.device)
M.load_state_dict(sd)


@torch.no_grad()
def forward(ids, am, T, use_map):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    pe = inner.rotary_emb(h, pos)
    for t in range(T):
        for l in range(L):
            if use_map and l == args.a and (args.apply == "all" or (args.apply == "first" and t == 0) or (args.apply == "last" and t == T - 1)):
                h = M(h)
            out = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe, past_key_value=None, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        h = inner.norm(h)
    return model.lm_head(h[:, -1]).float()


def evaluate(T, use_map, d):
    rng = random.Random(71_000 + d)
    pool = (one + two) if args.pool == "both" else (one if args.chains * d <= len(one) else two)
    items = [make_vb(rng, d, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=pool) for _ in range(args.n)]
    ex = ch = 0
    bs = 25 if d <= 32 else 10
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = tok([it.prompt for it in chunk], return_tensors="pt", padding=True).to(args.device)
        lg = forward(enc.input_ids, enc.attention_mask, T, use_map)
        for k, it in enumerate(chunk):
            ex += int(lg[k].argmax().item() == vids[it.answer])
            rl = torch.stack([lg[k, vids[x]] for x in it.roots])
            ch += int(it.roots[rl.argmax().item()] == it.answer)
    return ex / len(items), ch / len(items)


res = {"args": vars(args), "acc": {}}
for T in [int(x) for x in args.Ts.split(",")]:
    for use_map in ((False, True) if args.frozen else (True,)):
        key = f"T{T}_{'map' if use_map else 'frozen'}"
        res["acc"][key] = {d: evaluate(T, use_map, d) for d in [int(x) for x in args.depths.split(",")]}
        print(f"T={T:2d} {'map   ' if use_map else 'frozen'} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in res["acc"][key].items()),
              flush=True)
        save_json(res, f"{RESULTS}/e71b_scaling_{args.tag}.json")
