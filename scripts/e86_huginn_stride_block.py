"""E86: are the long strides needed in Huginn? As E85 for Ouro: in every block (prelude, every recurrence of the core, coda), the
tokens of each program line at level k may attend only to the header, to lines of their own level and to the lines of level k-1
("long": attention to levels <= k-2 blocked), or attention to level k-1 is blocked instead ("near"); the query tokens read freely.
Evaluated with an E81/E81b map on the core's input adapter, for several numbers of recurrences. The minimal Huginn code calls
scaled_dot_product_attention with is_causal=True; the wrapper replaces it by an explicit boolean mask when blocking is on.

Runs in loopdyn/.venv-loop. usage: python e86_huginn_stride_block.py --map PATH --tag TAG [--rs 4,8,16] [--depths 4,8,12,16] [--n 60]
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
ap.add_argument("--map", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--rs", default="4,8,16")
ap.add_argument("--depths", default="4,8,12,16")
ap.add_argument("--conds", default="none,long,near")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--pool", default="letters")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
model = model.to(args.device).eval()
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}


def single(x):
    return len(tok.encode(" " + x, add_special_tokens=False)) == 1 and len(tok.encode(x, add_special_tokens=False)) == 1


letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz") if single(c)]
if args.pool == "both":
    letters += [x + y for x in LETTERS for y in LETTERS if single(x + y)]


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
M = Map(model.config.n_embd, sd["A.weight"].shape[0]).to(args.device)
M.load_state_dict(sd)
ST = {"on": True, "allowed": None}
model.transformer.adapter.register_forward_hook(lambda mod, inp, out: M(out) if ST["on"] else out)
_sdpa = F.scaled_dot_product_attention


def sdpa(q, k, v, *a, **kw):
    if ST["allowed"] is not None and q.shape[2] == k.shape[2] == ST["allowed"].shape[0]:
        kw.pop("is_causal", None)
        return _sdpa(q, k, v, attn_mask=ST["allowed"], dropout_p=0.0)
    return _sdpa(q, k, v, *a, **kw)


F.scaled_dot_product_attention = sdpa
torch.nn.functional.scaled_dot_product_attention = sdpa


def allowed_mask(it, cond):
    enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
    off = [tuple(x) for x in enc.pop("offset_mapping")[0].tolist()]
    S = enc.input_ids.shape[1]
    allowed = torch.tril(torch.ones(S, S, dtype=torch.bool))
    if cond != "none":
        cur, by_level, spans = 0, {}, []
        for ln in it.lines:
            text = f"{ln['lhs']} = {ln['rhs']}\n"
            i = it.prompt.index(text, cur)
            j = i + len(text)
            tt = [t for t, (s, e) in enumerate(off) if e > i and s < j and e > s]
            by_level.setdefault(ln["level"], []).extend(tt)
            spans.append((ln["level"], tt))
            cur = j
        for lev, rows in spans:
            cols = ([t for lv, tt in by_level.items() if lv <= lev - 2 for t in tt] if cond == "long"
                    else list(by_level.get(lev - 1, [])))
            if rows and cols:
                allowed[torch.tensor(rows)[:, None], torch.tensor(cols)[None, :]] = False
    return enc.input_ids.to(args.device), allowed.to(args.device)


@torch.no_grad()
def accuracy(d, r, cond):
    rng = random.Random(8600 + d)
    ex = ch = 0
    for i in range(args.n):
        it = make_vb(rng, d, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
        ids, allowed = allowed_mask(it, cond)
        ST["allowed"] = None if cond == "none" else allowed
        torch.manual_seed(i)
        lg = model(input_ids=ids, num_steps=r).logits[0, -1].float()
        ST["allowed"] = None
        ex += int(lg.argmax().item() == vids[it.answer])
        rl = torch.stack([lg[vids[x]] for x in it.roots])
        ch += int(it.roots[rl.argmax().item()] == it.answer)
    return ex / args.n, ch / args.n


res = {"args": vars(args), "acc": {}}
for r in [int(x) for x in args.rs.split(",")]:
    for cond in args.conds.split(","):
        row = {d: accuracy(d, r, cond) for d in [int(x) for x in args.depths.split(",")]}
        res["acc"][f"r{r}_{cond}"] = row
        print(f"r={r:2d} {cond:5s} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()), flush=True)
        save_json(res, f"{RESULTS}/e86_huginn_block_{args.tag}.json")
print("done")
