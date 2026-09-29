"""E6: variable-binding accuracy of a looped LM (Ouro) as a function of the
number of recurrent steps T and the chain depth d.

Runs in the transformers-4.56 venv (loopdyn/.venv-loop). Uses the model's own
forward; T is changed by setting model.model.total_ut_steps; logits after every
loop are read from the per-step hidden-state list (the model's early-exit path).

usage: python e6_ouro_vb.py MODEL --tag TAG [--Tmax 8] [--n 120]
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--Tmax", type=int, default=8)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--depths", default="1,2,3,4,5,6,8")
ap.add_argument("--order", default="forward")
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--qf", default="print({q})\nOutput:")
args = ap.parse_args()

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

# single-token values/variables under this tokenizer
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
letters = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
print("single-token values", len(vals), "letters", len(letters))

inner = model.model
inner.total_ut_steps = args.Tmax


@torch.no_grad()
def step_logits(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True).to(args.device)
    out = inner(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    hs_list = out[1]
    return torch.stack([model.lm_head(h[:, -1]).float() for h in hs_list], 1)  # [B, T, V]


res = {"model": args.model, "Tmax": args.Tmax, "n": args.n, "cells": []}
for d in [int(x) for x in args.depths.split(",")]:
    rng = random.Random(7919 * d + 17)
    items = [make_vb(rng, d, args.chains, args.order, header=HDR, query_fmt=args.qf.replace("\\n", "\n"),
                     value_pool=vals, var_pool=letters) for _ in range(args.n)]
    L = []
    for i in range(0, len(items), args.bs):
        L.append(step_logits([it.prompt for it in items[i:i + args.bs]]).cpu())
    L = torch.cat(L)  # [n, T, V]
    for t in range(args.Tmax):
        acc = racc = 0
        for i, it in enumerate(items):
            a = vids[it.answer]
            acc += int(L[i, t].argmax().item() == a)
            rl = torch.stack([L[i, t, vids[r]] for r in it.roots])
            racc += int(it.roots[rl.argmax().item()] == it.answer)
        cell = dict(depth=d, T=t + 1, acc=acc / len(items), racc=racc / len(items))
        res["cells"].append(cell)
    print(d, " ".join(f"T{c['T']}:{c['acc']:.2f}/{c['racc']:.2f}" for c in res["cells"] if c["depth"] == d), flush=True)
save_json(res, f"{RESULTS}/e6_ouro_{args.tag}.json")
