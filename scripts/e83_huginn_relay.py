"""E83: the relay per recurrence in Huginn. As E77 for Ouro: a binary ridge read-out of which chain each line of the queried chain
belongs to, at the line's right-hand-side token, from the state after every recurrence of the 4-layer core (and after each of its
layers for the first recurrences), frozen and with an E81/E81b map on the core's input adapter. Programs have two chains in level
order, so a line's position says nothing about its chain. The latent state is initialised with the same seed per program in both modes.

Runs in loopdyn/.venv-loop. usage: python e83_huginn_relay.py --map PATH --tag TAG [--r 16] [--depth 16] [--n 500] [--pool letters|both]
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
ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
ap.add_argument("--map", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=500)
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
STATE = {"on": False, "rec": [], "pos": None}
core = model.transformer.core_block
NL = len(core)


def map_hook(mod, inp, out):
    return M(out) if STATE["on"] else out


def rec_hook(mod, inp, out):
    STATE["rec"].append(out[0, STATE["pos"]].float().cpu())


model.transformer.adapter.register_forward_hook(map_hook)
for blk in core:
    blk.register_forward_hook(rec_hook)


def locate_rhs(it, prompt, offsets):
    out, cur = {}, 0
    for ln in it.lines:
        text = f"{ln['lhs']} = {ln['rhs']}\n"
        i = prompt.index(text, cur)
        end = i + len(ln["lhs"]) + 3 + len(ln["rhs"])
        out[(ln["chain"], ln["level"])] = max(t for t, (a, b) in enumerate(offsets) if a < end and b > 0 and a >= i)
        cur = i + len(text)
    return out


def ridge_acc(X, y, ntr):
    X = torch.nan_to_num(X.double())
    mu = X[:ntr].mean(0, keepdim=True)
    Xc = X - mu
    Xc = Xc / Xc[:ntr].norm(dim=1).mean().clamp(min=1e-12)
    Xtr = Xc[:ntr]
    Y = torch.nn.functional.one_hot(y[:ntr], 2).double()
    Y = Y - Y.mean(0, keepdim=True)
    K = Xtr @ Xtr.T
    lam = 0.1 * K.diagonal().mean().item() + 1e-6
    alpha = torch.linalg.solve(K + lam * torch.eye(ntr, dtype=K.dtype), Y)
    return (Xc[ntr:] @ Xtr.T @ alpha).argmax(-1).eq(y[ntr:]).float().mean().item()


out = {"args": vars(args), "layers_per_rec": NL}
for mode in ("frozen", "map"):
    STATE["on"] = mode == "map"
    rng = random.Random(8383)
    feats, ys, racc = [], [], 0
    with torch.no_grad():
        for i in range(args.n):
            it = make_vb(rng, args.depth, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
            enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            pos = locate_rhs(it, it.prompt, offsets)
            q = it.meta["query_chain"]
            STATE["pos"] = torch.tensor([pos[(q, k)] for k in range(2, args.depth + 1)])
            STATE["rec"] = []
            torch.manual_seed(i)
            lg = model(input_ids=enc.input_ids.to(args.device), num_steps=args.r).logits[0, -1].float()
            rl = torch.stack([lg[vids[x]] for x in it.roots])
            racc += int(it.roots[rl.argmax().item()] == it.answer)
            order = sorted((pos[(c, 1)], c) for c in range(2))
            ys.append([c for _, c in order].index(q))
            feats.append(torch.stack(STATE["rec"]).half())            # [r * NL, depth - 1, d]
    y = torch.tensor(ys)
    X = torch.stack(feats)                                             # [n, r * NL, depth - 1, d]
    ntr = int(0.75 * args.n)
    steps = list(range(X.shape[1]))
    grid = {k: [ridge_acc(X[:, s, k - 2], y, ntr) for s in steps] for k in range(2, args.depth + 1)}
    out[mode] = {"racc": racc / args.n, "grid": grid}
    print(f"{mode}: task choice accuracy after {args.r} recurrences {racc / args.n:.3f}", flush=True)
    for k in range(2, args.depth + 1):
        per_rec = [max(grid[k][s] for s in steps if s // NL == t) for t in range(args.r)]
        first = next((t + 1 for t, a in enumerate(per_rec) if a >= 0.75), None)
        print(f"  line {k:2d}: first recurrence >= .75: {first}; best per recurrence " + " ".join(f"{a:.2f}" for a in per_rec),
              flush=True)
    save_json(out, f"{RESULTS}/e83_huginn_relay_{args.tag}.json")
print("done")
