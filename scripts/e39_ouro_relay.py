"""E39: does a looped model relay chain identity through the program one step per loop? Ouro, two chains, level-ordered
programs; binary read-out of which chain a line belongs to (named by the order of the root lines) at the right-hand-side
token of each line of the queried chain, at every step (loop, layer); dual-form ridge, held-out accuracy.

usage (loopdyn/.venv-loop): python e39_ouro_relay.py ByteDance/Ouro-1.4B --tag ouro14 [--T 6] [--depth 8] [--n 600]
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import NOUNS, RESULTS, make_vb, save_json, LETTERS  # noqa: E402
from ld_loop import OuroRunner  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:\n"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--T", type=int, default=6)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                          dtype=torch.bfloat16).to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
R = OuroRunner(model, tok)
L = R.L
S = args.T * L
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


def locate(it, toks):
    out = {}
    i = 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        if i >= len(toks) - 3:
            return None
        out[(ln["chain"], ln["level"])] = i + 2
        i += 4
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
    alpha = torch.linalg.solve(K + lam * torch.eye(ntr, device=X.device, dtype=K.dtype), Y)
    return (Xc[ntr:] @ Xtr.T @ alpha).argmax(-1).eq(y[ntr:]).float().mean().item()


rng = random.Random(2026)
steps = [s for s in range(S) if s % 4 == 3 or s % L == 0]  # every 4th layer and each loop start
feats = {k: [] for k in range(2, args.depth + 1)}
ys = []
racc = 0
n = 0
with torch.no_grad():
    while n < args.n:
        it = make_vb(rng, args.depth, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
        enc = tok(it.prompt, return_tensors="pt").to(args.device)
        toks = [tok.decode([t]) for t in enc.input_ids[0].tolist()]
        pos = locate(it, toks)
        if pos is None:
            continue
        r = R.run(enc.input_ids, None, T=args.T, record=True)
        lg = R.logits(r["h"][:, -1]).float()[0]
        rl = torch.stack([lg[tok.encode(" " + v, add_special_tokens=False)[0]] for v in it.roots])
        racc += int(it.roots[rl.argmax().item()] == it.answer)
        q = it.meta["query_chain"]
        order = sorted((pos[(c, 1)], c) for c in range(2))
        ys.append([c for _, c in order].index(q))
        H = torch.stack([r["rec"][s][0] for s in steps])  # [len(steps), T, d]
        for k in range(2, args.depth + 1):
            feats[k].append(H[:, pos[(q, k)]].float().cpu())
        n += 1
y = torch.tensor(ys, device=args.device)
ntr = int(0.75 * n)
grid = {}
for k in range(2, args.depth + 1):
    X = torch.stack(feats[k]).to(args.device)  # [n, steps, d]
    grid[k] = [ridge_acc(X[:, i], y, ntr) for i in range(len(steps))]
out = {"model": args.model, "T": args.T, "L": L, "steps": steps, "racc_final": racc / n, "grid": grid}
save_json(out, f"{RESULTS}/e39_ouro_relay_{args.tag}.json")
print("task racc after", args.T, "loops:", racc / n, flush=True)
for k in range(2, args.depth + 1):
    first = next((steps[i] for i, a in enumerate(grid[k]) if a >= 0.75), None)
    loop_first = None if first is None else divmod(first, L)
    per_loop = []
    for t in range(args.T):
        idx = [i for i, s in enumerate(steps) if s // L == t]
        per_loop.append(max(grid[k][i] for i in idx))
    print(f"line {k}: first step >=.75 {first} (loop, layer) {loop_first}; best per loop " +
          " ".join(f"{a:.2f}" for a in per_loop), flush=True)
