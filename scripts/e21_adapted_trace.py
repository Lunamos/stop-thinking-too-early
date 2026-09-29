"""E21: how does a small adapter change the clock?

Loads a no-loop re-entry adapter (applied once to all positions at the input of
block a) and runs (i) the pointer causal trace (level K pointer counterfactual,
restoration patching of chain + query positions) and (ii) the value trace, for a
chosen depth, with and without the adapter. Also (iii) linear read-outs of the
chain ROOT variable at every line's RHS token across layers (does every line
learn its root?).

usage: python e21_adapted_trace.py MODEL --adapter PATH --a 14 --tag TAG --depth 5
"""
import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--adapter", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--nprobe", type=int, default=600)
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
A_ = R.A
vids = value_token_ids(tok)
vals = list(vids)
letters = [c for c in LETTERS if len(tok.encode(" " + c, add_special_tokens=False)) == 1
           and len(tok.encode(c, add_special_tokens=False)) == 1]
L2I = {c: i for i, c in enumerate(letters)}
d_model = model.config.get_text_config().hidden_size


class ReEntry(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


M = ReEntry(d_model, args.rank).to(args.device)
M.load_state_dict(torch.load(args.adapter, map_location=args.device))
USE = {"on": True}


def hook(s, li, h):
    # Runner.run calls hook AFTER step s; the adapter must act on the INPUT of block a,
    # i.e. after step a-1.
    if USE["on"] and s == args.a - 1:
        return M(h)
    return h


@torch.no_grad()
def run(ids, am, h0=None, start=0, record=False):
    return R.run(ids, am, h0=h0, start=start, record_steps=set(range(N)) if record else None, hook=hook)


def label_positions(it, toks, kinds):
    labels = [("hdr",)] * len(toks)
    i = 0
    L = len(toks)
    for ln in it.lines:
        while i < L - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        if i >= L - 3:
            return None
        for k, role in enumerate(["lhs", "eq", "rhs", "nl"]):
            labels[i + k] = (kinds[ln["chain"]], ln["level"], role)
        i += 4
    for k in range(i, L):
        labels[k] = ("query", "last" if k == L - 1 else f"q{k - i}")
    return labels


def mk(rng, d):
    return make_vb(rng, d, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)


@torch.no_grad()
def trace(kind, D, K=None, n=24, seed=0):
    rng = random.Random(seed)
    agg = defaultdict(lambda: torch.zeros(N + 1))
    agg_n = defaultdict(int)
    kept = tries = 0
    while kept < n and tries < 80 * n:
        tries += 1
        it = mk(rng, D)
        q = it.meta["query_chain"]
        ch = it.meta["chains"]
        if kind == "val":
            v = it.answer
            v2 = rng.choice([w for w in vals if w not in it.roots])
            cfp = it.prompt.replace(f" = {v}\n", f" = {v2}\n")
            kinds = {c: ("q" if c == q else "o") for c in range(args.chains)}
        else:
            r = rng.choice([c for c in range(args.chains) if c != q])
            tgt, old, new = ch[q][K - 1], ch[q][K - 2], ch[r][K - 2]
            cfp = it.prompt.replace(f"{tgt} = {old}\n", f"{tgt} = {new}\n")
            v, v2 = it.roots[q], it.roots[r]
            kinds = {c: ("q" if c == q else ("r" if c == r else "o")) for c in range(args.chains)}
        ids, am = R.encode([it.prompt, cfp])
        if (am == 0).any():
            continue
        rc = run(ids, am, record=True)
        lg = R.unembed(rc["h"][:, -1]).float()
        a_, b_ = vids[v], vids[v2]
        if not (lg[0].argmax().item() == a_ and lg[1].argmax().item() == b_):
            continue
        ld_c = (lg[0, a_] - lg[0, b_]).item()
        ld_f = (lg[1, a_] - lg[1, b_]).item()
        toks = [tok.decode([t]) for t in ids[0].tolist()]
        labels = label_positions(it, toks, kinds)
        if labels is None:
            continue
        keep = [p for p in range(len(toks)) if labels[p][0] in ("q", "r", "query")]
        P = len(keep)
        inputs = [A_.embed(ids)] + [rc["rec"][s] for s in range(N)]
        idx = torch.tensor(keep, device=R.device)
        ar = torch.arange(P, device=R.device)
        grid = torch.zeros(N + 1, P)
        for s in range(N + 1):
            base = inputs[s][1:2].expand(P, -1, -1).clone()
            base[ar, idx] = inputs[s][0, idx]
            o = run(ids[1:2].expand(P, -1), am[1:2].expand(P, -1), h0=base, start=s)["h"] if s < N else base
            lp = R.unembed(o[:, -1]).float()
            grid[s] = (((lp[:, a_] - lp[:, b_]) - ld_f) / (ld_c - ld_f)).cpu()
        for j, p in enumerate(keep):
            agg[labels[p]] += grid[:, j]
            agg_n[labels[p]] += 1
        kept += 1
    return {"kept": kept, "tries": tries,
            "agg": {"|".join(map(str, k)): (v / agg_n[k]).tolist() for k, v in agg.items()}}


@torch.no_grad()
def root_probe(D, n):
    """decode the chain ROOT letter at each query-chain line's RHS token, per layer (ridge, held-out)."""
    rng = random.Random(4242)
    X = defaultdict(list)
    Y = defaultdict(list)
    for _ in range(n):
        it = mk(rng, D)
        q = it.meta["query_chain"]
        ch = it.meta["chains"]
        ids, am = R.encode([it.prompt])
        toks = [tok.decode([t]) for t in ids[0].tolist()]
        labels = label_positions(it, toks, {c: ("q" if c == q else "o") for c in range(args.chains)})
        if labels is None:
            continue
        rc = run(ids, am, record=True)
        for p, lab in enumerate(labels):
            if lab[0] == "q" and lab[2] == "rhs" and lab[1] >= 2:
                X[lab[1]].append(torch.stack([rc["rec"][s][0, p] for s in range(N)]).float().cpu())
                Y[lab[1]].append(L2I[ch[q][0]])
    res = {}
    for lvl in sorted(X):
        Xs = torch.stack(X[lvl])  # [n, N, d]
        ys = torch.tensor(Y[lvl])
        ntr = int(0.75 * len(ys))
        curve = []
        for s in range(N):
            Xl = Xs[:, s].to(R.device)
            mu = Xl[:ntr].mean(0, keepdim=True)
            Xc = Xl[:ntr] - mu
            Yo = torch.nn.functional.one_hot(ys[:ntr], len(letters)).float().to(R.device)
            lam = max(0.1 * Xc.var(0).sum().item(), 1e-2)
            W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1], device=R.device), Xc.T @ (Yo - Yo.mean(0)))
            pred = ((Xl[ntr:] - mu) @ W).argmax(-1).cpu()
            curve.append((pred == ys[ntr:]).float().mean().item())
        res[lvl] = curve
    return res


out = {"model": args.model, "adapter": args.adapter, "a": args.a, "depth": args.depth, "on": {}, "off": {}}
for mode in ("on", "off"):
    USE["on"] = mode == "on"
    o = out[mode]
    o["val"] = trace("val", args.depth, n=args.n, seed=11)
    for K in range(2, args.depth + 1):
        o[f"ptr{K}"] = trace("ptr", args.depth, K=K, n=args.n, seed=20 + K)
        src = o[f"ptr{K}"]["agg"].get(f"q|{K}|rhs")
        dep = next((l for l, x in enumerate(src) if x <= 0.5), None) if src else None
        print(mode, "ptr K", K, "kept", o[f"ptr{K}"]["kept"], "departure", dep, flush=True)
    o["root_probe"] = root_probe(args.depth, args.nprobe)
    for lvl, c in o["root_probe"].items():
        print(mode, "root decodable at line", lvl, " ".join(f"{x:.2f}" for x in c[::2]), flush=True)
    save_json(out, f"{RESULTS}/e21_adapted_{args.tag}.json")
print("done")
