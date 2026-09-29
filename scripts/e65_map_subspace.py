"""E65: does the relay run inside the map's own subspace? A rank-r map writes rms(h) B A h/rms(h), i.e. only into span(B)
(r dimensions). With the map on, read out each line's chain (which chain's root line comes first) at the token naming the
previous variable, per layer, from (i) the full state, (ii) its projection on span(B), (iii) the projection on a random
r-dimensional subspace, and (iv) the state with span(B) removed. Frozen vs mapped, two chains of sixteen lines, level order.

usage: python e65_map_subspace.py MODEL --map PATH --a 14 --tag TAG [--n 400]
"""
import argparse
import random

import torch

from ld_common import RESULTS, load_model, make_vb, save_json, value_token_ids, LETTERS
from mqa_common import Map, attach_map

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--order", default="forward")
ap.add_argument("--layers", default="12:32")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
L0, L1 = (int(x) for x in args.layers.split(":"))
LAYERS = list(range(L0, L1 + 1))
sd = torch.load(args.map, map_location="cpu")
M = Map.from_state(sd, d_model).to(args.device)
B = sd["B.weight"].float().to(args.device)                      # [d, r]
QB, _ = torch.linalg.qr(B.double())                             # orthonormal basis of span(B)
g = torch.Generator().manual_seed(0)
QR_, _ = torch.linalg.qr(torch.randn(d_model, B.shape[1], generator=g, dtype=torch.float64).to(args.device))


def locate(it, toks):
    out, i = {}, 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        out[(ln["chain"], ln["level"])] = i
        i += 4
    return out


def collect(use_map):
    h = attach_map(model, M, args.a) if use_map else None
    rng = random.Random(65)
    feats = {l: [] for l in LAYERS}
    meta = []
    with torch.no_grad():
        for p in range(args.n):
            it = make_vb(rng, args.depth, 2, args.order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
            enc = tok(it.prompt, return_tensors="pt").to(args.device)
            toks = [tok.decode([t]) for t in enc.input_ids[0].tolist()]
            pos = locate(it, toks)
            first = min((pos[(c, 1)], c) for c in (0, 1))[1]           # the chain whose root line comes first
            hs = model(input_ids=enc.input_ids, output_hidden_states=True).hidden_states
            idx, lab, lvl = [], [], []
            for (c, k), i0 in pos.items():
                idx.append(i0 + 2)
                lab.append(int(c == first))
                lvl.append(k)
            meta += [(p, y, k) for y, k in zip(lab, lvl)]
            for l in LAYERS:
                feats[l].append(hs[l][0, idx].float().cpu())
    if h is not None:
        h.remove()
    return {l: torch.cat(v) for l, v in feats.items()}, meta


def ridge_acc(X, y, tr, te):
    X = X.double()
    mu = X[tr].mean(0, keepdim=True)
    Xc = X - mu
    G = Xc[tr].T @ Xc[tr]
    w = torch.linalg.solve(G + (1e-2 * G.diagonal().mean() + 1e-8) * torch.eye(G.shape[0], dtype=G.dtype, device=G.device),
                           Xc[tr].T @ y[tr])
    return ((Xc[te] @ w).sign() == y[te]).double().mean().item()


res = {"model": args.model, "map": args.map, "a": args.a, "rank": int(B.shape[1]), "layers": LAYERS}
for mode in ("frozen", "map"):
    feats, meta = collect(mode == "map")
    prog = torch.tensor([m[0] for m in meta])
    y = torch.tensor([2.0 * m[1] - 1 for m in meta], dtype=torch.float64, device=args.device)
    lvl = torch.tensor([m[2] for m in meta])
    tr = (prog < int(0.75 * args.n)).to(args.device)
    far = (lvl >= 5).to(args.device)                            # lines beyond the default relay
    out = {}
    for l in LAYERS:
        X = feats[l].to(args.device).double()
        XB = X @ QB
        XR = X @ QR_
        Xperp = X - (X @ QB) @ QB.T
        row = {}
        for name, F_ in (("full", X), ("mapspace", XB), ("random", XR), ("without_mapspace", Xperp)):
            row[name] = ridge_acc(F_, y, tr, (~tr) & far)
        out[l] = row
        print(f"{mode} L{l}: " + " ".join(f"{k} {v:.2f}" for k, v in row.items()), flush=True)
    res[mode] = out
    save_json(res, f"{RESULTS}/e65_subspace_{args.tag}.json")
