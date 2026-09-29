"""E62: which ancestors does each program line know, at which layer, and in what code?
At the right-hand-side token of line k (chain c), the target anc_j is the name of the variable defined j levels earlier in the
same chain (anc_1 is the right-hand-side token itself; anc_j for j >= k is replaced by the chain's root value and skipped).
Serial relay predicts anc_2, anc_3, anc_4 ... appear one per step; pointer doubling predicts anc_2, anc_4, anc_8 appear in
successive rounds. A ridge read-out per (target, layer) gives decodability; applying the read-out fit for anc_j at layer i to
anc_j' at layer i' tests whether successive rounds write the pointer in the same code (closure).

usage: python e62_ancestor_code.py MODEL --map PATH --a 14 --tag TAG [--depth 16] [--n 500] [--order forward]
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
ap.add_argument("--map", default=None)
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--order", default="forward")
ap.add_argument("--layers", default="10:34")
ap.add_argument("--jmax", type=int, default=8)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
L0, L1 = (int(x) for x in args.layers.split(":"))
LAYERS = list(range(L0, L1 + 1))
cls = {c: i for i, c in enumerate(letters)}


def locate(it, toks):
    out = {}
    i = 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        out[(ln["chain"], ln["level"])] = i
        i += 4
    return out


def collect(use_map):
    handles = []
    if use_map:
        M = Map.from_state(torch.load(args.map, map_location="cpu"), d_model).to(args.device)
        handles.append(attach_map(model, M, args.a))
    rng = random.Random(62)
    feats = {l: [] for l in LAYERS}
    meta = []   # (program id, chain, level, names in program, anc names)
    with torch.no_grad():
        for p in range(args.n):
            it = make_vb(rng, args.depth, 2, args.order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
            enc = tok(it.prompt, return_tensors="pt").to(args.device)
            toks = [tok.decode([t]) for t in enc.input_ids[0].tolist()]
            pos = locate(it, toks)
            hs = model(input_ids=enc.input_ids, output_hidden_states=True).hidden_states
            chains = {}
            for ln in it.lines:
                chains.setdefault(ln["chain"], {})[ln["level"]] = ln["lhs"]
            names = [cls[ln["lhs"]] for ln in it.lines]
            idx = []
            for (c, k), i0 in pos.items():
                if k < 2:
                    continue
                anc = [cls[chains[c][k - j]] if k - j >= 1 else -1 for j in range(1, args.jmax + 1)]
                meta.append((p, c, k, names, anc))
                idx.append(i0 + 2)
            for l in LAYERS:
                feats[l].append(hs[l][0, idx].float().cpu().to(torch.bfloat16))
    for h in handles:
        h.remove()
    return {l: torch.cat(v) for l, v in feats.items()}, meta


def ridge_fit(X, Y, alpha=1.0):
    X = X.double()
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    G = Xc.T @ Xc
    lam = alpha * G.diagonal().mean()
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), Xc.T @ Y.double())
    return W, mu


def score(W, mu, X, tgt, mask):
    s = (X.double() - mu) @ W                                   # [n, classes]
    return ((s + mask).argmax(-1) == tgt).double().mean().item()


def analyse(feats, meta):
    dev = args.device
    progs = sorted({m[0] for m in meta})
    ntr = int(0.75 * len(progs))
    tr = torch.tensor([m[0] < ntr for m in meta])
    nmask = torch.full((len(meta), len(letters)), float("-inf"), dtype=torch.float64)
    for r, m in enumerate(meta):
        nmask[r, m[3]] = 0.0                                    # arg-max over the names that occur in this program
    anc = torch.tensor([m[4] for m in meta])                    # [n, jmax]
    out = {"decode": {}, "transfer": {}}
    probes = {}
    F = {l: feats[l].to(dev) for l in LAYERS}
    for j in range(1, args.jmax + 1):
        ok = anc[:, j - 1] >= 0
        trm, tem = (tr & ok).to(dev), (~tr & ok).to(dev)
        Y = torch.nn.functional.one_hot(anc[:, j - 1].clamp(min=0), len(letters)).to(dev)[trm]
        tgt, msk = anc[:, j - 1].to(dev)[tem], nmask.to(dev)[tem]
        row = {}
        for l in LAYERS:
            W, mu = ridge_fit(F[l][trm], Y)
            probes[(j, l)] = (W, mu)
            row[l] = score(W, mu, F[l][tem], tgt, msk)
        out["decode"][j] = row
        print(f"anc_{j} decode by layer: " + " ".join(f"L{l}:{row[l]:.2f}" for l in LAYERS), flush=True)
    # closure: read-out fit for anc_j at layer i, applied to anc_j2 at layer i2 (test programs)
    js = [j for j in (1, 2, 3, 4, 6, 8) if j <= args.jmax]
    for j in js:
        for j2 in js:
            ok = anc[:, j2 - 1] >= 0
            tem = (~tr & ok).to(dev)
            tgt, msk = anc[:, j2 - 1].to(dev)[tem], nmask.to(dev)[tem]
            mat = {}
            for l in LAYERS:
                W, mu = probes[(j, l)]
                mat[l] = {l2: score(W, mu, F[l2][tem], tgt, msk) for l2 in LAYERS}
            out["transfer"][f"{j}->{j2}"] = mat
        best = {j2: max(max(v.values()) for v in out["transfer"][f"{j}->{j2}"].values()) for j2 in js}
        print(f"probe anc_{j} -> best transfer: " + " ".join(f"anc_{k}:{v:.2f}" for k, v in best.items()), flush=True)
    return out


res = {"model": args.model, "map": args.map, "a": args.a, "depth": args.depth, "order": args.order, "layers": LAYERS}
for mode in (["frozen", "map"] if args.map else ["frozen"]):
    print(mode, flush=True)
    feats, meta = collect(mode == "map")
    res[mode] = analyse(feats, meta)
    save_json(res, f"{RESULTS}/e62_anc_{args.tag}.json")
