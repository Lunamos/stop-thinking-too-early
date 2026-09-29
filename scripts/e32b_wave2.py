"""E32b: where does a mapped model resolve long chains? For each line of the queried chain, fit per-layer linear
read-outs at two positions of that line (the right-hand-side token and the line-ending newline) for two targets:
  root  - the chain's root variable (one of ~50 letters)
  slot  - which chain the line belongs to, named by the order of the root lines in the prompt (binary for 2 chains)
Frozen vs with a map at layer a. Dual-form ridge (n << d). Prompts: 2 chains, forward or interleaved order.

usage: python e32b_wave2.py MODEL --map PATH --a 12 --tag TAG [--depth 16] [--n 600] [--order forward]
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
ap.add_argument("--map", default="none", help="path to a map, or 'none' for the frozen model only")
ap.add_argument("--a", type=int, default=0)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--order", default="forward")
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
L2I = {c: i for i, c in enumerate(letters)}


class Map(nn.Module):
    """rank-r map or steering vector, inferred from the state dict"""
    def __init__(self, sd):
        super().__init__()
        if "v" in sd:
            self.v = nn.Parameter(sd["v"].float())
        else:
            r = sd["A.weight"].shape[0]
            self.A = nn.Linear(d_model, r, bias=False)
            self.B = nn.Linear(r, d_model, bias=False)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        if hasattr(self, "v"):
            return (self.s * x + rms * self.v).to(h.dtype)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


if args.map != "none":
    _sd = torch.load(args.map, map_location=args.device)
    M = Map(_sd).to(args.device)
    M.load_state_dict(_sd)


def locate(it, toks):
    """(chain, level) -> (rhs position, newline position)"""
    out = {}
    i = 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        out[(ln["chain"], ln["level"])] = (i + 2, i + 3)
        i += 4
    return out


def ridge_acc(X, y, ncls, ntr):
    """dual ridge: X [n, d] float on device, y [n] long"""
    X = torch.nan_to_num(X.double(), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X[:ntr].mean(0, keepdim=True)
    Xc = X - mu
    Xc = Xc / Xc[:ntr].norm(dim=1).mean().clamp(min=1e-12)  # scale-free (Gemma residuals are very large)
    Xtr = Xc[:ntr]
    Y = torch.nn.functional.one_hot(y[:ntr], ncls).double()
    Y = Y - Y.mean(0, keepdim=True)
    K = Xtr @ Xtr.T
    lam = 0.1 * K.diagonal().mean().item() + 1e-6
    try:
        alpha = torch.linalg.solve(K + lam * torch.eye(ntr, device=X.device, dtype=K.dtype), Y)
    except RuntimeError:
        return float("nan")
    pred = (Xc[ntr:] @ Xtr.T @ alpha).argmax(-1)
    return pred.eq(y[ntr:]).float().mean().item()


rng = random.Random(2026)
items = [make_vb(rng, args.depth, args.chains, args.order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
         for _ in range(args.n)]
res = {"model": args.model, "a": args.a, "depth": args.depth, "order": args.order, "chains": args.chains}
for mode in (("frozen", "map") if args.map != "none" else ("frozen",)):
    hook = (lambda s, li, h: M(h) if s == args.a - 1 else h) if mode == "map" else None
    feats = {"rhs": defaultdict(list), "nl": defaultdict(list)}
    yroot, yslot = defaultdict(list), defaultdict(list)
    fin = []
    correct = 0
    with torch.no_grad():
        for it in items:
            ids, am = R.encode([it.prompt])
            toks = [tok.decode([t]) for t in ids[0].tolist()]
            pos = locate(it, toks)
            r = R.run(ids, am, record_steps=set(range(N)), hook=hook)
            correct += int(R.unembed(r["h"][:, -1]).float()[0].argmax().item() == vids[it.answer])
            q = it.meta["query_chain"]
            root = it.meta["chains"][q][0]
            root_pos = sorted((pos[(c, 1)][0], c) for c in range(args.chains))
            slot = [c for _, c in root_pos].index(q)
            H = torch.stack([r["rec"][s][0] for s in range(N)], 0)  # [N, T, d]
            for lvl in range(1, args.depth + 1):
                prhs, pnl = pos[(q, lvl)]
                feats["rhs"][lvl].append(H[:, prhs].half().cpu())
                feats["nl"][lvl].append(H[:, pnl].half().cpu())
                yroot[lvl].append(L2I[root])
                yslot[lvl].append(slot)
            fin.append(H[:, -1].half().cpu())
    ntr = int(0.75 * len(items))
    grids = {}
    for role in ("rhs", "nl"):
        for target, Y, ncls in (("root", yroot, len(letters)), ("slot", yslot, args.chains)):
            g = torch.zeros(args.depth + 1, N)
            for lvl in range(2, args.depth + 1):
                Xs = torch.stack(feats[role][lvl]).to(args.device).float()  # [n, N, d]
                ys = torch.tensor(Y[lvl], device=args.device)
                for s in range(N):
                    g[lvl, s] = ridge_acc(Xs[:, s], ys, ncls, ntr)
            grids[f"{role}_{target}"] = g.tolist()
    # final token read-out of the root and of the slot
    Xf = torch.stack(fin).to(args.device).float()
    yr = torch.tensor(yroot[1], device=args.device)
    ysl = torch.tensor(yslot[1], device=args.device)
    grids["final_root"] = [ridge_acc(Xf[:, s], yr, len(letters), ntr) for s in range(N)]
    grids["final_slot"] = [ridge_acc(Xf[:, s], ysl, args.chains, ntr) for s in range(N)]
    res[mode] = {"acc": correct / len(items), "grids": grids}
    print(mode, "task acc", correct / len(items), flush=True)
    for key in ("rhs_root", "nl_root", "rhs_slot", "nl_slot"):
        thr = 0.5 if key.endswith("root") else (0.75 if args.chains == 2 else 0.6)
        firsts = []
        for lvl in range(2, args.depth + 1):
            row = grids[key][lvl]
            firsts.append(next((s for s in range(N) if row[s] >= thr), None))
        mx = [max(grids[key][lvl]) for lvl in range(2, args.depth + 1)]
        print(f"  {key:9s} first layer >= {thr}: {firsts}", flush=True)
        print(f"  {key:9s} max: {' '.join(f'{m:.2f}' for m in mx)}", flush=True)
    print("  final_root", " ".join(f"{x:.2f}" for x in grids["final_root"]), flush=True)
    print("  final_slot", " ".join(f"{x:.2f}" for x in grids["final_slot"]), flush=True)
save_json(res, f"{RESULTS}/e32b_wave_{args.tag}.json")
