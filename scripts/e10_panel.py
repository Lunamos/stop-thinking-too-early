"""E10: cross-model panel for the depth clock.

For one standard HF decoder:
  1. VB accuracy (acc, racc) vs depth d=1..6 (n per depth).
  2. Value trace (root-value counterfactual) for d in --val_depths: share of the
     answer carried by the root token and by the final token at every layer.
  3. Pointer trace for a d=3 chain, levels 2 and 3.
Tracing patches only the positions of the queried chain (+ redirect chain) and
the query suffix, which is where all causal mass lives (checked on Qwen3-8B).

usage: python e10_panel.py MODEL --tag TAG [--n 200] [--ntrace 24]
"""
import argparse
import json
import random
import time
from collections import defaultdict

import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, NOUNS, LETTERS

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--ntrace", type=int, default=24)
ap.add_argument("--depths", default="1,2,3,4,5,6")
ap.add_argument("--val_depths", default="1,2,3")
ap.add_argument("--ptr_depth", type=int, default=3)
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--skip_trace", action="store_true")
args = ap.parse_args()

t0 = time.time()
model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
vids = value_token_ids(tok)
vals = list(vids)
letters = [c for c in LETTERS if len(tok.encode(" " + c, add_special_tokens=False)) == 1
           and len(tok.encode(c, add_special_tokens=False)) == 1]
print(args.model, "layers", N, "values", len(vals), "letters", len(letters), flush=True)
out = {"model": args.model, "N": N, "acc": {}, "val": {}, "ptr": {}}


def mk(rng, d):
    return make_vb(rng, d, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)


# ---------------- 1. accuracy
for d in [int(x) for x in args.depths.split(",")]:
    rng = random.Random(7919 * d + 17)
    items = [mk(rng, d) for _ in range(args.n)]
    lg = R.last_logits([it.prompt for it in items], bs=args.bs)
    acc = sum(int(lg[i].argmax().item() == vids[it.answer]) for i, it in enumerate(items)) / len(items)
    racc = sum(int(it.roots[torch.stack([lg[i, vids[r]] for r in it.roots]).argmax().item()] == it.answer)
               for i, it in enumerate(items)) / len(items)
    out["acc"][d] = dict(acc=acc, racc=racc)
    print(f"d={d} acc={acc:.3f} racc={racc:.3f}", flush=True)
save_json(out, f"{RESULTS}/e10_panel_{args.tag}.json")
if args.skip_trace:
    raise SystemExit


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
        labels[k] = ("query", k - i, toks[k])
    return labels


@torch.no_grad()
def trace(kind, D, K=None, n=24, seed=0):
    rng = random.Random(seed)
    agg = defaultdict(lambda: torch.zeros(N + 1))
    agg_n = defaultdict(int)
    kept = tries = 0
    while kept < n and tries < 60 * n:
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
        rc = R.run(ids, am, record_steps=set(range(N)))
        lg = R.unembed(rc["h"][:, -1]).float()
        a, b = vids[v], vids[v2]
        cand = sorted(set([vids[x] for x in it.roots] + [a, b]))
        ci = torch.tensor(cand, device=lg.device)
        if not (cand[lg[0, ci].argmax().item()] == a and cand[lg[1, ci].argmax().item()] == b):
            continue
        ld_clean = (lg[0, a] - lg[0, b]).item()
        ld_cf = (lg[1, a] - lg[1, b]).item()
        toks = [tok.decode([t]) for t in ids[0].tolist()]
        labels = label_positions(it, toks, kinds)
        if labels is None:
            continue
        keep = [p for p in range(len(toks)) if labels[p][0] in ("q", "r", "query")]
        P = len(keep)
        inputs = [R.A.embed(ids)] + [rc["rec"][s] for s in range(N)]
        idx = torch.tensor(keep, device=R.device)
        ar = torch.arange(P, device=R.device)
        grid = torch.zeros(N + 1, P)
        for s in range(N + 1):
            base = inputs[s][1:2].expand(P, -1, -1).clone()
            base[ar, idx] = inputs[s][0, idx]
            o = R.run(ids[1:2].expand(P, -1), am[1:2].expand(P, -1), h0=base, start=s)["h"] if s < N else base
            lp = R.unembed(o[:, -1]).float()
            grid[s] = (((lp[:, a] - lp[:, b]) - ld_cf) / (ld_clean - ld_cf)).cpu()
        for j, p in enumerate(keep):
            lab = labels[p]
            if lab[0] == "query":
                lab = ("query", "last" if p == len(toks) - 1 else f"q{lab[1]}")
            agg[lab] += grid[:, j]
            agg_n[lab] += 1
        kept += 1
    return {"kept": kept, "tries": tries,
            "agg": {"|".join(map(str, k)): (v / agg_n[k]).tolist() for k, v in agg.items()}}


for d in [int(x) for x in args.val_depths.split(",")]:
    out["val"][d] = trace("val", d, n=args.ntrace, seed=100 + d)
    r = out["val"][d]
    q = r["agg"].get("query|last")
    cross = next((l for l, x in enumerate(q) if x >= 0.5), None) if q else None
    print(f"val d={d} kept={r['kept']}/{r['tries']} fetch-crossing layer={cross} (N={N}) t={time.time()-t0:.0f}", flush=True)
    save_json(out, f"{RESULTS}/e10_panel_{args.tag}.json")
D = args.ptr_depth
for K in range(2, D + 1):
    out["ptr"][K] = trace("ptr", D, K=K, n=args.ntrace, seed=200 + K)
    r = out["ptr"][K]
    src = r["agg"].get(f"q|{K}|rhs")
    dep = next((l for l, x in enumerate(src) if x <= 0.5), None) if src else None
    print(f"ptr d={D} K={K} kept={r['kept']}/{r['tries']} departure(<=0.5) layer={dep} t={time.time()-t0:.0f}", flush=True)
    save_json(out, f"{RESULTS}/e10_panel_{args.tag}.json")
print("done", time.time() - t0)
