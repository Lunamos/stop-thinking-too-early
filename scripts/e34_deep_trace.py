"""E34: where does pointer information flow when a mapped model follows a long chain?
Pointer counterfactuals on long two-chain programs (line K of the queried chain redirected to the other chain), with
restoration patching at every token of the queried chain's lines and of the query suffix, frozen vs mapped.
For each K we record the share of the causal effect held at each (position, layer).

usage: python e34_deep_trace.py MODEL --map PATH --a 14 --tag TAG --depth 16 --Ks 2,5,9,13,16 [--n 16]
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
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--Ks", default="2,5,9,13,16")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--modes", default="map,frozen")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
vals = list(vids)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


class Map(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(d_model, args.rank, bias=False)
        self.B = nn.Linear(args.rank, d_model, bias=False)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


M = Map().to(args.device)
M.load_state_dict(torch.load(args.map, map_location=args.device))
USE = {"on": True}


def hook(s, li, h):
    return M(h) if (USE["on"] and s == args.a - 1) else h


def run(ids, am, h0=None, start=0, record=False):
    return R.run(ids, am, h0=h0, start=start, record_steps=set(range(N)) if record else None, hook=hook)


def labels_for(it, toks, q):
    labels = [None] * len(toks)
    i = 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        if i >= len(toks) - 3:
            return None
        if ln["chain"] == q:
            for k, role in enumerate(["lhs", "eq", "rhs", "nl"]):
                labels[i + k] = f"q|{ln['level']}|{role}"
        i += 4
    for k in range(i, len(toks)):
        labels[k] = "query|last" if k == len(toks) - 1 else f"query|q{k - i}"
    return labels


@torch.no_grad()
def trace(K, seed):
    rng = random.Random(seed)
    agg = defaultdict(lambda: torch.zeros(N + 1))
    cnt = defaultdict(int)
    kept = tries = 0
    while kept < args.n and tries < 60 * args.n:
        tries += 1
        it = make_vb(rng, args.depth, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
        q = it.meta["query_chain"]
        r = 1 - q
        ch = it.meta["chains"]
        tgt, old, new = ch[q][K - 1], ch[q][K - 2], ch[r][K - 2]
        cfp = it.prompt.replace(f"\n{tgt} = {old}\n", f"\n{tgt} = {new}\n")
        v, v2 = it.roots[q], it.roots[r]
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
        labels = labels_for(it, toks, q)
        if labels is None:
            continue
        keep = [p for p in range(len(toks)) if labels[p] is not None and not labels[p].endswith("|lhs")
                and not labels[p].endswith("|eq")]
        P = len(keep)
        inputs = [R.A.embed(ids)] + [rc["rec"][s] for s in range(N)]
        idx = torch.tensor(keep, device=R.device)
        ar = torch.arange(P, device=R.device)
        grid = torch.zeros(N + 1, P)
        for s in range(N + 1):
            if s < N:
                # each chunk carries one unpatched row: re-running from layer s in a batch is not bit-identical to the
                # original run, so the share is measured against the chunk's own unpatched re-run
                for c0 in range(0, P, 31):
                    sl = slice(c0, min(P, c0 + 31))
                    m = sl.stop - sl.start
                    base = inputs[s][1:2].expand(m + 1, -1, -1).clone()
                    base[ar[:m], idx[sl]] = inputs[s][0, idx[sl]]
                    o = run(ids[1:2].expand(m + 1, -1), am[1:2].expand(m + 1, -1), h0=base, start=s)["h"]
                    lp = R.unembed(o[:, -1]).float()
                    ld = lp[:, a_] - lp[:, b_]
                    grid[s, sl] = ((ld[:m] - ld[m]) / (ld_c - ld[m])).cpu()
            else:
                base = inputs[s][1:2].expand(P, -1, -1).clone()
                base[ar, idx] = inputs[s][0, idx]
                lp = R.unembed(base[:, -1]).float()
                grid[s] = (((lp[:, a_] - lp[:, b_]) - ld_f) / (ld_c - ld_f)).cpu()
        for j, p in enumerate(keep):
            agg[labels[p]] += grid[:, j]
            cnt[labels[p]] += 1
        kept += 1
    return {"kept": kept, "tries": tries, "agg": {k: (v / cnt[k]).tolist() for k, v in agg.items()}}


out = {"model": args.model, "map": args.map, "a": args.a, "depth": args.depth}
for mode in args.modes.split(","):
    USE["on"] = mode == "map"
    out[mode] = {}
    for K in [int(x) for x in args.Ks.split(",")]:
        t = trace(K, seed=100 + K)
        out[mode][f"K{K}"] = t
        a = t["agg"]
        own = a.get(f"q|{K}|rhs")
        fin = a.get("query|last")
        dep = next((l for l, x in enumerate(own) if x <= 0.5), None) if own else None
        arr = next((l for l, x in enumerate(fin) if x >= 0.5), None) if fin else None
        # where is the effect held at each layer (top position other than own token)
        tops = []
        for l in range(0, N + 1, 2):
            best = max(((k, v[l]) for k, v in a.items()), key=lambda kv: kv[1], default=(None, 0))
            tops.append(f"{l}:{best[0]}({best[1]:.2f})")
        print(f"{mode} K={K} kept {t['kept']}/{t['tries']}  own-token departure L{dep}  final-token arrival L{arr}", flush=True)
        print("   top holder by layer:", " ".join(tops), flush=True)
        save_json(out, f"{RESULTS}/e34_deep_trace_{args.tag}.json")
print("done")
