"""E4: causal tracing of value propagation in variable binding.

For each prompt we build a counterfactual (cf) prompt that differs only in the
root value of the QUERY chain (v -> v'). Keep pairs where the model answers v
on the clean prompt and v' on the cf prompt.

Restoration patching: run the cf prompt, but at schedule step s (input to
step s) and position p replace the residual by the clean one; measure
    rec(p, s) = (LD_patch - LD_cf) / (LD_clean - LD_cf),
    LD = logit(v) - logit(v') at the final position.
rec = 1: the single state (p, s) carries everything needed to flip the answer.

Aggregation: positions are labelled by (chain in {query, other, none}, level,
token role in {lhs, eq, rhs, nl}) and query-suffix roles.

usage: python e4_trace.py MODEL --tag TAG --depth 4 --n 60 [--schedule ...]
"""
import argparse
import random
from collections import defaultdict

import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--n", type=int, default=60, help="number of kept (both-correct) pairs")
ap.add_argument("--order", default="forward")
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--schedule", default=None)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max_tries", type=int, default=2000)
args = ap.parse_args()


def parse_schedule(s, N):
    if s is None:
        return list(range(N))
    out = []
    for part in s.split(","):
        a, b = part.split("-")
        out += list(range(int(a), int(b) + 1))
    return out


model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
sched = parse_schedule(args.schedule, N)
S = len(sched)
vids = value_token_ids(tok)


def label_positions(it, toks):
    """label every token position; returns list of labels (tuples)."""
    labels = [("hdr",)] * len(toks)
    i = 0
    L = len(toks)
    q = it.meta["query_chain"]
    for ln in it.lines:
        while i < L - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1] == " =" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        kind = "q" if ln["chain"] == q else "o"
        for k, role in enumerate(["lhs", "eq", "rhs", "nl"]):
            labels[i + k] = (kind, ln["level"], role)
        i += 4
    # query suffix: everything after the last line
    for k in range(i, L):
        labels[k] = ("query", k - i, toks[k])
    return labels


rng = random.Random(4242 + args.depth)
agg = defaultdict(lambda: torch.zeros(S + 1))
agg_n = defaultdict(int)
kept = 0
tries = 0
examples = []
while kept < args.n and tries < args.max_tries:
    tries += 1
    it = make_vb(rng, args.depth, args.chains, args.order, header=HDR, query_fmt=QF)
    v = it.answer
    v2 = rng.choice([w for w in vids if w not in it.roots])
    cfp = it.prompt.replace(f" = {v}\n", f" = {v2}\n")
    ids, am = R.encode([it.prompt, cfp])
    if (am == 0).any():
        continue
    rc = R.run(ids, am, schedule=sched, record_steps=set(range(S)))
    lg = R.unembed(rc["h"][:, -1]).float()
    a, b = vids[v], vids[v2]
    if not (lg[0].argmax().item() == a and lg[1].argmax().item() == b):
        continue
    ld_clean = (lg[0, a] - lg[0, b]).item()
    ld_cf = (lg[1, a] - lg[1, b]).item()
    T = ids.shape[1]
    toks = [tok.decode([t]) for t in ids[0].tolist()]
    labels = label_positions(it, toks)
    # residual inputs to each step: step 0 input = embeddings
    emb = R.A.embed(ids)
    inputs = [emb] + [rc["rec"][s] for s in range(S)]  # inputs[s] = input to step s (s=S: final)
    rec_grid = torch.zeros(S + 1, T)
    for s in range(S + 1):
        base = inputs[s][1:2].expand(T, -1, -1).clone()  # cf residual, batch of T copies
        idx = torch.arange(T, device=R.device)
        base[idx, idx] = inputs[s][0, idx]  # patch clean state at position p for copy p
        am_b = am[1:2].expand(T, -1)
        ids_b = ids[1:2].expand(T, -1)
        if s < S:
            out = R.run(ids_b, am_b, schedule=sched, h0=base, start=s)["h"]
        else:
            out = base
        lp = R.unembed(out[:, -1]).float()
        ld = lp[:, a] - lp[:, b]
        rec_grid[s] = ((ld - ld_cf) / (ld_clean - ld_cf)).cpu()
    for p in range(T):
        agg[labels[p]] += rec_grid[:, p]
        agg_n[labels[p]] += 1
    if kept < 3:
        examples.append(dict(prompt=it.prompt, answer=v, cf=v2, toks=toks,
                             labels=[list(map(str, l)) for l in labels], grid=rec_grid.tolist()))
    kept += 1
    if kept % 10 == 0:
        print("kept", kept, "tries", tries, flush=True)

out = {"model": args.model, "depth": args.depth, "kept": kept, "tries": tries,
       "schedule": sched,
       "agg": {"|".join(map(str, k)): (v / agg_n[k]).tolist() for k, v in agg.items()},
       "agg_n": {"|".join(map(str, k)): n for k, n in agg_n.items()},
       "examples": examples}
save_json(out, f"{RESULTS}/e4_trace_{args.tag}.json")
print("done kept", kept, "tries", tries)
