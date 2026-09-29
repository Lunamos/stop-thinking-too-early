"""A frontier-scale data point: DeepSeek-V4-Flash-Base (43 layers, mixture of experts) on the panel's three-chain programs.

The model was run by the neighbouring depthread project (layer-streamed, likelihood only; depthread/scripts/dsv4_loglik.py with the
prompts of depthread/data/ext/vb_dsv4.jsonl, 100 programs per chain length); its per-item scores were copied, unchanged, to
results/external/dsv4/ (see the README there). Per item the file holds the three candidates' logits at the last position after every
layer (row 0: embedding; row k+1: after layer k) through the model's final norm and head.

Prints and writes checks/dsv4.json: choice accuracy by chain length, reach (80%), and for each length the first layer after which the
correct candidate is the top candidate in at least half of the programs.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS, RES  # noqa: E402

rows = [json.loads(line) for line in open(f"{RES}/external/dsv4/vb_dsv4_items.jsonl")]
by = defaultdict(list)
for r in rows:
    by[r["depth"]].append(r)


def top(lc, ans):
    return int(max(range(len(lc)), key=lambda i: lc[i]) == ans)


acc, readable = {}, {}
for d, rs in sorted(by.items()):
    L = len(rs[0]["layer_cand_logit"])
    per_layer = [sum(top(r["layer_cand_logit"][k], r["answer"]) for r in rs) / len(rs) for k in range(L)]
    acc[d] = per_layer[-1]
    readable[d] = next((k for k, a in enumerate(per_layer) if a >= 0.5), None)   # k = layers applied
ds = sorted(acc)
reach = float(ds[-1])
for d0, d1 in zip(ds, ds[1:]):
    if acc[d0] >= 0.8 > acc[d1]:
        reach = d0 + (acc[d0] - 0.8) / (acc[d0] - acc[d1]) * (d1 - d0)
        break
out = {"model": "deepseek-ai/DeepSeek-V4-Flash-Base", "layers": L - 1, "n_per_length": len(by[ds[0]]),
       "choice_acc": acc, "reach": reach, "first_layer_top_candidate_half": readable}
json.dump(out, open(f"{CHECKS}/dsv4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
