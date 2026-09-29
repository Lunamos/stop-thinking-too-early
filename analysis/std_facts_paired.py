"""Qwen3-8B on chains of fictional facts (synth_mhqa.py): exact match by number of hops, frozen and with a rank-8 map at layer 14
trained on two- to five-hop questions, with fixed prompts (PYTHONHASHSEED=0 re-evaluations e52_mqa_rv_q8_{frozen_synth,
syn_map_a14}.json; 500 questions, 100 per hop count 2-6, the evaluation set of e52_mqa_train.py), and the paired gain with a 95%
bootstrap interval over questions. Prints; writes checks/facts_std.json.
"""
import json
import os
import random
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS, RES  # noqa: E402
from mqa_common import em_score  # noqa: E402
from synth_mhqa import make_item  # noqa: E402

rng = random.Random(777)                                   # as in e52_mqa_train.py
EXS = [make_item(rng, h, 2) for h in (2, 3, 4, 5, 6) for _ in range(100)]


def per_question(tag):
    r = json.load(open(f"{RES}/e52_mqa_rv_{tag}.json"))
    preds = r["preds"]["synth_gold"]
    em = np.array([max(em_score(p, a) for a in (ex.get("aliases") or [ex["answer"]])) for ex, p in zip(EXS, preds)], float)
    assert abs(em.mean() - r["res"]["synth_gold"]["all_em"]) < 1e-9
    return em, r["res"]["synth_gold"]


m, rm = per_question("q8_syn_map_a14")
f, rf = per_question("q8_frozen_synth")
d = m - f
idx = np.random.default_rng(zlib.crc32(b"facts|map|frozen")).integers(0, len(d), size=(2000, len(d)))
lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
out = {"frozen": {k: v for k, v in rf.items() if k.endswith("_em")}, "map": {k: v for k, v in rm.items() if k.endswith("_em")},
       "gain": float(d.mean()), "ci": [float(lo), float(hi)]}
json.dump(out, open(f"{CHECKS}/facts_std.json", "w"), indent=1)
print(json.dumps(out, indent=1))
