"""Appendix (masks in detail): reach of Ouro-1.4B with the layer-6 map when reads of far lines are removed (weights zeroed after
the softmax), with 95% parametric-bootstrap intervals (each accuracy cell redrawn from a binomial with its sample size, 2,000 draws,
the routine of tab_reach.py). Two chains, 300 programs per length (e85_stride_block_o14_a6_nr_n300.json) and three chains, 200 per
length (e85_stride_block_o14_a6_nr_c3.json); rows: unmasked, all far lines, the own chain's far lines, the other chains', and as many
random far lines. Prints the rows; writes checks/masks_ci.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS, RES  # noqa: E402

import numpy as np  # noqa: E402


def reach(acc, thr=0.8):
    acc = dict(sorted(acc.items()))
    ds = list(acc)
    if acc[ds[0]] < thr:
        return float(ds[0]), "lt"
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0), ""
    return float(ds[-1]), "ge"


def ci(row, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ds = sorted(row)
    p = np.array([row[d] for d in ds])
    point, flag = reach(dict(zip(ds, p)))
    draws = rng.binomial(n, p[None, :], size=(B, len(ds))) / n
    rs = [reach(dict(zip(ds, dr)))[0] for dr in draws]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return point, float(lo), float(hi), flag


out = {}
for name, n in (("e85_stride_block_o14_a6_nr_n300.json", 300), ("e85_stride_block_o14_a6_nr_c3.json", 200)):
    acc = json.load(open(f"{RES}/{name}"))["acc"]
    rows = {}
    for key, row in acc.items():
        T, cond = key.split("_", 1)
        r = {int(d): v[1] for d, v in row.items()}                  # choice accuracy
        pt, lo, hi, flag = ci(r, n)
        rows[key] = {"reach": round(pt, 1), "ci": [round(lo, 1), round(hi, 1)], "flag": flag}
        print(f"{name[:-5]:36s} {T} {cond:10s} {'>=' if flag == 'ge' else ''}{pt:.1f} [{lo:.1f}, {hi:.1f}]")
    out[name] = rows
json.dump(out, open(f"{CHECKS}/masks_ci.json", "w"), indent=1)
