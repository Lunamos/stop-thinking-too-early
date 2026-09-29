"""Panel of thirteen standard base models (tables/tab_panel.tex).

Columns: choice accuracy among the three chains' roots by chain length (e10_panel_*.json, 200 programs per cell); reach (first
downward crossing of 80%, interpolated); cutoff layer (commit_depth.py: first layer at which a pointer's own token holds less than
half of its effect, lines 2 and 3 of three-line programs, averaged; relative to the number of layers); value handoff (first layer at
which the final token holds half of the root value's effect, chains of one to three lines, averaged; relative); the default relay
front (furthest line whose chain a linear read-out at the pointer token gives with accuracy >= 0.75, all earlier lines included,
two-chain programs of eight lines; e32b_wave_relay_*.json); and the reach unlocked by a rank-8 map with the text penalty at its best
placement among the layers tested (exact accuracy, two chains, 150 programs per length; e19_reentry_place_*_a*.json; the dagger
fallback, a rank-64 map without the penalty, applies only when no rank-8 placement run exists, which is no longer the case).
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RES, TABLES  # noqa: E402
from commit_depth import commit  # noqa: E402

ROWS = [("Llama-3.2-1B", "llama32_1b", "llama32_1b", "place_l1b", "vdeep_llama32_1b"),
        ("Qwen3-0.6B", "q06base", "q06", "place_q06", "vdeep_q06"),
        ("Qwen3-1.7B", "q17base", "q17", "place_q17", "vdeep_q17"),
        ("Llama-3.2-3B", "llama32_3b", "llama32_3b", "place_l32_3b", "vdeep_llama32_3b"),
        ("Qwen3-4B", "q4base", "q4", "place_q4", "vdeep_q4"),
        ("Gemma-3-4B", "gemma3_4b", "gemma3_4b", None, None),
        ("Llama-3.1-8B", "llama31_8b", "llama31_8b", "place_l31", "vdeep_llama31_8b"),
        ("OLMo-3-7B", "olmo3_7b", "olmo3_7b", "place_olmo", "vdeep_olmo3_7b"),
        ("Qwen3-8B", "q8base", "q8", "place_q8", "vdeep_q8_kl1"),
        ("Gemma-3-12B", "gemma3_12b", "gemma3_12b", "place_g12", None),
        ("Qwen3-14B", "q14base", "q14", "place_q14", "vdeep_q14"),
        ("Gemma-3-27B", "gemma3_27b", "gemma3_27b", None, None),
        ("OLMo-3-32B", "olmo3_32b", "olmo3_32b", "place_o32", None)]


def reach(acc, thr=0.8, one=None):
    acc = dict(acc)
    if one is not None:
        acc.setdefault(1, one)
    ds = sorted(acc)
    best = 0.0
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0)
        if acc[d1] >= thr:
            best = d1
    return best if acc[ds[0]] >= thr else 0.0


def relay_front(tag, thr=0.75):
    p = f"{RES}/e32b_wave_relay_{tag}.json"
    if not os.path.exists(p):
        return None
    g = json.load(open(p))["frozen"]["grids"]["rhs_slot"]
    best = 1
    for j in range(len(g[2])):
        k = 1
        for ln in range(2, len(g)):
            if not g[ln] or g[ln][j] < thr:
                break
            k = ln
        best = max(best, k)
    return best


def unlocked(prefix, fallback):
    best = None
    for f in glob.glob(f"{RES}/e19_reentry_{prefix}_a*.json") if prefix else []:
        if not re.search(r"_a(\d+)\.json$", f):                   # seed 0 of the placement sweep
            continue
        ev = json.load(open(f)).get("eval")
        if not ev:
            continue
        ds = sorted({int(k.split("_")[0][1:]) for k in ev})
        r = reach({d: ev[f"d{d}_K1"] for d in ds}, one=1.0)
        best = r if best is None else max(best, r)
    if best is not None:
        return r"$\geq$24" if best >= 24 else f"{best:.0f}"
    p = f"{RES}/e19_reentry_{fallback}.json" if fallback else None
    if p and os.path.exists(p):
        ev = json.load(open(p))["eval"]
        ds = sorted({int(k.split("_")[0][1:]) for k in ev if k.endswith("_K1")})
        r = reach({d: ev[f"d{d}_K1"] for d in ds}, one=1.0)
        return (r"$\geq$24" if r >= 24 else f"{r:.0f}") + r"$^\dagger$"
    return "--"


out = [r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
       r"Model & Layers & \multicolumn{6}{c}{Choice accuracy by chain length (chance $1/3$)} & Reach & Cutoff & Value copy & Relay & Unlocked \\",
       r"\cmidrule(lr){3-8}",
       r" & & 1 & 2 & 3 & 4 & 5 & 6 & (lines) & (rel.\ depth) & (rel.\ depth) & (line) & (lines) \\", r"\midrule"]
for name, tag, rtag, ptag, ftag in ROWS:
    r = json.load(open(f"{RES}/e10_panel_{tag}.json"))
    N = r["N"]
    racc = {int(d): v["racc"] for d, v in r["acc"].items()}
    hand = []
    for d in ("1", "2", "3"):
        y = r["val"][d]["agg"]["query|last"]
        hand.append(next((i for i, x in enumerate(y) if x >= 0.5), N))
    c, _, _, _ = commit(f"{RES}/e10_panel_{tag}.json")
    fr = relay_front(rtag)
    cells = [name, str(N)] + [f"{racc[d]:.2f}" for d in range(1, 7)] + [f"{reach(racc):.1f}", f"{c:.2f}", f"{np.mean(hand) / N:.2f}",
                                                                       "--" if fr is None else str(fr), unlocked(ptag, ftag)]
    out.append(" & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/tab_panel.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out))
