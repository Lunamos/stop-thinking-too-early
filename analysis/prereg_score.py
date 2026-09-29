"""Score the placement-edge rules fixed before the held-out models were run. For each model: reach of the rank-8 map at every tested
layer (80% exact accuracy, K=1, one-line accuracy taken as 1), frozen reach (median of the K=0 evaluations), the observed edge
(midpoint between the last working placement and the next tested one; works = reach >= R0 + 0.5 (max R - R0)), and each rule's
prediction and error. Prints a table and writes results/prereg_edges.json (read by routing_window.py) and
out/tables/tab_prereg.tex. The paper's own scoring of the prospective test is in std_analyze.py."""
import glob
import json
import os
import re

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "tables")
# prefix, layers, R1 commit layer, R3 handoff - 11.5 (all fixed in advance); swept = the five models known beforehand
MODELS = [("place_q8", "Qwen3-8B", 36, 20.5, 20.2, "swept"), ("place_olmo", "OLMo-3-7B", 32, 15.0, 12.7, "swept"),
          ("place_l31", "Llama-3.1-8B", 32, 12.5, 13.4, "swept"), ("place_q17", "Qwen3-1.7B", 28, 18.0, 13.5, "swept"),
          ("place_q06", "Qwen3-0.6B", 28, 10.0, 13.7, "swept"),
          ("place_l1b", "Llama-3.2-1B", 16, 8.0, 2.8, "held-out"), ("place_q4", "Qwen3-4B", 36, 22.0, 20.6, "held-out"),
          ("place_g12", "Gemma-3-12B", 48, 23.0, 27.5, "held-out"), ("place_o32", "OLMo-3-32B", 64, 19.0, 30.4, "held-out")]


R4_SRC = {"Qwen3-8B": "q8kl_d16_inter", "OLMo-3-7B": "o3a5_d16_inter", "Llama-3.1-8B": "l31a8_d16_inter",
          "Qwen3-1.7B": "q17a3_d16_inter", "Qwen3-0.6B": "q06a6_d16_inter", "Llama-3.2-1B": "frozen_l1b_d16_inter",
          "Qwen3-4B": "frozen_q4_d16_inter", "Gemma-3-12B": "frozen_g12_d16_inter", "OLMo-3-32B": "frozen_o32_d16_inter"}


def r4(name):
    """last layer at which the frozen model's stride-one same-chain-minus-other attention is at least half of its maximum"""
    import os
    p = f"{RES}/e37b_strides_{R4_SRC[name]}.json"
    if not os.path.exists(p):
        return None
    d1 = np.array(json.load(open(p))["frozen"]["diff"])[:, 0]
    return float(max(i for i, x in enumerate(d1) if x >= 0.5 * d1.max()))


def reach(acc, thr=0.8):
    acc = dict(acc)
    acc.setdefault(1, 1.0)
    ds = sorted(acc)
    best = 0.0
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0)
        if acc[d1] >= thr:
            best = d1
    return best if acc[ds[0]] >= thr else 1.0


def curve(prefix):
    pts, frozen = [], []
    for f in glob.glob(f"{RES}/e19_reentry_{prefix}_a*.json"):
        m = re.search(r"_a(\d+)\.json$", f)
        if not m:
            continue
        ev = json.load(open(f))["eval"]
        ds = sorted({int(k.split("_")[0][1:]) for k in ev})
        pts.append((int(m.group(1)), reach({d: ev[f"d{d}_K1"] for d in ds})))
        frozen.append(reach({d: ev[f"d{d}_K0"] for d in ds}))
    return sorted(pts), (float(np.median(frozen)) if frozen else None)


out = {}
print(f"{'model':13s} {'N':>3s} {'R0':>5s} {'maxR':>5s}  reach by layer                                   edge   "
      f"R1 commit   R2 45%   R3 handoff-11.5")
for prefix, name, N, r1, r3, kind in MODELS:
    pts, R0 = curve(prefix)
    if not pts:
        continue
    mx = max(r for _, r in pts)
    thr = R0 + 0.5 * (mx - R0)
    works = [a for a, r in pts if r >= thr]
    edge = bracket = None
    if works:
        last = max(works)
        later = [a for a, _ in pts if a > last]
        if later:
            bracket = (last, min(later))
            edge = (last + min(later)) / 2
    preds = {"R1": r1, "R2": 0.45 * N, "R3": r3}
    if r4(name) is not None:
        preds["R4"] = r4(name)
    errs = {k: (abs(edge - v) if edge is not None else None) for k, v in preds.items()}
    inside = {k: (bracket[0] - 1 <= v <= bracket[1] + 1 if bracket else None) for k, v in preds.items()}
    out[name] = dict(kind=kind, N=N, R0=R0, max_reach=mx, reach={a: r for a, r in pts}, bracket=bracket, edge=edge,
                     predictions=preds, errors=errs, inside=inside)
    rs = " ".join(f"{a}:{r:.1f}" for a, r in pts)
    e = f"{bracket[0]}|{bracket[1]}" if bracket else "--"
    fmt = lambda k: f"{preds[k]:5.1f} ({'in' if inside[k] else 'out'} {errs[k]:.1f})" if edge is not None else f"{preds[k]:5.1f}"
    print(f"{name:13s} {N:3d} {R0:5.1f} {mx:5.1f}  {rs:48s} {e:6s} {fmt('R1'):16s} {fmt('R2'):16s} {fmt('R3'):16s} "
          f"{fmt('R4') if 'R4' in preds else '--'}")
for kind in ("swept", "held-out"):
    rows = [v for v in out.values() if v["kind"] == kind and v["edge"] is not None]
    if rows:
        ks = [k for k in ("R1", "R2", "R3", "R4") if all(k in r["errors"] for r in rows)]
        print(kind, "mean abs error:", {k: round(float(np.mean([r["errors"][k] for r in rows])), 2) for k in ks},
              " inside bracket:", {k: sum(bool(r["inside"][k]) for r in rows) for k in ks}, "of", len(rows))
json.dump(out, open(f"{RES}/prereg_edges.json", "w"), indent=1)

# LaTeX table for the appendix
lines = [r"\begin{tabular}{llrlrrrrr}", r"\toprule",
         r"Model & & Layers & Edge (works $|$ fails) & Frozen reach & Commit (R1) & 45\% (R2) & Copy $-$ 11.5 (R3) & Pointer heads (R4) \\",
         r"\midrule"]
for name, v in out.items():
    e = f"{v['bracket'][0]} $|$ {v['bracket'][1]}" if v["bracket"] else "--"
    cell = lambda k: ("--" if k not in v["predictions"] else f"{v['predictions'][k]:.1f}" + ("" if v["edge"] is None else
                      (r"$^\checkmark$" if v["inside"][k] else "")))
    lines.append(f"{name} & {v['kind']} & {v['N']} & {e} & {v['R0']:.1f} & {cell('R1')} & {cell('R2')} & {cell('R3')} & {cell('R4')}"
                 + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/tab_prereg.tex", "w").write("\n".join(lines) + "\n")
