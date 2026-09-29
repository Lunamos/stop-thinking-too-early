"""Paper 2: reach with parametric-bootstrap 95% intervals. Each accuracy cell (n programs) is redrawn from Binomial(n, p)/n, reach
(longest chain with 80% choice accuracy, linear interpolation, one-line chains counted as answered when the grid starts at one or two
lines; "<d" when the shortest tested length d was not answered) is recomputed, 2000 times. Ouro-1.4B rows by loops, Huginn-0125 rows
by recurrences (without the one-line assumption). Writes out/tables/tab_reach.tex and prints the table."""
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RES, TABLES  # noqa: E402

OUT = os.path.join(TABLES, "tab_reach.tex")

rng = np.random.default_rng(0)


def reach(acc, thr=0.8, assume_one=True):
    """(reach, flag): flag "ge" if every tested length was answered, "lt" if the shortest tested length (> 2) was not (reach below
    it; one-line chains are assumed answered only when the grid starts at one or two lines)."""
    acc = dict(sorted(acc.items()))
    if assume_one and min(acc) <= 2:
        acc.setdefault(1, 1.0)
        acc = dict(sorted(acc.items()))
    ds = list(acc)
    if acc[ds[0]] < thr:
        return (0.0, "") if ds[0] == 1 else (float(ds[0]), "lt")
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0), ""
    return float(ds[-1]), "ge"


def load(names, key="eval"):
    """merge rows over files; later files win at the same depth; returns {row: {d: (choice, n)}}. An entry may carry a third element,
    a dict renaming rows (e.g. the unmasked condition of a masking run)."""
    out = {}
    for entry in names:
        name, n = entry[:2]
        ren = entry[2] if len(entry) > 2 else None
        p = f"{RES}/{name}"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        ev = r.get(key) or r.get("acc") or {}
        for k, row in ev.items():
            if ren is not None:
                if k not in ren:
                    continue
                k = ren[k]
            out.setdefault(k, {}).update({int(d): (v[1], n) for d, v in row.items()})
    return out


def ci(row, assume_one=True, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ds = sorted(row)
    p = np.array([row[d][0] for d in ds])
    n = np.array([row[d][1] for d in ds])
    point, flag = reach({d: row[d][0] for d in ds}, assume_one=assume_one)
    draws = rng.binomial(n[None, :], p[None, :], size=(B, len(ds))) / n[None, :]
    rs = [reach(dict(zip(ds, dr)), assume_one=assume_one)[0] for dr in draws]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return point, lo, hi, flag


def cell(row, assume_one=True, seed=0):
    pt, lo, hi, flag = ci(row, assume_one=assume_one, seed=seed)
    if flag == "lt":
        return f"$<${pt:.0f}"
    return (r"$\geq$" if flag == "ge" else "") + f"{pt:.1f} [{lo:.1f}, {hi:.1f}]"


ROWS = [
    ("Ouro-1.4B frozen", [("e71_ouro_o14_a6.json", 150)], "T{}_frozen", [1, 2, 3, 4, 6, 8]),
    ("steering vector", [("e71_ouro_o14_a6_steer.json", 150)], "T{}_map", [2, 3, 4, 6, 8]),
    ("map, every loop (seed 0)", [("e71b_scaling_o14_a6.json", 100), ("e71_ouro_o14_a6.json", 150)], "T{}_map", [1, 2, 3, 4, 6, 8]),
    ("map, every loop (seed 1)", [("e71_ouro_o14_a6_s1.json", 150)], "T{}_map", [1, 2, 3, 4, 6, 8]),
    ("map, every loop (seed 2)", [("e71_ouro_o14_a6_s2.json", 150)], "T{}_map", [1, 2, 3, 4, 6, 8]),
    ("map, first loop only", [("e71b_scaling_o14_a6_first.json", 100), ("e71_ouro_o14_a6_first.json", 150)], "T{}_map",
     [2, 3, 4, 6, 8]),
    ("map at layer 10", [("e71_ouro_o14_a10.json", 150)], "T{}_map", [1, 2, 3, 4, 6, 8]),
    ("map trained with one loop", [("e71_ouro_o14_a6_T1.json", 150)], "T{}_map", [1, 2, 3, 4]),
    ("map trained with two loops", [("e71b_scaling_o14_a6_T2.json", 100), ("e71_ouro_o14_a6_T2.json", 150)], "T{}_map",
     [1, 2, 3, 4, 6, 8]),
    ("long map (up to 40 lines)", [("e71b_scaling_o14_a6long_128.json", 60), ("e71_ouro_o14_a6_long.json", 100)], "T{}_map",
     [2, 3, 4, 6, 8, 12]),
    ("long map, second seed", [("e71_ouro_o14_a6_long_s1.json", 100)], "T{}_map", [2, 3, 4, 6, 8]),
    ("long map, 2-4 chains: two chains", [("e71_ouro_o14_a6_long_c234.json", 100)], "T{}_map_c2", [2, 3, 4, 6, 8]),
    ("\\quad three chains", [("e71_ouro_o14_a6_long_c234.json", 100)], "T{}_map_c3", [2, 3, 4, 6, 8]),
    ("\\quad four chains", [("e71_ouro_o14_a6_long_c234.json", 100)], "T{}_map_c4", [2, 3, 4, 6, 8]),
]
HUG = [
    ("Huginn-0125 frozen", [("e81_huginn_hug_r8.json", 60)], "r{}_frozen", [2, 4, 8, 16, 32], False),
    ("map (up to 12 lines)", [("e81_huginn_hug_r8.json", 60), ("e81_huginn_hug_r8_fine.json", 60),
                              ("e86_huginn_block_hug_r8.json", 60, {"r16_none": "r16_map", "r32_none": "r32_map"})],
     "r{}_map", [2, 4, 8, 16, 32], False),
    ("long map (up to 24 lines)", [("e81_huginn_hug_long_r8.json", 60)], "r{}_map", [2, 4, 8, 16, 32, 64], False),
]
Ts = [1, 2, 3, 4, 6, 8, 12]
Rs = [2, 4, 8, 16, 32, 64]
lines = [r"\begin{tabular}{l" + "c" * len(Ts) + "}", r"\toprule",
         "Ouro-1.4B & " + " & ".join(f"$T={t}$" for t in Ts) + r" \\", r"\midrule"]


def emit(name, files, pat, tlist, cols, assume_one=True):
    data = load(files)
    cells = []
    for t in cols:
        k = pat.format(t)
        seed = zlib.crc32(f"{name}|{k}".encode())
        cells.append(cell(data[k], assume_one, seed) if t in tlist and k in data else "--")
    cells += ["--"] * (len(Ts) - len(cols))
    lines.append(name + " & " + " & ".join(cells) + r" \\")
    print(f"{name:28s} " + "  ".join(f"{c:>18s}" for c in cells))


for name, files, pat, tlist in ROWS:
    emit(name, files, pat, tlist, Ts)
lines += [r"\midrule", "Huginn-0125 & " + " & ".join(f"$r={r}$" for r in Rs) + " & " * (len(Ts) - len(Rs)) + r" \\", r"\midrule"]
for name, files, pat, tlist, one in HUG:
    emit(name, files, pat, tlist, Rs, one)
lines += [r"\bottomrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(lines) + "\n")
