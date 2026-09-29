"""Paper 2, overview figure: the chain a looped model follows against the number of loops (reach: longest chain answered with 80%
choice accuracy between the two chains' roots, interpolated; two-chain programs). (a) Ouro-1.4B: frozen, a constant steering vector,
the rank-8 map at layer 6 in every loop, the same map in the first loop only, the map trained on longer chains with one name pool
(dmax 40), and the map trained with two loops. (b) Huginn-0125: frozen and with its map, against recurrences. (The relay front per
layer, formerly panel c, is in fig_band.py.)
Open markers: the reach reached the longest chain tested (a lower bound)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401
import numpy as np  # noqa: E402

from figstyle import C, panel_label, plt, setup  # noqa: E402

setup()


def reach(acc, thr=0.8, assume_one=True):
    """longest chain with accuracy >= thr, linearly interpolated; returns (reach, censored). One-line chains count as answered
    unless assume_one is False (then a first tested length below thr gives 0)."""
    acc = dict(sorted(acc.items()))
    if assume_one:
        acc.setdefault(1, 1.0)
    acc = dict(sorted(acc.items()))
    ds = list(acc)
    if acc[ds[0]] < thr:
        return 0.0, False
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0), False
    return float(ds[-1]), True


def load_eval(*names, key="eval"):
    """merge {T{t}_{kind}: {d: [exact, choice]}} over several result files"""
    out = {}
    for n in names:
        p = f"{RES}/{n}"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        ev = r.get(key) or r.get("acc") or {}
        for k, row in ev.items():
            out.setdefault(k, {}).update({int(d): v for d, v in row.items()})
    return out


def series(ev, kind, prefix="T", assume_one=True):
    pts = []
    for k, row in ev.items():
        if not k.endswith("_" + kind) or not k.startswith(prefix):
            continue
        t = int(k[len(prefix):].split("_")[0])
        r, cens = reach({d: v[1] for d, v in row.items()}, assume_one=assume_one)
        pts.append((t, r, cens))
    return sorted(pts)


def plot(ax, pts, color, label, ls="-", marker="o", dx=0.0):
    if not pts:
        return
    ts = [t * (1 + dx) if ax.get_xscale() == "log" else t + dx for t, _, _ in pts]
    rs = [max(r, 0.8) for _, r, _ in pts]
    ax.plot(ts, rs, color=color, ls=ls, lw=1.6, label=label, zorder=3)
    for t, (_, r, c) in zip(ts, pts):
        ax.plot([t], [max(r, 0.8)], marker=marker, ms=3.6, color=color, mfc="white" if c else color, zorder=4)


fig = plt.figure(figsize=(8.6, 3.3))
gs = fig.add_gridspec(1, 2, wspace=0.28)

ax = fig.add_subplot(gs[0])
base = load_eval("e71_ouro_o14_a6.json")                               # single letters, chains up to 24 lines, n=150
plot(ax, series(base, "frozen"), C["muted"], "frozen", ls=(0, (3, 1.5)))
plot(ax, series(load_eval("e71_ouro_o14_a6_steer.json"), "map"), C["yellow"], "constant steering vector")
plot(ax, series(base, "map"), C["blue"], "rank-8 map, every loop")
plot(ax, series(load_eval("e71_ouro_o14_a6_first.json"), "map"), C["aqua"], "same, first loop only", marker="s", dx=0.12)
plot(ax, series(load_eval("e71_ouro_o14_a6_T2.json"), "map"), C["orange"], "map trained with two loops", marker="^", dx=-0.12)
plot(ax, series(load_eval("e71b_scaling_o14_a6long_128.json", "e71_ouro_o14_a6_long.json"), "map"), C["violet"],
     "map trained on up to 40 lines", marker="D")
ax.axvline(4, color=C["muted"], lw=0.8, ls=":")
ax.text(4.15, 1.05, "trained\nloops", fontsize=6.2, color=C["ink2"])
ax.set_yscale("log", base=2)
ax.set_yticks([0.8, 1, 2, 4, 8, 16, 32, 64, 128])
ax.set_yticklabels(["<1", "1", "2", "4", "8", "16", "32", "64", "128"])
ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 16])
ax.set_xlabel("loops at inference")
ax.set_ylabel("reach (lines)")
ax.legend(fontsize=5.8, loc="upper left")
ax.set_title("Ouro-1.4B: loops pay once the relay is on", fontsize=8.6, loc="left")
panel_label(ax, "a")

ax = fig.add_subplot(gs[1])
hug0 = load_eval("e81_huginn_hug_r8.json")                              # includes one-line chains (frozen needs them)
hug = load_eval("e81_huginn_hug_r8.json", "e81_huginn_hug_r8_fine.json")
b86 = json.load(open(f"{RES}/e86_huginn_block_hug_r8.json"))["acc"] if os.path.exists(f"{RES}/e86_huginn_block_hug_r8.json") else {}
for k, row in b86.items():
    r_, cond = k.split("_")
    if cond == "none" and r_ in ("r16", "r32"):
        hug.setdefault(f"{r_}_map", {}).update({int(d): v for d, v in row.items()})
plot(ax, series(hug0, "frozen", prefix="r", assume_one=False), C["muted"], "frozen", ls=(0, (3, 1.5)))
plot(ax, series(hug, "map", prefix="r"), C["blue"], "rank-8 map, every recurrence")
hl = load_eval("e81_huginn_hug_long_r8.json")
if len(hl) >= 4:                                                          # once the long-chain evaluation has several rows
    plot(ax, series(hl, "map", prefix="r", assume_one=False), C["violet"], "map trained on up to 24 lines (text penalty)", marker="D")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_yticks([0.8, 1, 2, 4, 8, 16, 32, 64])
ax.set_yticklabels(["<1", "1", "2", "4", "8", "16", "32", "64"])
ax.set_xlabel("recurrences at inference")
ax.set_ylabel("reach (lines)")
ax.legend(fontsize=5.8, loc="upper left")
ax.set_title("Huginn-0125: the same", fontsize=8.6, loc="left")
panel_label(ax, "b")

plt.savefig(f"{OUT}/fig_reach_loop.pdf")
plt.savefig(f"{OUT}/fig_reach_loop.png")
for name, ev, pre in (("letters map", base, "T"), ("long map", load_eval("e71_ouro_o14_a6_long.json"), "T"),
                      ("first loop", load_eval("e71_ouro_o14_a6_first.json"), "T"), ("steer", load_eval("e71_ouro_o14_a6_steer.json"), "T"),
                      ("huginn", hug, "r")):
    print(name, [(t, round(r, 1), c) for t, r, c in series(ev, "map", prefix=pre)])
