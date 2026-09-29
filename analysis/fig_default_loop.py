"""Paper 2, Figure 1: what an extra loop computes by default. (a) Ouro-1.4B and 2.6B, three-chain programs: choice accuracy against
loops, chains of one to four lines. (b) Questions over chains of fictional facts: accuracy by hops against loops (Ouro-1.4B), with
Qwen3-1.7B-Base in one pass for reference. (c) Pointer counterfactuals in Ouro-1.4B (three-line chains, four loops): share of the
pointer's effect held by its own token over the 96 steps. (d) Relay read-out per loop: best accuracy of a read-out of which chain a
line belongs to: furthest line of the queried chain labelled, against layers applied, frozen Ouro-1.4B (per loop) and
Huginn (per recurrence)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401

import numpy as np

from figstyle import C, panel_label, plt, setup

setup()
fig = plt.figure(figsize=(13.6, 3.2))
gs = fig.add_gridspec(1, 4, wspace=0.38)
LC = [C["blue"], C["aqua"], C["yellow"], C["orange"]]

# (a) programs, accuracy vs loops
ax = fig.add_subplot(gs[0])
for f, ls, lab in (("e6_ouro_ouro14_nl_n300.json", "-", "1.4B"), ("e6_ouro_ouro26_nl.json", (0, (3, 1.5)), "2.6B")):
    r = json.load(open(f"{RES}/{f}"))
    tab = {}
    for c in r["cells"]:
        tab.setdefault(c["depth"], {})[c["T"]] = c["racc"]
    for i, d in enumerate([1, 2, 3, 4]):
        Ts = sorted(tab[d])
        ax.plot(Ts, [tab[d][t] for t in Ts], color=LC[i], ls=ls, lw=1.5, marker="o" if ls == "-" else None, ms=2.6,
                label=f"{d} line{'s' if d > 1 else ''}" if ls == "-" else None)
ax.axhline(1 / 3, color=C["ink2"], lw=0.7, ls=":")
ax.axvline(4, color=C["muted"], lw=0.8, ls=":")
ax.plot([], [], color=C["ink2"], ls=(0, (3, 1.5)), label="Ouro-2.6B")
ax.set_xlabel("loops at inference (trained with 4)")
ax.set_ylabel("accuracy (three chains)")
ax.set_ylim(0.2, 1.03)
ax.legend(fontsize=6.3, loc="lower right", ncol=2)
ax.set_title("programs: about a line per loop", fontsize=8.6, loc="left")
panel_label(ax, "a")

# (b) questions by hops vs loops
ax = fig.add_subplot(gs[1])
r = json.load(open(f"{RES}/e74_hops_o14.json"))["by_hops"]
q = json.load(open(f"{RES}/e74_hops_q17.json"))["by_hops"]
HC = [C["blue"], C["aqua"], C["yellow"], C["orange"], C["red"]]
for i, h in enumerate(["1", "2", "3", "4"]):
    ch = r[h]["choice"]
    ax.plot(range(1, len(ch) + 1), ch, color=HC[i], lw=1.5, marker="o", ms=2.6, label=f"{h} hop{'s' if h != '1' else ''}")
    ax.scatter([0.4], [q[h]["choice"][0]], color=HC[i], marker="D", s=16, zorder=4)
ax.text(0.9, 0.16, "diamonds: Qwen3-1.7B, one pass", fontsize=6.0, color=C["ink2"], ha="left")
ax.axhline(1 / 3, color=C["ink2"], lw=0.7, ls=":")
ax.axvline(4, color=C["muted"], lw=0.8, ls=":")
ax.set_xlim(0, 8.5)
ax.set_ylim(0.15, 1.03)
ax.set_xlabel("loops (Ouro-1.4B)")
ax.set_ylabel("accuracy (three chains)")
ax.legend(fontsize=6.3, loc="upper right")
ax.set_title("questions: about a hop per loop", fontsize=8.6, loc="left")
panel_label(ax, "b")

# (c) pointer effect over steps
ax = fig.add_subplot(gs[2])
for f, K, lab, col in (("e8_ouro_o14_ptr_d3_T4_l3.json", 3, "pointer on the queried line", C["blue"]),
                       ("e8_ouro_o14_ptr_d3_T4_l2.json", 2, "pointer one line back", C["aqua"])):
    a = json.load(open(f"{RES}/{f}"))["agg"]
    own = np.array(a[f"q|{K}|rhs"])
    fin = np.array(a[sorted([k for k in a if k.startswith("query")], key=lambda k: int(k.split("|")[1]))[-1]])
    ax.plot(np.arange(len(own)), own, color=col, lw=1.5, label=lab)
    ax.plot(np.arange(len(fin)), fin, color=col, lw=1.2, ls=(0, (3, 1.5)))
for t in range(1, 4):
    ax.axvline(24 * t, color=C["grid"], lw=1.0)
for t in range(4):
    ax.text(24 * t + 12, 1.06, f"loop {t + 1}", ha="center", fontsize=6.5, color=C["ink2"])
ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("step (4 loops x 24 layers)")
ax.set_ylabel("share of the pointer's effect")
ax.legend(fontsize=6.0, loc="center left")
ax.set_title("pointers leave only in late loops", fontsize=8.6, loc="left", pad=12)
panel_label(ax, "c")

# (d) relay, frozen: furthest line of the queried chain labelled (read-out >= .75, best so far), against layers applied
ax = fig.add_subplot(gs[3])


def prefix(grid_rows, per_unit):
    """grid_rows: {line: [acc per recorded step]}, per_unit: list of step-index lists per loop/recurrence -> labelled prefix"""
    lines = sorted(grid_rows)
    out = []
    for u in range(len(per_unit)):
        seen = [i for idx in per_unit[: u + 1] for i in idx]
        n = 1
        for k in lines:
            if max(grid_rows[k][i] for i in seen) >= 0.75:
                n = k
            else:
                break
        out.append(n)
    return out


r = json.load(open(f"{RES}/e39_ouro_relay_ouro14.json"))
steps, L = r["steps"], r["L"]
g = {int(k): v for k, v in r["grid"].items()}
units = [[i for i, s in enumerate(steps) if s // L == t] for t in range(r["T"])]
po = prefix(g, units)
ax.plot([L * (t + 1) for t in range(len(po))], po, color=C["blue"], lw=1.6, marker="o", ms=3, label="Ouro-1.4B (per loop)")
hp = f"{RES}/e83_huginn_relay_hug_r8_d16.json"
try:
    h = json.load(open(hp))
    NL = h["layers_per_rec"]
    gh = {int(k): v for k, v in h["frozen"]["grid"].items()}
    R = len(next(iter(gh.values()))) // NL
    ph = prefix(gh, [list(range(t * NL, (t + 1) * NL)) for t in range(R)])
    ax.plot([2 + NL * (t + 1) for t in range(R)], ph, color=C["aqua"], lw=1.6, marker="^", ms=3, label="Huginn (per recurrence)")
except (FileNotFoundError, KeyError):
    pass
ax.set_xscale("log", base=2)
ax.set_xticks([8, 16, 32, 64, 128])
ax.set_xticklabels(["8", "16", "32", "64", "128"])
ax.set_ylim(0, 8.5)
ax.set_xlabel("layers applied")
ax.set_ylabel("furthest line readable (read-out >= .75)")
ax.legend(fontsize=6.3, loc="upper left")
ax.set_title("the program relays three or four lines, then stops", fontsize=8.6, loc="left")
panel_label(ax, "d")

plt.savefig(f"{OUT}/fig_default_loop.pdf")
plt.savefig(f"{OUT}/fig_default_loop.png")
print("saved")
