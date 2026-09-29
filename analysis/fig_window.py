"""Paper 2, Figure 2: a fixed routing window in standard models. (a) Window width D = b - a (layers between the queried pointer's
departure from its token and the final token holding 90% of both pointers' effects; notes/routing_window.md) against the number of
layers, thirteen models; dotted: a width proportional to depth through the mean. (b) Where in depth the window sits: a/N and b/N,
and the value copy H/N. (c) Blocking the query's attention to the program's pointer lines in a sliding band of three layers:
two-line accuracy against the band's first layer (relative depth), for the models measured (E76); shaded: the window."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401

import numpy as np

from figstyle import C, panel_label, plt, setup

setup()
T = json.load(open(f"{NOTES}/routing_window_table.json"))["rows"]
FAM = {"Qwen": C["blue"], "Llama": C["orange"], "OLMo": C["aqua"], "Gemma": C["violet"]}
fig = plt.figure(figsize=(12.6, 3.2))
gs = fig.add_gridspec(1, 3, wspace=0.34)

# (a) width vs depth
ax = fig.add_subplot(gs[0])
N = np.array([r["N"] for r in T])
D = np.array([r["W80"] for r in T])          # threshold-free width: 90% -> 10% of the pointer's own share
for r in T:
    ax.scatter(r["N"], r["W80"], color=FAM[r["fam"]], s=26, zorder=3)
    if r["name"] in ("Llama-3.2-1B", "OLMo-3-32B", "Gemma-3-27B", "OLMo-3-7B", "Qwen3-14B"):
        off = {"Gemma-3-27B": (-38, -11), "OLMo-3-32B": (-8, 6), "OLMo-3-7B": (-6, -11)}.get(r["name"].replace("-Base", ""), (3, 4))
        ax.annotate(r["name"].replace("-Base", ""), (r["N"], r["W80"]), fontsize=5.8, color=C["ink2"], xytext=off,
                    textcoords="offset points")
xs = np.linspace(12, 68, 50)
ax.plot(xs, D.mean() / N.mean() * xs, color=C["muted"], lw=1.0, ls=(0, (1, 1.6)), label="proportional to depth")
ax.axhline(D.mean(), color=C["ink2"], lw=0.9, ls=(0, (4, 2)), label=f"mean {D.mean():.1f} layers")
for fam, col in FAM.items():
    ax.scatter([], [], color=col, s=20, label=fam)
ax.set_xlabel("layers")
ax.set_ylabel("layers over which the question reads (W80)")
ax.set_ylim(0, 20)
ax.legend(fontsize=6.0, loc="upper left", ncol=2)
ax.set_title("the reading range does not widen with depth", fontsize=8.6, loc="left")
panel_label(ax, "a")

# (b) position of the window and of the value copy
ax = fig.add_subplot(gs[1])
order = sorted(T, key=lambda r: r["N"])
for i, r in enumerate(order):
    ax.plot([r["a"] / r["N"], r["b"] / r["N"]], [i, i], color=FAM[r["fam"]], lw=4, solid_capstyle="butt")
    ax.scatter([r["H"] / r["N"]], [i], color=C["orange"], marker="|", s=60, zorder=3)
ax.scatter([], [], color=C["orange"], marker="|", s=60, label="value copied")
ax.plot([], [], color=C["ink2"], lw=4, label="layers where the question reads")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{r['name'].replace('-Base', '')} ({r['N']})" for r in order], fontsize=6.2)
ax.set_xlim(0, 1)
ax.set_xlabel("relative depth")
ax.legend(fontsize=6.2, loc="upper left")
ax.set_title("where that range sits", fontsize=8.6, loc="left")
panel_label(ax, "b", x=-0.42)

# (c) blocking scan
ax = fig.add_subplot(gs[2])
BL = [("l31", "Llama-3.1-8B", C["orange"]), ("q8", "Qwen3-8B", C["blue"]), ("o3", "OLMo-3-7B", C["aqua"])]
for tag, lab, col in BL:
    p = f"{RES}/e76_block_{tag}.json"
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    if not r.get("scan"):
        continue
    Nm = r["N"]
    xs = sorted(int(s) for s in r["scan"])
    base = r["named"]["none"]["2"]
    ax.plot([(s + 1) / Nm for s in xs], [r["scan"][str(s)]["2"] for s in xs], color=col, lw=1.5, marker="o", ms=2.4,
            label=f"{lab} (unblocked {base:.2f})")
    ax.axvspan(r["a"] / Nm, r["b"] / Nm, color=col, alpha=0.08, lw=0)
ax.axhline(1 / 3, color=C["ink2"], lw=0.7, ls=":")
ax.set_xlim(0, 1)
ax.set_ylim(0.2, 1.0)
ax.set_xlabel("blocked layers (relative depth of their centre)")
ax.set_ylabel("two-line accuracy (three chains)")
ax.legend(fontsize=6.2, loc="lower right")
ax.set_title("the question reads the program in a few layers", fontsize=8.6, loc="left")
panel_label(ax, "c")

plt.savefig(f"{OUT}/fig_window.pdf")
plt.savefig(f"{OUT}/fig_window.png")
print("saved")
