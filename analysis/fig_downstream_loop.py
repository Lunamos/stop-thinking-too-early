"""Paper 2, downstream figure. (a) Questions over chains of fictional facts (three chains, chance 1/3), Ouro-1.4B: choice accuracy
against loops for 2-8 hops, frozen (dashed) and with a map trained on 1-4-hop questions (solid). (b) MuSiQue (300 dev questions):
exact match against loops, frozen, with a map trained on MuSiQue at layer 6, and (when available) the same map at layer 20 and the
program-trained map without MuSiQue training. (c) MuSiQue exact match by hops at four loops."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401
import numpy as np  # noqa: E402

from figstyle import C, panel_label, plt, setup  # noqa: E402

setup()
fig = plt.figure(figsize=(12.0, 3.2))
gs = fig.add_gridspec(1, 3, wspace=0.32, width_ratios=[1.15, 1.0, 0.8])

ax = fig.add_subplot(gs[0])
ev = json.load(open(f"{RES}/e79_ouro_nl_o14_a6.json"))["eval"]
Ts = sorted({int(k[1:].split("_")[0]) for k in ev})
cols = {2: C["blue"], 3: C["aqua"], 4: C["yellow"], 5: C["orange"], 6: C["magenta"], 8: C["violet"]}
for d, col in cols.items():
    for kind, ls, mk in (("map", "-", "o"), ("frozen", (0, (3, 1.5)), None)):
        ys = [ev[f"T{t}_{kind}"][str(d)][1] for t in Ts]
        ax.plot(Ts, ys, color=col, ls=ls, lw=1.5 if kind == "map" else 1.0, marker=mk, ms=2.8,
                label=f"{d} hops" if kind == "map" else None)
ax.axhline(1 / 3, color=C["ink2"], lw=0.7, ls=":")
ax.axvline(4, color=C["muted"], lw=0.8, ls=":")
ax.plot([], [], color=C["ink2"], ls=(0, (3, 1.5)), label="frozen")
ax.set_xticks(Ts)
ax.set_xlabel("loops at inference")
ax.set_ylabel("accuracy (three chains)")
ax.set_ylim(0.2, 1.03)
ax.legend(fontsize=6.0, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.17))
ax.set_title("fact chains: map trained on 1-4 hops", fontsize=8.6, loc="left")
panel_label(ax, "a")


def parse(path):
    """{T: {kind: (EM, {hops: EM})}} from an E80 log"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"T=(\d+) (frozen|map)\s+EM ([\d.]+) F1 [\d.]+ 2:([\d.]+) 3:([\d.]+) 4:([\d.]+)", line)
        if m:
            t, kind = int(m.group(1)), m.group(2)
            out.setdefault(t, {})[kind] = (float(m.group(3)), {2: float(m.group(4)), 3: float(m.group(5)), 4: float(m.group(6))})
    return out


def pick(*names):
    """the first existing log (fixed-prompt *_h0 runs first)"""
    for n in names:
        if os.path.exists(f"{RES}/{n}"):
            return parse(f"{RES}/{n}")
    return {}


base = pick("e80_o14_a6_h0.log", "e80_o14_a6.log")
late = pick("e80_o14_a20_h0.log", "e80_o14_a20.log")
prog = pick("e80_o14_progmap_h0.log", "e80_o14_a6_progmap.log")
lora = pick("e80_o14_a6_lora_h0.log")
lora20 = pick("e80_o14_a20_lora_h0.log")
soft = pick("e80_o14_prompt_h0.log")
steer = pick("e80_o14_a6_steer_h0.log")
ax = fig.add_subplot(gs[1])
for src, kind, col, ls, lw, lab in ((base, "frozen", C["muted"], (0, (3, 1.5)), 1.4, "frozen"),
                                    (base, "map", C["blue"], "-", 1.8, "map, layer 6 (32,769 parameters)"),
                                    (late, "map", C["orange"], "-", 1.4, "same map at layer 20"),
                                    (lora, "map", C["violet"], "-", 1.4, "LoRA at layer 6 (32,768)"),
                                    (lora20, "map", C["violet"], (0, (3, 1.5)), 1.2, "same LoRA at layer 20"),
                                    (soft, "map", C["magenta"], "-", 1.4, "soft prompt (32,768)"),
                                    (steer, "map", C["yellow"], "-", 1.2, "steering vector (2,049)")):
    pts = sorted((t, v[kind][0]) for t, v in src.items() if kind in v)
    if pts:
        ax.plot([t for t, _ in pts], [e for _, e in pts], color=col, ls=ls, lw=lw, marker="o", ms=2.8, label=lab)
ax.axvline(4, color=C["muted"], lw=0.8, ls=":")
ax.set_xticks([2, 3, 4, 6, 8])
ax.set_xlabel("loops at inference")
ax.set_ylabel("exact match")
ax.legend(fontsize=5.4, loc="lower right", ncol=1)
ax.set_title("MuSiQue (same prompts in every run)", fontsize=8.6, loc="left")
panel_label(ax, "b")

ax = fig.add_subplot(gs[2])
T = 4
hops = [2, 3, 4]
n = {2: 142, 3: 105, 4: 53}
w = 0.36
f = [base[T]["frozen"][1][h] for h in hops]
m = [base[T]["map"][1][h] for h in hops]
x = np.arange(len(hops))
ax.bar(x - w / 2, f, w, color=C["muted"], label="frozen")
ax.bar(x + w / 2, m, w, color=C["blue"], label="map")
for i, h in enumerate(hops):
    se = np.sqrt(m[i] * (1 - m[i]) / n[h])
    ax.errorbar(x[i] + w / 2, m[i], yerr=se, color=C["ink2"], lw=0.8, capsize=2)
    se = np.sqrt(f[i] * (1 - f[i]) / n[h])
    ax.errorbar(x[i] - w / 2, f[i], yerr=se, color=C["ink2"], lw=0.8, capsize=2)
ax.set_xticks(x)
ax.set_xticklabels([f"{h} hops\n(n={n[h]})" for h in hops], fontsize=7)
ax.set_ylabel("exact match, four loops")
ax.set_ylim(0, 0.8)
ax.legend(fontsize=6.3, loc="upper right")
ax.set_title("exact match by hops, four loops", fontsize=8.6, loc="left")
panel_label(ax, "c")
plt.savefig(f"{OUT}/fig_downstream_loop.pdf")
plt.savefig(f"{OUT}/fig_downstream_loop.png")
print("saved")
