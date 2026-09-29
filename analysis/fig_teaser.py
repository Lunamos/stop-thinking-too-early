"""Figure 1 (first page): the problem, the fix and the gains.
(a) Accuracy against chain length on three-chain programs (the model's top choice among the three chains' root values; chance 1/3):
    thirteen standard base models (0.6B-32B, 16-64 layers; e10_panel_*.json), DeepSeek-V4-Flash-Base (292B; results/external/dsv4),
    and the looped Ouro-1.4B after 4 and 8 loops (e6_ouro_ouro14_nl_n300.json).
(b) The longest chain followed with at least 80% accuracy (two chains), frozen and with a rank-8 map at one layer: Qwen3-8B in one
    pass (exact accuracy for both bars: e33_order_r8kl_matrix.json frozen; map trained on up to 40 lines, e19_reentry_long_q8_r8_kl1.json),
    Ouro-1.4B after 4 and 8 loops (e71_ouro_o14_a6_long.json, e71b_scaling_o14_a6long_128.json), Huginn-0125 after 16 recurrences
    (e81_huginn_hug_r8.json frozen, e81_huginn_hug_long_r8.json).
    Ouro and Huginn use choice accuracy between the two chains' roots for both bars.
(c) MuSiQue exact-match gain over the frozen model with paired 95% bootstrap intervals, for a map at the earliest and at the latest
    layer tested (standard models: layers 6/30, 4/28, 4/28, 900 development questions, checks/musique_std.json; Ouro-1.4B: layers 6
    and 20 of the loop body, maps in every loop, four loops, 300 questions, checks/musique_loop.json).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS, OUT, RES  # noqa: E402
import numpy as np  # noqa: E402

from figstyle import C, panel_label, plt, setup  # noqa: E402

setup()


def reach(acc, thr=0.8):
    acc = dict(sorted(acc.items()))
    if min(acc) <= 2:
        acc.setdefault(1, 1.0)
        acc = dict(sorted(acc.items()))
    ds = list(acc)
    if acc[ds[0]] < thr:
        return 0.0, ""
    for a, b in zip(ds, ds[1:]):
        if acc[a] >= thr > acc[b]:
            return a + (acc[a] - thr) / (acc[a] - acc[b]) * (b - a), ""
    return float(ds[-1]), "ge"


fig = plt.figure(figsize=(12.6, 3.05))
gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[1.0, 1.05, 1.15])

# (a) the problem
ax = fig.add_subplot(gs[0])
FAM = {"q": C["blue"], "l": C["orange"], "o": C["aqua"], "g": C["violet"]}
for p in sorted(glob.glob(f"{RES}/e10_panel_*.json")):
    tag = os.path.basename(p)[len("e10_panel_"):-len(".json")]
    if tag.startswith(("c2_", "post_")):
        continue
    r = json.load(open(p))
    ds = sorted(int(d) for d in r["acc"])
    ax.plot(ds, [r["acc"][str(d)]["racc"] for d in ds], color=FAM[tag[0]], lw=0.9, alpha=0.55)
rows = [json.loads(line) for line in open(f"{RES}/external/dsv4/vb_dsv4_items.jsonl")]
dd = sorted({r["depth"] for r in rows})
acc = [np.mean([r["layer_cand_logit"][-1].index(max(r["layer_cand_logit"][-1])) == r["answer"] for r in rows if r["depth"] == d]) for d in dd]
ax.plot(dd, acc, color=C["ink"], lw=2.0, label="DeepSeek-V4-Flash (292B)")
cells = json.load(open(f"{RES}/e6_ouro_ouro14_nl_n300.json"))["cells"]
for T, ls in ((4, (0, (4, 1.5))), (8, (0, (1, 1.2)))):
    pts = sorted((c["depth"], c["racc"]) for c in cells if c["T"] == T)
    ax.plot([d for d, _ in pts], [a for _, a in pts], color=C["red"], lw=1.8, ls=ls, label=f"Ouro-1.4B, {T} loops")
for name, col in (("Qwen3 0.6-14B", C["blue"]), ("Llama-3 1-8B", C["orange"]), ("OLMo-3 7B, 32B", C["aqua"]), ("Gemma-3 4-27B", C["violet"])):
    ax.plot([], [], color=col, lw=0.9, alpha=0.7, label=name)
ax.axhline(1 / 3, color=C["muted"], lw=0.7, ls=":")
ax.text(1.05, 0.345, "chance", fontsize=6.5, color=C["muted"], ha="left", va="bottom")
ax.set_xlim(0.8, 8.2)
ax.set_ylim(0.2, 1.03)
ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
ax.set_xlabel("chain length (lines)")
ax.set_ylabel("accuracy (three chains)")
ax.legend(fontsize=5.9, loc="upper right", ncol=1)
ax.set_title("every model stops after about four lines", fontsize=8.8, loc="left")
panel_label(ax, "a")

# (b) the fix
ax = fig.add_subplot(gs[1])
bars = []
m = json.load(open(f"{RES}/e33_order_r8kl_matrix.json"))["res"]
fr = {int(k.split("_d")[1]): v["frozen"]["acc"] for k, v in m.items() if k.startswith("c2_forward_d")}
lm = {int(k[1:].split("_")[0]): v for k, v in json.load(open(f"{RES}/e19_reentry_long_q8_r8_kl1.json"))["eval"].items() if k.endswith("_K1")}
bars.append(("Qwen3-8B\none pass", reach(fr), reach(lm)))
ol = json.load(open(f"{RES}/e71_ouro_o14_a6_long.json"))["eval"]
o8 = dict({int(d): v[1] for d, v in ol["T8_map"].items()})
o8.update({int(d): v[1] for d, v in json.load(open(f"{RES}/e71b_scaling_o14_a6long_128.json"))["acc"]["T8_map"].items()})
ob = json.load(open(f"{RES}/e71_ouro_o14_a6.json"))["eval"]
bars.append(("Ouro-1.4B\n4 loops", reach({int(d): v[1] for d, v in ob["T4_frozen"].items()}),
             reach({int(d): v[1] for d, v in ol["T4_map"].items()})))
bars.append(("Ouro-1.4B\n8 loops", reach({int(d): v[1] for d, v in ob["T8_frozen"].items()}), reach(o8)))
hf = json.load(open(f"{RES}/e81_huginn_hug_r8.json"))["eval"]["r16_frozen"]
hl = json.load(open(f"{RES}/e81_huginn_hug_long_r8.json"))["eval"]["r16_map"]
bars.append(("Huginn\n16 recurrences", reach({int(d): v[1] for d, v in hf.items()}), reach({int(d): v[1] for d, v in hl.items()})))
x = np.arange(len(bars))
w = 0.36
for i, (name, (f0, _), (f1, flag)) in enumerate(bars):
    ax.bar(i - w / 2, max(f0, 0.8), w, color=C["muted"], alpha=0.55, label="frozen" if i == 0 else None)
    ax.bar(i + w / 2, f1, w, color=C["blue"], label="+ rank-8 map (under 0.01% of the weights)" if i == 0 else None)
    ax.text(i - w / 2, max(f0, 0.8) * 1.12, f"{f0:.1f}", ha="center", fontsize=6.8, color=C["ink2"])
    ax.text(i + w / 2, f1 * 1.12, ("≥" if flag == "ge" else "") + f"{f1:.0f}", ha="center", fontsize=7.5, color=C["ink"],
            fontweight="bold")
ax.set_yscale("log", base=2)
ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
ax.set_ylim(0.8, 2400)
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=7)
ax.set_ylabel("longest chain followed (lines)")
ax.legend(fontsize=6.3, loc="upper left")
ax.grid(axis="x", visible=False)
ax.set_title("one small map at one layer fixes it", fontsize=8.8, loc="left")
panel_label(ax, "b")

# (c) the gains: exact-match gain over the frozen model, paired 95% intervals, map at the earliest and at the latest layer tested
ax = fig.add_subplot(gs[2])
Q = json.load(open(f"{CHECKS}/musique_std.json"))["models"]
LQ = json.load(open(f"{CHECKS}/musique_loop.json"))
groups = []
for tag, name, early, late in (("q8", "Qwen3-8B", "map@6", "map@30"), ("o3", "OLMo-3-7B", "map@4", "map@28"),
                               ("l31", "Llama-3.1-8B", "map@4", "map@28")):
    a = Q[tag]["arms"]
    groups.append((name, Q[tag]["frozen_em"], a[early]["gain"], a[early]["ci"], a[late]["gain"], a[late]["ci"]))
e6, e20 = LQ["o14_a6_h0"]["T4"], LQ["o14_a20_h0"]["T4"]
groups.append(("Ouro-1.4B\n4 loops", e6["frozen"], e6["gain"], e6["ci"], e20["gain"], e20["ci"]))
x = np.arange(len(groups))
w = 0.34
for i, (name, f0, ge, ce, gl, cl) in enumerate(groups):
    ax.bar(i - w / 2, ge, w, color=C["blue"], label="+ map, early layer" if i == 0 else None)
    ax.bar(i + w / 2, gl, w, color=C["orange"], alpha=0.55, label="+ map, late layer" if i == 0 else None)
    for xx, g, c in ((i - w / 2, ge, ce), (i + w / 2, gl, cl)):
        ax.errorbar(xx, g, yerr=[[g - c[0]], [c[1] - g]], color=C["ink2"], lw=0.8, capsize=1.8)
    ax.text(i - w / 2, ce[1] + 0.7, f"+{ge:.1f}", ha="center", fontsize=7.3, fontweight="bold", color=C["ink"])
ax.axhline(0, color=C["muted"], lw=0.8)
ax.set_ylim(-7, 31)
ax.set_yticks([-5, 0, 5, 10, 15, 20, 25])
ax.set_xticks(x)
ax.set_xticklabels([f"{g[0]}\nfrozen {g[1]:.1f}" for g in groups], fontsize=6.6)
ax.set_ylabel("MuSiQue exact-match gain (points)")
ax.legend(fontsize=6.3, loc="upper center", ncol=2, handlelength=1.2, columnspacing=1.0)
ax.grid(axis="x", visible=False)
ax.set_title("and lifts multi-hop question answering", fontsize=8.8, loc="left")
panel_label(ax, "c")
plt.savefig(f"{OUT}/fig_teaser.pdf")
plt.savefig(f"{OUT}/fig_teaser.png")
for b in bars:
    print(b[0].replace("\n", " "), "frozen %.2f" % b[1][0], "map %.1f %s" % b[2])
for g in groups:
    print(g[0].replace("\n", " "), "frozen %.1f  early %+.1f %s  late %+.1f %s" % (g[1], g[2], g[3], g[4], g[5]))
