"""Paper 2, appendix figure: the same relay in looped transformers trained from scratch (e60_arch_reach.py, loops of a two-layer
block, adaptive curriculum). Top: (a) one model (eight layers, training chains of up to 16 lines): relay front and each block's widest
chain-selective read; (b) front advance per step against that step's read width, every finished run (toy layers) and Ouro-1.4B's long
map inside its band (two-layer steps; width at thresholds .1 and .05 as a range); (c) names defined 3-5 levels up, decoded at the
pointer token per layer, unmasked and with the parent reads removed at layer 3 only (every later name one layer late).
Bottom, grid over depth and training length (two seeds): (d) longest chain the curriculum reached; (e) widest read; (f) accuracy lost
when the reads two or more levels up are removed (6-line chains for 8-line training, 12-line for 16).
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401
import numpy as np  # noqa: E402

from figstyle import C, panel_label, plt, setup  # noqa: E402

setup()


def toy_front(tag, d):
    m = json.load(open(f"{RES}/e60_mech_{tag}.json"))
    if str(d) not in m:
        return None
    fr = {int(k): v for k, v in m[str(d)]["front"].items()}
    L = len(m[str(d)]["relay_acc"])
    per = []
    for layer in range(L):
        k = 1
        for lev in sorted(fr):
            if lev == 1:
                continue
            if fr[lev] is not None and fr[lev] <= layer:
                k = lev
            else:
                break
        per.append(k)
    return per


def toy_widths(tag, thr=0.1):
    sp = sorted(glob.glob(f"{RES}/e60_strides_{tag}_d*.json"))
    if not sp:
        return None
    st = json.load(open(sp[-1]))
    ri = st["roles"].index("rhs")
    return [max([j + 1 for j, x in enumerate(layer[ri]) if x >= thr], default=0) for layer in st["diff"]]


def finished():
    out = []
    for p in sorted(glob.glob(f"{RES}/e60_arch_p3*_loop_L*_d*.json")):
        t = os.path.basename(p)[len("e60_arch_"):-len(".json")]
        if "p3w_" in t:
            continue
        out.append(t)
    return out


fig = plt.figure(figsize=(12.4, 6.6))
gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.5)

# (a)
ax = fig.add_subplot(gs[0, 0])
ex = "p3_loop_L8_d16"
per, w = toy_front(ex, 16), toy_widths(ex)
if per:
    ax.step(range(len(per)), per, where="post", color=C["violet"], lw=1.6, label="furthest line reached (layer input)")
if w:
    ax.bar(np.arange(len(w)) + 0.5, w, width=0.5, color=C["aqua"], alpha=0.7, label="furthest attention of the layer")
ax.set_xlabel("layer")
ax.set_ylabel("lines")
ax.legend(fontsize=6.3, loc="upper left")
ax.set_title("eight layers, chains of up to 16 lines", fontsize=8.6, loc="left")
panel_label(ax, "a")

# (b)
ax = fig.add_subplot(gs[0, 1])
xs, ys = [], []
for t in finished():
    d = int(re.search(r"_d(\d+)", t).group(1))
    if not os.path.exists(f"{RES}/e60_mech_{t}.json"):
        continue
    dd = 16 if d >= 16 else 8
    per = toy_front(t, dd)
    if per is None:                                        # grid runs were analysed on 8-line programs
        dd, per = 8, toy_front(t, 8)
    w = toy_widths(t)
    if not per or not w:
        continue
    for layer in range(1, min(len(per), len(w) + 1)):      # the front is read at each block's input: block layer-1 moves it
        if per[layer - 1] >= dd - 1:
            break
        xs.append(w[layer - 1])
        ys.append(per[layer] - per[layer - 1])
n_exact = sum(y == x for x, y in zip(xs, ys))
n_more = sum(y > x for x, y in zip(xs, ys))
print(f"toy layer steps: {len(xs)}; front advance equal to the widest read in {n_exact}, larger in {n_more}, smaller in "
      f"{len(xs) - n_exact - n_more}")
if xs:
    jit = np.random.default_rng(0).uniform(-0.12, 0.12, size=(2, len(xs)))
    ax.scatter(np.array(xs) + jit[0], np.array(ys) + jit[1], s=11, color=C["blue"], alpha=0.6, label=f"trained from scratch ({len(xs)} layers)")
pr, ps = f"{RES}/e77_ouro_relay_o14_a6long_d40_map.json", f"{RES}/e82_strides_o14_a6long_d40.json"
if os.path.exists(pr) and os.path.exists(ps):
    fr = json.load(open(pr))
    st = json.load(open(ps))
    L, steps, g = fr["L"], fr["steps"], fr["grid"]
    lines = sorted(int(k) for k in g)
    S = (np.array(st["map"]["same_full"]) - np.array(st["map"]["other_full"])).max(1)

    def front(i):
        k = 1
        for ln in lines:
            if g[str(ln)][i] < 0.75:
                break
            k = ln
        return k

    def width(s, thr):
        return max([j + 1 for j, x in enumerate(S[s]) if x >= thr], default=0)
    for t in range(3):
        idx = [i for i, s_ in enumerate(steps) if s_ // L == t]
        for a, b in zip(idx, idx[1:]):
            s0, s1 = steps[a], steps[b]
            if not (5 <= s0 % L and s1 % L <= 15):
                continue
            w1 = max(width(s, 0.1) for s in range(s0 + 1, s1 + 1))
            w2 = max(width(s, 0.05) for s in range(s0 + 1, s1 + 1))
            adv = front(b) - front(a)
            ax.plot([w1, w2], [adv, adv], color=C["orange"], lw=1.1)
            ax.scatter([w1], [adv], s=13, color=C["orange"], zorder=3)
    ax.scatter([], [], s=13, color=C["orange"], label="Ouro-1.4B long map, steps in the middle layers")
ax.plot([0, 8], [0, 8], color=C["muted"], lw=0.8, ls=(0, (3, 2)), label="advance = width")
ax.set_xlim(-0.5, 8)
ax.set_ylim(-0.5, 8)
ax.set_xlabel("furthest attention in the step (lines up the chain)")
ax.set_ylabel("lines gained in the step")
ax.legend(fontsize=6.0, loc="upper left")
ax.set_title("the relay moves as far as attention reaches", fontsize=8.6, loc="left")
panel_label(ax, "b")

# (c)
ax = fig.add_subplot(gs[0, 2])
p0, p1 = f"{RES}/e60_names_{ex}.json", f"{RES}/e60_names_{ex}_parent_L3.json"
if os.path.exists(p0) and os.path.exists(p1):
    r0 = list(json.load(open(p0)).values())[0]
    r1 = list(json.load(open(p1)).values())[0]
    for j, col in ((2, C["aqua"]), (3, C["yellow"]), (4, C["orange"])):          # index j -> j+1 levels up
        ax.plot(range(len(r0)), [r[j] for r in r0], color=col, lw=1.6, marker="o", ms=2.8, label=f"{j + 1} lines up")
        ax.plot(range(len(r1)), [r[j] for r in r1], color=col, lw=1.1, ls=(0, (3, 1.5)))
    ax.plot([], [], color=C["ink2"], lw=1.1, ls=(0, (3, 1.5)), label="attention to the parent line removed at layer 3")
    ax.axvline(3, color=C["muted"], lw=0.7, ls=":")
ax.axhline(1 / 64, color=C["muted"], lw=0.6, ls=":")
ax.set_xlabel("layer")
ax.set_ylabel("name decoded at the pointer token")
ax.legend(fontsize=6.0, loc="upper left")
ax.set_title("blocking the parent line delays the next name", fontsize=8.6, loc="left")
panel_label(ax, "c")

# (d)-(f) grid
R = []
for p in sorted(glob.glob(f"{RES}/e60_arch_p3g_loop_L*_d*_s*.json")):
    tag = os.path.basename(p)[len("e60_arch_"):-len(".json")]
    L_, d_, s_ = (int(x) for x in re.search(r"L(\d+)_d(\d+)_s(\d+)", tag).groups())
    reached = 1
    for line in open(f"{RES}/e60_{tag}.log"):
        mm = re.match(r"step (\d+): curriculum -> dmax (\d+)", line)
        if mm:
            reached = int(mm.group(2))
    w = toy_widths(tag)
    reach2 = json.load(open(p)).get("reach", {}).get("reach_level_c2", 0)
    cost = None
    mk = f"{RES}/e60_masks_{tag}.json"
    if os.path.exists(mk):
        mm_ = json.load(open(mk))
        dd = "6" if d_ == 8 else "12"
        if dd in mm_["none"]:
            cost = mm_["none"][dd] - mm_["far"][dd]
    R.append(dict(L=L_, d=d_, s=s_, reached=reached, reach2=reach2, wmax=max(w) if w else None, cost=cost))
cols = {8: C["blue"], 16: C["orange"]}
mks = {1: "o", 2: "s"}
for pi, (key, ylab, title) in enumerate((("reached", "longest chain learned (lines)", "what a depth can learn"),
                                          ("wmax", "widest read (lines back)", "reads widen when depth is short"),
                                          ("cost", "accuracy lost without far reads", "far reads matter where reads widened"))):
    ax = fig.add_subplot(gs[1, pi])
    for r in R:
        if r[key] is None:
            continue
        learned = r["reached"] >= r["d"] and r["reach2"] >= r["d"] - 1
        if key == "cost" and not learned:
            continue                                    # a model that never learned its chains is at chance there anyway
        ax.scatter(r["L"] + (r["s"] - 1.5) * 0.3, r[key], marker=mks[r["s"]], s=24, zorder=3,
                   facecolors=cols[r["d"]] if learned else "none", edgecolors=cols[r["d"]], linewidths=1.0)
    for d_, col in cols.items():
        ax.scatter([], [], color=col, s=24, label=f"training chains up to {d_} lines")
        if key == "reached":
            ax.axhline(d_, color=col, lw=0.7, ls=":")
    ax.set_xticks([4, 6, 8, 12, 16])
    if key == "wmax":
        ax.set_yticks(range(0, 9))
    ax.set_xlabel("depth (layers; loops of a two-layer module)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=8.6, loc="left")
    if pi == 0:
        ax.scatter([], [], facecolors="none", edgecolors=C["ink2"], s=24, label="did not learn its training chains")
        ax.legend(fontsize=6.0, loc="lower right")
    panel_label(ax, "def"[pi])
plt.savefig(f"{OUT}/fig_toy.pdf")
plt.savefig(f"{OUT}/fig_toy.png")
print("runs", len(R), "points", len(xs))
