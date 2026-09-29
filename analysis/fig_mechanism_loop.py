"""Paper 2, mechanism figure. (a, b) Chain-selective attention at each line's pointer token (attention to the line j levels up in the
same chain minus attention to the other chain's line at that level; best head per step and stride chosen on held-out programs;
maximum over the layers of a loop), Ouro-1.4B, 24-line two-chain programs, frozen and with the layer-6 map. (c) Read-out of which
chain each line of the queried chain belongs to (best in each loop), with the map in the first loop only: loops 2-4 run the
unmodified weights. (d) The reading window of single heads: mean distance (lines back, weighted by same-minus-other attention) per
loop, for the Ouro and Huginn heads with the largest selectivity beyond one line, with the map and frozen."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401
import numpy as np  # noqa: E402

from figstyle import C, panel_label, plt, setup  # noqa: E402

setup()
fig = plt.figure(figsize=(12.4, 6.6))
gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.52)


def per_loop(r, mode):
    L = r.get("L") or r.get("layers_per_rec")
    D = np.array(r[mode]["diff"])
    T = D.shape[0] // L
    return np.stack([D[t * L:(t + 1) * L].max(0) for t in range(T)])          # [T, J]


def heat(ax, M, title, label, jmax=12):
    T = M.shape[0]
    M = M[:, :jmax]
    im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=0.6, extent=(0.5, T + 0.5, 0.5, jmax + 0.5))
    ax.set_xlabel("loop")
    ax.set_ylabel("lines up the chain")
    ax.set_xticks(range(1, T + 1))
    ax.grid(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("same minus other chain", fontsize=6.5)
    ax.set_title(title, fontsize=8.6, loc="left")
    panel_label(ax, label)


r24 = json.load(open(f"{RES}/e82_strides_o14_a6_d24.json"))
heat(fig.add_subplot(gs[0, 0]), per_loop(r24, "frozen"), "frozen: one line back, every loop", "a")
heat(fig.add_subplot(gs[0, 1]), per_loop(r24, "map"), "with the map: reads reach further", "b")

# (d) relay with the map in the first loop only
ax = fig.add_subplot(gs[1, 0])
r = json.load(open(f"{RES}/e77_ouro_relay_o14_a6_first_map.json"))
steps, L = r["steps"], r["L"]
lines = sorted(int(k) for k in r["grid"])
M = np.array([[max(r["grid"][str(k)][i] for i, s in enumerate(steps) if s // L == t) for t in range(r["T"])] for k in lines])
im = ax.imshow(M, aspect="auto", origin="lower", cmap="Blues", vmin=0.5, vmax=1.0,
               extent=(0.5, r["T"] + 0.5, lines[0] - 0.5, lines[-1] + 0.5))
ax.axvline(1.5, color=C["orange"], lw=1.2, ls="--")
ax.text(1.0, lines[-1] - 0.2, "map", ha="center", va="top", fontsize=6.5, color=C["orange"])
ax.text(3.0, lines[-1] - 0.2, "unmodified loops", ha="center", va="top", fontsize=6.5, color=C["orange"])
ax.set_xlabel("loop")
ax.set_ylabel("line of the queried chain")
ax.set_xticks(range(1, r["T"] + 1))
ax.grid(False)
cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.ax.tick_params(labelsize=6.5)
cb.set_label("chain read-out (best in loop)", fontsize=6.5)
ax.set_title("map in the first loop only: later loops relay", fontsize=8.6, loc="left")
panel_label(ax, "d")

# (c) which heads read two or more lines back: far-read mass (same-minus-other attention summed over strides >= 2) per loop
ax = fig.add_subplot(gs[0, 2])


def far_mass(r, mode, l, h):
    L = r.get("L") or r.get("layers_per_rec")
    S = np.array(r[mode]["same_full"]) - np.array(r[mode]["other_full"])
    T = S.shape[0] // L
    return [float(S[t * L + l, h, 1:].clip(min=0).sum()) for t in range(T)]


def top_far(r, k):
    L = r.get("L") or r.get("layers_per_rec")
    S = np.array(r["map"]["same_full"]) - np.array(r["map"]["other_full"])
    T = S.shape[0] // L
    far = S[:, :, 1:].clip(min=0).sum(2).reshape(T, L, S.shape[1]).max(0)
    idx = np.dstack(np.unravel_index(np.argsort(-far, axis=None), far.shape))[0][:k]
    return [(int(l), int(h)) for l, h in idx]


def frozen_one_back(r, k):
    L = r.get("L") or r.get("layers_per_rec")
    S = np.array(r["frozen"]["same_full"]) - np.array(r["frozen"]["other_full"])
    T = S.shape[0] // L
    s1 = S.reshape(T, L, S.shape[1], S.shape[2])[:, :, :, 0].max(0)
    idx = np.dstack(np.unravel_index(np.argsort(-s1, axis=None), s1.shape))[0][:k]
    return [(int(l), int(h)) for l, h in idx]


for (l, h), col in zip(top_far(r24, 3), [C["blue"], C["violet"], C["magenta"]]):
    ax.plot(range(1, 7), far_mass(r24, "map", l, h), color=col, lw=1.6, marker="o", ms=3.0, label=f"Ouro L{l} H{h} (furthest attention)")
    ax.plot(range(1, 7), far_mass(r24, "frozen", l, h), color=col, lw=0.9, ls=(0, (3, 1.5)))
for (l, h), col in zip(frozen_one_back(r24, 3), [C["orange"], C["yellow"], C["red"]]):
    ax.plot(range(1, 7), far_mass(r24, "map", l, h), color=col, lw=1.4, marker="s", ms=2.8, label=f"Ouro L{l} H{h} (parent-line head)")
pH = f"{RES}/e84_huginn_strides_hug_r8_d16.json"
if os.path.exists(pH):
    rh = json.load(open(pH))
    (l, h) = frozen_one_back(rh, 1)[0]
    fm = far_mass(rh, "map", l, h)
    ax.plot(range(1, len(fm) + 1), fm, color=C["aqua"], lw=1.4, marker="^", ms=2.8, label=f"Huginn L{l} H{h} (parent-line head)")
    ax.plot(range(1, len(fm) + 1), far_mass(rh, "frozen", l, h), color=C["aqua"], lw=0.9, ls=(0, (3, 1.5)))
ax.plot([], [], color=C["ink2"], lw=0.9, ls=(0, (3, 1.5)), label="dashed: same head, frozen")
ax.set_xticks([1, 2, 3, 4, 6, 8, 10, 12])
ax.set_xlabel("loop (Ouro) or recurrence (Huginn)")
ax.set_ylabel("attention 2+ lines up (same minus other)")
ax.legend(fontsize=5.2, loc="upper left")
ax.set_title("which heads attend further up the chain", fontsize=8.6, loc="left")
panel_label(ax, "c")

# (e) ancestor names at lines the chain has not reached yet (loop 2)
ax = fig.add_subplot(gs[1, 1])


def beyond(probe, relay, loop=1, thr=0.75):
    r = json.load(open(f"{RES}/{probe}"))
    T, REC = r["args"]["T"], r["layers"]
    if relay is None:
        P = [3] * T
    else:
        e = json.load(open(f"{RES}/{relay}"))
        steps, L = e["steps"], e["L"]
        P = []
        for t in range(T):
            n = 1
            for k in sorted(int(x) for x in e["grid"]):
                if max(e["grid"][str(k)][i] for i, s_ in enumerate(steps) if s_ // L <= t) >= thr:
                    n = k
                else:
                    break
            P.append(n)
    xs, ys = [], []
    for j in sorted(int(x) for x in r["acc_by_line"]):
        if j < 2:
            continue
        blk = r["acc_by_line"][str(j)]
        A = np.array(blk["by_step"])
        best = A[loop * len(REC):(loop + 1) * len(REC)].max(0)
        ks = np.arange(blk["first_line"], blk["first_line"] + best.shape[0])
        sel = best[ks > P[loop]]
        if sel.size:
            xs.append(j)
            ys.append(sel.mean())
    return xs, ys


for probe, relay, col, lab, mk in (("e87_ancestor_o14_frozen_byline.json", None, C["muted"], "frozen", "o"),
                                   ("e87_ancestor_o14_a6_byline.json", "e77_ouro_relay_o14_a6_map.json", C["blue"], "map, every loop", "o"),
                                   ("e87_ancestor_o14_a6first_byline.json", "e77_ouro_relay_o14_a6_first_map.json", C["aqua"],
                                    "map in loop 1 only (loop 2 unmodified)", "s")):
    if os.path.exists(f"{RES}/{probe}"):
        xs, ys = beyond(probe, relay)
        ax.plot(xs, ys, color=col, lw=1.6, marker=mk, ms=3.2, label=lab)
ax.axhline(1 / 52, color=C["ink2"], lw=0.7, ls=":")
ax.text(7.9, 1 / 52 + 0.015, "chance", fontsize=6.3, color=C["ink2"], ha="right")
ax.set_xlabel("earlier line (lines up the chain)")
ax.set_ylabel("name decoded (lines not yet reached)")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=6.0, loc="upper right")
ax.set_title("names arrive before the chain (loop 2)", fontsize=8.6, loc="left")
panel_label(ax, "e")

# (f) which reads the relay needs
ax = fig.add_subplot(gs[1, 2])


def reach_(acc, thr=0.8):
    acc = dict(sorted(acc.items()))
    ds = list(acc)
    if acc[ds[0]] < thr:
        return 0.0
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0)
    return float(ds[-1])


ch = json.load(open(f"{RES}/e85_stride_block_o14_a6_nr.json"))["acc"]            # weights zeroed after the softmax
chr_ = json.load(open(f"{RES}/e85_stride_block_o14_a6_chain.json"))["acc"]      # renormalizing masks, for comparison
conds = [("none", "nothing"), ("rand_far", "random lines, 2+ up"), ("other_far", "other chain, 2+ up"),
         ("long", "both chains, 2+ up"), ("same_far", "own chain, 2+ up"), ("near_other", "other chain, 1 up"),
         ("near_same", "own parent")]
vals_ = [reach_({int(d): v[1] for d, v in ch[f"T3_{c}"].items()}) for c, _ in conds]
y = np.arange(len(conds))[::-1]
cols = [C["blue"] if c == "none" else (C["orange"] if c == "long" else C["muted"]) for c, _ in conds]
vr = [reach_({int(d): v[1] for d, v in chr_[f"T3_{c}"].items()}) for c, _ in conds]
ax.barh(y, vals_, color=cols, height=0.62, label="weights zeroed (no renormalization)")
ax.scatter(vr, y, marker="|", s=90, color=C["ink"], zorder=4, label="renormalizing mask")
for yy, v, w in zip(y, vals_, vr):
    ax.text(max(v, w) + 0.4, yy, f"{v:.1f}", va="center", fontsize=6.8, color=C["ink2"])
ax.legend(fontsize=5.6, loc="lower right")
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab in conds], fontsize=7)
ax.set_xlabel("reach after three loops (lines)")
ax.set_xlim(0, 22)
ax.grid(axis="y", visible=False)
ax.set_title("attention removed from each line", fontsize=8.6, loc="left")
panel_label(ax, "f", x=-0.42)

plt.savefig(f"{OUT}/fig_mechanism_loop.pdf")
plt.savefig(f"{OUT}/fig_mechanism_loop.png")
print("fig4 saved")
