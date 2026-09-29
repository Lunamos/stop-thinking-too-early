"""One band per pass. The relay front (the longest part of a chain whose lines a linear read-out at the pointer token assigns to
their chain with accuracy >= 0.75) against depth.
(a) Standard models by default: the furthest line the front reaches in one pass, against the number of layers (13 base models,
    two-chain programs of 8 lines; e32b_wave2.py with --a 0), and with a map (Qwen3-8B, Llama-3.1-8B; 16-line programs, the
    programs' ceiling).
(b) Qwen3-8B and Llama-3.1-8B: front after each layer, frozen and with the rank-8 map (layer 14 in Qwen3-8B, level and
    interleaved order; layer 11 in Llama-3.1-8B, its placement-sweep map), 16-line programs. Shaded: the blocks in which Qwen3-8B's mapped front advances
    past line 4. Dotted: Qwen3-8B's placement edge (rank-8 maps entering block 20 work, block 21 do not).
(c) Ouro-1.4B: front after every second layer of the first four loops (e77_ouro_relay_map.py), frozen and with maps at layer 6;
    shaded: layers 7-15 of each loop.
Prints the numbers quoted in the text.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, RES  # noqa: E402
import numpy as np  # noqa: E402

from figstyle import C, panel_label, plt, setup  # noqa: E402

setup()
THR = 0.75
NAMES = {"q06": "Qwen3-0.6B", "q17": "Qwen3-1.7B", "q4": "Qwen3-4B", "q8": "Qwen3-8B", "q14": "Qwen3-14B",
         "llama32_1b": "Llama-3.2-1B", "llama32_3b": "Llama-3.2-3B", "llama31_8b": "Llama-3.1-8B", "olmo3_7b": "OLMo-3-7B",
         "olmo3_32b": "OLMo-3-32B", "gemma3_4b": "Gemma-3-4B", "gemma3_12b": "Gemma-3-12B", "gemma3_27b": "Gemma-3-27B"}
FAMILY = {"Qwen": C["blue"], "Llam": C["orange"], "OLMo": C["aqua"], "Gemm": C["violet"]}


def block_front(grid):
    """front after each block: longest prefix of lines 2, 3, ... whose read-out is >= THR (line 1 counts as known)"""
    L = len(grid[2])
    out = []
    for j in range(L):
        k = 1
        for ln in range(2, len(grid)):
            if not grid[ln] or grid[ln][j] < THR:
                break
            k = ln
        out.append(k)
    return np.array(out)


def wave(name, mode):
    return block_front(json.load(open(f"{RES}/{name}"))[mode]["grids"]["rhs_slot"])


fig = plt.figure(figsize=(12.6, 3.3))
gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[0.8, 1.1, 1.1])

# (a) the default front in thirteen standard models
ax = fig.add_subplot(gs[0])
rows = []
for p in sorted(glob.glob(f"{RES}/e32b_wave_relay_*.json")):
    tag = os.path.basename(p)[len("e32b_wave_relay_"):-len(".json")]
    f = wave(os.path.basename(p), "frozen")
    rows.append((NAMES[tag], len(f), int(f.max()), int(np.argmax(f))))
seen = {}
for name, L, mx, _ in sorted(rows, key=lambda r: (r[1], r[0])):
    n = seen.get((L, mx), 0)                                  # models with the same depth and front: spread vertically
    seen[(L, mx)] = n + 1
    ax.scatter([L], [mx + 0.25 * n], s=26, color=FAMILY[name[:4]], zorder=3)
for name, col in (("Qwen3", C["blue"]), ("Llama-3", C["orange"]), ("OLMo-3", C["aqua"]), ("Gemma-3", C["violet"])):
    ax.scatter([], [], s=26, color=col, label=f"{name}, frozen (8-line programs)")
for name, L, f, col in (("Qwen3-8B", 36, wave("e32b_wave_q8r8kl_d16.json", "map"), C["blue"]),
                        ("Llama-3.1-8B", 32, wave("e32b_wave_l31r8_a11_d16.json", "map"), C["orange"])):
    ax.scatter([L], [f.max()], s=46, marker="*", color=col, zorder=4)
ax.scatter([], [], s=46, marker="*", color=C["ink2"], label="with a map (16-line programs)")
ax.set_xscale("log", base=2)
ax.set_xticks([16, 24, 32, 48, 64])
ax.set_xticklabels(["16", "24", "32", "48", "64"])
ax.set_ylim(0, 17.5)
ax.set_yticks([1, 2, 3, 4, 5, 8, 12, 16])
ax.set_xlabel("layers")
ax.set_ylabel("furthest line reached by the relay")
ax.legend(fontsize=5.8, loc="center left", bbox_to_anchor=(0.0, 0.62))
ax.set_title("by default the relay stops at line 3 to 5", fontsize=8.6, loc="left")
panel_label(ax, "a")

# (b) Qwen3-8B and Llama-3.1-8B, block by block
ax = fig.add_subplot(gs[1])
q_map, q_int, q_fr = wave("e32b_wave_q8r8kl_d16.json", "map"), wave("e32b_wave_q8r8kl_d16_inter.json", "map"), wave("e32b_wave_q8r8kl_d16.json", "frozen")
l_map, l_fr = wave("e32b_wave_l31r8_a11_d16.json", "map"), wave("e32b_wave_l31r8_a11_d16.json", "frozen")
adv = [j for j in range(1, len(q_map)) if q_map[j] > q_map[j - 1] and q_map[j] > 4]
b0, b1 = min(adv), max(adv)                                   # blocks whose output moves the front past line 4
ax.axvspan(b0 - 0.5, b1 + 0.5, color=C["band2"], lw=0, zorder=0)
x = np.arange(len(q_map))
ax.step(x, q_fr, where="mid", color=C["blue"], ls=(0, (3, 1.5)), lw=1.3, label="Qwen3-8B, frozen")
ax.step(x, q_map, where="mid", color=C["blue"], lw=1.7, label="Qwen3-8B, map at layer 14")
ax.step(x, q_int, where="mid", color=C["blue"], lw=1.0, alpha=0.55, label="same, interleaved chains")
xl = np.arange(len(l_map))
ax.step(xl, l_fr, where="mid", color=C["orange"], ls=(0, (3, 1.5)), lw=1.3, label="Llama-3.1-8B, frozen")
ax.step(xl, l_map, where="mid", color=C["orange"], lw=1.7, label="Llama-3.1-8B, map at layer 11")
for a, col in ((14, C["blue"]), (11, C["orange"])):
    ax.plot([a - 0.5], [0.35], marker="^", color=col, ms=6, clip_on=False, zorder=5)
ax.axvline(20, color=C["ink2"], lw=0.9, ls=":", zorder=1)          # maps entering block 20 work, block 21 do not (rank 8)
ax.text(20.3, 1.2, "Qwen3-8B\nplacement\nedge", fontsize=5.8, va="bottom", ha="left", color=C["ink2"], linespacing=1.1)
ax.set_xlim(-0.5, 35.5)
ax.set_ylim(0, 17.8)
ax.set_xticks([0, 6, 12, 18, 24, 30, 35])
ax.set_yticks([1, 2, 4, 8, 12, 16])
ax.set_xlabel("layer (triangles: where the map acts)")
ax.set_ylabel("furthest line reached (lines)")
ax.legend(fontsize=5.8, loc="upper left")
ax.set_title("with the map, the relay runs in a few middle layers", fontsize=8.6, loc="left")
panel_label(ax, "b")

# (c) Ouro-1.4B, loop by loop
ax = fig.add_subplot(gs[2])


def loop_front(path, T=4):
    r = json.load(open(f"{RES}/{path}"))
    steps, L, g = r["steps"], r["L"], r["grid"]
    lines = sorted(int(k) for k in g)
    xs, ys = [], []
    for i, s_ in enumerate(steps):
        if s_ >= T * L:
            break
        k = 1
        for ln in lines:
            if g[str(ln)][i] < THR:
                break
            k = ln
        xs.append(s_ + 1)
        ys.append(k)
    return np.array(xs), np.array(ys)


L_ = 24
for t in range(4):
    ax.axvspan(t * L_ + 7, t * L_ + 16, color=C["band2"], lw=0, zorder=0)          # layers 7-15 (0-based) of loop t
    if t:
        ax.axvline(t * L_, color=C["muted"], lw=0.6, ls=":")
for path, col, ls, lab in (("e77_ouro_relay_o14_a6long_d40_frozen.json", C["muted"], (0, (3, 1.5)), "frozen"),
                           ("e77_ouro_relay_o14_a6_map.json", C["blue"], "-", "map at layer 6 (16-line programs)"),
                           ("e77_ouro_relay_o14_a6_first_map.json", C["aqua"], "-", "same, first loop only"),
                           ("e77_ouro_relay_o14_a6long_d40_map.json", C["violet"], "-", "map trained on up to 40 lines")):
    xs, ys = loop_front(path)
    ax.step(xs, ys, where="post", color=col, ls=ls, lw=1.5, label=lab)
ax.set_xlim(0, 96)
ax.set_ylim(0, 41)
ax.set_xticks([12, 36, 60, 84])
ax.set_xticklabels(["loop 1", "loop 2", "loop 3", "loop 4"])
ax.tick_params(axis="x", length=0)
ax.set_xlabel("layers applied (shaded: layers 7-15 of each loop)")
ax.set_ylabel("furthest line reached (lines)")
ax.legend(fontsize=5.8, loc="upper left")
ax.set_title("in a looped model, in the same layers of every loop", fontsize=8.6, loc="left")
panel_label(ax, "c")

plt.savefig(f"{OUT}/fig_band.pdf")
plt.savefig(f"{OUT}/fig_band.png")

print("default front, 13 standard models (name, layers, furthest line, first block reaching it):")
for r in sorted(rows, key=lambda r: r[1]):
    print("  ", r)
print("Qwen3-8B map: front passes line 4 in blocks", b0, "to", b1, "; front per block:",
      " ".join(f"{j}:{v}" for j, v in enumerate(q_map) if j == 0 or v != q_map[j - 1]))
print("Qwen3-8B interleaved:", " ".join(f"{j}:{v}" for j, v in enumerate(q_int) if j == 0 or v != q_int[j - 1]))
print("Llama-3.1-8B map:", " ".join(f"{j}:{v}" for j, v in enumerate(l_map) if j == 0 or v != l_map[j - 1]))


def widest_read(name, thr=0.1):
    """furthest line up the chain that each layer attends to: largest j with same-chain minus other-chain attention >= thr"""
    D = np.array(json.load(open(f"{RES}/{name}"))["map"]["diff"])
    return [max([j + 1 for j in range(D.shape[1]) if D[L, j] >= thr], default=0) for L in range(D.shape[0])]


for order, fr in (("level", q_map), ("interleaved", q_int)):
    w = widest_read(f"e37b_strides_q8r8kl_d16_{'level' if order == 'level' else 'inter'}.json")
    print(f"Qwen3-8B {order}: layer: widest read / front advance:",
          " ".join(f"{L}:{w[L]}/{fr[L] - fr[L - 1]}" for L in range(15, 25)))
