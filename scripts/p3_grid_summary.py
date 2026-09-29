"""Paper 3: summary of the toy grid (looped 2-layer blocks trained from scratch with an adaptive curriculum, e60_arch_reach.py).

For every run: the step at which the curriculum reached its maximal chain length (training chains of up to dmax lines), reach on
two and three chains, the relay front per layer (the longest prefix of the queried chain whose levels are read out at >= .75, from
--analyze), how many lines the front advances at each layer, and the longest chain-selective read at each layer (same-minus-other
attention >= .1 at the pointer (rhs) and newline tokens, from --strides). The widening account predicts that the front advances by
about as many lines at a layer as that layer reads back, and that runs whose training chains are long relative to their depth read
further back.

usage: python p3_grid_summary.py [--tags p3g_loop_L4_d8_s1,...] (default: every e60_arch_p3g_* and p3_* run found)
"""
import argparse
import glob
import json
import os
import re

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="")
ap.add_argument("--thr", type=float, default=0.1)
args = ap.parse_args()


def tags_found():
    out = []
    for p in sorted(glob.glob(f"{RES}/e60_arch_p3*_loop_L*_d*.json")):
        out.append(os.path.basename(p)[len("e60_arch_"):-len(".json")])
    return out


def meta(tag):
    m = re.search(r"L(\d+)_d(\d+)", tag)
    return int(m.group(1)), int(m.group(2))


def solved_step(tag, dmax):
    for name in (f"{RES}/e60_{tag}.log", f"{RES}/e60_{tag.replace('p3_', 'p3_')}.log"):
        if os.path.exists(name):
            last = None
            for line in open(name):
                mm = re.match(r"step (\d+): curriculum -> dmax (\d+)", line)
                if mm:
                    last = (int(mm.group(1)), int(mm.group(2)))
            if last is None:
                return None, 1
            return (last[0] if last[1] >= dmax else None), last[1]
    return None, None


def front_per_layer(mech, L):
    """{d: [front after layer 0..L-1]} from the level -> first labelled layer map"""
    out = {}
    for d, v in mech.items():
        fr = {int(k): x for k, x in v.get("front", {}).items()}
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
        out[int(d)] = per
    return out


def widths(strides, role, thr):
    ri = strides["roles"].index(role)
    return [max([j + 1 for j, x in enumerate(layer[ri]) if x >= thr], default=0) for layer in strides["diff"]]


rows = []
for tag in (args.tags.split(",") if args.tags else tags_found()):
    L, dmax = meta(tag)
    arch = f"{RES}/e60_arch_{tag}.json"
    if not os.path.exists(arch):
        continue
    r = json.load(open(arch))
    step, reached = solved_step(tag, dmax)
    line = [f"{tag:24s} L={L:2d} dmax={dmax:2d} solved@{step} (reached {reached})",
            f"reach c2 {r.get('reach', r).get('reach_level_c2')} c3 {r.get('reach', r).get('reach_level_c3')}"]
    mp = f"{RES}/e60_mech_{tag}.json"
    fr = None
    if os.path.exists(mp):
        fr = front_per_layer(json.load(open(mp)), L)
        d = max(fr)
        per = fr[d]
        adv = [per[0] - 1] + [b - a for a, b in zip(per, per[1:])]
        line.append(f"front(d={d}) {per} advance {adv}")
    sp = sorted(glob.glob(f"{RES}/e60_strides_{tag}_d*.json"))
    if sp:
        st = json.load(open(sp[-1]))
        line.append(f"reads(rhs) {widths(st, 'rhs', args.thr)} reads(nl) {widths(st, 'nl', args.thr)}")
    rows.append("\n    ".join(line))
print("\n".join(rows))
