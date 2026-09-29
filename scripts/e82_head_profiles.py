"""Per-head stride profiles from E82 (Ouro) or E84 (Huginn) runs that saved same_full / other_full ([step, head, stride], held-out
programs). For the heads with the largest chain selectivity at strides >= 2, print their selectivity (same minus other) at each
stride in each loop. Pointer doubling predicts that a head's peak stride moves out from loop to loop; pooling over the whole chain
predicts a flat profile over strides; a relay with fixed offsets predicts the same peak in every loop.

usage: python e82_head_profiles.py RESULT_JSON [--mode map] [--top 6]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--mode", default="map")
ap.add_argument("--top", type=int, default=6)
args = ap.parse_args()
r = json.load(open(args.path))
L = r.get("L") or r.get("layers_per_rec")
S = np.array(r[args.mode]["same_full"])
O = np.array(r[args.mode]["other_full"])
D = S - O                                                     # [steps, H, J]
T = D.shape[0] // L
J = D.shape[2]
# score each (layer, head) by its best selectivity at strides >= 2 over loops
Dl = D.reshape(T, L, D.shape[1], J)                           # [T, L, H, J]
score = Dl[:, :, :, 1:].max(axis=(0, 3))                      # [L, H]
cand = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0][: args.top]
print(f"{args.path}  mode={args.mode}  loops={T}  layers/loop={L}  strides={J}")
for l, h in cand:
    print(f"layer {l:2d} head {h:2d}  (best selectivity at stride >= 2: {score[l, h]:.2f})")
    for t in range(T):
        prof = Dl[t, l, h]
        peak = int(prof.argmax()) + 1
        mass = prof.clip(min=0)
        mean_stride = float((mass * np.arange(1, J + 1)).sum() / max(mass.sum(), 1e-9))
        print(f"   loop {t + 1}: peak stride {peak:2d}  mean stride {mean_stride:5.2f}  total {mass.sum():.2f}  | "
              + " ".join(f"{x:+.2f}" for x in prof[:min(J, 16)]))
