"""Split the E87 ancestor-name probe by the labelled prefix: for each loop t, lines k inside the prefix that the chain read-out (E77)
has labelled by the end of loop t, and lines beyond it. If lines gather their ancestors' names before their chain arrives (a cause
of the relay's speed), the names j >= 3 levels up should be decodable beyond the prefix; if names only follow labels, they should be
decodable inside it only.

usage: python e87_split_prefix.py RESULT_JSON E77_RELAY_JSON [--thr 0.75]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("probe")
ap.add_argument("relay", nargs="?", default=None)
ap.add_argument("--thr", type=float, default=0.75)
ap.add_argument("--prefix", default=None, help="comma list of labelled prefix per loop, instead of an E77 file")
args = ap.parse_args()
r = json.load(open(args.probe))
T, REC = r["args"]["T"], r["layers"]
D = r["args"]["depth"]
if args.prefix:
    P = [int(x) for x in args.prefix.split(",")]
else:
    e = json.load(open(args.relay))
    steps, L = e["steps"], e["L"]
    P = []
    for t in range(T):
        n = 1
        for k in sorted(int(x) for x in e["grid"]):
            if max(e["grid"][str(k)][i] for i, s in enumerate(steps) if s // L <= t) >= args.thr:
                n = k
            else:
                break
        P.append(n)
print("labelled prefix per loop:", P)
chance = 1 / 52
for j in sorted(int(x) for x in r["acc_by_line"]):
    if j < 2:
        continue
    blk = r["acc_by_line"][str(j)]
    k0 = blk["first_line"]
    A = np.array(blk["by_step"])                                  # [S, D - j]: accuracy per recorded step and line k = k0 + i
    cells = []
    for t in range(T):
        best = A[t * len(REC):(t + 1) * len(REC)].max(0)          # best recorded layer of loop t, per line
        ks = np.arange(k0, k0 + best.shape[0])
        inside, beyond = best[ks <= P[t]], best[ks > P[t]]
        cells.append(f"loop {t + 1}: in {inside.mean() if inside.size else float('nan'):.2f} (n={inside.size}) "
                     f"beyond {beyond.mean() if beyond.size else float('nan'):.2f} (n={beyond.size})")
    print(f"name {j} levels up: " + " | ".join(cells))
