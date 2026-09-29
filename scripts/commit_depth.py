"""Commit depth from pointer traces (e10 panel JSONs): for pointer counterfactuals on three-line chains, the first layer at which the
pointer's own token holds less than half of the effect, for the pointers on lines 2 and 3, averaged; relative depth = layer / N.
usage: python commit_depth.py [JSON ...]"""
import glob
import os
import json
import sys

import numpy as np


def commit(path):
    r = json.load(open(path))
    N = r["N"]
    layers = []
    for d, v in r["ptr"].items():
        key = f"q|{d}|rhs"
        arr = np.array(v["agg"][key])
        first = next((i for i, x in enumerate(arr) if x < 0.5), None)
        if first is not None:
            layers.append(first)
    return (float(np.mean(layers)) / N if layers else None), layers, N, {d: v["kept"] for d, v in r["ptr"].items()}


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e10_panel_*.json")))
    for f in files:
        c, ls, N, kept = commit(f)
        print(f"{f.split('/')[-1]:40s} N={N:2d} layers={ls} commit={c if c is None else round(c, 3)} kept={kept}")
