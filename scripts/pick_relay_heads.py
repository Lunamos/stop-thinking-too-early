"""Pick the heads whose attention reaches furthest up the chain, from an E37b stride file: the best one-line head at the first
layer given, and at each later layer the distinct heads that are best for two or more lines up with a same-minus-other attention
of at least 0.1. Prints them as e68_head_profile.py --heads expects (layer:head,...).
usage: python pick_relay_heads.py STRIDES_JSON [--first 16] [--layers 20:25]"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--first", type=int, default=16)
ap.add_argument("--layers", default="20:25")
args = ap.parse_args()
r = json.load(open(args.path))["map"]
heads = [(args.first, r["head"][args.first][0])]
lo, hi = (int(x) for x in args.layers.split(":"))
for layer in range(lo, hi):
    for j in range(1, len(r["head"][layer])):
        h = r["head"][layer][j]
        if r["diff"][layer][j] >= 0.1 and (layer, h) not in heads:
            heads.append((layer, h))
print(",".join(f"{a}:{b}" for a, b in heads))
