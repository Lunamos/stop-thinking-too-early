"""Ouro-1.4B on MuSiQue: paired exact-match differences on MuSiQue (300 development questions, fixed prompts: PYTHONHASHSEED=0 runs *_h0).
For each run and loop count: exact match frozen and with the intervention, and the paired difference with a 95% bootstrap interval
over questions (2,000 resamples). Also compares interventions with each other on the same questions (their frozen rows are
identical, which the script checks). Writes checks/musique_loop.json (per run and loop count: frozen and mapped exact match, the
paired gain and its interval, in percent) and prints a summary.

usage: python loop_musique_paired.py [--tags ...]
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS, DATA, NOTES, OUT, RES, TABLES  # noqa: E402,F401
from mqa_common import em_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="o14_a6_h0,o14_a6_lora_h0,o14_a20_lora_h0,o14_prompt_h0,o14_a6_steer_h0,o14_a20_h0,o14_a6_last_h0,o14_a20_last_h0")
ap.add_argument("--val", default="musique_val900_seed1.json")
args = ap.parse_args()
val = json.load(open(f"{DATA}/{args.val}"))
def per_q(preds):
    out = []
    for ex, p in zip(val, preds):
        al = ex.get("aliases") or [ex["answer"]]
        out.append(max(em_score(p, a) for a in al))
    return np.array(out, dtype=float)


def boot(diff, key, B=2000):
    """percentile interval of the mean difference; one generator per comparison (seeded by its name), so intervals do not depend on
    which other comparisons are run"""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


runs = {}
for tag in args.tags.split(","):
    p = f"{RES}/e80_ouro_musique_{tag}.json"
    if not os.path.exists(p):
        print(f"{tag}: missing")
        continue
    r = json.load(open(p))
    if "preds" not in r:
        print(f"{tag}: no per-question predictions")
        continue
    runs[tag] = {k: per_q(v) for k, v in r["preds"].items()}
frozen_ref = {}
summary = {}
for tag, rows in runs.items():
    line = [f"{tag:18s}"]
    for key in sorted(rows, key=lambda k: (int(k.split("_")[0][1:]), k)):
        if not key.endswith("_map"):
            continue
        T = key.split("_")[0]
        fz = rows.get(f"{T}_frozen")
        if fz is None:
            continue
        if T in frozen_ref and not np.array_equal(frozen_ref[T], fz):
            line.append(f"[{T}: frozen differs from other runs]")
        frozen_ref.setdefault(T, fz)
        d = rows[key] - fz
        lo, hi = boot(d, f"{tag}|{key}|frozen")
        line.append(f"{T}: {100 * fz.mean():.1f}->{100 * rows[key].mean():.1f} ({100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}])")
        summary.setdefault(tag, {})[T] = {"frozen": round(100 * fz.mean(), 1), "map": round(100 * rows[key].mean(), 1),
                                          "gain": round(100 * d.mean(), 1), "ci": [round(100 * lo, 1), round(100 * hi, 1)],
                                          "n": int(len(d))}
    print("  ".join(line))
# intervention vs intervention (same questions)
base = runs.get("o14_a6_h0")
if base:
    for tag, rows in runs.items():
        if tag == "o14_a6_h0":
            continue
        cmp_ = []
        for key in rows:
            if key.endswith("_map") and key in base:
                d = base[key] - rows[key]
                lo, hi = boot(d, f"o14_a6_h0|{tag}|{key}")
                cmp_.append(f"{key.split('_')[0]}: map-{tag} {100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        if cmp_:
            print("  ".join(cmp_))
if "o14_a6_lora_h0" in runs and "o14_a20_lora_h0" in runs:
    a6, a20 = runs["o14_a6_lora_h0"], runs["o14_a20_lora_h0"]
    cmp_ = []
    for key in a6:
        if key.endswith("_map") and key in a20:
            d = a6[key] - a20[key]
            lo, hi = boot(d, f"lora6-lora20|{key}")
            cmp_.append(f"{key.split('_')[0]}: LoRA6-LoRA20 {100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print("  ".join(cmp_))
json.dump(summary, open(f"{CHECKS}/musique_loop.json", "w"), indent=1)
