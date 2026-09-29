"""Export the question-answering selections to loopdyn/data/ as JSON (for the looped-model environment, whose older `datasets`
cannot load the benchmarks, and for the analysis scripts, which need no GPU).

  musique_train_seed0.json    load_musique("train", seed=0)             training questions (e80_ouro_musique.py)
  musique_val900_seed1.json   load_musique("validation", n=900, seed=1) the 900 development questions used everywhere
                              (e52_mqa_train.py evaluates the same selection; e80 uses its first 300)
  2wiki_dev300_seed1.json, hotpot_bridge_dev300_seed1.json          other benchmarks (exploratory runs only)

usage (standard-model environment): python export_qa_data.py [--which musique_val,musique_train,2wiki,hotpot] [--check]
With --check, compares against the existing files instead of writing.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mqa_common import load_2wiki, load_hotpot, load_musique  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EXPORTS = {
    "musique_val": ("musique_val900_seed1.json", lambda: load_musique("validation", n=900, seed=1)),
    "musique_train": ("musique_train_seed0.json", lambda: load_musique("train", seed=0)),
    "2wiki": ("2wiki_dev300_seed1.json", lambda: load_2wiki("dev", n=300, seed=1)),
    "hotpot": ("hotpot_bridge_dev300_seed1.json", lambda: load_hotpot("validation", n=300, seed=1, types=("bridge",))),
}
ap = argparse.ArgumentParser()
ap.add_argument("--which", default="musique_val,musique_train")
ap.add_argument("--check", action="store_true")
args = ap.parse_args()
os.makedirs(DATA, exist_ok=True)
for key in args.which.split(","):
    name, fn = EXPORTS[key]
    rows = fn()
    path = os.path.join(DATA, name)
    if args.check:
        old = json.load(open(path)) if os.path.exists(path) else None
        same = old is not None and json.loads(json.dumps(rows)) == old
        print(f"{name}: {len(rows)} rows; {'identical to' if same else 'DIFFERENT from'} the existing file")
    else:
        json.dump(rows, open(path, "w"))
        print(f"wrote {path} ({len(rows)} rows)")
