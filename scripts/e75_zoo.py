"""E75: post-training zoo. For post-trained descendants of a base model: default reach on setup A (three chains, choice among roots,
chains of 1-6 lines) and whether a map trained on the BASE model, applied unchanged, still unlocks long chains (two chains,
upper/lower-case letters, exact accuracy by length). One JSON per model.

usage: python e75_zoo.py --base_map PATH:LAYER --models m1,m2,... --tag TAG
"""
import argparse
import os
import random
import time

import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS
from mqa_common import Map, attach_map

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--base_map", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--n_map", type=int, default=100)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()


def reach(acc, thr=0.8, one=None):
    acc = dict(acc)
    if one is not None:
        acc.setdefault(1, one)
    ds = sorted(acc)
    best = 0.0
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0)
        if acc[d1] >= thr:
            best = d1
    return best if acc[ds[0]] >= thr else 0.0


path, layer = args.base_map.rsplit(":", 1)
for name in args.models.split(","):
    out_p = f"{RESULTS}/e75_zoo_{args.tag}_{name.replace('/', '__')}.json"
    if os.path.exists(out_p):
        print("skip", name, flush=True)
        continue
    t0 = time.time()
    try:
        model, tok = load_model(name, args.device)
    except Exception as e:                                                  # noqa: BLE001
        print("could not load", name, repr(e)[:200], flush=True)
        continue
    R = Runner(model, tok)
    vids = value_token_ids(tok)
    caps = [c for c in LETTERS if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
    both = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
            if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
    res = {"model": name, "setupA": {}, "frozen2": {}, "map2": {}}

    @torch.no_grad()
    def score(items):
        ex = ch = 0
        for i in range(0, len(items), 25):
            chunk = items[i:i + 25]
            lg = R.last_logits([it.prompt for it in chunk], bs=25).float()
            for k, it in enumerate(chunk):
                ex += int(lg[k].argmax().item() == vids[it.answer])
                rl = torch.stack([lg[k, vids[x]] for x in it.roots])
                ch += int(it.roots[rl.argmax().item()] == it.answer)
        return ex / len(items), ch / len(items)

    for d in range(1, 7):
        rng = random.Random(7919 * d + 17)
        res["setupA"][d] = score([make_vb(rng, d, 3, "forward", header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=caps)
                                  for _ in range(args.n)])
    res["reachA"] = reach({d: c for d, (e, c) in res["setupA"].items()})
    d_model = model.config.get_text_config().hidden_size
    M = Map.from_state(torch.load(path, map_location="cpu"), d_model).to(args.device)
    for mode in ("frozen2", "map2"):
        h = attach_map(model, M, int(layer)) if mode == "map2" else None
        for d in (2, 4, 8, 12, 16, 24):
            rng = random.Random(10_000 + d)
            res[mode][d] = score([make_vb(rng, d, 2, "forward", header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=both)
                                  for _ in range(args.n_map)])
        if h is not None:
            h.remove()
    res["reach_frozen2"] = reach({d: e for d, (e, c) in res["frozen2"].items()}, one=1.0)
    res["reach_map2"] = reach({d: e for d, (e, c) in res["map2"].items()}, one=1.0)
    save_json(res, out_p)
    print(f"{name}: reach A {res['reachA']:.2f} | 2-chain exact reach frozen {res['reach_frozen2']:.1f}, with base map "
          f"{res['reach_map2']:.1f} | t={time.time() - t0:.0f}s", flush=True)
    del model, R, M
    torch.cuda.empty_cache()
