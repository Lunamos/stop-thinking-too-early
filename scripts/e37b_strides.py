"""E37b: chain-selective relay strides with fixed heads. At the token naming the previous variable on line k (level k of the queried
chain), attention to the same token on the line j levels earlier in the same chain, minus attention to the same token on the other
chain's line at that level (control for distance and position). For each (layer, head) the difference is averaged over lines and
programs; heads are chosen on half of the programs (largest mean difference per layer and stride) and scored on the other half.
Frozen vs with a map. Level order or interleaved.

usage: python e37b_strides.py MODEL --map PATH --a 14 --tag TAG [--depth 16] [--n 60] [--order forward|interleave] [--var_pool letters|both]
"""
import argparse
import random

import numpy as np
import torch

from ld_common import RESULTS, load_model, make_vb, save_json, value_token_ids, LETTERS
from mqa_common import Map, attach_map

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--map", default=None)
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--order", default="forward")
ap.add_argument("--var_pool", default="letters")
ap.add_argument("--jmax", type=int, default=16)
ap.add_argument("--direction", default="down", help="down: line k reads line k-j (level order); up: line k reads line k+j (reversed programs, where later levels come first)")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--frozen_only", action="store_true", help="only the frozen model (no map needed)")
args = ap.parse_args()

model, tok = load_model(args.model, args.device, attn_implementation="eager")
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
if args.var_pool == "both":
    two = [a + b for a in LETTERS for b in LETTERS if len(tok.encode(" " + a + b, add_special_tokens=False)) == 1]
    letters = letters + two
M = Map.from_state(torch.load(args.map, map_location="cpu"), d_model).to(args.device) if args.map else None


def locate_rhs(it, prompt, offsets):
    """token index of the right-hand side of each line (last token of the name), robust to multi-token names"""
    out, cur = {}, 0
    for ln in it.lines:
        text = f"{ln['lhs']} = {ln['rhs']}\n"
        i = prompt.index(text, cur)
        end = i + len(ln["lhs"]) + 3 + len(ln["rhs"])          # character just after the rhs
        tok_i = max(t for t, (a, b) in enumerate(offsets) if a < end and b > 0 and a >= i)
        out[(ln["chain"], ln["level"])] = tok_i
        cur = i + len(text)
    return out


def collect(use_map):
    h = attach_map(model, M, args.a) if use_map else None
    rng = random.Random(3737)
    same, other = [], []          # per program: [L, H, J] sums and counts
    with torch.no_grad():
        for _ in range(args.n):
            it = make_vb(rng, args.depth, 2, args.order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
            enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = enc.to(args.device)
            pos = locate_rhs(it, it.prompt, offsets)
            out = model(input_ids=enc.input_ids, output_attentions=True)
            A = torch.stack([x[0].float() for x in out.attentions])       # [L, H, T, T]
            Lh, Hh = A.shape[0], A.shape[1]
            s = torch.zeros(Lh, Hh, args.jmax)
            o = torch.zeros(Lh, Hh, args.jmax)
            cnt = torch.zeros(args.jmax)
            for c in (0, 1):
                for k in range(1, args.depth + 1):
                    src = pos[(c, k)]
                    for j in range(1, args.jmax + 1):
                        t = k - j if args.direction == "down" else k + j
                        if t < 1 or t > args.depth:
                            break
                        s[:, :, j - 1] += A[:, :, src, pos[(c, t)]].cpu()
                        o[:, :, j - 1] += A[:, :, src, pos[(1 - c, t)]].cpu()
                        cnt[j - 1] += 1
            same.append((s / cnt.clamp(min=1)).numpy())
            other.append((o / cnt.clamp(min=1)).numpy())
    if h is not None:
        h.remove()
    return np.stack(same), np.stack(other)                                # [n, L, H, J]


res = {"model": args.model, "map": args.map, "a": args.a, "depth": args.depth, "order": args.order}
for mode in (("frozen",) if args.frozen_only else ("frozen", "map")):
    S, O = collect(mode == "map")
    D = S - O
    half = S.shape[0] // 2
    sel = D[:half].mean(0)                                                 # [L, H, J]
    best = sel.argmax(1)                                                   # [L, J] head chosen on the first half
    Dt, St, Ot = D[half:].mean(0), S[half:].mean(0), O[half:].mean(0)
    L, H, J = Dt.shape
    diff = np.array([[Dt[l, best[l, j], j] for j in range(J)] for l in range(L)])
    same = np.array([[St[l, best[l, j], j] for j in range(J)] for l in range(L)])
    oth = np.array([[Ot[l, best[l, j], j] for j in range(J)] for l in range(L)])
    res[mode] = {"diff": diff.tolist(), "same": same.tolist(), "other": oth.tolist(), "head": best.tolist()}
    print(mode, flush=True)
    for j in range(J):
        row = diff[:, j]
        top = int(np.argmax(row))
        print(f"  stride {j + 1:2d}: max same-minus-other {row.max():.2f} at layer {top}   (same {same[top, j]:.2f}, other {oth[top, j]:.2f})",
              flush=True)
    save_json(res, f"{RESULTS}/e37b_strides_{args.tag}.json")
