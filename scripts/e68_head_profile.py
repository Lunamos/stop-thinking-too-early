"""E68: per-head attention profiles of the relay heads. For chosen heads, at the token naming the previous variable on each line k
of the queried programs, the attention to the same token on every ancestor line of the same chain (levels 1..k-1) and on every line
of the other chain. Tells a pointer-like head (attention concentrated on one ancestor, whose distance varies) from a window (attention
spread over the chain's ancestors). With a map, frozen, interleaved programs.

usage: python e68_head_profile.py MODEL --map PATH --a 14 --heads 16:23,21:18,... --tag TAG [--depth 16] [--n 60]
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
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--heads", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--order", default="interleave")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device, attn_implementation="eager")
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
M = Map.from_state(torch.load(args.map, map_location="cpu"), d_model).to(args.device)
heads = [tuple(int(x) for x in h.split(":")) for h in args.heads.split(",")]


def locate_rhs(it, prompt, offsets):
    out, cur = {}, 0
    for ln in it.lines:
        text = f"{ln['lhs']} = {ln['rhs']}\n"
        i = prompt.index(text, cur)
        end = i + len(ln["lhs"]) + 3 + len(ln["rhs"])
        tok_i = max(t for t, (a, b) in enumerate(offsets) if a < end and b > 0 and a >= i)
        out[(ln["chain"], ln["level"])] = tok_i
        cur = i + len(text)
    return out


def collect(use_map):
    h = attach_map(model, M, args.a) if use_map else None
    rng = random.Random(6868)
    D = args.depth
    same = np.zeros((len(heads), args.n, 2, D + 1, D + 1))       # [head, prog, chain, k, level]
    other = np.zeros((len(heads), args.n, 2, D + 1, D + 1))
    with torch.no_grad():
        for p in range(args.n):
            it = make_vb(rng, D, 2, args.order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
            enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = enc.to(args.device)
            pos = locate_rhs(it, it.prompt, offsets)
            out = model(input_ids=enc.input_ids, output_attentions=True)
            for hi, (l, hh) in enumerate(heads):
                A = out.attentions[l][0, hh].float().cpu().numpy()
                for c in (0, 1):
                    for k in range(2, D + 1):
                        src = pos[(c, k)]
                        for t in range(1, D + 1):
                            if pos[(c, t)] < src:
                                same[hi, p, c, k, t] = A[src, pos[(c, t)]]
                            if pos[(1 - c, t)] < src:
                                other[hi, p, c, k, t] = A[src, pos[(1 - c, t)]]
    if h is not None:
        h.remove()
    return same, other


res = {"model": args.model, "map": args.map, "a": args.a, "heads": heads, "depth": args.depth, "order": args.order}
for mode in ("map", "frozen"):
    S, O = collect(mode == "map")
    rows = []
    for hi, (l, hh) in enumerate(heads):
        conc, spread, tot_same, tot_other, argj = [], [], [], [], []
        for p in range(args.n):
            for c in (0, 1):
                for k in range(6, args.depth + 1):                     # lines with at least five ancestors
                    anc = S[hi, p, c, k, 1:k]                           # levels 1..k-1
                    s = anc.sum()
                    tot_same.append(s)
                    tot_other.append(O[hi, p, c, k, 1:].sum())
                    if s > 0.05:
                        conc.append(anc.max() / s)
                        q = anc / s
                        spread.append(float(np.exp(-(q * np.log(q + 1e-12)).sum())))   # effective number of ancestors attended
                        argj.append(int(k - (np.argmax(anc) + 1)))                     # stride of the most attended ancestor
        row = {"layer": l, "head": hh, "same": float(np.mean(tot_same)), "other": float(np.mean(tot_other)),
               "n_active": len(conc), "concentration": float(np.mean(conc)) if conc else None,
               "eff_ancestors": float(np.mean(spread)) if spread else None,
               "argmax_stride_hist": np.bincount(argj, minlength=args.depth).tolist() if argj else None}
        rows.append(row)
        print(f"{mode:6s} L{l:2d}H{hh:2d} same {row['same']:.2f} other {row['other']:.2f} active {len(conc):4d} "
              f"conc {row['concentration'] if conc else float('nan'):.2f} eff.anc {row['eff_ancestors'] if spread else float('nan'):.2f} "
              f"argmax strides {row['argmax_stride_hist'][:12] if argj else None}", flush=True)
    res[mode] = rows
    res[mode + "_mean_profile"] = {f"{l}:{hh}": [[float(S[hi, :, :, k, t].mean()) for t in range(1, args.depth + 1)]
                                                 for k in range(1, args.depth + 1)] for hi, (l, hh) in enumerate(heads)}
    save_json(res, f"{RESULTS}/e68_heads_{args.tag}.json")
