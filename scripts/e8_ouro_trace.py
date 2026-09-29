"""E8: causal tracing inside a looped LM (Ouro), value and pointer counterfactuals.

Same design as e4_trace.py / e4b_pointer.py but steps run over (loop, layer).
Scored with the restricted metric LD = logit(v_clean) - logit(v_cf) at the last
position after T loops (prompt ends with 'Output:\\n' for Ouro).

usage: python e8_ouro_trace.py MODEL --tag TAG --depth 2 --T 4 --kind val|ptr [--level K] --n 30
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import NOUNS, RESULTS, make_vb, save_json  # noqa: E402
from ld_loop import OuroRunner  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--T", type=int, default=4)
ap.add_argument("--kind", default="val")
ap.add_argument("--level", type=int, default=None, help="pointer level K (ptr kind)")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--qf", default="print({q})\\nOutput:\\n")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max_tries", type=int, default=3000)
args = ap.parse_args()
QF = args.qf.replace("\\n", "\n")

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
R = OuroRunner(model, tok)
S = args.T * R.L

vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
letters = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


def label_positions(it, toks, kinds):
    labels = [("hdr",)] * len(toks)
    i = 0
    L = len(toks)
    for ln in it.lines:
        while i < L - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1] == " =" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        assert i < L - 3, (ln, toks)
        for k, role in enumerate(["lhs", "eq", "rhs", "nl"]):
            labels[i + k] = (kinds[ln["chain"]], ln["level"], role)
        i += 4
    for k in range(i, L):
        labels[k] = ("query", k - i, toks[k])
    return labels


rng = random.Random(555 + args.depth * 7 + (args.level or 0))
agg = defaultdict(lambda: torch.zeros(S + 1))
agg_n = defaultdict(int)
kept = tries = 0
D = args.depth
while kept < args.n and tries < args.max_tries:
    tries += 1
    it = make_vb(rng, D, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
    q = it.meta["query_chain"]
    chains = it.meta["chains"]
    if args.kind == "val":
        r = None
        v = it.answer
        v2 = rng.choice([w for w in vals if w not in it.roots])
        cfp = it.prompt.replace(f" = {v}\n", f" = {v2}\n")
        kinds = {c: ("q" if c == q else "o") for c in range(args.chains)}
    else:
        K = args.level
        r = rng.choice([c for c in range(args.chains) if c != q])
        tgt, old, new = chains[q][K - 1], chains[q][K - 2], chains[r][K - 2]
        cl, cf_line = f"{tgt} = {old}\n", f"{tgt} = {new}\n"
        assert it.prompt.count(cl) == 1
        cfp = it.prompt.replace(cl, cf_line)
        v, v2 = it.roots[q], it.roots[r]
        kinds = {c: ("q" if c == q else ("r" if c == r else "o")) for c in range(args.chains)}
    enc = tok([it.prompt, cfp], return_tensors="pt", padding=True).to(args.device)
    ids, am = enc.input_ids, enc.attention_mask
    if (am == 0).any() or ids.shape[1] != len(tok(it.prompt).input_ids):
        continue
    rc = R.run(ids, am, T=args.T, record=True)
    lg = R.logits(rc["h"][:, -1]).float()
    a, b = vids[v], vids[v2]
    # restricted correctness: clean prefers v over v2 and all roots; cf prefers v2
    cand = [vids[x] for x in it.roots] + ([b] if args.kind == "val" else [])
    if not (lg[0, cand].argmax().item() == cand.index(a) and lg[1, a] < lg[1, b]):
        continue
    ld_clean = (lg[0, a] - lg[0, b]).item()
    ld_cf = (lg[1, a] - lg[1, b]).item()
    T_ = ids.shape[1]
    toks = [tok.decode([t]) for t in ids[0].tolist()]
    labels = label_positions(it, toks, kinds)
    emb = model.model.embed_tokens(ids)
    inputs = [emb] + [rc["rec"][s] for s in range(S)]
    keep = [p for p in range(T_) if labels[p][0] in ("q", "r", "query")]
    P = len(keep)
    grid = torch.zeros(S + 1, P)
    idx = torch.tensor(keep, device=args.device)
    ar = torch.arange(P, device=args.device)
    for s in range(S + 1):
        base = inputs[s][1:2].expand(P, -1, -1).clone()
        base[ar, idx] = inputs[s][0, idx]
        if s < S:
            out = R.run(ids[1:2].expand(P, -1), am[1:2].expand(P, -1), T=args.T, h0=base, start=s)["h"]
        else:
            out = base
        lp = R.logits(out[:, -1]).float()
        grid[s] = (((lp[:, a] - lp[:, b]) - ld_cf) / (ld_clean - ld_cf)).cpu()
    for j, p in enumerate(keep):
        agg[labels[p]] += grid[:, j]
        agg_n[labels[p]] += 1
    kept += 1
    if kept % 10 == 0:
        print("kept", kept, "tries", tries, flush=True)

save_json({"model": args.model, "depth": D, "T": args.T, "L": R.L, "kind": args.kind, "level": args.level,
           "kept": kept, "tries": tries,
           "agg": {"|".join(map(str, k)): (v / agg_n[k]).tolist() for k, v in agg.items()},
           "agg_n": {"|".join(map(str, k)): n for k, n in agg_n.items()}},
          f"{RESULTS}/e8_ouro_{args.tag}.json")
print("done kept", kept, "tries", tries)
