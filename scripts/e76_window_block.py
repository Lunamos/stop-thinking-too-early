"""E76: a causal test of the routing window. Block the attention from the query tokens (print(X)\\nOutput:) to every program line
except the root lines (so the value can still be copied) in a band of layers, and measure choice accuracy among the chain roots
(setup A: three chains, capital letters, one to four lines). One-line chains have no pointer line and are a control.
  - named bands: before the window [0, a), the window [a, b), between the window and the value copy [b, H), after the copy [H, N)
  - a scan with a sliding band of `width` layers
Prediction (notes/routing_window.md, P3): blocking inside the window sends two- to four-line chains to chance; blocking between the
window's end and the value copy changes accuracy by 0.03 or less.

usage: python e76_window_block.py MODEL --tag TAG --a 17 --b 24 --H 32 [--width 3] [--n 100]
"""
import argparse
import random

import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--b", type=int, required=True)
ap.add_argument("--H", type=int, required=True)
ap.add_argument("--width", type=int, default=3)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--depths", default="1,2,3,4")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device, attn_implementation="eager")
R = Runner(model, tok)
A_ = R.A
N = R.n_layers
vids = value_token_ids(tok)
letters = [c for c in LETTERS if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


def spans(it, prompt, offsets):
    """token index ranges of each program line (with its newline), marked root or not, and the first query token"""
    out, cur = [], 0
    for ln in it.lines:
        text = f"{ln['lhs']} = {ln['rhs']}\n"
        i = prompt.index(text, cur)
        j = i + len(text)
        toks = [t for t, (s, e) in enumerate(offsets) if e > i and s < j and e > s]
        out.append((toks, ln["level"] == 1))
        cur = j
    q0 = prompt.index("print(", cur)
    qtok = [t for t, (s, e) in enumerate(offsets) if s >= q0 and e > s]
    return out, qtok


def build(items):
    prompts = [it.prompt for it in items]
    enc = tok(prompts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    ids, am = enc.input_ids.to(args.device), enc.attention_mask.to(args.device)
    T = ids.shape[1]
    block = torch.zeros(len(items), T, T, dtype=torch.bool)
    for b, it in enumerate(items):
        pad = int((am[b] == 0).sum())                          # left padding
        off = [tuple(x) for x in offs[b].tolist()]
        lines, qtok = spans(it, it.prompt, off)
        rows = torch.tensor(qtok)
        cols = torch.tensor([t for toks, root in lines if not root for t in toks], dtype=torch.long)
        if len(cols):
            block[b][rows[:, None], cols[None, :]] = True
        assert pad == 0 or tokenizer_left_padding_ok(pad)
    return ids, am, block.to(args.device)


def tokenizer_left_padding_ok(pad):
    return True


@torch.no_grad()
def run(ids, am, block, layers):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = A_.embed(ids)
    masks = R._masks(h, am, pos)
    pe = {}
    for lt in set(A_.layer_types):
        try:
            pe[lt] = A_.rotary(h, pos, lt)
        except TypeError:
            pe[lt] = A_.rotary(h, pos)
    neg = torch.finfo(h.dtype).min
    blocked = {}
    for lt, m in masks.items():
        mm = m.clone()
        mm[:, 0] = mm[:, 0].masked_fill(block, neg)
        blocked[lt] = mm
    for j in range(N):
        lt = A_.layer_types[j]
        out = A_.layers[j](h, attention_mask=(blocked[lt] if j in layers else masks[lt]), position_embeddings=pe[lt],
                           position_ids=pos, past_key_values=None, use_cache=False)
        h = out if torch.is_tensor(out) else out[0]
    return R.unembed(h[:, -1]).float()


def accuracy(d, layers):
    rng = random.Random(7600 + d)
    items = [make_vb(rng, d, 3, "forward", header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters) for _ in range(args.n)]
    ok = 0
    for i in range(0, len(items), 25):
        chunk = items[i:i + 25]
        ids, am, block = build(chunk)
        lg = run(ids, am, block, layers)
        for k, it in enumerate(chunk):
            rl = torch.stack([lg[k, vids[x]] for x in it.roots])
            ok += int(it.roots[rl.argmax().item()] == it.answer)
    return ok / len(items)


depths = [int(x) for x in args.depths.split(",")]
res = {"model": args.model, "N": N, "a": args.a, "b": args.b, "H": args.H, "named": {}, "scan": {}}
bands = {"none": set(), "before [0,a)": set(range(0, args.a)), "window [a,b)": set(range(args.a, args.b)),
         "idle [b,H)": set(range(args.b, args.H)), "after [H,N)": set(range(args.H, N))}
for name, layers in bands.items():
    res["named"][name] = {d: accuracy(d, layers) for d in depths}
    print(f"{name:14s} " + " ".join(f"d{d}:{v:.2f}" for d, v in res["named"][name].items()), flush=True)
    save_json(res, f"{RESULTS}/e76_block_{args.tag}.json")
for s in range(0, N - args.width + 1):
    layers = set(range(s, s + args.width))
    res["scan"][s] = {d: accuracy(d, layers) for d in depths if d > 1}
    print(f"band {s:2d}-{s + args.width - 1:2d} " + " ".join(f"d{d}:{v:.2f}" for d, v in res["scan"][s].items()), flush=True)
    save_json(res, f"{RESULTS}/e76_block_{args.tag}.json")
