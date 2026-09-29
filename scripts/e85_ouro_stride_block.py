"""E85: are the long strides needed? Ouro-1.4B with an E71 map (or frozen). In every step of every loop, the tokens of each program
line at level k may attend only to the header, to lines of their own level and to the lines of level k-1 (both chains): attention to
all lines of levels <= k-2 is blocked ("long" blocked, so a relay can only advance one level per round). Control: attention to level
k-1 blocked instead ("near"). The query tokens (print(X) Output:) read freely. If the growing strides of E82 carry the relay (pointer
doubling), blocking them cuts reach at a fixed number of loops; if they only pool over lines already labelled, it changes little.

Runs in loopdyn/.venv-loop (eager attention).
usage: python e85_ouro_stride_block.py --tag TAG [--map PATH --a 6] [--Ts 2,3,4,6] [--depths 4,8,12,16,20,24] [--n 100]
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
ap.add_argument("--tag", required=True)
ap.add_argument("--map", default=None)
ap.add_argument("--a", type=int, default=6)
ap.add_argument("--Ts", default="2,3,4,6")
ap.add_argument("--depths", default="4,8,12,16,20,24")
ap.add_argument("--conds", default="none,long,near")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--pool", default="letters", help="letters or both (plus single-token two-letter names)")
ap.add_argument("--renorm", type=int, default=1, help="1: blocked keys get -inf before the softmax (attention renormalizes over the rest); 0: blocked weights are zeroed after the softmax, without renormalizing")
ap.add_argument("--ablate", default="", help="heads to zero, as layer:head pairs separated by commas, e.g. 12:6,14:12")
ap.add_argument("--ablate_loops", default="", help="loops (1-based) in which the heads are zeroed; empty = all loops")
ap.add_argument("--ablate_sets", default="", help="JSON file with a list of head sets (each 'l:h,l:h,...'); runs condition 'none' once per set")
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--mask_layers", default="", help="layers of the loop body (0-based, e.g. 7-15 or 0-6,16-23) in which the block applies; empty = all")
ap.add_argument("--mask_loops", default="", help="loops (1-based, e.g. 1 or 2-8) in which the block applies; empty = all")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                          attn_implementation="eager").to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
inner = model.model
import sys as _sys  # noqa: E402
_mod = _sys.modules[model.__class__.__module__]
_repeat_kv = _mod.repeat_kv
STATE = {"keep": None, "cur": None}
ABL = {}
for _pair in [x for x in args.ablate.split(",") if x]:
    _l, _h = (int(v) for v in _pair.split(":"))
    ABL.setdefault(_l, []).append(_h)
ABL_LOOPS = {int(x) - 1 for x in args.ablate_loops.split(",") if x}


def _parse_layers(spec):
    out = set()
    for part in [x for x in spec.split(",") if x]:
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


MASK_LAYERS = _parse_layers(args.mask_layers)
MASK_LOOPS = {x - 1 for x in _parse_layers(args.mask_loops)}


def _masked(t, l):
    return (not MASK_LAYERS or l in MASK_LAYERS) and (not MASK_LOOPS or t in MASK_LOOPS)


def _eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks = _repeat_kv(key, module.num_key_value_groups)
    vs = _repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : ks.shape[-2]]
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    t, l = STATE["cur"] if STATE["cur"] is not None else (None, None)
    if STATE["keep"] is not None and _masked(t, l):
        w = w * STATE["keep"]
    out = torch.matmul(w, vs)                                       # [B, H, T, dh]
    if l in ABL and (not ABL_LOOPS or t in ABL_LOOPS):
        out[:, ABL[l]] = 0
    return out.transpose(1, 2).contiguous(), w


_mod.eager_attention_forward = _eager
L = len(inner.layers)
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
if args.pool == "both":
    letters = letters + [x + y for x in LETTERS for y in LETTERS
                         if len(tok.encode(" " + x + y, add_special_tokens=False)) == 1 and len(tok.encode(x + y, add_special_tokens=False)) == 1]
MAP = None
if args.map:
    class Map(nn.Module):
        def __init__(self, d, r):
            super().__init__()
            self.A = nn.Linear(d, r, bias=False)
            self.B = nn.Linear(r, d, bias=False)
            self.s = nn.Parameter(torch.ones(1))

        def forward(self, h):
            x = h.float()
            rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)
    sd = torch.load(args.map, map_location="cpu")
    MAP = Map(model.config.hidden_size, sd["A.weight"].shape[0]).to(args.device)
    MAP.load_state_dict(sd)


def build(items, cond):
    enc = tok([it.prompt for it in items], return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    ids, am = enc.input_ids.to(args.device), enc.attention_mask.to(args.device)
    T = ids.shape[1]
    block = torch.zeros(len(items), T, T, dtype=torch.bool)
    for b, it in enumerate(items):
        off = [tuple(x) for x in offs[b].tolist()]
        cur, toks_by_level = 0, {}
        spans = []
        for ln in it.lines:
            text = f"{ln['lhs']} = {ln['rhs']}\n"
            i = it.prompt.index(text, cur)
            j = i + len(text)
            tt = [t for t, (s, e) in enumerate(off) if e > i and s < j and e > s]
            toks_by_level.setdefault(ln["level"], []).extend(tt)
            spans.append((ln["level"], tt))
            cur = j
        if cond == "none":
            continue
        by_cl = {}
        for (lev, tt), ln in zip(spans, it.lines):
            by_cl[(ln["chain"], lev)] = tt
        chains = sorted({ln["chain"] for ln in it.lines})
        far_all = None
        for (lev, rows), ln in zip(spans, it.lines):
            c = ln["chain"]
            if cond == "long":                    # both chains, levels <= k-2
                cols = [t for lv, tt in toks_by_level.items() if lv <= lev - 2 for t in tt]
            elif cond == "near":                  # both chains, level k-1
                cols = list(toks_by_level.get(lev - 1, []))
            elif cond == "same_far":              # own chain only, levels <= k-2
                cols = [t for lv in range(1, lev - 1) for t in by_cl.get((c, lv), [])]
            elif cond == "other_far":             # the other chains only, levels <= k-2
                cols = [t for cc in chains if cc != c for lv in range(1, lev - 1) for t in by_cl.get((cc, lv), [])]
            elif cond == "rand_far":              # as many lines of levels <= k-2 as same_far, drawn at random from both chains
                cand = [(cc, lv) for cc in chains for lv in range(1, lev - 1)]
                pick = random.Random(hash((b, lev, c)) % 10**9).sample(cand, lev - 2) if lev > 2 else []
                cols = [t for key in pick for t in by_cl.get(key, [])]
            elif cond == "near_same":             # own chain's line at level k-1 only (the parent)
                cols = list(by_cl.get((c, lev - 1), []))
            elif cond == "near_other":            # other chain's line at level k-1 only
                cols = [t for cc in chains if cc != c for t in by_cl.get((cc, lev - 1), [])]
            else:
                raise ValueError(cond)
            if rows and cols:
                block[b][torch.tensor(rows)[:, None], torch.tensor(cols)[None, :]] = True
    return ids, am, block.to(args.device)


@torch.no_grad()
def run(ids, am, block, T):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    if args.renorm:
        bl = mask.clone()
        bl[:, 0] = bl[:, 0].masked_fill(block, torch.finfo(h.dtype).min)
        STATE["keep"] = None
    else:
        bl = mask
        STATE["keep"] = (~block)[:, None].to(h.dtype)
    pe = inner.rotary_emb(h, pos)
    for t in range(T):
        for l in range(L):
            if MAP is not None and l == args.a:
                h = MAP(h)
            STATE["cur"] = (t, l)
            am_l = bl if _masked(t, l) else mask
            out = inner.layers[l](h, attention_mask=am_l, position_ids=pos, position_embeddings=pe, past_key_value=None, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        h = inner.norm(h)
    return model.lm_head(h[:, -1]).float()


def accuracy(d, T, cond):
    rng = random.Random(8500 + d)
    items = [make_vb(rng, d, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters) for _ in range(args.n)]
    ex = ch = 0
    for i in range(0, len(items), 20):
        chunk = items[i:i + 20]
        ids, am, block = build(chunk, cond)
        lg = run(ids, am, block, T)
        for k, it in enumerate(chunk):
            ex += int(lg[k].argmax().item() == vids[it.answer])
            rl = torch.stack([lg[k, vids[x]] for x in it.roots])
            ch += int(it.roots[rl.argmax().item()] == it.answer)
    return ex / len(items), ch / len(items)


res = {"args": vars(args), "acc": {}}
if args.ablate_sets:
    sets = json.load(open(args.ablate_sets))
    res["sets"] = sets
    for i, hs in enumerate(sets):
        ABL.clear()
        for _pair in [x for x in hs.split(",") if x]:
            _l, _h = (int(v) for v in _pair.split(":"))
            ABL.setdefault(_l, []).append(_h)
        for T in [int(x) for x in args.Ts.split(",")]:
            row = {d: accuracy(d, T, "none") for d in [int(x) for x in args.depths.split(",")]}
            res["acc"][f"set{i}_T{T}_none"] = row
            print(f"set {i} ({hs}) T={T} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()), flush=True)
            save_json(res, f"{RESULTS}/e85_stride_block_{args.tag}.json")
    print("done")
    raise SystemExit(0)
for T in [int(x) for x in args.Ts.split(",")]:
    for cond in args.conds.split(","):
        row = {d: accuracy(d, T, cond) for d in [int(x) for x in args.depths.split(",")]}
        res["acc"][f"T{T}_{cond}"] = row
        print(f"T={T} {cond:5s} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()), flush=True)
        save_json(res, f"{RESULTS}/e85_stride_block_{args.tag}.json")
print("done")
