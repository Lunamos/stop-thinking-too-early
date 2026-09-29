"""E87: does each line accumulate its chain's ancestors across loops? Ouro-1.4B, frozen or with an E71 map. At the token naming the
previous variable on line k (the pointer token, which is itself the name of the ancestor one level up), decode the name of the
ancestor j levels up (the variable defined on line k-j, j >= 2) with a linear ridge probe over the 52 single-letter names (chance
about 1/52), trained on 300 programs and tested on 100 others, at several layers of every loop. If each loop unions a line's set of
known ancestors with its parents' sets, the largest decodable j should grow from loop to loop with the map and stay at 1-2 frozen.

Runs in loopdyn/.venv-loop.
usage: python e87_ouro_ancestor_probe.py --tag TAG [--map PATH --a 6 --apply all|first] [--T 4] [--depth 16] [--n 400] [--jmax 10]
"""
import argparse
import os
import random
import sys

import numpy as np
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
ap.add_argument("--apply", default="all")
ap.add_argument("--T", type=int, default=4)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--ntrain", type=int, default=300)
ap.add_argument("--jmax", type=int, default=10)
ap.add_argument("--layers", default="3,7,11,15,19,23")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--mask", default="", help="block one kind of read while probing: cond:layers[:loops], cond = parent (each line's reads of its "
                "parent's line) or far (lines two or more levels up, both chains); layers/loops as 0-6 or 7-15 / 2 or 2-4 (1-based loops); "
                "weights zeroed after the softmax")
args = ap.parse_args()

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                          **({"attn_implementation": "eager"} if args.mask else {})).to(args.device).eval()
inner = model.model
L = len(inner.layers)
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
lid = {c: i for i, c in enumerate(letters)}
REC = [int(x) for x in args.layers.split(",")]


def _rng(spec):
    out = set()
    for part in [x for x in spec.split(",") if x]:
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


MSTATE = {"keep": None, "cur": None}
if args.mask:
    _parts = args.mask.split(":")
    M_COND, M_LAYERS = _parts[0], _rng(_parts[1])
    M_LOOPS = {x - 1 for x in _rng(_parts[2])} if len(_parts) > 2 else None
    import sys as _sys
    _mod = _sys.modules[model.__class__.__module__]
    _repeat_kv = _mod.repeat_kv

    def _eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = _repeat_kv(key, module.num_key_value_groups)
        vs = _repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, :, : ks.shape[-2]]
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        t, l = MSTATE["cur"]
        if MSTATE["keep"] is not None and l in M_LAYERS and (M_LOOPS is None or t in M_LOOPS):
            w = w * MSTATE["keep"]
        return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
    _mod.eager_attention_forward = _eager
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


def locate_rhs(it, prompt, offsets):
    out, cur = {}, 0
    for ln in it.lines:
        text = f"{ln['lhs']} = {ln['rhs']}\n"
        i = prompt.index(text, cur)
        end = i + len(ln["lhs"]) + 3 + len(ln["rhs"])
        out[(ln["chain"], ln["level"])] = max(t for t, (a, b) in enumerate(offsets) if a < end and b > 0 and a >= i)
        cur = i + len(text)
    return out


@torch.no_grad()
def states(it):
    enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    pos = locate_rhs(it, it.prompt, offsets)
    idx = torch.tensor([pos[(c, k)] for c in (0, 1) for k in range(1, args.depth + 1)], device=args.device)
    ids = enc.input_ids.to(args.device)
    am = torch.ones_like(ids)
    p = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=p)
    pe = inner.rotary_emb(h, p)
    if args.mask:
        spans, cur = {}, 0
        for ln in it.lines:
            text = f"{ln['lhs']} = {ln['rhs']}\n"
            i = it.prompt.index(text, cur)
            j = i + len(text)
            spans[(ln["chain"], ln["level"])] = [t_ for t_, (a_, b_) in enumerate(offsets) if b_ > i and a_ < j and b_ > a_]
            cur = j
        keep = torch.ones(1, 1, ids.shape[1], ids.shape[1], dtype=h.dtype, device=args.device)
        for (c, k), rows in spans.items():
            if M_COND == "parent":
                cols = spans.get((c, k - 1), [])
            else:
                cols = [t_ for (c2, k2), tt in spans.items() if k2 <= k - 2 for t_ in tt]
            if rows and cols:
                keep[0, 0, torch.tensor(rows)[:, None], torch.tensor(cols)[None, :]] = 0
        MSTATE["keep"] = keep
    out = []
    for t in range(args.T):
        for l in range(L):
            MSTATE["cur"] = (t, l)
            if MAP is not None and l == args.a and (args.apply == "all" or t == 0):
                h = MAP(h)
            o = inner.layers[l](h, attention_mask=mask, position_ids=p, position_embeddings=pe, past_key_value=None, use_cache=False)
            h = o[0] if isinstance(o, tuple) else o
            if l in REC:
                out.append(h[0, idx].float().cpu().half())
        h = inner.norm(h)
    return torch.stack(out)                                            # [T*len(REC), 2*D, d]


rng = random.Random(8787)
X, names = [], []
for i in range(args.n):
    it = make_vb(rng, args.depth, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
    X.append(states(it))
    lhs = {(ln["chain"], ln["level"]): ln["lhs"] for ln in it.lines}
    names.append([[lid[lhs[(c, k)]] for k in range(1, args.depth + 1)] for c in (0, 1)])
X = torch.stack(X)                                                     # [n, S, 2*D, d]
names = torch.tensor(names)                                            # [n, 2, D]: name defined on line (c, k)
D, S = args.depth, X.shape[1]


def probe_acc(s, j):
    """decode the name defined on line k-j from the pointer token of line k (k > j), both chains"""
    rows, ys = [], []
    for c in (0, 1):
        for k in range(j + 1, D + 1):
            rows.append(X[:, s, c * D + (k - 1)])                     # [n, d]
            ys.append(names[:, c, k - j - 1])                         # [n]
    Xs = torch.stack(rows, 1).float().to(args.device)                 # [n, m, d]
    Ys = torch.stack(ys, 1).to(args.device)                           # [n, m]
    tr, te = slice(0, args.ntrain), slice(args.ntrain, args.n)
    Xtr, Ytr = Xs[tr].reshape(-1, Xs.shape[-1]), Ys[tr].reshape(-1)
    Xte, Yte = Xs[te].reshape(-1, Xs.shape[-1]), Ys[te].reshape(-1)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-3)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.nn.functional.one_hot(Ytr, len(letters)).float()
    Y = Y - Y.mean(0)
    lam = 1e-1 * Xtr.shape[0]
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=Xtr.device), Xtr.T @ Y)
    ok = (Xte @ W).argmax(-1).eq(Yte).float().reshape(-1, Xs.shape[1])      # [n_test, m]: rows (c, k) in the order built above
    m_per_c = D - j
    by_k = [(ok[:, i].sum().item() + ok[:, m_per_c + i].sum().item()) / (2 * ok.shape[0]) for i in range(m_per_c)]
    return ok.mean().item(), by_k                                             # by_k[i]: line k = j + 1 + i


res = {"args": vars(args), "L": L, "layers": REC, "acc": {}, "acc_by_line": {}}
for j in range(1, args.jmax + 1):
    out = [probe_acc(s, j) for s in range(S)]
    row = [o[0] for o in out]
    res["acc"][j] = row
    res["acc_by_line"][j] = {"first_line": j + 1, "by_step": [o[1] for o in out]}
    per_loop = [max(row[t * len(REC):(t + 1) * len(REC)]) for t in range(args.T)]
    print(f"ancestor {j:2d} levels up: best per loop " + " ".join(f"{a:.2f}" for a in per_loop), flush=True)
    save_json(res, f"{RESULTS}/e87_ancestor_{args.tag}.json")
print("done")
