"""E38: is the relay necessary? Cut the attention from the right-hand-side token of every program line to the line that
defines its variable (the four tokens of that line), in a band of layers, for all heads; re-normalize. Measure accuracy
on chains of several lengths, frozen and mapped. Control: cut the same amount of attention to the same-level line of
another chain. Optional 'redirect': move attention from the defining line's newline to its right-hand-side token.

usage: python e38_relay_cut.py MODEL --map PATH --a 14 --tag TAG
"""
import argparse
import random

import torch
import torch.nn as nn
import transformers.models.qwen3.modeling_qwen3 as mq
import transformers.models.llama.modeling_llama as ml

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--cut", default="16:22", help="band (python range) where the mapped relay runs")
ap.add_argument("--late", default="22:28", help="band after the relay")
ap.add_argument("--fcut", default="4:18", help="band of the default relay (frozen)")
ap.add_argument("--redirect", default="17:22")
ap.add_argument("--depths", default="3,4,6,8,12")
ap.add_argument("--renorm", type=int, default=1, help="1: re-normalize after cutting; 0: remove the mass without re-normalizing")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

EDIT = {"layers": set(), "mode": None, "pairs": []}  # pairs: (q_pos, [src positions], dst position or None)


def edited_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = mq.repeat_kv(key, module.num_key_value_groups)
    value_states = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32)
    if module.layer_idx in EDIT["layers"] and EDIT["pairs"]:
        for qp, srcs, dst in EDIT["pairs"]:
            if EDIT["mode"] == "redirect":
                w[:, :, qp, dst] += w[:, :, qp, srcs].sum(-1)
                w[:, :, qp, srcs] = 0.0
            elif EDIT["mode"] == "massmatch":
                # remove the same attention mass the cut would remove, taken proportionally from all OTHER keys
                m = w[:, :, qp, srcs].sum(-1, keepdim=True)
                keep = torch.ones(w.shape[-1], dtype=torch.bool, device=w.device)
                keep[srcs] = False
                others = w[:, :, qp, :] * keep
                tot = others.sum(-1, keepdim=True).clamp(min=1e-9)
                w[:, :, qp, :] = torch.where(keep, others * (1 - m / tot).clamp(min=0), w[:, :, qp, :])
            else:
                w[:, :, qp, srcs] = 0.0
        if EDIT["mode"] not in ("redirect", "massmatch") and args.renorm:
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
    w = w.to(query.dtype)
    out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
    return out, w


mq.eager_attention_forward = edited_eager
ml.eager_attention_forward = edited_eager
model, tok = load_model(args.model, args.device, attn_implementation="eager")
R = Runner(model, tok)
N = R.n_layers
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


class Map(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(d_model, args.rank, bias=False)
        self.B = nn.Linear(args.rank, d_model, bias=False)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


M = Map().to(args.device)
M.load_state_dict(torch.load(args.map, map_location=args.device))


def starts(it, toks):
    out = {}
    i = 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        out[(ln["chain"], ln["level"])] = i
        i += 4
    return out


def pairs_for(it, toks, kind):
    st = starts(it, toks)
    nch = it.n_chains
    pr = []
    for (c, k), s in st.items():
        if k < 2:
            continue
        p = s + 2
        if kind == "ref":
            s0 = st[(c, k - 1)]
        else:  # control: same-level line of the next chain
            s0 = st[((c + 1) % nch, k - 1)]
        pr.append((p, [s0, s0 + 1, s0 + 2, s0 + 3], None))
    return pr


def redirect_pairs(it, toks):
    st = starts(it, toks)
    pr = []
    for (c, k), s in st.items():
        if k < 2:
            continue
        s0 = st[(c, k - 1)]
        pr.append((s + 2, [s0 + 3], s0 + 2))
    return pr


def evaluate(items, use_map, layers, kind):
    hook = (lambda s, li, h: M(h) if s == args.a - 1 else h) if use_map else None
    acc = 0
    for it in items:
        ids, am = R.encode([it.prompt])
        toks = [tok.decode([t]) for t in ids[0].tolist()]
        EDIT["layers"] = set(layers)
        if kind is None:
            EDIT["pairs"], EDIT["mode"] = [], None
        elif kind == "redirect":
            EDIT["pairs"], EDIT["mode"] = redirect_pairs(it, toks), "redirect"
        elif kind == "massmatch":
            EDIT["pairs"], EDIT["mode"] = pairs_for(it, toks, "ref"), "massmatch"
        else:
            EDIT["pairs"], EDIT["mode"] = pairs_for(it, toks, kind), "cut"
        lg = R.unembed(R.run(ids, am, hook=hook)["h"][:, -1]).float()[0]
        rl = torch.stack([lg[vids[r]] for r in it.roots])
        acc += int(it.roots[rl.argmax().item()] == it.answer)
    EDIT["pairs"] = []
    return acc / len(items)


out = {"model": args.model, "res": {}}
def band(x):
    a_, b_ = (int(v) for v in x.split(":"))
    return range(a_, b_), f"L{a_}-{b_ - 1}"


(cb, cn), (lb, ln_), (fb, fn), (rb, rn) = band(args.cut), band(args.late), band(args.fcut), band(args.redirect)
conds = [
    ("map", "none", True, [], None),
    ("map", f"cut {cn}", True, cb, "ref"),
    ("map", f"control {cn}", True, cb, "other"),
    ("map", f"massmatched {cn}", True, cb, "massmatch"),
    ("map", f"cut {ln_}", True, lb, "ref"),
    ("frozen", "none", False, [], None),
    ("frozen", f"cut {fn}", False, fb, "ref"),
    ("frozen", f"control {fn}", False, fb, "other"),
    ("frozen", f"redirect {rn}", False, rb, "redirect"),
]
with torch.no_grad():
    for d in [int(x) for x in args.depths.split(",")]:
        rng = random.Random(500 + d)
        items = [make_vb(rng, d, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
                 for _ in range(args.n)]
        for mode, name, use_map, layers, kind in conds:
            a = evaluate(items, use_map, layers, kind)
            out["res"][f"d{d}|{mode}|{name}"] = a
            print(f"d={d:2d} {mode:6s} {name:18s} acc(choice among roots) {a:.2f}", flush=True)
        save_json(out, f"{RESULTS}/e38_relaycut_{args.tag}.json")
