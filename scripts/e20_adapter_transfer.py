"""E20: does the re-entry adapter (trained on code-format forward-order VB)
transfer to other surface forms of pointer chasing?

Formats: code (training), code_shuffled (random line order), js, natural.
Accuracy by depth and number of re-entries K (K=0 = frozen model).

usage: python e20_adapter_transfer.py MODEL --adapter PATH --band 14:23 --tag TAG [--rank 64]
"""
import argparse
import math
import random

import torch
import torch.nn as nn

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--adapter", required=True)
ap.add_argument("--band", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--depths", default="2,3,4,5,6,7")
ap.add_argument("--Ks", default="0,1,2")
ap.add_argument("--formats", default="code,code_shuffled,js,natural")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--no_loop", action="store_true", help="apply the map once at the input of block a (no band repeat)")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
A_ = R.A
a, b = (int(x) for x in args.band.split(":"))
vids = value_token_ids(tok)
vals = list(vids)
letters = [c for c in LETTERS if len(tok.encode(" " + c, add_special_tokens=False)) == 1
           and len(tok.encode(c, add_special_tokens=False)) == 1]
d_model = model.config.get_text_config().hidden_size


class ReEntry(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


M = ReEntry(d_model, args.rank).to(args.device)
M.load_state_dict(torch.load(args.adapter, map_location=args.device))


def layer_call(j, h, masks, pe, pos):
    lt = A_.layer_types[j]
    out = A_.layers[j](h, attention_mask=masks[lt], position_embeddings=pe[lt], position_ids=pos,
                       past_key_values=None, use_cache=False)
    return out if torch.is_tensor(out) else out[0]


@torch.no_grad()
def forward(ids, am, K):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = A_.embed(ids)
    masks = R._masks(h, am, pos)
    pe = {}
    for lt in set(A_.layer_types):
        try:
            pe[lt] = A_.rotary(h, pos, lt)
        except TypeError:
            pe[lt] = A_.rotary(h, pos)
    if args.no_loop:
        for j in range(0, a):
            h = layer_call(j, h, masks, pe, pos)
        if K > 0:
            h = M(h)
        for j in range(a, N):
            h = layer_call(j, h, masks, pe, pos)
        return h
    for j in range(0, b + 1):
        h = layer_call(j, h, masks, pe, pos)
    for _ in range(K):
        h = M(h)
        for j in range(a, b + 1):
            h = layer_call(j, h, masks, pe, pos)
    for j in range(b + 1, N):
        h = layer_call(j, h, masks, pe, pos)
    return h


def render(it, fmt):
    lines = it.lines
    q = it.query_var
    if fmt in ("code", "code_shuffled"):
        body = "".join(f"{l['lhs']} = {l['rhs']}\n" for l in lines)
        return "Here is a short program. Each line assigns a value to a variable.\n" + body + f"print({q})\nOutput:"
    if fmt == "js":
        def rhs(l):
            return f'"{l["rhs"]}"' if l["level"] == 1 else l["rhs"]
        body = "".join(f"let {l['lhs']} = {rhs(l)};\n" for l in lines)
        return "// JavaScript\n" + body + f"console.log({q});\n// prints:"
    if fmt == "natural":
        def sent(l):
            return f"{l['lhs']} means {l['rhs']}." if l["level"] == 1 else f"{l['lhs']} means the same as {l['rhs']}."
        body = " ".join(sent(l) for l in lines)
        return "Here are some definitions. " + body + f"\nQuestion: What does {q} mean?\nAnswer: {q} means"
    raise ValueError(fmt)


out = {"model": args.model, "band": [a, b], "adapter": args.adapter, "res": {}}
Ks = [int(x) for x in args.Ks.split(",")]
for fmt in args.formats.split(","):
    for d in [int(x) for x in args.depths.split(",")]:
        rng = random.Random(55_000 + d)
        order = "shuffled" if fmt == "code_shuffled" else "forward"
        nch = min(3, max(2, len(letters) // d))
        items = [make_vb(rng, d, nch, order, value_pool=vals, var_pool=letters) for _ in range(args.n)]
        prompts = [render(it, fmt) for it in items]
        row = {}
        for K in Ks:
            acc = 0
            for i in range(0, len(items), 25):
                ids, am = R.encode(prompts[i:i + 25])
                lg = R.unembed(forward(ids, am, K)[:, -1]).float()
                acc += sum(int(lg[k].argmax().item() == vids[it.answer]) for k, it in enumerate(items[i:i + 25]))
            row[K] = acc / len(items)
        out["res"][f"{fmt}_d{d}"] = row
        print(fmt, d, " ".join(f"K{K}={row[K]:.2f}" for K in Ks), flush=True)
    save_json(out, f"{RESULTS}/e20_transfer_{args.tag}.json")
print("done")
