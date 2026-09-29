"""E84: chain-selective attention strides per recurrence in Huginn. As E82 for Ouro: at the token naming the previous variable on
line k of a chain, attention to the same token on the line j levels earlier in the same chain minus attention to the other chain's
line at that level; the best head per (recurrence, core layer, stride) is chosen on half of the programs and scored on the other half;
frozen and with an E81/E81b map on the core's input adapter. The minimal Huginn code calls scaled_dot_product_attention, so the
attention probabilities of the core layers are recomputed from q and k inside a wrapper (causal softmax), for the rows needed only.

Runs in loopdyn/.venv-loop. usage: python e84_huginn_strides.py --map PATH --tag TAG [--r 12] [--depth 16] [--n 60] [--pool letters]
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
ap.add_argument("--map", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--r", type=int, default=12)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--jmax", type=int, default=15)
ap.add_argument("--pool", default="letters")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
model = model.to(args.device).eval()
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]


def single(x):
    return len(tok.encode(" " + x, add_special_tokens=False)) == 1 and len(tok.encode(x, add_special_tokens=False)) == 1


letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz") if single(c)]
if args.pool == "both":
    letters += [x + y for x in LETTERS for y in LETTERS if single(x + y)]


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
M = Map(model.config.n_embd, sd["A.weight"].shape[0]).to(args.device)
M.load_state_dict(sd)
core = model.transformer.core_block
NL = len(core)
ST = {"on": False, "cur": None, "rec": -1, "rows": None, "cap": []}


def map_hook(mod, inp, out):
    return M(out) if ST["on"] else out


model.transformer.adapter.register_forward_hook(map_hook)


def pre(l):
    def f(mod, inp):
        if l == 0:
            ST["rec"] += 1
        ST["cur"] = l
    return f


def post(mod, inp, out):
    ST["cur"] = None


for l, blk in enumerate(core):
    blk.attn.register_forward_pre_hook(pre(l))
    blk.attn.register_forward_hook(post)

_sdpa = F.scaled_dot_product_attention


def sdpa(q, k, v, *a, **kw):
    if ST["cur"] is not None:
        rows = ST["rows"]
        qq = q[0, :, rows].float()                                  # [H, R, hd]
        kk = k[0].float()                                           # [Hkv, S, hd]
        if kk.shape[0] != qq.shape[0]:
            kk = kk.repeat_interleave(qq.shape[0] // kk.shape[0], 0)
        s = qq @ kk.transpose(1, 2) / math.sqrt(q.shape[-1])        # [H, R, S]
        S = kk.shape[1]
        s = s.masked_fill(torch.arange(S, device=s.device)[None, None, :] > rows.to(s.device)[None, :, None], float("-inf"))
        ST["cap"].append(torch.softmax(s, -1))
    return _sdpa(q, k, v, *a, **kw)


F.scaled_dot_product_attention = sdpa
torch.nn.functional.scaled_dot_product_attention = sdpa


def locate_rhs(it, prompt, offsets):
    out, cur = {}, 0
    for ln in it.lines:
        text = f"{ln['lhs']} = {ln['rhs']}\n"
        i = prompt.index(text, cur)
        end = i + len(ln["lhs"]) + 3 + len(ln["rhs"])
        out[(ln["chain"], ln["level"])] = max(t for t, (a, b) in enumerate(offsets) if a < end and b > 0 and a >= i)
        cur = i + len(text)
    return out


D, J = args.depth, args.jmax
res = {"args": vars(args), "layers_per_rec": NL}
for mode in ("frozen", "map"):
    ST["on"] = mode == "map"
    rng = random.Random(8282)
    Ss, Os = [], []
    with torch.no_grad():
        for i in range(args.n):
            it = make_vb(rng, D, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
            enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            pos = locate_rhs(it, it.prompt, offsets)
            src = [(c, k) for c in (0, 1) for k in range(2, D + 1)]
            ST["rows"] = torch.tensor([pos[x] for x in src], device=args.device)
            ST["cap"], ST["rec"] = [], -1
            torch.manual_seed(i)
            model(input_ids=enc.input_ids.to(args.device), num_steps=args.r)
            steps = []
            for A in ST["cap"]:                                     # one per (recurrence, core layer), in order
                H = A.shape[0]
                s = torch.zeros(H, J, device=A.device)
                o = torch.zeros(H, J, device=A.device)
                cnt = torch.zeros(J, device=A.device)
                for ri, (c, k) in enumerate(src):
                    for j in range(1, min(J, k - 1) + 1):
                        s[:, j - 1] += A[:, ri, pos[(c, k - j)]]
                        o[:, j - 1] += A[:, ri, pos[(1 - c, k - j)]]
                        cnt[j - 1] += 1
                steps.append(((s / cnt.clamp(min=1)).cpu().numpy(), (o / cnt.clamp(min=1)).cpu().numpy()))
            Ss.append(np.stack([x[0] for x in steps]))
            Os.append(np.stack([x[1] for x in steps]))
    Ss, Os = np.stack(Ss), np.stack(Os)                             # [n, r*NL, H, J]
    Dd = Ss - Os
    half = args.n // 2
    best = Dd[:half].mean(0).argmax(1)
    Dt, St, Ot = Dd[half:].mean(0), Ss[half:].mean(0), Os[half:].mean(0)
    S_, H_, J_ = Dt.shape
    diff = np.array([[Dt[s, best[s, j], j] for j in range(J_)] for s in range(S_)])
    res[mode] = {"diff": diff.tolist(), "head": best.tolist(), "same_full": St.round(5).tolist(), "other_full": Ot.round(5).tolist()}
    print(mode, flush=True)
    for t in range(args.r):
        blk = diff[t * NL:(t + 1) * NL]
        cells = []
        for j in range(J_):
            l = int(blk[:, j].argmax())
            cells.append(f"j{j + 1}:{blk[l, j]:.2f}@{l}")
        print(f"  rec {t + 1:2d}: " + " ".join(cells), flush=True)
    save_json(res, f"{RESULTS}/e84_huginn_strides_{args.tag}.json")
print("done")
