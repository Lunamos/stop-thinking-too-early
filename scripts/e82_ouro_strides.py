"""E82: chain-selective attention strides per loop in Ouro. As E37b (at the token naming the previous variable on line k of a chain,
attention to the same token on the line j levels earlier in the same chain, minus attention to the other chain's line at that level;
for each step and stride the best head is chosen on half of the programs and scored on the other half), but at every (loop, layer)
step of a looped model, frozen and with the E71 map applied at layer a of the loop body in every loop. Pointer doubling across loops
predicts that the longest chain-selective stride roughly doubles from loop to loop; a relay of one line per round predicts stride 1
in every loop.

Runs in loopdyn/.venv-loop (eager attention).
usage: python e82_ouro_strides.py --map PATH --a 6 --tag TAG [--T 4] [--depth 16] [--n 60] [--order forward|interleave]
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
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--T", type=int, default=4)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--jmax", type=int, default=15)
ap.add_argument("--order", default="forward")
ap.add_argument("--pool", default="letters")
ap.add_argument("--apply", default="all", help="all: map in every loop; first: only in the first loop")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                          attn_implementation="eager").to(args.device).eval()
inner = model.model
L = len(inner.layers)
d_model = model.config.hidden_size
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
if args.pool == "both":
    letters = letters + [x + y for x in LETTERS for y in LETTERS
                         if len(tok.encode(" " + x + y, add_special_tokens=False)) == 1 and len(tok.encode(x + y, add_special_tokens=False)) == 1]


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
M = Map(d_model, sd["A.weight"].shape[0]).to(args.device)
M.load_state_dict(sd)

CAP = {}


def make_hook(l):
    def hook(mod, inp, out):
        CAP[l] = out[1]
    return hook


for l in range(L):
    inner.layers[l].self_attn.register_forward_hook(make_hook(l))


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
def run_program(it, use_map):
    enc = tok(it.prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc.input_ids.to(args.device)
    pos_tok = locate_rhs(it, it.prompt, offsets)
    am = torch.ones_like(ids)
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    pe = inner.rotary_emb(h, pos)
    D, J = args.depth, args.jmax
    trip = [(pos_tok[(c, k)], pos_tok[(c, k - j)], pos_tok[(1 - c, k - j)], j - 1)
            for c in (0, 1) for k in range(1, D + 1) for j in range(1, min(J, k - 1) + 1)]
    Qi, Ks, Ko, Ji = (torch.tensor(x, device=args.device) for x in zip(*trip))
    cnt = torch.zeros(J, device=args.device).index_add_(0, Ji, torch.ones_like(Ji, dtype=torch.float)).clamp(min=1)
    rows = []
    for t in range(args.T):
        for l in range(L):
            if use_map and l == args.a and (args.apply == "all" or t == 0):
                h = M(h)
            out = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe, past_key_value=None,
                                  use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
            A = CAP[l][0].float()                                     # [H, T, T]
            H = A.shape[0]
            s = torch.zeros(H, J, device=A.device).index_add_(1, Ji, A[:, Qi, Ks])
            o = torch.zeros(H, J, device=A.device).index_add_(1, Ji, A[:, Qi, Ko])
            rows.append(((s / cnt).cpu().numpy(), (o / cnt).cpu().numpy()))
        h = inner.norm(h)
    same = np.stack([r[0] for r in rows])                              # [S, H, J]
    other = np.stack([r[1] for r in rows])
    return same, other


res = {"args": vars(args), "L": L}
for mode in ("frozen", "map"):
    rng = random.Random(8282)
    Ss, Os = [], []
    for i in range(args.n):
        it = make_vb(rng, args.depth, 2, args.order, header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
        s, o = run_program(it, mode == "map")
        Ss.append(s)
        Os.append(o)
    Ss, Os = np.stack(Ss), np.stack(Os)                                # [n, S, H, J]
    Dd = Ss - Os
    half = args.n // 2
    best = Dd[:half].mean(0).argmax(1)                                 # [S, J]
    Dt, St, Ot = Dd[half:].mean(0), Ss[half:].mean(0), Os[half:].mean(0)
    S_, H_, J_ = Dt.shape
    diff = np.array([[Dt[s, best[s, j], j] for j in range(J_)] for s in range(S_)])
    same = np.array([[St[s, best[s, j], j] for j in range(J_)] for s in range(S_)])
    oth = np.array([[Ot[s, best[s, j], j] for j in range(J_)] for s in range(S_)])
    res[mode] = {"diff": diff.tolist(), "same": same.tolist(), "other": oth.tolist(), "head": best.tolist(),
                 "same_full": St.round(5).tolist(), "other_full": Ot.round(5).tolist()}   # [S, H, J], held-out half
    print(mode, flush=True)
    for t in range(args.T):
        blk = diff[t * L:(t + 1) * L]                                  # [L, J]
        cells = []
        for j in range(J_):
            l = int(blk[:, j].argmax())
            cells.append(f"j{j + 1}:{blk[l, j]:.2f}@{l}")
        print(f"  loop {t + 1}: " + " ".join(cells), flush=True)
    save_json(res, f"{RESULTS}/e82_strides_{args.tag}.json")
print("done")
