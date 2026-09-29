"""E77: the relay in a looped model with a map switched on. As E39 (binary read-out of which chain each line of the queried chain
belongs to, at its right-hand-side token, at every recorded step (loop, layer)), but with the rank-8 map of E71 applied at layer a of
the loop body in every loop, and the prompt format the map was trained with. Frozen and with the map.

usage (loopdyn/.venv-loop): python e77_ouro_relay_map.py ByteDance/Ouro-1.4B --map PATH --a 6 --tag TAG [--T 4] [--depth 16]
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import NOUNS, RESULTS, make_vb, save_json, LETTERS  # noqa: E402
from ld_loop import OuroRunner  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--T", type=int, default=6)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--mode", default="map", help="map | frozen")
ap.add_argument("--apply", default="all", help="all: map in every loop; first: only in the first loop")
ap.add_argument("--pool", default="letters", help="letters or both (plus single-token two-letter names)")
args = ap.parse_args()

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                          dtype=torch.bfloat16).to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
R = OuroRunner(model, tok)
import torch.nn as nn  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402


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


_sd = torch.load(args.map, map_location="cpu")
MAP = Map(model.config.hidden_size, _sd["A.weight"].shape[0]).to(args.device)
MAP.load_state_dict(_sd)


def run_rec(ids, T):
    inner = model.model
    am = torch.ones_like(ids)
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    pe = inner.rotary_emb(h, pos)
    rec = {}
    for t in range(T):
        for l in range(L):
            if args.mode == "map" and l == args.a and (args.apply == "all" or t == 0):
                h = MAP(h)
            out = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe, past_key_value=None, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
            if l == L - 1:
                h = inner.norm(h)
            rec[t * L + l] = h
    return {"h": h, "rec": rec}
L = R.L
S = args.T * L
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
if args.pool == "both":
    letters = letters + [x + y for x in LETTERS for y in LETTERS
                         if len(tok.encode(" " + x + y, add_special_tokens=False)) == 1 and len(tok.encode(x + y, add_special_tokens=False)) == 1]


def locate(it, toks):
    out = {}
    i = 0
    for ln in it.lines:
        while i < len(toks) - 3:
            if toks[i].strip() == ln["lhs"] and toks[i + 1].strip() == "=" and toks[i + 2].strip() == ln["rhs"] and toks[i + 3] == "\n":
                break
            i += 1
        if i >= len(toks) - 3:
            return None
        out[(ln["chain"], ln["level"])] = i + 2
        i += 4
    return out


def ridge_acc(X, y, ntr):
    X = torch.nan_to_num(X.double())
    mu = X[:ntr].mean(0, keepdim=True)
    Xc = X - mu
    Xc = Xc / Xc[:ntr].norm(dim=1).mean().clamp(min=1e-12)
    Xtr = Xc[:ntr]
    Y = torch.nn.functional.one_hot(y[:ntr], 2).double()
    Y = Y - Y.mean(0, keepdim=True)
    K = Xtr @ Xtr.T
    lam = 0.1 * K.diagonal().mean().item() + 1e-6
    alpha = torch.linalg.solve(K + lam * torch.eye(ntr, device=X.device, dtype=K.dtype), Y)
    return (Xc[ntr:] @ Xtr.T @ alpha).argmax(-1).eq(y[ntr:]).float().mean().item()


rng = random.Random(2026)
steps = [s for s in range(S) if s % 2 == 1 or s % L == 0]  # every 2nd layer and each loop start
feats = {k: [] for k in range(2, args.depth + 1)}
ys = []
racc = 0
n = 0
with torch.no_grad():
    while n < args.n:
        it = make_vb(rng, args.depth, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
        enc = tok(it.prompt, return_tensors="pt").to(args.device)
        toks = [tok.decode([t]) for t in enc.input_ids[0].tolist()]
        pos = locate(it, toks)
        if pos is None:
            continue
        r = run_rec(enc.input_ids, args.T)
        lg = R.logits(r["h"][:, -1]).float()[0]
        rl = torch.stack([lg[tok.encode(" " + v, add_special_tokens=False)[0]] for v in it.roots])
        racc += int(it.roots[rl.argmax().item()] == it.answer)
        q = it.meta["query_chain"]
        order = sorted((pos[(c, 1)], c) for c in range(2))
        ys.append([c for _, c in order].index(q))
        H = torch.stack([r["rec"][s][0] for s in steps])  # [len(steps), T, d]
        for k in range(2, args.depth + 1):
            feats[k].append(H[:, pos[(q, k)]].float().cpu())
        n += 1
y = torch.tensor(ys, device=args.device)
ntr = int(0.75 * n)
grid = {}
for k in range(2, args.depth + 1):
    X = torch.stack(feats[k]).to(args.device)  # [n, steps, d]
    grid[k] = [ridge_acc(X[:, i], y, ntr) for i in range(len(steps))]
out = {"model": args.model, "T": args.T, "L": L, "steps": steps, "racc_final": racc / n, "grid": grid}
save_json(out, f"{RESULTS}/e77_ouro_relay_{args.tag}_{args.mode}.json")
print("task racc after", args.T, "loops:", racc / n, flush=True)
for k in range(2, args.depth + 1):
    first = next((steps[i] for i, a in enumerate(grid[k]) if a >= 0.75), None)
    loop_first = None if first is None else divmod(first, L)
    per_loop = []
    for t in range(args.T):
        idx = [i for i, s in enumerate(steps) if s // L == t]
        per_loop.append(max(grid[k][i] for i in idx))
    print(f"line {k}: first step >=.75 {first} (loop, layer) {loop_first}; best per loop " +
          " ".join(f"{a:.2f}" for a in per_loop), flush=True)
