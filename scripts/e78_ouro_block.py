"""E78: where in a looped model does the query read the program? Ouro-1.4B (T loops of 24 layers), frozen or with an E71 map.
Block the attention from the query tokens (print(X)\\nOutput:) to every program line except the root lines, in a sliding band of
`width` steps over the T x 24 steps, and measure choice accuracy among the roots (three chains, one to four lines; two chains when a
map is used, at the lengths given). Prediction (notes/routing_window.md, P2): blocking inside a loop's window (about layers 8-13)
of the last loops lowers accuracy to the level of one loop fewer; blocking elsewhere does not.

Runs in loopdyn/.venv-loop.
usage: python e78_ouro_block.py --tag TAG [--T 4] [--width 3] [--map PATH --a 6 --chains 2 --depths 8,16]
"""
import argparse
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
ap.add_argument("--T", type=int, default=4)
ap.add_argument("--width", type=int, default=3)
ap.add_argument("--map", default=None)
ap.add_argument("--a", type=int, default=6)
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--depths", default="2,3")
ap.add_argument("--n", type=int, default=100)
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
L = len(inner.layers)
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
pool = LETTERS if args.chains == 3 else LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
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


def build(items):
    enc = tok([it.prompt for it in items], return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    ids, am = enc.input_ids.to(args.device), enc.attention_mask.to(args.device)
    T = ids.shape[1]
    block = torch.zeros(len(items), T, T, dtype=torch.bool)
    for b, it in enumerate(items):
        off = [tuple(x) for x in offs[b].tolist()]
        cur, cols = 0, []
        for ln in it.lines:
            text = f"{ln['lhs']} = {ln['rhs']}\n"
            i = it.prompt.index(text, cur)
            j = i + len(text)
            if ln["level"] > 1:
                cols += [t for t, (s, e) in enumerate(off) if e > i and s < j and e > s]
            cur = j
        q0 = it.prompt.index("print(", cur)
        rows = [t for t, (s, e) in enumerate(off) if s >= q0 and e > s]
        if cols:
            block[b][torch.tensor(rows)[:, None], torch.tensor(cols)[None, :]] = True
    return ids, am, block.to(args.device)


@torch.no_grad()
def run(ids, am, block, steps):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    bl = mask.clone()
    bl[:, 0] = bl[:, 0].masked_fill(block, torch.finfo(h.dtype).min)
    pe = inner.rotary_emb(h, pos)
    for t in range(args.T):
        for l in range(L):
            if MAP is not None and l == args.a:
                h = MAP(h)
            out = inner.layers[l](h, attention_mask=(bl if t * L + l in steps else mask), position_ids=pos, position_embeddings=pe,
                                  past_key_value=None, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        h = inner.norm(h)
    return model.lm_head(h[:, -1]).float()


def accuracy(d, steps):
    rng = random.Random(7800 + d)
    items = [make_vb(rng, d, args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters) for _ in range(args.n)]
    ok = 0
    for i in range(0, len(items), 25):
        chunk = items[i:i + 25]
        ids, am, block = build(chunk)
        lg = run(ids, am, block, steps)
        for k, it in enumerate(chunk):
            rl = torch.stack([lg[k, vids[x]] for x in it.roots])
            ok += int(it.roots[rl.argmax().item()] == it.answer)
    return ok / len(items)


depths = [int(x) for x in args.depths.split(",")]
S = args.T * L
res = {"args": vars(args), "none": {d: accuracy(d, set()) for d in depths}, "all": {d: accuracy(d, set(range(S))) for d in depths},
       "loop": {}, "scan": {}}
print("none", res["none"], "| all steps blocked", res["all"], flush=True)
for t in range(args.T):
    res["loop"][t] = {d: accuracy(d, set(range(t * L, (t + 1) * L))) for d in depths}
    print(f"block loop {t + 1}: " + " ".join(f"d{d}:{v:.2f}" for d, v in res["loop"][t].items()), flush=True)
for s in range(0, S - args.width + 1, args.width):
    res["scan"][s] = {d: accuracy(d, set(range(s, s + args.width))) for d in depths}
    print(f"steps {s:3d}-{s + args.width - 1:3d} (loop {s // L + 1}, layers {s % L}-{(s + args.width - 1) % L}): " +
          " ".join(f"d{d}:{v:.2f}" for d, v in res["scan"][s].items()), flush=True)
    save_json(res, f"{RESULTS}/e78_ouro_block_{args.tag}.json")
