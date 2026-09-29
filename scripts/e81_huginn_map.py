"""E81: a map in a second looped architecture. Huginn-0125 (2 prelude layers, a 4-layer recurrent core applied r times, 2 coda
layers). A rank-r map is applied to the output of the core's input adapter, i.e. at the start of the core in every recurrence, and
trained with r_train recurrences (gradients through all of them) on two-chain programs of up to dmax lines, answer cross-entropy
only. Evaluated for r = 1..32 recurrences, frozen and with the map: exact accuracy and choice between the two roots. The initial
latent state is drawn with a fixed seed for every forward, so frozen and mapped runs see the same initialisation.

Runs in loopdyn/.venv-loop. Batch size 1 per forward (the minimal Huginn model does not support padding masks).
usage: python e81_huginn_map.py --tag TAG [--r_train 8] [--steps 600] [--accum 8]
"""
import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
ap.add_argument("--tag", required=True)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--r_train", type=int, default=8)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dmax", type=int, default=12)
ap.add_argument("--rs_eval", default="1,2,4,8,16,32")
ap.add_argument("--depths_eval", default="1,2,3,4,6,8,12,16")
ap.add_argument("--neval", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
torch.manual_seed(args.seed)

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
model = model.to(args.device).eval()
for p in model.parameters():
    p.requires_grad_(False)
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
d_model = model.config.n_embd
print("letters", len(letters), "values", len(vals), "width", d_model, flush=True)


class Map(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.B.weight)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


M = Map(d_model, args.rank).to(args.device)
STATE = {"on": False}


def hook(mod, inp, out):
    return M(out) if STATE["on"] else out


model.transformer.adapter.register_forward_hook(hook)


def logits_last(prompt, r, grad=False, seed=0):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)
    torch.manual_seed(seed)
    steps = (torch.tensor(0), torch.tensor(r)) if grad else r
    out = model(input_ids=ids, num_steps=steps)
    return out.logits[0, -1].float()


def gen(rng, d):
    return make_vb(rng, d, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)


@torch.no_grad()
def evaluate(r, on, d):
    STATE["on"] = on
    rng = random.Random(81_000 + d)
    ex = ch = 0
    for i in range(args.neval):
        it = gen(rng, d)
        lg = logits_last(it.prompt, r, seed=i)
        ex += int(lg.argmax().item() == vids[it.answer])
        rl = torch.stack([lg[vids[x]] for x in it.roots])
        ch += int(it.roots[rl.argmax().item()] == it.answer)
    return ex / args.neval, ch / args.neval


opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=0.0)
rng = random.Random(args.seed)
t0 = time.time()
log = []
for step in range(args.steps):
    STATE["on"] = True
    opt.zero_grad()
    tot, acc = 0.0, 0
    for k in range(args.accum):
        it = gen(rng, rng.randint(1, args.dmax))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = logits_last(it.prompt, args.r_train, grad=True, seed=step * args.accum + k)
        loss = nn.functional.cross_entropy(lg[None], torch.tensor([vids[it.answer]], device=args.device)) / args.accum
        loss.backward()
        tot += loss.item()
        acc += int(lg.argmax().item() == vids[it.answer])
    torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
    opt.step()
    if step % 25 == 0:
        log.append(dict(step=step, loss=tot, acc=acc / args.accum))
        print(f"step {step} loss {tot:.3f} acc {acc / args.accum:.2f} s={M.s.item():.3f} t={time.time() - t0:.0f}", flush=True)
torch.save(M.state_dict(), f"{RESULTS}/e81_map_{args.tag}.pt")
res = {"args": vars(args), "log": log, "eval": {}}
for r in [int(x) for x in args.rs_eval.split(",")]:
    for on in (False, True):
        row = {d: evaluate(r, on, d) for d in [int(x) for x in args.depths_eval.split(",")]}
        res["eval"][f"r{r}_{'map' if on else 'frozen'}"] = row
        print(f"r={r:2d} {'map   ' if on else 'frozen'} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()), flush=True)
        save_json(res, f"{RESULTS}/e81_huginn_{args.tag}.json")
