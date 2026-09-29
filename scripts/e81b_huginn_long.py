"""E81b: the Huginn map on long chains. Same map and hook as E81 (rank-r map on the output of the core's input adapter, so it acts
at the start of the 4-layer core in every recurrence), with three changes: variable names come from single letters plus single-token
two-letter names (so chains up to 64 lines use one name pool in training and evaluation), a KL penalty to the frozen model on WikiText
keeps the map's text predictions (as for the Ouro maps), and the evaluation extends to 64-line chains and 64 recurrences. With --load
the training is skipped and a saved map is evaluated.

Runs in loopdyn/.venv-loop. Batch size 1 per forward (the minimal Huginn model does not support padding masks).
usage: python e81b_huginn_long.py --tag TAG [--r_train 8] [--dmax 24] [--steps 800] [--load results/e81_map_X.pt]
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
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dmax", type=int, default=24)
ap.add_argument("--dmin", type=int, default=0, help="if > 0: curriculum, the longest training chain grows linearly from dmin to dmax over the first half of training")
ap.add_argument("--text_kl", type=float, default=1.0)
ap.add_argument("--rs_eval", default="1,2,4,8,16,32,64")
ap.add_argument("--depths_eval", default="2,4,8,12,16,24,32,48,64")
ap.add_argument("--neval", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--load", default="")
ap.add_argument("--apply", default="all", help="all: the map acts at the start of every recurrence; first: only in the first recurrence")
ap.add_argument("--pool", default="both", help="both: single letters plus two-letter names; letters: single letters only")
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


def single(x):
    return len(tok.encode(" " + x, add_special_tokens=False)) == 1 and len(tok.encode(x, add_special_tokens=False)) == 1


letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz") if single(c)]
if args.pool == "both":
    letters += [x + y for x in LETTERS for y in LETTERS if single(x + y)]
d_model = model.config.n_embd
print("names", len(letters), "values", len(vals), "width", d_model, flush=True)


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
STATE = {"on": False, "rec": 0}


def reset(mod, inp):
    STATE["rec"] = 0


def hook(mod, inp, out):
    STATE["rec"] += 1
    return M(out) if STATE["on"] and (args.apply == "all" or STATE["rec"] == 1) else out


model.transformer.prelude[0].register_forward_pre_hook(reset)
model.transformer.adapter.register_forward_hook(hook)


def run(ids, r, grad=False, seed=0):
    torch.manual_seed(seed)
    steps = (torch.tensor(0), torch.tensor(r)) if grad else r
    return model(input_ids=ids, num_steps=steps).logits[0].float()


def enc(text):
    return tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)


def gen(rng, d):
    return make_vb(rng, d, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)


@torch.no_grad()
def evaluate(r, on, d):
    STATE["on"] = on
    rng = random.Random(81_000 + d)
    ex = ch = 0
    for i in range(args.neval):
        it = gen(rng, d)
        lg = run(enc(it.prompt), r, seed=i)[-1]
        ex += int(lg.argmax().item() == vids[it.answer])
        rl = torch.stack([lg[vids[x]] for x in it.roots])
        ch += int(it.roots[rl.argmax().item()] == it.answer)
    return ex / args.neval, ch / args.neval


res = {"args": vars(args), "log": [], "eval": {}}
t0 = time.time()
if args.load:
    M.load_state_dict(torch.load(args.load, map_location=args.device))
else:
    TEXTS = []
    if args.text_kl > 0:
        from datasets import load_dataset
        for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"):
            t = row["text"].strip()
            if len(t) > 600 and not t.startswith("="):
                TEXTS.append(t[:700])
            if len(TEXTS) >= 4000:
                break
    opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    for step in range(args.steps):
        opt.zero_grad()
        tot, kl_tot, acc = 0.0, 0.0, 0
        for k in range(args.accum):
            dcur = args.dmax if args.dmin <= 0 else min(args.dmax, int(args.dmin + (args.dmax - args.dmin) * step / (0.5 * args.steps)))
            it = gen(rng, rng.randint(1, dcur))
            STATE["on"] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = run(enc(it.prompt), args.r_train, grad=True, seed=step * args.accum + k)[-1]
            loss = nn.functional.cross_entropy(lg[None], torch.tensor([vids[it.answer]], device=args.device))
            if args.text_kl > 0 and k % 4 == 0:
                tids = enc(rng.choice(TEXTS))[:, :160]
                STATE["on"] = False
                with torch.no_grad():
                    lp0 = torch.log_softmax(run(tids, args.r_train, seed=10**6 + step), -1)
                STATE["on"] = True
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp1 = torch.log_softmax(run(tids, args.r_train, grad=True, seed=10**6 + step), -1)
                kl = (lp0.exp() * (lp0 - lp1)).sum(-1).mean()
                loss = loss + args.text_kl * kl
                kl_tot += kl.item()
            (loss / args.accum).backward()
            tot += loss.item() / args.accum
            acc += int(lg.argmax().item() == vids[it.answer])
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            res["log"].append(dict(step=step, loss=tot, kl=kl_tot, acc=acc / args.accum))
            print(f"step {step} loss {tot:.3f} kl {kl_tot:.4f} acc {acc / args.accum:.2f} s={M.s.item():.3f} "
                  f"t={time.time() - t0:.0f}", flush=True)
    torch.save(M.state_dict(), f"{RESULTS}/e81_map_{args.tag}.pt")
for r in [int(x) for x in args.rs_eval.split(",")]:
    for on in (False, True):
        row = {d: evaluate(r, on, d) for d in [int(x) for x in args.depths_eval.split(",")]}
        res["eval"][f"r{r}_{'map' if on else 'frozen'}"] = row
        print(f"r={r:2d} {'map   ' if on else 'frozen'} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()),
              flush=True)
        save_json(res, f"{RESULTS}/e81_huginn_{args.tag}.json")
print("done", time.time() - t0)
