"""E19: retrofit a loop into a frozen pretrained LM by learning ONLY a small
re-entry adapter at the loop boundary.

Schedule with K re-entries of band [a, b]:
    blocks 0..b ; repeat K times { h <- M(h) ; blocks a..b } ; blocks b+1..N-1
M(h) = s*h + rms(h) * B(A(h / rms(h)))   (rank r, B zero-init => starts as a naive loop)
Only A, B, s are trained (model frozen). Training: VB depth ~ U{1..dmax}, K = k_train.
Evaluation: accuracy for depth 1..8 and K = 0..Kmax (K=0 is the unmodified model).
Control: --no_loop inserts M once at block a without repeating the band.

usage: python e19_reentry.py MODEL --tag TAG --band 14:23 [--rank 64] [--steps 600]
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn

from ld_common import RESULTS, Runner, load_model, make_vb, make_contain, save_json, value_token_ids, LETTERS

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--band", required=True)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dmax", type=int, default=5)
ap.add_argument("--k_train", type=int, default=1)
ap.add_argument("--k_train_set", default=None, help="comma list; sample K per step")
ap.add_argument("--text_kl", type=float, default=0.0, help="weight of KL(base||adapted) on WikiText")
ap.add_argument("--depths_eval", default="1,2,3,4,5,6,7,8")
ap.add_argument("--kmax", type=int, default=4)
ap.add_argument("--no_loop", action="store_true")
ap.add_argument("--task", default="vb", help="vb | contain_in | contain_out | mix (vb+contain_in)")
ap.add_argument("--var_pool", default="upper", help="upper | upperlower")
ap.add_argument("--order", default="forward", help="program order for training and evaluation: forward | reverse | interleave")
ap.add_argument("--steer", action="store_true", help="constant steering vector: M(h)=s*h+rms(h)*v")
ap.add_argument("--neval", type=int, default=200)
ap.add_argument("--chains", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing of the layers after the map (large models)")
args = ap.parse_args()
torch.manual_seed(args.seed)

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N = R.n_layers
A_ = R.A
a, b = (int(x) for x in args.band.split(":"))
vids = value_token_ids(tok)
vals = list(vids)
_pool = LETTERS + (list("abcdefghijklmnopqrstuvwxyz") if args.var_pool in ("upperlower", "both") else [])
letters = [c for c in _pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1
           and len(tok.encode(c, add_special_tokens=False)) == 1]
if args.var_pool == "both":
    letters = letters + [x + y for x in LETTERS for y in LETTERS
                         if len(tok.encode(" " + x + y, add_special_tokens=False)) == 1
                         and len(tok.encode(x + y, add_special_tokens=False)) == 1]
d_model = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size


class ReEntry(nn.Module):
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
        y = self.s * x + rms * self.B(self.A(x / rms))
        return y.to(h.dtype)


class Steer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(d))
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.v).to(h.dtype)


M = (Steer(d_model) if args.steer else ReEntry(d_model, args.rank)).to(args.device)
TEXTS = []
if args.text_kl > 0:
    from datasets import load_dataset
    _ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    for _r in _ds:
        _t = _r["text"].strip()
        if len(_t) > 600 and not _t.startswith("="):
            TEXTS.append(_t[:700])
        if len(TEXTS) >= 4000:
            break


def _layer_call(j, h, masks, pe, pos):
    lt = A_.layer_types[j]
    out = A_.layers[j](h, attention_mask=masks[lt], position_embeddings=pe[lt], position_ids=pos,
                       past_key_values=None, use_cache=False)
    return out if torch.is_tensor(out) else out[0]


def layer_call(j, h, masks, pe, pos):
    if args.ckpt and torch.is_grad_enabled() and h.requires_grad:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(_layer_call, j, h, masks, pe, pos, use_reentrant=False)
    return _layer_call(j, h, masks, pe, pos)


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


def gen(rng, d, task, nch=None):
    nch = nch or args.chains
    if task == "mix":
        task = rng.choice(["vb", "contain_in"])
    if task == "vb":
        return make_vb(rng, d, nch, args.order, header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)
    return make_contain(rng, d, nch, "inward" if task == "contain_in" else "outward", value_pool=vals)


def batch(rng, n, dmin, dmax):
    items = [gen(rng, rng.randint(dmin, dmax), args.task) for _ in range(n)]
    ids, am = R.encode([it.prompt for it in items])
    tgt = torch.tensor([vids[it.answer] for it in items], device=args.device)
    return ids, am, tgt, items


@torch.no_grad()
def evaluate(Ks, depths, n):
    res = {}
    for d in depths:
        rng = random.Random(10_000 + d)
        nch = min(args.chains, max(2, len(letters) // d))
        items = [gen(rng, d, EVAL_TASK, nch) for _ in range(n)]
        for K in Ks:
            acc = 0
            for i in range(0, n, 25):
                chunk = items[i:i + 25]
                ids, am = R.encode([it.prompt for it in chunk])
                lg = R.unembed(forward(ids, am, K)[:, -1]).float()
                acc += sum(int(lg[k].argmax().item() == vids[it.answer]) for k, it in enumerate(chunk))
            res[f"d{d}_K{K}"] = acc / n
    return res


depths = [int(x) for x in args.depths_eval.split(",")]
Ks = list(range(0, args.kmax + 1))
t0 = time.time()
out = {"model": args.model, "band": [a, b], "rank": args.rank, "args": vars(args), "log": []}
EVAL_TASK = "vb" if args.task in ("vb", "mix") else args.task
pre = evaluate([0, 1], [3, 4, 5, 6], 100)
out["pre"] = pre
print("PRE (naive loop, B=0):", json.dumps(pre), flush=True)
opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=0.0)
rng = random.Random(args.seed)
for step in range(args.steps):
    ids, am, tgt, _ = batch(rng, args.bs, 1, args.dmax)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kk = rng.choice([int(x) for x in args.k_train_set.split(",")]) if args.k_train_set else args.k_train
        h = forward(ids, am, kk)
        lg = R.unembed(h[:, -1]).float()
    loss = torch.nn.functional.cross_entropy(lg, tgt)
    if args.text_kl > 0:
        tb = rng.sample(TEXTS, 8)
        tids, tam = R.encode(tb)
        tids, tam = tids[:, :160], tam[:, :160]
        with torch.no_grad():
            h0 = forward(tids, tam, 0)
            lp0 = torch.log_softmax(R.unembed(h0).float(), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h1 = forward(tids, tam, max(1, args.k_train))
        lp1 = torch.log_softmax(R.unembed(h1).float(), -1)
        m = tam.bool().unsqueeze(-1)
        kl = ((lp0.exp() * (lp0 - lp1)) * m).sum() / tam.sum()
        loss = loss + args.text_kl * kl
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        acc = (lg.argmax(-1) == tgt).float().mean().item()
        out["log"].append(dict(step=step, loss=loss.item(), acc=acc))
        print(f"step {step} loss {loss.item():.3f} acc {acc:.2f} s={M.s.item():.3f} t={time.time()-t0:.0f}", flush=True)
EVAL_TASK = "vb" if args.task in ("vb", "mix") else args.task
res = evaluate(Ks, depths, args.neval)
out["eval"] = res
if args.task == "mix":
    EVAL_TASK = "contain_in"
    out["eval_contain_in"] = evaluate(Ks, depths, args.neval)
    for d in depths:
        print(f"contain_in d={d}: " + " ".join(f"K{K}={out['eval_contain_in'][f'd{d}_K{K}']:.2f}" for K in Ks), flush=True)
    EVAL_TASK = "vb"
for d in depths:
    print(f"d={d}: " + " ".join(f"K{K}={res[f'd{d}_K{K}']:.2f}" for K in Ks), flush=True)
torch.save(M.state_dict(), f"{RESULTS}/e19_adapter_{args.tag}.pt")
save_json(out, f"{RESULTS}/e19_reentry_{args.tag}.json")
print("done", time.time() - t0)
