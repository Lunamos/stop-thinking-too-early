"""E44: where must the map act? Train the same single-layer map (rank r, at the input of block a) applied only at the
program's tokens, only at the query tokens (the 'print(X)\\nOutput:' suffix), or at all tokens, on two-chain programs of up
to dmax lines; evaluate on two- and three-chain programs, level-ordered and interleaved.
If program-only reproduces the unlock and query-only does not, the unlock is a change to what the context computes while it
is read (the relay); if query-only suffices, the relay is not needed.

usage: python e44_position_maps.py MODEL --tag TAG --where program|query|all [--a 14] [--rank 8] [--steps 1200]
"""
import argparse
import json
import random
import time

import torch

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS
from mqa_common import Map

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--where", required=True)
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dmax", type=int, default=16)
ap.add_argument("--n_eval", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
torch.manual_seed(args.seed)

model, tok = load_model(args.model, args.device)
for p in model.parameters():
    p.requires_grad_(False)
R = Runner(model, tok)
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
n_suffix = len(tok.encode(QF.format(q="K"), add_special_tokens=False))  # same for every single-token variable
M = Map(d_model, rank=args.rank).to(args.device)
USE = {"on": True}


def hook(s, li, h):
    if not USE["on"] or s != args.a - 1:
        return h
    if args.where == "all":
        return M(h)
    h2 = h.clone()
    if args.where == "program":
        h2[:, :-n_suffix] = M(h[:, :-n_suffix])
    else:
        h2[:, -n_suffix:] = M(h[:, -n_suffix:])
    return h2


def items(rng, n, dmin, dmax, nch=2, order="forward"):
    return [make_vb(rng, rng.randint(dmin, dmax), nch, order, header=HDR, query_fmt=QF, value_pool=list(vids),
                    var_pool=letters) for _ in range(n)]


@torch.no_grad()
def evaluate():
    res = {}
    for nch, order in ((2, "forward"), (2, "interleave"), (3, "forward"), (3, "interleave")):
        for d in (2, 4, 6, 8, 12, 16):
            if nch * d > len(letters):
                continue
            rng = random.Random(9000 + 100 * nch + d)
            its = [make_vb(rng, d, nch, order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
                   for _ in range(args.n_eval)]
            row = {}
            for mode in ("frozen", "map"):
                USE["on"] = mode == "map"
                acc = 0
                for i in range(0, len(its), 25):
                    ch = its[i:i + 25]
                    ids, am = R.encode([it.prompt for it in ch])
                    lg = R.unembed(R.run(ids, am, hook=hook)["h"][:, -1]).float()
                    for k, it in enumerate(ch):
                        rl = torch.stack([lg[k, vids[r]] for r in it.roots])
                        acc += int(it.roots[rl.argmax().item()] == it.answer)
                row[mode] = acc / len(its)
            USE["on"] = True
            res[f"c{nch}_{order}_d{d}"] = row
            print(f"  chains={nch} {order:10s} d={d:2d}: frozen {row['frozen']:.2f}  map({args.where}) {row['map']:.2f}", flush=True)
    return res


opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=0.0)
rng = random.Random(args.seed)
t0 = time.time()
log = []
for step in range(args.steps):
    its = items(rng, args.bs, 1, args.dmax, 2, "forward")
    ids, am = R.encode([it.prompt for it in its])
    tgt = torch.tensor([vids[it.answer] for it in its], device=args.device)
    USE["on"] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = R.unembed(R.run_grad(ids, am, hook=hook)["h"][:, -1]).float()
    loss = torch.nn.functional.cross_entropy(lg, tgt)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        acc = (lg.argmax(-1) == tgt).float().mean().item()
        log.append(dict(step=step, loss=loss.item(), acc=acc))
        print(f"step {step} loss {loss.item():.3f} acc {acc:.2f} t={time.time() - t0:.0f}", flush=True)
torch.save(M.state_dict(), f"{RESULTS}/e44_map_{args.tag}.pt")
out = {"args": vars(args), "log": log, "eval": evaluate()}
save_json(out, f"{RESULTS}/e44_posmap_{args.tag}.json")
print("done")
