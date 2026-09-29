"""E33: do deep maps chase pointers, or exploit the level structure of forward-ordered prompts?
With two chains in forward order, every level holds one line of each chain, so the root could in principle be
found from the parity of 'crossings' between levels without following pointers. Evaluate a map (and the frozen
model) on prompts whose lines are (i) forward ordered, (ii) randomly interleaved while keeping each chain in order
(no level structure), and (iii) with three chains.

usage: python e33_order_control.py MODEL --map PATH --a 14 --tag TAG [--depths 4,8,12,16,20]
"""
import argparse
import random

import torch
import torch.nn as nn

from ld_common import RESULTS, Runner, load_model, make_vb, save_json, value_token_ids, LETTERS

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--tag", required=True)
ap.add_argument("--depths", default="4,8,12,16,20")
ap.add_argument("--conds", default="2:forward,2:interleave,3:forward,3:interleave")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--var_pool", default="letters", help="letters | two | both")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
d_model = model.config.get_text_config().hidden_size
vids = value_token_ids(tok)
pool = LETTERS + list("abcdefghijklmnopqrstuvwxyz")
letters = [c for c in pool if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
two = [a + b for a in LETTERS for b in LETTERS
       if len(tok.encode(" " + a + b, add_special_tokens=False)) == 1 and len(tok.encode(a + b, add_special_tokens=False)) == 1]
letters = {"letters": letters, "two": two, "both": letters + two}[args.var_pool]
print("variable pool", args.var_pool, len(letters), flush=True)


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


from mqa_common import Map as _QMap
_sd = torch.load(args.map, map_location="cpu")
M = _QMap.from_state(_sd, d_model).to(args.device)
hooks = {"frozen": None, "map": lambda s, li, h: M(h) if s == args.a - 1 else h}
out = {"model": args.model, "map": args.map, "a": args.a, "res": {}}
with torch.no_grad():
    for cond in args.conds.split(","):
        nch, order = cond.split(":")
        nch = int(nch)
        for d in [int(x) for x in args.depths.split(",")]:
            if nch * d > len(letters):
                continue
            rng = random.Random(7000 + 100 * nch + d)
            items = [make_vb(rng, d, nch, order, header=HDR, query_fmt=QF, value_pool=list(vids), var_pool=letters)
                     for _ in range(args.n)]
            row = {}
            for mode, hook in hooks.items():
                acc = racc = 0
                bs = 20 if d <= 12 else 10
                for i in range(0, len(items), bs):
                    chunk = items[i:i + bs]
                    ids, am = R.encode([it.prompt for it in chunk])
                    lg = R.unembed(R.run(ids, am, hook=hook)["h"][:, -1]).float()
                    for k, it in enumerate(chunk):
                        acc += int(lg[k].argmax().item() == vids[it.answer])
                        rl = torch.stack([lg[k, vids[r]] for r in it.roots])
                        racc += int(it.roots[rl.argmax().item()] == it.answer)
                row[mode] = dict(acc=acc / len(items), racc=racc / len(items))
            out["res"][f"c{nch}_{order}_d{d}"] = row
            print(f"chains={nch} order={order:10s} d={d:2d}  frozen acc {row['frozen']['acc']:.2f} racc {row['frozen']['racc']:.2f}"
                  f"   map acc {row['map']['acc']:.2f} racc {row['map']['racc']:.2f}", flush=True)
save_json(out, f"{RESULTS}/e33_order_{args.tag}.json")
