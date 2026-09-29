"""E81c: does the Huginn map keep the model's text predictions? WikiText-103 perplexity of Huginn-0125, frozen and with an E81/E81b
map on the core's input adapter, at several recurrence counts (same random latent initialisation per chunk in both runs).

Runs in loopdyn/.venv-loop. usage: python e81c_huginn_ppl.py --map PATH --tag TAG [--rs 8,16] [--n 200] [--len 256]
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import RESULTS, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
ap.add_argument("--map", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--rs", default="8,16")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--len", type=int, default=256)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
model = model.to(args.device).eval()


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
ST = {"on": False}
model.transformer.adapter.register_forward_hook(lambda mod, inp, out: M(out) if ST["on"] else out)
chunks = []
for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test"):
    t = row["text"].strip()
    if len(t) > 1200 and not t.startswith("="):
        ids = tok(t, return_tensors="pt", add_special_tokens=True).input_ids[:, : args.len]
        if ids.shape[1] == args.len:
            chunks.append(ids)
    if len(chunks) >= args.n:
        break
res = {"args": vars(args), "ppl": {}}
with torch.no_grad():
    for r in [int(x) for x in args.rs.split(",")]:
        for on in (False, True):
            ST["on"] = on
            nll, cnt = 0.0, 0
            for i, ids in enumerate(chunks):
                ids = ids.to(args.device)
                torch.manual_seed(i)
                lg = model(input_ids=ids, num_steps=r).logits[0, :-1].float()
                nll += nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum").item()
                cnt += ids.shape[1] - 1
            ppl = math.exp(nll / cnt)
            res["ppl"][f"r{r}_{'map' if on else 'frozen'}"] = ppl
            print(f"r={r:2d} {'map' if on else 'frozen':6s} WikiText ppl {ppl:.3f} ({len(chunks)} chunks of {args.len})", flush=True)
            save_json(res, f"{RESULTS}/e81c_huginn_ppl_{args.tag}.json")
print("done")
