"""E71c: controls for the map in Ouro. (1) WikiText perplexity at T loops, frozen and with the map. (2) Specificity: programs in
reversed order (every line names a variable defined later) and interleaved order, frozen and with the forward-trained map, at T=4
and T=8. Exact accuracy and choice between the two roots.

Runs in loopdyn/.venv-loop.
usage: python e71c_ouro_controls.py --map PATH --a 6 --tag TAG
"""
import argparse
import math
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
ap.add_argument("--map", required=True)
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
inner = model.model
L = len(inner.layers)
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
one = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
       if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]


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
M = Map(model.config.hidden_size, sd["A.weight"].shape[0]).to(args.device)
M.load_state_dict(sd)


@torch.no_grad()
def hidden(ids, am, T, use_map):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    pe = inner.rotary_emb(h, pos)
    for t in range(T):
        for l in range(L):
            if use_map and l == args.a:
                h = M(h)
            out = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe, past_key_value=None, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        h = inner.norm(h)
    return h


res = {"args": vars(args), "ppl": {}, "order": {}}
from datasets import load_dataset  # noqa: E402

texts = [r["text"].strip() for r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")]
texts = [t for t in texts if len(t) > 800 and not t.startswith("=")][:48]
tok.padding_side = "right"
for T in (4,):
    for use_map in (False, True):
        nll = cnt = 0.0
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256).to(args.device)
            lg = model.lm_head(hidden(enc.input_ids, enc.attention_mask, T, use_map)).float()
            lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0]
            m = enc.attention_mask[:, 1:].float()
            nll += -(lp * m).sum().item()
            cnt += m.sum().item()
        res["ppl"][f"T{T}_{'map' if use_map else 'frozen'}"] = math.exp(nll / cnt)
        print(f"WikiText perplexity T={T} {'map' if use_map else 'frozen'}: {math.exp(nll / cnt):.3f}", flush=True)
tok.padding_side = "left"
for order in ("reverse", "interleave"):
    for T in (4, 8):
        for use_map in (False, True):
            row = {}
            for d in (2, 4, 8, 16):
                rng = random.Random(7100 + d)
                items = [make_vb(rng, d, 2, order, header=HDR, query_fmt=QF, value_pool=vals, var_pool=one) for _ in range(args.n)]
                ex = ch = 0
                for i in range(0, len(items), 25):
                    chunk = items[i:i + 25]
                    enc = tok([it.prompt for it in chunk], return_tensors="pt", padding=True).to(args.device)
                    lg = model.lm_head(hidden(enc.input_ids, enc.attention_mask, T, use_map)[:, -1]).float()
                    for k, it in enumerate(chunk):
                        ex += int(lg[k].argmax().item() == vids[it.answer])
                        rl = torch.stack([lg[k, vids[x]] for x in it.roots])
                        ch += int(it.roots[rl.argmax().item()] == it.answer)
                row[d] = (ex / len(items), ch / len(items))
            key = f"{order}_T{T}_{'map' if use_map else 'frozen'}"
            res["order"][key] = row
            print(key, " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()), flush=True)
            save_json(res, f"{RESULTS}/e71c_controls_{args.tag}.json")
