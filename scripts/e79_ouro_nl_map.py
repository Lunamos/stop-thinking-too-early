"""E79: a map in a looped model, trained on natural-language multi-hop questions (synthetic chains of facts about fictional
entities, 1-4 hops, two distractor chains; target: first token of the answer entity). Evaluated by hops (1-8) for 1-8 loops, with and
without the map: choice among the first tokens of the three chains' final entities. (Derived from E71.)

E71: a map in a looped model. Ouro-1.4B applies its 24 layers T times (trained with T=4). We insert the rank-r map of Eq. 1 at the
input of layer a of the loop body, applied in every loop (or only in the first), train it like the maps on standard models (answer
cross-entropy on two-chain programs of up to dmax lines, KL penalty on WikiText at T=4), and evaluate exact accuracy and choice
among the roots against chain length for several numbers of loops, with and without the map. Does the map extend the relay, and do
more loops then extend the reach?

Runs in loopdyn/.venv-loop (transformers 4.56).
usage: python e71_ouro_map.py --a 6 --tag TAG [--apply all|first] [--rank 8] [--steps 1200]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
ap.add_argument("--a", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--apply", default="all", help="all: the map acts in every loop; first: only in the first loop")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dmax", type=int, default=4)
ap.add_argument("--T_train", type=int, default=4)
ap.add_argument("--text_kl", type=float, default=1.0)
ap.add_argument("--Ts_eval", default="1,2,3,4,6,8")
ap.add_argument("--depths_eval", default="1,2,3,4,5,6,8")
ap.add_argument("--neval", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--load", default="", help="evaluate a saved map (e.g. a program-trained E71 map) instead of training one")
args = ap.parse_args()
torch.manual_seed(args.seed)

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).to(args.device).eval()
for p in model.parameters():
    p.requires_grad_(False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
inner = model.model
L = len(inner.layers)
d_model = model.config.hidden_size
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
print("layers per loop", L, "values", len(vals), "letters", len(letters), flush=True)


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


def encode(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True).to(args.device)
    return enc.input_ids, enc.attention_mask


def _layer(l, h, mask, pos, pe):
    out = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe, past_key_value=None, use_cache=False)
    return out[0] if isinstance(out, tuple) else out


def forward(ids, am, T, use_map):
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = inner.embed_tokens(ids)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    pe = inner.rotary_emb(h, pos)
    for t in range(T):
        for l in range(L):
            if use_map and l == args.a and (args.apply == "all" or t == 0):
                h = M(h)
            if torch.is_grad_enabled() and h.requires_grad:
                h = checkpoint(_layer, l, h, mask, pos, pe, use_reentrant=False)
            else:
                h = _layer(l, h, mask, pos, pe)
        h = inner.norm(h)
    return h


from synth_mhqa import REL, Names, chain  # noqa: E402
INSTR = "Answer the question using the facts.\n\n"
DEMO = ("Facts: Lumen Press was founded by Aria Holt. Aria Holt was born in Pellmoor. Brook Guild was founded by Evan Stone. "
        "Evan Stone was born in Tamridge.\nQuestion: Who or what is the birthplace of the founder of Lumen Press?\nAnswer: Pellmoor\n\n")


class Q:
    pass


def gen(rng, hops):
    names = Names(rng)
    ents, rels = chain(rng, names, hops)
    facts = [REL[r][2].format(s=ents[i], t=ents[i + 1]) for i, r in enumerate(rels)]
    finals = [ents[-1]]
    for _ in range(2):
        d = [names.make(REL[rels[0]][0])]
        for r in rels:
            d.append(names.make(REL[r][1]))
        facts += [REL[r][2].format(s=d[i], t=d[i + 1]) for i, r in enumerate(rels)]
        finals.append(d[-1])
    rng.shuffle(facts)
    x = ents[0]
    for r in rels:
        x = REL[r][3].format(x=x)
    q = Q()
    q.prompt = INSTR + DEMO + "Facts: " + " ".join(facts) + f"\nQuestion: Who or what is {x}?\nAnswer:"
    q.ids = [tok.encode(" " + f, add_special_tokens=False)[0] for f in finals]
    return q


TEXTS = []
if args.text_kl > 0:
    from datasets import load_dataset
    for r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"):
        t = r["text"].strip()
        if len(t) > 600 and not t.startswith("="):
            TEXTS.append(t[:700])
        if len(TEXTS) >= 4000:
            break


@torch.no_grad()
def evaluate(T, use_map, d, n):
    rng = random.Random(10_000 + d)
    items = [q for q in (gen(rng, d) for _ in range(n)) if len(set(q.ids)) == 3]
    ex = ch = 0
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        ids, am = encode([it.prompt for it in chunk])
        lg = model.lm_head(forward(ids, am, T, use_map)[:, -1]).float()
        for k, it in enumerate(chunk):
            ex += int(lg[k].argmax().item() == it.ids[0])
            ch += int(lg[k, it.ids].argmax().item() == 0)
    return ex / len(items), ch / len(items)


out = {"args": vars(args), "log": []}
t0 = time.time()
opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=0.0)
rng = random.Random(args.seed)
if args.load:
    M.load_state_dict(torch.load(args.load, map_location=args.device))
for step in range(0 if args.load else args.steps):
    items = [gen(rng, rng.randint(1, args.dmax)) for _ in range(args.bs)]
    ids, am = encode([it.prompt for it in items])
    tgt = torch.tensor([it.ids[0] for it in items], device=args.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model.lm_head(forward(ids, am, args.T_train, True)[:, -1]).float()
    loss = torch.nn.functional.cross_entropy(lg, tgt)
    if args.text_kl > 0:
        tids, tam = encode(rng.sample(TEXTS, 8))
        tids, tam = tids[:, :160], tam[:, :160]
        with torch.no_grad():
            lp0 = torch.log_softmax(model.lm_head(forward(tids, tam, args.T_train, False)).float(), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp1 = torch.log_softmax(model.lm_head(forward(tids, tam, args.T_train, True)).float(), -1)
        m = tam.bool().unsqueeze(-1)
        loss = loss + args.text_kl * ((lp0.exp() * (lp0 - lp1)) * m).sum() / tam.sum()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        acc = (lg.argmax(-1) == tgt).float().mean().item()
        out["log"].append(dict(step=step, loss=loss.item(), acc=acc))
        print(f"step {step} loss {loss.item():.3f} acc {acc:.2f} s={M.s.item():.3f} t={time.time() - t0:.0f}", flush=True)
if not args.load:
    torch.save(M.state_dict(), f"{RESULTS}/e79_map_{args.tag}.pt")
res = {}
for T in [int(x) for x in args.Ts_eval.split(",")]:
    for use_map in (False, True):
        row = {d: evaluate(T, use_map, d, args.neval) for d in [int(x) for x in args.depths_eval.split(",")]}
        res[f"T{T}_{'map' if use_map else 'frozen'}"] = row
        print(f"T={T} {'map   ' if use_map else 'frozen'} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()),
              flush=True)
        out["eval"] = res
        save_json(out, f"{RESULTS}/e79_ouro_nl_{args.tag}.json")
print("done", time.time() - t0)
