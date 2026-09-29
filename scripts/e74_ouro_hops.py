"""E74: does a looped model add about one hop per loop on natural-language multi-hop questions too? Synthetic questions about
fictional entities (synth_mhqa: a chain of typed facts shuffled among two distractor chains with the same relations; "Who or what
is the founder of the employer of X?"), 1 to 5 hops. Ouro-1.4B run for T = 1..8 loops (trained with 4); accuracy is the choice,
by first token, among the final entities of the queried chain and of the distractor chains (chance 1/3), and exact match of the
first token. Also a standard model for reference (same prompts), accuracy by hops.

Runs in loopdyn/.venv-loop (transformers 4.56).
usage: python e74_ouro_hops.py --model ByteDance/Ouro-1.4B --tag o14 [--n 150]
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import RESULTS, save_json  # noqa: E402
from synth_mhqa import REL, Names, chain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--Tmax", type=int, default=8)
ap.add_argument("--hops", default="1,2,3,4,5")
ap.add_argument("--looped", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(args.device).eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
INSTR = "Answer the question using the facts.\n\n"
DEMO = ("Facts: Lumen Press was founded by Aria Holt. Aria Holt was born in Pellmoor. Brook Guild was founded by Evan Stone. "
        "Evan Stone was born in Tamridge.\nQuestion: Who or what is the birthplace of the founder of Lumen Press?\nAnswer: Pellmoor\n\n")


def item(rng, hops):
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
    prompt = INSTR + DEMO + "Facts: " + " ".join(facts) + f"\nQuestion: Who or what is {x}?\nAnswer:"
    return prompt, finals


def first_tok(s):
    return tok.encode(" " + s, add_special_tokens=False)[0]


@torch.no_grad()
def per_loop_logits(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True).to(args.device)
    if args.looped:
        out = model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        return torch.stack([model.lm_head(h[:, -1]).float() for h in out[1]], 1)          # [B, T, V]
    return model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, -1:].float()


if args.looped:
    model.model.total_ut_steps = args.Tmax
res = {"model": args.model, "Tmax": args.Tmax, "by_hops": {}}
for h in [int(x) for x in args.hops.split(",")]:
    rng = random.Random(7400 + h)
    items = [item(rng, h) for _ in range(args.n)]
    items = [(p, f) for p, f in items if len({first_tok(x) for x in f}) == len(f)]
    ch = None
    ex = None
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        L = per_loop_logits([p for p, _ in chunk]).cpu()
        if ch is None:
            ch = torch.zeros(L.shape[1])
            ex = torch.zeros(L.shape[1])
        for k, (_, finals) in enumerate(chunk):
            ids = [first_tok(x) for x in finals]
            for t in range(L.shape[1]):
                ch[t] += int(L[k, t, ids].argmax().item() == 0)
                ex[t] += int(L[k, t].argmax().item() == ids[0])
    res["by_hops"][h] = {"n": len(items), "choice": (ch / len(items)).tolist(), "exact": (ex / len(items)).tolist()}
    print(f"hops {h} (n={len(items)}): choice by loop " + " ".join(f"{x:.2f}" for x in res["by_hops"][h]["choice"]) +
          " | exact " + " ".join(f"{x:.2f}" for x in res["by_hops"][h]["exact"]), flush=True)
    save_json(res, f"{RESULTS}/e74_hops_{args.tag}.json")
