"""E88: few-shot baseline for the default. Frozen Ouro (T loops) or Huginn (r recurrences) on two-chain programs, with k solved
demonstration programs (lengths 1-6, answers given) before the test program, and zero-shot for comparison. Exact accuracy and choice
between the two roots, against chain length. If the default is a prompt artefact, demonstrations should raise reach well above the
zero-shot 2-3 lines; if it is the model's computation, they should fix the answer format but not the reach.

Runs in loopdyn/.venv-loop.
usage: python e88_fewshot.py --arch ouro|huginn --tag TAG [--shots 0,4] [--steps 1,2,3,4,6,8] [--depths 1,2,3,4,6,8] [--n 100]
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ld_common import LETTERS, NOUNS, RESULTS, make_vb, save_json  # noqa: E402

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"
ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="ouro")
ap.add_argument("--model", default=None)
ap.add_argument("--tag", required=True)
ap.add_argument("--shots", default="0,4")
ap.add_argument("--steps", default="1,2,3,4,6,8")
ap.add_argument("--depths", default="1,2,3,4,6,8")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
name = args.model or ("ByteDance/Ouro-1.4B" if args.arch == "ouro" else "tomg-group-umd/huginn-0125")

import transformers  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
vals = [w for w in NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
vids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in vals}
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
if args.arch == "ouro":
    from ld_loop import OuroRunner
    R = OuroRunner(model, tok)


def gen(rng, d):
    return make_vb(rng, d, 2, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)


def prompt_with_shots(rng, it, k):
    demos = []
    for _ in range(k):
        dm = gen(rng, rng.randint(1, 6))
        demos.append(dm.prompt + " " + dm.answer + "\n\n")
    return "".join(demos) + it.prompt


@torch.no_grad()
def last_logits(prompt, steps, seed):
    enc = tok(prompt, return_tensors="pt").to(args.device)
    if args.arch == "ouro":
        h = R.run(enc.input_ids, T=steps)["h"]
        return R.logits(h[:, -1]).float()[0]
    torch.manual_seed(seed)
    return model(input_ids=enc.input_ids, num_steps=steps).logits[0, -1].float()


res = {"args": vars(args), "acc": {}}
for k in [int(x) for x in args.shots.split(",")]:
    for s in [int(x) for x in args.steps.split(",")]:
        row = {}
        for d in [int(x) for x in args.depths.split(",")]:
            rng = random.Random(88_000 + 97 * d + k)
            ex = ch = 0
            for i in range(args.n):
                it = gen(rng, d)
                lg = last_logits(prompt_with_shots(rng, it, k), s, i)
                ex += int(lg.argmax().item() == vids[it.answer])
                rl = torch.stack([lg[vids[x]] for x in it.roots])
                ch += int(it.roots[rl.argmax().item()] == it.answer)
            row[d] = (ex / args.n, ch / args.n)
        res["acc"][f"k{k}_s{s}"] = row
        print(f"shots={k} steps={s:2d} " + " ".join(f"d{d}:{e:.2f}/{c:.2f}" for d, (e, c) in row.items()), flush=True)
        save_json(res, f"{RESULTS}/e88_fewshot_{args.tag}.json")
print("done")
