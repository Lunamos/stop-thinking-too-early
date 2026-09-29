"""E80: does a map make extra loops pay on a real multi-hop benchmark? Ouro-1.4B, rank-8 map at the input of layer a of the loop body
(every loop), trained with T=4 loops on MuSiQue training questions (supporting paragraphs, with or without two distractors; answer
cross-entropy; KL penalty on WikiText), then MuSiQue validation exact match (same 900-question split as the standard-model runs, first
n) for 1-8 loops, frozen and with the map, by greedy decoding (full recomputation each step, no cache).

Runs in loopdyn/.venv-loop; MuSiQue is read from loopdyn/data/*.json (exported with the main environment).
usage: python e80_ouro_musique.py --a 6 --tag TAG [--steps 1500] [--n_eval 300]
       zero-shot transfer: --load ../results/e80_map_o14_a6.pt --val 2wiki_dev300_seed1.json --max_para_chars 2000
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
from ld_common import RESULTS, save_json  # noqa: E402
from mqa_common import build_prompt, score_aliases  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
ap.add_argument("--a", type=int, default=6)
ap.add_argument("--tag", required=True)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--T_train", type=int, default=4)
ap.add_argument("--text_kl", type=float, default=1.0)
ap.add_argument("--Ts_eval", default="2,3,4,6,8")
ap.add_argument("--n_eval", type=int, default=300)
ap.add_argument("--max_new", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--load", default="", help="evaluate a saved map (e.g. a program-trained E71 map) instead of training one")
ap.add_argument("--setting", default="gold", help="evaluation paragraphs: gold, or distractor (gold plus two distractors)")
ap.add_argument("--apply", default="all", help="all: the map acts in every loop; first / last: only in the first / last loop")
ap.add_argument("--val", default="musique_val900_seed1.json", help="evaluation questions in loopdyn/data (e.g. 2wiki_dev300_seed1.json)")
ap.add_argument("--max_para_chars", type=int, default=900)
ap.add_argument("--kind", default="map", help="map (rank-r map at layer a) | steer (constant vector at layer a, 2,049 parameters) | "
                "lora (rank-r LoRA on q_proj and v_proj of layer a; r=4 matches the map's parameters) | prompt (n_soft learned input vectors)")
ap.add_argument("--n_soft", type=int, default=16)
args = ap.parse_args()
torch.manual_seed(args.seed)

import transformers  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(args.device).eval()
for p in model.parameters():
    p.requires_grad_(False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
inner = model.model
L = len(inner.layers)


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


class Steer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(d))
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.v).to(h.dtype)


class LoRA(nn.Module):
    """rank-r updates of q_proj and v_proj of layer a (weight-tied, so they act in every loop)"""
    def __init__(self, attn, r, alpha=16.0):
        super().__init__()
        self.mods = nn.ModuleDict()
        for name in ("q_proj", "v_proj"):
            lin = getattr(attn, name)
            A = nn.Linear(lin.in_features, r, bias=False)
            B = nn.Linear(r, lin.out_features, bias=False)
            nn.init.normal_(A.weight, std=1.0 / math.sqrt(lin.in_features))
            nn.init.zeros_(B.weight)
            self.mods[name] = nn.Sequential(A, B)
            lin.register_forward_hook(self._hook(name))
        self.scale = alpha / r
        self.on = False

    def _hook(self, name):
        def f(mod, inp, out):
            if not self.on:
                return out
            return out + (self.scale * self.mods[name](inp[0].float())).to(out.dtype)
        return f

    def forward(self, h):
        return h


class Soft(nn.Module):
    def __init__(self, d, n, emb):
        super().__init__()
        idx = torch.randint(0, emb.weight.shape[0], (n,))
        self.P = nn.Parameter(emb.weight[idx].detach().float().clone())

    def forward(self, h):
        return h


if args.kind == "map":
    M = Map(model.config.hidden_size, args.rank).to(args.device)
elif args.kind == "steer":
    M = Steer(model.config.hidden_size).to(args.device)
elif args.kind == "lora":
    M = LoRA(inner.layers[args.a].self_attn, args.rank).to(args.device)
else:
    M = Soft(model.config.hidden_size, args.n_soft, inner.embed_tokens).to(args.device)
print(f"kind {args.kind}: {sum(p.numel() for p in M.parameters())} parameters", flush=True)


def _layer(l, h, mask, pos, pe):
    out = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe, past_key_value=None, use_cache=False)
    return out[0] if isinstance(out, tuple) else out


def forward(ids, am, T, use_map):
    n_pre = 0
    h = inner.embed_tokens(ids)
    if args.kind == "prompt" and use_map:
        n_pre = M.P.shape[0]
        h = torch.cat([M.P.to(h.dtype)[None].expand(h.shape[0], -1, -1), h], 1)
        am = torch.cat([torch.ones(am.shape[0], n_pre, dtype=am.dtype, device=am.device), am], 1)
    if args.kind == "lora":
        M.on = bool(use_map)
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    mask = create_causal_mask(config=inner.config, input_embeds=h, attention_mask=am,
                              cache_position=torch.arange(h.shape[1], device=h.device), past_key_values=None, position_ids=pos)
    pe = inner.rotary_emb(h, pos)
    for t in range(T):
        for l in range(L):
            if use_map and args.kind in ("map", "steer") and l == args.a and (args.apply == "all" or (args.apply == "first" and t == 0) or (args.apply == "last" and t == T - 1)):
                h = M(h)
            if torch.is_grad_enabled() and h.requires_grad:
                h = checkpoint(_layer, l, h, mask, pos, pe, use_reentrant=False)
            else:
                h = _layer(l, h, mask, pos, pe)
        h = inner.norm(h)
    return h[:, n_pre:]


train = json.load(open(f"{DATA}/musique_train_seed0.json"))
val = json.load(open(f"{DATA}/{args.val}"))[:args.n_eval]
TEXTS = []
if args.text_kl > 0:
    from datasets import load_dataset
    for r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"):
        t = r["text"].strip()
        if len(t) > 600 and not t.startswith("="):
            TEXTS.append(t[:700])
        if len(TEXTS) >= 4000:
            break
rng = random.Random(args.seed)


def batch(i):
    exs = [train[(i * args.bs + k) % len(train)] for k in range(args.bs)]
    prompts = [build_prompt(ex, rng.choice(["gold", "distractor"]), seed=rng.randrange(10 ** 6)) for ex in exs]
    ids, labels = [], []
    for p, ex in zip(prompts, exs):
        pi = tok(p, add_special_tokens=True).input_ids
        ti = tok(" " + ex["answer"] + "\n", add_special_tokens=False).input_ids
        ids.append(pi + ti)
        labels.append([-100] * len(pi) + ti)
    Lm = max(len(x) for x in ids)
    am = [[1] * len(x) + [0] * (Lm - len(x)) for x in ids]
    ids = [x + [tok.pad_token_id] * (Lm - len(x)) for x in ids]
    labels = [x + [-100] * (Lm - len(x)) for x in labels]
    T_ = lambda v: torch.tensor(v, device=args.device)
    return T_(ids), T_(am), T_(labels)


opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=0.0)
t0 = time.time()
log = []
tok.padding_side = "right"
if args.load:
    M.load_state_dict(torch.load(args.load, map_location=args.device))
for step in range(0 if args.load else args.steps):
    ids, am, labels = batch(step)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = forward(ids, am, args.T_train, True)
        sel = labels[:, 1:] != -100
        logits = model.lm_head(h[:, :-1][sel]).float()
    loss = nn.functional.cross_entropy(logits, labels[:, 1:][sel])
    kl = torch.zeros((), device=args.device)
    if args.text_kl > 0:
        enc = tok(rng.sample(TEXTS, 4), return_tensors="pt", padding=True, truncation=True, max_length=160).to(args.device)
        with torch.no_grad():
            lp0 = torch.log_softmax(model.lm_head(forward(enc.input_ids, enc.attention_mask, args.T_train, False)).float(), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp1 = torch.log_softmax(model.lm_head(forward(enc.input_ids, enc.attention_mask, args.T_train, True)).float(), -1)
        m = enc.attention_mask.bool().unsqueeze(-1)
        kl = ((lp0.exp() * (lp0 - lp1)) * m).sum() / enc.attention_mask.sum()
    opt.zero_grad()
    (loss + args.text_kl * kl).backward()
    torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        log.append(dict(step=step, ce=loss.item(), kl=kl.item()))
        print(f"step {step} ce {loss.item():.3f} kl {kl.item():.4f} t={time.time() - t0:.0f}", flush=True)
if not args.load:
    torch.save(M.state_dict(), f"{RESULTS}/e80_map_{args.tag}.pt")

tok.padding_side = "left"
nl = set(tok.encode("\n", add_special_tokens=False))


@torch.no_grad()
def generate(prompts, T, use_map, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(args.device)
        ids, am = enc.input_ids, enc.attention_mask
        done = torch.zeros(ids.shape[0], dtype=torch.bool, device=args.device)
        gen = [[] for _ in range(ids.shape[0])]
        for _ in range(args.max_new):
            nxt = model.lm_head(forward(ids, am, T, use_map)[:, -1]).float().argmax(-1)
            for k in range(ids.shape[0]):
                if not done[k]:
                    if nxt[k].item() in nl or nxt[k].item() == tok.eos_token_id:
                        done[k] = True
                    else:
                        gen[k].append(nxt[k].item())
            if done.all():
                break
            ids = torch.cat([ids, nxt[:, None]], 1)
            am = torch.cat([am, torch.ones_like(nxt[:, None])], 1)
        outs += [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
    return outs


res = {"args": vars(args), "log": log, "em": {}}
mpc = args.max_para_chars
prompts = ([build_prompt(ex, args.setting, seed=i, max_para_chars=mpc) for i, ex in enumerate(val)] if args.setting != "gold"
           else [build_prompt(ex, "gold", max_para_chars=mpc) for ex in val])
types = sorted({ex["type"] for ex in val})
for T in [int(x) for x in args.Ts_eval.split(",")]:
    for use_map in (False, True):
        preds = generate(prompts, T, use_map)
        r = score_aliases(val, preds)
        key = f"T{T}_{'map' if use_map else 'frozen'}"
        res.setdefault("preds", {})[key] = preds
        res["em"][key] = {k: v for k, v in r.items() if k.endswith("_em") or k.endswith("_f1") or k == "n"}
        print(f"T={T} {'map   ' if use_map else 'frozen'} EM {r['all_em']:.3f} F1 {r['all_f1']:.3f} " +
              (" ".join(f"{h}:{r.get(f'{h}hop_em', float('nan')):.2f}" for h in (2, 3, 4)) if "2hop" in types else
               " ".join(f"{t}:{r.get(f'{t}_em', float('nan')):.2f}" for t in types)), flush=True)
        save_json(res, f"{RESULTS}/e80_ouro_musique_{args.tag}.json")
