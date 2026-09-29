"""E52: post-training for direct-answer multi-hop QA with a tiny single-layer map vs LoRA (and FLAS-style flows).
Trains on MuSiQue (or HotpotQA) train with teacher-forced cross-entropy on the answer, plus KL(frozen || adapted) on WikiText,
then evaluates by greedy generation (EM/F1) on MuSiQue and HotpotQA validation.

kinds:  map   - rank-r map at the input of block a (h <- s h + rms(h) B A h/rms(h))
        steer - single vector at the input of block a
        lora  - rank-r LoRA on all attention and MLP projections of all layers (or of --lora_layers)
        flow  - FLAS-style low-rank flow at the input of block a, N Euler steps (interventions.FlowLowRank)
        flowmlp - the FLAS flow block without concept encoder or self-attention at block a (interventions.FlowMLP)
        none  - the frozen model (evaluation only)

usage: python e52_mqa_train.py MODEL --tag TAG --kind map --a 14 [--rank 8] [--train musique] [--steps 1500]
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn

from ld_common import RESULTS, load_model, save_json
from interventions import FlowLowRank, FlowMLP  # noqa: E402
from mqa_common import (Map, attach_map, build_prompt, decoder_layers, generate_answers, load_2wiki, load_hotpot,
                        load_musique, load_squad, score_aliases)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--kind", default="map", help="map | steer | lora | flow | flowmlp | none (frozen model, evaluation only); flow and "
                "flowmlp are FLAS-style iterated edits (interventions.py)")
ap.add_argument("--N", type=int, default=3, help="flow, flowmlp: Euler steps")
ap.add_argument("--time", type=int, default=1, help="flowmlp: time embedding (1) or not (0)")
ap.add_argument("--evals", default="musique,hotpot,synth,2wiki")
ap.add_argument("--lora_layers", default=None, help="LoRA only on layers lo:hi (python slice), e.g. 0:18")
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--train", default="musique", help="musique | hotpot | mix | synth | squad | 2wiki")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--micro", type=int, default=8, help="micro-batch size (gradient accumulation)")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--text_kl", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_n", type=int, default=900)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--load", default="", help="evaluate a saved adapter (results/e52_adapter_TAG.pt) instead of training; build it with the same"
                " --kind/--a/--rank/--lora_layers as when it was trained")
ap.add_argument("--settings", default="", help="restrict evaluation settings (comma list, e.g. gold)")
ap.add_argument("--save_preds", type=int, default=0, help="store per-question predictions (for paired comparisons)")
args = ap.parse_args()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
trng = random.Random(args.seed + 1)        # flow time (flow, flowmlp), separate so that the batches match the other kinds

model, tok = load_model(args.model, args.device)
for p in model.parameters():
    p.requires_grad_(False)
d = model.config.get_text_config().hidden_size
N = len(decoder_layers(model))
STATE = {"on": True}

# ---------------- adapters
params = []
if args.kind in ("map", "steer", "flow", "flowmlp"):
    if args.kind == "flow":
        M = FlowLowRank(d, r=args.rank, N=args.N).to(args.device)
    elif args.kind == "flowmlp":
        M = FlowMLP(d, model.config.get_text_config().intermediate_size, N=args.N, time=bool(args.time)).to(args.device)
    else:
        M = Map(d, rank=args.rank, steer=args.kind == "steer").to(args.device)
    base_fwd = M.forward

    def gated(h):
        return base_fwd(h) if STATE["on"] else h
    M.forward = gated
    attach_map(model, M, args.a)
    params = list(M.parameters())
elif args.kind == "lora":
    class LoRALinear(nn.Module):
        def __init__(self, base, r):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

        def forward(self, x):
            y = self.base(x)
            if STATE["on"]:
                y = y + ((x.float() @ self.A.T) @ self.B.T).to(y.dtype)
            return y
    _layers = list(decoder_layers(model))
    if args.lora_layers:
        _lo, _hi = (int(x) for x in args.lora_layers.split(":"))
        _layers = _layers[_lo:_hi]
    for layer in _layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                if hasattr(parent, name):
                    lo = LoRALinear(getattr(parent, name), args.rank)
                    setattr(parent, name, lo)
                    params += [lo.A, lo.B]
if args.load:
    _sd = torch.load(args.load, map_location="cpu")
    if args.kind in ("map", "steer", "flow", "flowmlp"):
        M.load_state_dict({k: v.to(args.device) for k, v in _sd.items()})
    elif args.kind == "lora":
        assert len(_sd) == len(params), (len(_sd), len(params))
        with torch.no_grad():
            for _i, _p in enumerate(params):
                _p.copy_(_sd[f"p{_i}"].to(_p.device, _p.dtype))
n_params = sum(p.numel() for p in params)
print(f"kind={args.kind} a={args.a} rank={args.rank} trainable params={n_params}", flush=True)
if args.kind == "none" or args.load:
    args.steps = 0
else:
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

# ---------------- data
if args.train == "squad":
    train = load_squad("train", n=30000, seed=args.seed)
elif args.train == "synth":
    from synth_mhqa import make_item
    train = [make_item(rng, rng.choice([2, 3, 4, 5]), 2) for _ in range(30000)]
elif args.train == "musique":
    train = load_musique("train", seed=args.seed)
elif args.train == "2wiki":
    train = load_2wiki("train", n=30000, seed=args.seed)
elif args.train == "hotpot":
    train = load_hotpot("train", n=30000, seed=args.seed)
else:
    train = load_musique("train", seed=args.seed) + load_hotpot("train", n=20000, seed=args.seed)
rng.shuffle(train)
print("train examples", len(train), flush=True)
TEXTS = []
if args.text_kl > 0:
    import datasets
    for r in datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"):
        t = r["text"].strip()
        if len(t) > 600 and not t.startswith("="):
            TEXTS.append(t[:700])
        if len(TEXTS) >= 4000:
            break


def batch(i):
    exs = [train[(i * args.bs + k) % len(train)] for k in range(args.bs)]
    prompts = [build_prompt(ex, rng.choice(["gold", "distractor"]), seed=rng.randrange(10 ** 6),
                            max_para_chars=2000 if args.train == "2wiki" else 900) for ex in exs]
    tgts = [" " + ex["answer"] + "\n" for ex in exs]
    ids, labels = [], []
    for p, t in zip(prompts, tgts):
        pi = tok(p, add_special_tokens=True).input_ids
        ti = tok(t, add_special_tokens=False).input_ids
        ids.append(pi + ti)
        labels.append([-100] * len(pi) + ti)
    L = max(len(x) for x in ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    am = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
    ids = [x + [pad] * (L - len(x)) for x in ids]
    labels = [x + [-100] * (L - len(x)) for x in labels]
    T = lambda v: torch.tensor(v, device=args.device)
    return T(ids), T(am), T(labels)


model.train(False)
t0 = time.time()
log = []
for step in range(args.steps):
    if args.kind in ("flow", "flowmlp"):
        M.T = trng.uniform(0.5, 2.0)         # flow time, as in FLAS
    ids, am, labels = batch(step)
    STATE["on"] = True
    opt.zero_grad()
    ce_tot = 0.0
    n_tgt = (labels[:, 1:] != -100).sum().item()
    for m0 in range(0, ids.shape[0], args.micro):
        sl = slice(m0, m0 + args.micro)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = model.model(input_ids=ids[sl], attention_mask=am[sl]).last_hidden_state
            sel = labels[sl, 1:] != -100
            logits = model.lm_head(hid[:, :-1][sel])
        ce = nn.functional.cross_entropy(logits.float(), labels[sl, 1:][sel], reduction="sum") / n_tgt
        ce.backward()
        ce_tot += ce.item()
        del hid, logits, ce
    loss = torch.tensor(ce_tot)
    kl = torch.zeros((), device=args.device)
    if args.text_kl > 0:
        tb = rng.sample(TEXTS, 8)
        enc = tok(tb, return_tensors="pt", padding=True, truncation=True, max_length=160).to(args.device)
        with torch.no_grad():
            STATE["on"] = False
            lp0 = torch.log_softmax(model(**enc).logits.float(), -1)
            STATE["on"] = True
        for m0 in range(0, enc.input_ids.shape[0], args.micro):
            sl = slice(m0, m0 + args.micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp1 = torch.log_softmax(model(input_ids=enc.input_ids[sl], attention_mask=enc.attention_mask[sl]).logits.float(), -1)
            m = enc.attention_mask[sl].bool().unsqueeze(-1)
            klm = ((lp0[sl].exp() * (lp0[sl] - lp1)) * m).sum() / enc.attention_mask.sum()
            (args.text_kl * klm).backward()
            kl = kl + klm.detach()
            del lp1, klm
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 50 == 0:
        log.append(dict(step=step, loss=loss.item(), kl=kl.item()))
        print(f"step {step} ce {loss.item():.3f} kl {kl.item():.4f} t={time.time() - t0:.0f}", flush=True)

if args.kind != "none" and not args.load:
    torch.save({k: v.detach().cpu() for k, v in (M.state_dict() if args.kind != "lora" else
                                                 {f"p{i}": p for i, p in enumerate(params)}).items()},
               f"{RESULTS}/e52_adapter_{args.tag}.pt")
# ---------------- evaluation
STATE["on"] = True
if args.kind in ("flow", "flowmlp"):
    M.T = 2.0
out = {"args": vars(args), "n_params": n_params, "log": log, "res": {}}
from synth_mhqa import make_item as _mk
_srng = random.Random(777)
EV = args.evals.split(",")
evals = []
if "musique" in EV:
    evals.append(("musique", load_musique("validation", n=args.eval_n, seed=1), ("gold", "distractor")))
if "hotpot" in EV:
    evals.append(("hotpot", load_hotpot("validation", n=600, seed=1), ("gold",)))
if "synth" in EV:
    evals.append(("synth", [_mk(_srng, h, 2) for h in (2, 3, 4, 5, 6) for _ in range(100)], ("gold",)))
if "2wiki" in EV:
    evals.append(("2wiki", load_2wiki("dev", n=600, seed=1), ("gold",)))
with torch.no_grad():
    for name, exs, settings in evals:
        for setting in settings:
            if args.settings and setting not in args.settings.split(","):
                continue
            preds = generate_answers(model, tok, [build_prompt(ex, setting, max_para_chars=2000 if name == "2wiki" else 900)
                                                  for ex in exs], bs=8)
            r = score_aliases(exs, preds)
            out["res"][f"{name}_{setting}"] = r
            if args.save_preds:
                out.setdefault("preds", {})[f"{name}_{setting}"] = preds
            print(name, setting, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            save_json(out, f"{RESULTS}/e52_mqa_{args.tag}.json")
