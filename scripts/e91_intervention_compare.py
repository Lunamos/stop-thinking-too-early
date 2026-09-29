"""E91: which small change starts the relay? Different interventions at one layer of a frozen standard model, trained and evaluated
exactly like the paper's standard map (e19_reentry.py --no_loop: two chains in level order, lengths up to 20, single-token upper-
and lowercase variable names, 1,200 steps of 16 programs, answer cross-entropy plus the KL penalty on WikiText with weight 1).

  map        the paper's map: h <- s h + rms(h) B A (h / rms(h)), rank r, at every position (a representation intervention in the
             sense of ReFT; with the scale s it is DiReFT without a bias)
  flow       a FLAS-style iterated edit (flow-based activation steering, Jin et al. 2026) of the same size: a per-token nonlinear
             low-rank velocity field v(z, t) = B silu(A z + e(t)) on the RMS-normalized state (e: FLAS's sinusoidal time embedding
             through a two-layer MLP), integrated with N forward-Euler steps over a flow time T (T ~ U[0.5, 2] in training, T = 2 at
             evaluation, N = 3, as in FLAS); the same module as interventions.FlowLowRank
  flowblock  the FLAS FlowBlock without the concept encoder (one task, no concept), following the FLAS code: the time embedding
             (sinusoidal, 128 dimensions, two-layer MLP with a zero-initialized output) is added to the state, then an optional causal
             self-attention phase (--attn 1; later FLAS versions drop it) and a gated MLP of the model's intermediate width, each
             phase RMSNorm -> operation -> RMSNorm -> residual with a per-channel gate initialized to 0.1; velocity = output minus
             input, integrated with N Euler steps as above. --time 0 removes the time embedding.
  lora       rank-r LoRA on the query, key, value, output, gate, up and down projections of layer a (a weight edit of that layer)
  multimap   one map, the same parameters, applied at the input of each layer in --layers (the map repeated across depth)

Evaluation: exact accuracy (argmax over the vocabulary) and choice accuracy (among the two chains' root values) at --depths_eval,
frozen and with the intervention, on fresh programs (the evaluation seeds of e19_reentry.py). Saves the module
(results/e91_module_TAG.pt) and results/e91_compare_TAG.json.

usage: python e91_intervention_compare.py MODEL --kind flow --a 14 --tag TAG [--rank 8] [--N 3] [--attn 0|1] [--time 0|1]
       [--layers 6,10,14] [--lr ...]
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from interventions import RMSNorm, time_embedding
from ld_common import LETTERS, RESULTS, Runner, load_model, make_vb, save_json, value_token_ids

HDR = "Here is a short program. Each line assigns a value to a variable.\n"
QF = "print({q})\nOutput:"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--kind", default="map", choices=["map", "flow", "flowblock", "lora", "multimap"])
ap.add_argument("--a", type=int, default=14)
ap.add_argument("--layers", default="", help="multimap: comma list of layers whose input the shared map edits")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--N", type=int, default=3, help="flow, flowblock: Euler steps")
ap.add_argument("--T_eval", type=float, default=2.0, help="flow, flowblock: flow time at evaluation")
ap.add_argument("--attn", type=int, default=0, help="flowblock: causal self-attention phase (1) or not (0)")
ap.add_argument("--time", type=int, default=1, help="flowblock: time embedding (1) or not (0)")
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 (map, flow, multimap), 3e-4 (lora), 1e-4 (flowblock)")
ap.add_argument("--dmax", type=int, default=20)
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--text_kl", type=float, default=1.0)
ap.add_argument("--depths_eval", default="4,8,12,16,20,24")
ap.add_argument("--neval", type=int, default=200)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
LR = args.lr or {"map": 1e-3, "flow": 1e-3, "multimap": 1e-3, "lora": 3e-4, "flowblock": 1e-4}[args.kind]
random.seed(args.seed)
torch.manual_seed(args.seed)

model, tok = load_model(args.model, args.device)
R = Runner(model, tok)
N_LAYERS = R.n_layers
A_ = R.A
vids = value_token_ids(tok)
vals = list(vids)
letters = [c for c in LETTERS + list("abcdefghijklmnopqrstuvwxyz")
           if len(tok.encode(" " + c, add_special_tokens=False)) == 1 and len(tok.encode(c, add_special_tokens=False)) == 1]
cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
D = cfg.hidden_size


def rms(x):
    return x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)


class Map(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.B.weight)
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, h, T=None):
        x = h.float()
        n = rms(x)
        return (self.s * x + n * self.B(self.A(x / n))).to(h.dtype)


class Flow(nn.Module):
    """per-token nonlinear low-rank velocity field on the RMS-normalized state, integrated with N Euler steps"""

    def __init__(self, d, r, n_freq=16):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.B.weight)
        self.n_freq = n_freq
        self.t1 = nn.Linear(2 * n_freq, r)
        self.t2 = nn.Linear(r, r)
        nn.init.zeros_(self.t2.weight)
        nn.init.zeros_(self.t2.bias)

    def forward(self, h, T):
        x = h.float()
        n = rms(x)
        z = x / n
        for k in range(args.N):
            t = torch.tensor(k * T / args.N, device=x.device)
            e = self.t2(F.silu(self.t1(time_embedding(t, 2 * self.n_freq, x.device))))
            z = z + (T / args.N) * self.B(F.silu(self.A(z) + e))
        return (z * n).to(h.dtype)


def rope(x, pos, theta=1e6):
    """rotary position embedding on the last dimension of x: [B, H, L, dh]"""
    dh = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
    ang = pos[:, None, :, None].float() * inv[None, None, None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class FlowBlock(nn.Module):
    """FLAS FlowBlock without the concept cross-attention (see the module docstring)"""

    def __init__(self, d, heads, mlp, attn=False, time=True, freq_dim=128):
        super().__init__()
        self.attn, self.time, self.freq_dim = attn, time, freq_dim
        if time:
            self.t1, self.t2 = nn.Linear(freq_dim, d), nn.Linear(d, d)
            nn.init.zeros_(self.t2.weight)
            nn.init.zeros_(self.t2.bias)
        if attn:
            self.h, self.dh = heads, d // heads
            self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
            self.n1a, self.n1b = RMSNorm(d), RMSNorm(d)
            self.g1 = nn.Parameter(torch.full((d,), 0.1))
        self.gate, self.up = nn.Linear(d, mlp, bias=False), nn.Linear(d, mlp, bias=False)
        self.down = nn.Linear(mlp, d, bias=False)
        self.n2a, self.n2b = RMSNorm(d), RMSNorm(d)
        self.g2 = nn.Parameter(torch.full((d,), 0.1))

    def velocity(self, x, t, mask, pos):
        y = x
        if self.time:
            y = y + self.t2(F.silu(self.t1(time_embedding(t, self.freq_dim, x.device))))
        if self.attn:
            b, L, d = y.shape
            z = self.n1a(y)
            q = rope(self.q(z).view(b, L, self.h, self.dh).transpose(1, 2), pos)
            k = rope(self.k(z).view(b, L, self.h, self.dh).transpose(1, 2), pos)
            v = self.v(z).view(b, L, self.h, self.dh).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            y = y + self.g1 * self.n1b(self.o(att.transpose(1, 2).reshape(b, L, d)))
        z = self.n2a(y)
        y = y + self.g2 * self.n2b(self.down(F.silu(self.gate(z)) * self.up(z)))
        return y - x

    def forward(self, h, T, mask, pos):
        x = h.float()
        for k in range(args.N):
            x = x + (T / args.N) * self.velocity(x, torch.tensor(k * T / args.N, device=x.device), mask, pos)
        return x.to(h.dtype)


class LoRA(nn.Module):
    """rank-r LoRA on the linear projections of one layer, applied through forward hooks while STATE['on']"""

    def __init__(self, layer, r):
        super().__init__()
        self.mods = nn.ModuleDict()
        self.targets = {}
        for name, mod in layer.named_modules():
            short = name.split(".")[-1]
            if isinstance(mod, nn.Linear) and short in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                A = nn.Linear(mod.in_features, r, bias=False)
                B = nn.Linear(r, mod.out_features, bias=False)
                nn.init.normal_(A.weight, std=1.0 / math.sqrt(mod.in_features))
                nn.init.zeros_(B.weight)
                key = name.replace(".", "_")
                self.mods[key] = nn.Sequential(A, B)
                self.targets[key] = mod

    def attach(self):
        for key, mod in self.targets.items():
            ab = self.mods[key]
            mod.register_forward_hook(lambda m, inp, out, ab=ab: out + ab(inp[0].to(ab[0].weight.dtype)).to(out.dtype)
                                      if STATE["on"] else out)


STATE = {"on": True}
if args.kind == "map":
    MOD = Map(D, args.rank)
elif args.kind == "multimap":
    MOD = Map(D, args.rank)
elif args.kind == "flow":
    MOD = Flow(D, args.rank)
elif args.kind == "flowblock":
    MOD = FlowBlock(D, cfg.num_attention_heads, cfg.intermediate_size, attn=bool(args.attn), time=bool(args.time))
else:
    MOD = LoRA(A_.layers[args.a], args.rank)
MOD = MOD.to(args.device)
if args.kind == "lora":
    MOD.attach()
EDIT_LAYERS = [int(x) for x in args.layers.split(",")] if args.kind == "multimap" else [args.a]
n_params = sum(p.numel() for p in MOD.parameters())
print(f"kind={args.kind} layers={EDIT_LAYERS} params={n_params} lr={LR}", flush=True)

TEXTS = []
if args.text_kl > 0:
    from datasets import load_dataset
    for _r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"):
        _t = _r["text"].strip()
        if len(_t) > 600 and not _t.startswith("="):
            TEXTS.append(_t[:700])
        if len(TEXTS) >= 4000:
            break


def forward(ids, am, on, T=None):
    STATE["on"] = on
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    h = A_.embed(ids)
    masks = R._masks(h, am, pos)
    pe = {}
    for lt in set(A_.layer_types):
        try:
            pe[lt] = A_.rotary(h, pos, lt)
        except TypeError:
            pe[lt] = A_.rotary(h, pos)
    if args.kind == "flowblock" and on and args.attn:
        L = ids.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=ids.device).tril()
        fb_mask = causal[None, None] & am.bool()[:, None, None, :]
        fb_mask = fb_mask | torch.eye(L, dtype=torch.bool, device=ids.device)[None, None]   # padded rows attend to themselves
    for j in range(N_LAYERS):
        if on and args.kind != "lora" and j in EDIT_LAYERS:
            if args.kind == "flowblock":
                h = MOD(h, T, fb_mask if args.attn else None, pos)
            else:
                h = MOD(h, T)
        lt = A_.layer_types[j]
        out = A_.layers[j](h, attention_mask=masks[lt], position_embeddings=pe[lt], position_ids=pos, past_key_values=None,
                           use_cache=False)
        h = out if torch.is_tensor(out) else out[0]
    STATE["on"] = True
    return h


def gen(rng, d, nch=None):
    return make_vb(rng, d, nch or args.chains, "forward", header=HDR, query_fmt=QF, value_pool=vals, var_pool=letters)


@torch.no_grad()
def evaluate(depths, n):
    res = {}
    for d in depths:
        rng = random.Random(10_000 + d)
        nch = min(args.chains, max(2, len(letters) // d))
        items = [gen(rng, d, nch) for _ in range(n)]
        for on in (False, True):
            ex = ch = 0
            for i in range(0, n, 25):
                chunk = items[i:i + 25]
                ids, am = R.encode([it.prompt for it in chunk])
                lg = R.unembed(forward(ids, am, on, args.T_eval)[:, -1]).float()
                for k, it in enumerate(chunk):
                    ex += int(lg[k].argmax().item() == vids[it.answer])
                    cand = [vids[r] for r in it.roots]
                    ch += int(cand[int(lg[k, cand].argmax())] == vids[it.answer])
            res[f"d{d}_{'map' if on else 'frozen'}"] = [ex / n, ch / n]
    return res


opt = torch.optim.AdamW(MOD.parameters(), lr=LR, weight_decay=0.0)
rng = random.Random(args.seed)
t0 = time.time()
out = {"args": vars(args), "lr": LR, "n_params": n_params, "log": []}
for step in range(args.steps):
    items = [gen(rng, rng.randint(1, args.dmax)) for _ in range(args.bs)]
    ids, am = R.encode([it.prompt for it in items])
    tgt = torch.tensor([vids[it.answer] for it in items], device=args.device)
    T = rng.uniform(0.5, 2.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = R.unembed(forward(ids, am, True, T)[:, -1]).float()
    loss = F.cross_entropy(lg, tgt)
    kl = torch.zeros(())
    if args.text_kl > 0:
        tids, tam = R.encode(rng.sample(TEXTS, 8))
        tids, tam = tids[:, :160], tam[:, :160]
        with torch.no_grad():
            lp0 = torch.log_softmax(R.unembed(forward(tids, tam, False)).float(), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp1 = torch.log_softmax(R.unembed(forward(tids, tam, True, T)).float(), -1)
        m = tam.bool().unsqueeze(-1)
        kl = ((lp0.exp() * (lp0 - lp1)) * m).sum() / tam.sum()
        loss = loss + args.text_kl * kl
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(MOD.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        acc = (lg.argmax(-1) == tgt).float().mean().item()
        out["log"].append(dict(step=step, loss=loss.item(), kl=float(kl), acc=acc))
        print(f"step {step} loss {loss.item():.3f} kl {float(kl):.4f} acc {acc:.2f} t={time.time() - t0:.0f}", flush=True)
out["eval"] = evaluate([int(x) for x in args.depths_eval.split(",")], args.neval)
for k, v in out["eval"].items():
    print(k, "exact %.3f choice %.3f" % tuple(v), flush=True)
torch.save(MOD.state_dict(), f"{RESULTS}/e91_module_{args.tag}.pt")
save_json(out, f"{RESULTS}/e91_compare_{args.tag}.json")
print("done", round(time.time() - t0))
