"""E60: does the residual design set how far a transformer can follow a chain in one pass?
Small transformers trained from scratch on in-context pointer chasing, identical except for how layers communicate across depth:
  resid    - pre-norm residual stream (h <- h + f(norm(h)))
  attnres  - Attention Residuals (Kimi 2026): the input of layer l is a softmax(depth) mixture of the embedding and all earlier
             layer outputs, with a learned per-layer pseudo-query over RMS-normalised outputs
  mhc      - manifold-constrained hyper-connections (DeepSeek 2025): n parallel residual streams; each layer reads a non-negative
             mixture of the streams, writes its output back with non-negative weights, and the streams are mixed by a doubly
             stochastic matrix (Sinkhorn); read/write/mix weights are input-dependent (static + dynamic)
  hc       - the same without the doubly stochastic constraint (unconstrained mixing)
  mhc2/hc2 - faithful versions (v3): streams start as copies of the embedding, layer l initially reads stream l mod n
             (one-hot static read, as in Hyper-Connections), static write weights 1, static mixing ~identity, dynamic parts
             from a randomly initialised projection gated by alpha = 0.01 (mHC); mhc2 applies sigmoid / 2 sigmoid / 20-step
             Sinkhorn (mHC eq. 8), hc2 uses static + alpha tanh(dynamic) without constraints (HC). The v1 mhc/hc start
             permutation-symmetric (all streams identical, zero dynamic init) and so never use more than one stream.
  loop     - a block of B layers applied T times (weight tied), same effective depth
Task: programs of c chains (level-ordered or interleaved), query the last variable of one chain, predict its root value.
Reports accuracy by chain length and diagnostics (residual norm by layer, radial cos(update, state)).

usage: python e60_arch_reach.py --arch attnres --layers 6 --tag TAG [--dmax 16] [--steps 20000]
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = os.environ.get("LOOPDYN_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--loop_block", type=int, default=2)
ap.add_argument("--streams", type=int, default=4)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--dmax", type=int, default=16)
ap.add_argument("--eval_dmax", type=int, default=0, help="evaluate up to this chain length (default: dmax)")
ap.add_argument("--chains", default="2,3")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--nvars", type=int, default=160)
ap.add_argument("--curriculum", type=float, default=0.0, help="fraction of training over which dmax ramps from 1 to dmax")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--analyze", type=int, default=0, help="load e60_model_TAG.pt and run the mechanistic analysis only")
ap.add_argument("--stop_after_solved", type=int, default=0, help="stop this many steps after the curriculum reaches dmax")
ap.add_argument("--eval_only", type=int, default=0, help="load e60_model_TAG.pt and run the evaluation only")
ap.add_argument("--ckpt_every", type=int, default=5000)
ap.add_argument("--mech_d", default="4,8", help="chain lengths for the mechanistic analysis")
ap.add_argument("--strides", type=int, default=0, help="load e60_model_TAG.pt and measure chain-selective attention strides per layer on programs of this many lines")
ap.add_argument("--masks", type=int, default=0, help="load e60_model_TAG.pt and measure accuracy when each program line's attention to its parent line / lines two or more levels up is zeroed after the softmax (no renormalization)")
ap.add_argument("--mask_depths", default="2,4,6,8,12,16,20")
ap.add_argument("--mask_chains", type=int, default=2, help="chains per program in the mask analysis")
ap.add_argument("--names_mask", default="", help="with --names: block one kind of read while probing, as cond:layers, e.g. parent:3 or far:5,6")
ap.add_argument("--names", type=int, default=0, help="load e60_model_TAG.pt and decode, at each line's pointer token and layer, the names defined 1-6 levels up (programs of --mech_d lines)")
ap.add_argument("--multi", type=int, default=1, help="1: supervise the last variable of every chain (default); 0: one queried chain per program, as in pretraining")
args = ap.parse_args()
EVAL_DMAX = args.eval_dmax or args.dmax
torch.manual_seed(args.seed)
random.seed(args.seed)

# ---------------- data
V, U = args.nvars, 32              # variable tokens, value tokens
PAD, BOS, EQ, NL, PRINT, OUT = range(6)
VAR0, VAL0 = 6, 6 + V
VOCAB = 6 + V + U
CHAINS = [int(x) for x in args.chains.split(",")]
MAXLEN = 1 + max(CHAINS) * EVAL_DMAX * 4 + 3 * max(CHAINS)


def program(rng, d, c, order, multi=False):
    vars_ = rng.sample(range(V), c * d)
    vals = rng.sample(range(U), c)
    lines = []
    for ch in range(c):
        for k in range(d):
            rhs = VAL0 + vals[ch] if k == 0 else VAR0 + vars_[ch * d + k - 1]
            lines.append((ch, k, VAR0 + vars_[ch * d + k], rhs))
    if order == "level":
        out = []
        for k in range(d):
            lv = [l for l in lines if l[1] == k]
            rng.shuffle(lv)
            out += lv
    else:  # interleave: random merge keeping each chain in order
        rest = [[l for l in lines if l[0] == ch] for ch in range(c)]
        out = []
        while any(rest):
            ch = rng.choices(range(c), weights=[len(r) for r in rest])[0]
            out.append(rest[ch].pop(0))
    toks = [BOS]
    for _, _, lhs, rhs in out:
        toks += [lhs, EQ, rhs, NL]
    qs = list(range(c)) if multi else [rng.randrange(c)]
    rng.shuffle(qs)
    tpos, tgt = [], []
    for q in qs:
        toks += [PRINT, VAR0 + vars_[q * d + d - 1], OUT]
        tpos.append(len(toks) - 1)
        tgt.append(VAL0 + vals[q])
    return toks, tpos, tgt, [VAL0 + v for v in vals]


def batch(rng, bs, dmin, dmax, order=None, c=None, multi=False):
    seqs, tposs, tgts, roots = [], [], [], []
    for _ in range(bs):
        d = rng.randint(dmin, dmax)
        cc = c or rng.choice(CHAINS)
        o = order or rng.choice(["level", "interleave"])
        t, tp, tg, r = program(rng, d, cc, o, multi)
        seqs.append(t)
        tposs.append(tp)
        tgts.append(tg)
        roots.append(r)
    L = max(len(s) for s in seqs)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    bi, pi, yy = [], [], []
    for i, s in enumerate(seqs):
        off = L - len(s)
        x[i, off:] = torch.tensor(s)          # left pad; the last query's answer position is the last token
        for p, g in zip(tposs[i], tgts[i]):
            bi.append(i)
            pi.append(off + p)
            yy.append(g)
    return x, (torch.tensor(bi), torch.tensor(pi)), torch.tensor(yy), roots


# ---------------- model
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


TOY_STATE = {"keep": None, "layers": None, "cur": None}


def rope(x, pos):
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    """attention + MLP computed from ONE input (the block's read); returns the update to be written"""
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def attn(self, x, mask, pos):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        q, k = rope(q, pos), rope(k, pos)
        if TOY_STATE["keep"] is None or (TOY_STATE["layers"] is not None and TOY_STATE["cur"] not in TOY_STATE["layers"]):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:                                  # attention weights multiplied by keep after the softmax (no renormalization)
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            sc = sc.masked_fill(~mask, float("-inf"))
            w = torch.softmax(sc.float(), -1).to(q.dtype) * TOY_STATE["keep"].to(q.dtype)
            y = w @ v
        return self.o(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, mask, pos):
        # standard pre-norm block as a function of its read x: returns delta such that x + delta is the block output
        a = self.attn(x, mask, pos)
        m = self.fc2(F.gelu(self.fc1(self.n2(x + a))))
        return a + m


def sinkhorn(logits, iters=10):
    m = logits.exp()
    for _ in range(iters):
        m = m / m.sum(-1, keepdim=True)
        m = m / m.sum(-2, keepdim=True)
    return m


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        d, L = args.dim, args.layers
        self.emb = nn.Embedding(VOCAB, d)
        nblocks = args.loop_block if args.arch == "loop" else L
        self.blocks = nn.ModuleList([Block(d, args.heads) for _ in range(nblocks)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        if args.arch == "attnres":
            self.q = nn.Parameter(torch.zeros(L + 1, d))      # pseudo-queries: one per layer input + final
            self.kn = RMSNorm(d)
        if args.arch in ("mhc", "hc"):
            n = args.streams
            self.n = n
            self.pre_s = nn.Parameter(torch.zeros(L, n))
            self.post_s = nn.Parameter(torch.zeros(L, n))
            self.res_s = nn.Parameter(torch.eye(n).repeat(L, 1, 1) * (4.0 if args.arch == "mhc" else 1.0))
            self.dyn = nn.ModuleList([nn.Linear(n * d, 2 * n + n * n, bias=False) for _ in range(L)])
            for m in self.dyn:
                nn.init.zeros_(m.weight)
            self.dn = nn.ModuleList([RMSNorm(n * d) for _ in range(L)])
        if args.arch in ("mhc2", "hc2"):
            n = args.streams
            self.n = n
            self.phi = nn.ModuleList([nn.Linear(n * d, 2 * n + n * n, bias=False) for _ in range(L)])
            for m in self.phi:
                nn.init.normal_(m.weight, std=(n * d) ** -0.5)
            self.alpha = nn.Parameter(torch.full((L, 3), 0.01))
            onehot = torch.zeros(L, n)
            for l in range(L):
                onehot[l, l % n] = 1.0
            if args.arch == "mhc2":
                self.b_pre = nn.Parameter(onehot * 6.0 - 3.0)          # sigmoid -> ~0.95 on stream l mod n, ~0.05 elsewhere
                self.b_post = nn.Parameter(torch.zeros(L, n))          # 2 sigmoid(0) = 1
                self.b_res = nn.Parameter(torch.eye(n).repeat(L, 1, 1) * 4.0)   # Sinkhorn(exp) ~ 0.95 on the diagonal
            else:
                self.b_pre = nn.Parameter(onehot.clone())
                self.b_post = nn.Parameter(torch.ones(L, n))
                self.b_res = nn.Parameter(torch.eye(n).repeat(L, 1, 1))
            self.dn = nn.ModuleList([RMSNorm(n * d) for _ in range(L)])

    def forward(self, x, record=False, sel=None, layers=None, save_at=None, patch=None):
        # save_at: return a copy of the state entering layer save_at; patch: {'layer', 'pos', 'state'} overwrites the state
        # entering that layer at positions pos (resid/loop: h; attnres: the list of sources; hyper-connections: the streams)
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        valid = (x != PAD)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None] & valid[:, None, None, :]
        mask = mask | torch.eye(T, dtype=torch.bool, device=x.device)[None, None]
        e = self.emb(x)
        rec = []
        saved = None
        L = layers or args.layers

        def _patch(st, l):
            if patch is None or patch['layer'] != l:
                return st
            if isinstance(st, list):
                st = [o.clone() for o in st]
                for j, o in enumerate(st):
                    o[:, patch['pos']] = patch['state'][j][:, patch['pos']]
                return st
            st = st.clone()
            st[:, patch['pos']] = patch['state'][:, patch['pos']]
            return st
        if args.arch in ("resid", "loop"):
            h = e
            for l in range(L):
                if save_at == l:
                    saved = h.clone()
                h = _patch(h, l)
                blk = self.blocks[l % len(self.blocks)] if args.arch == "loop" else self.blocks[l]
                TOY_STATE["cur"] = l
                dlt = blk(h, mask, pos)
                if record:
                    rec.append((h.detach(), dlt.detach()))
                h = h + dlt
            out = h
        elif args.arch == "attnres":
            outs = [e]                                       # v_0 = embedding, v_j = output (update) of layer j
            for l in range(L + 1):
                if save_at == l:
                    saved = [o.clone() for o in outs]
                outs = _patch(outs, l)
                V_ = torch.stack(outs, 2)                   # [B, T, l+1, d]
                sc = torch.einsum("btjd,d->btj", self.kn(V_), self.q[l])
                a = sc.softmax(-1)
                h = torch.einsum("btj,btjd->btd", a, V_)
                if l == L:
                    out = h
                    break
                dlt = self.blocks[l](h, mask, pos)
                if record:
                    rec.append((h.detach(), dlt.detach(), a.detach()))
                outs.append(dlt)
        elif args.arch in ("mhc2", "hc2"):
            n = self.n
            S = e[:, :, None, :].repeat(1, 1, n, 1)              # n copies of the embedding
            for l in range(L):
                if save_at == l:
                    saved = S.clone()
                S = _patch(S, l)
                z = self.phi[l](self.dn[l](S.flatten(-2)))
                zp, zq, zr = z[..., :n], z[..., n:2 * n], z[..., 2 * n:].view(B, T, n, n)
                ap_, aq_, ar_ = self.alpha[l]
                if args.arch == "mhc2":
                    pre = torch.sigmoid(ap_ * zp + self.b_pre[l])
                    post = 2 * torch.sigmoid(aq_ * zq + self.b_post[l])
                    R = sinkhorn(ar_ * zr + self.b_res[l], iters=20)
                else:
                    pre = self.b_pre[l] + ap_ * torch.tanh(zp)
                    post = self.b_post[l] + aq_ * torch.tanh(zq)
                    R = self.b_res[l] + ar_ * torch.tanh(zr)
                hin = torch.einsum("btn,btnd->btd", pre, S)
                dlt = self.blocks[l](hin, mask, pos)
                if record:
                    rec.append((hin.detach(), dlt.detach(), S.detach()))
                S = torch.einsum("btmn,btnd->btmd", R, S) + post[..., None] * dlt[:, :, None, :]
            out = S.sum(2)
        else:  # mhc / hc (v1)
            n = self.n
            S = e[:, :, None, :].repeat(1, 1, n, 1) / n       # [B, T, n, d]; sum over streams = embedding
            for l in range(L):
                if save_at == l:
                    saved = S.clone()
                S = _patch(S, l)
                z = self.dyn[l](self.dn[l](S.flatten(-2)))  # dynamic part
                zp, zq, zr = z[..., :n], z[..., n:2 * n], z[..., 2 * n:].view(B, T, n, n)
                pre = torch.sigmoid(self.pre_s[l] + zp) * 2 / n              # non-negative read weights
                post = torch.sigmoid(self.post_s[l] + zq) * 2                 # non-negative write weights
                rl = self.res_s[l] + zr
                R = sinkhorn(rl) if args.arch == "mhc" else rl                # doubly stochastic vs unconstrained
                hin = torch.einsum("btn,btnd->btd", pre, S)
                dlt = self.blocks[l](hin, mask, pos)
                if record:
                    rec.append((hin.detach(), dlt.detach(), S.detach()))
                S = torch.einsum("btmn,btnd->btmd", R, S) + post[..., None] * dlt[:, :, None, :]
            out = S.sum(2)
        hsel = out[:, -1] if sel is None else out[sel[0], sel[1]]
        logits = self.head(self.nf(hsel))
        if save_at is not None:
            return logits, rec, saved
        return logits, rec




@torch.no_grad()
def build_keep(seqs, d, c, cond, mrng):
    """[n, 1, T, T] multipliers: 0 where a program line's tokens would attend to the stated lines (parent / far / same_far /
    other_far / rand_far), 1 elsewhere"""
    n, T = len(seqs), len(seqs[0])
    keep = torch.ones(n, 1, T, T)
    nl = c * d
    if cond == "none":
        return keep
    for p_ in range(n):
        ln = [(seqs[p_][1 + 4 * i], seqs[p_][1 + 4 * i + 2]) for i in range(nl)]
        at = {l_: i for i, (l_, _) in enumerate(ln)}
        lev, chn = [], []
        for i, (_, rhs) in enumerate(ln):
            k, cur = 1, rhs
            while cur in at:
                cur, k = ln[at[cur]][1], k + 1
            lev.append(k)
            chn.append(cur)
        for i in range(nl):
            if cond == "parent":
                tgt_lines = [at[ln[i][1]]] if ln[i][1] in at else []
            else:
                far = [j for j in range(nl) if lev[j] <= lev[i] - 2]
                same = [j for j in far if chn[j] == chn[i]]
                other = [j for j in far if chn[j] != chn[i]]
                tgt_lines = {"far": far, "same_far": same, "other_far": other,
                             "rand_far": mrng.sample(far, len(same)) if far else []}[cond]
            qpos = [1 + 4 * i + t for t in range(4)]
            for j in tgt_lines:
                for t in range(4):
                    keep[p_, 0, qpos, 1 + 4 * j + t] = 0.0
    return keep


@torch.no_grad()
def mask_accuracy(d, cond, n=300, seed=0, c=2):
    """choice accuracy on two-chain level-order programs of d lines when every program line's tokens lose their attention to the
    stated lines (all layers): parent = the line defining its pointer; far = every line two or more levels up (both chains);
    same_far / other_far = those of its own / the other chain; rand_far = as many lines as same_far, drawn from far."""
    model.eval()
    rng = random.Random(7700 + d + 1000 * seed)
    mrng = random.Random(7800 + d + 1000 * seed)
    seqs, tgts, roots = [], [], []
    for _ in range(n):
        toks, tpos, tgt, r = program(rng, d, c, "level", multi=False)
        seqs.append(toks)
        tgts.append(tgt[0])
        roots.append(r)
    x = torch.tensor(seqs, device=args.device)
    T = x.shape[1]
    keep = build_keep(seqs, d, c, cond, mrng)
    TOY_STATE["keep"] = keep.to(args.device)
    lg, _ = model(x)
    TOY_STATE["keep"] = None
    lg = lg.float().cpu()
    return sum(int(r[int(torch.stack([lg[i, t] for t in r]).argmax())] == tgts[i]) for i, r in enumerate(roots)) / n


def names_probe(d, n=600, ntr=450, J=6):
    """accuracy of decoding the variable defined j levels up (j = 1: the pointer token's own name) from the residual stream at each
    program line's pointer (rhs) token, per layer; ridge regression over the V variable names, held-out programs, lines whose
    ancestor exists. Also split by whether the line's chain is already linearly readable there (>= .75 front from mech())."""
    model.eval()
    rng = random.Random(9500 + d)
    seqs = [program(rng, d, 2, "level", multi=False)[0] for _ in range(n)]
    x = torch.tensor(seqs, device=args.device)
    J = min(J, d - 1)
    anc = torch.full((n, 2 * d, J), -1, dtype=torch.long)
    for p_ in range(n):
        ln = [(seqs[p_][1 + 4 * i], seqs[p_][1 + 4 * i + 2]) for i in range(2 * d)]
        at = {l_: i for i, (l_, _) in enumerate(ln)}
        for i, (_, rhs) in enumerate(ln):
            cur, j = rhs, 0
            while j < J and cur in at:
                anc[p_, i, j] = cur - VAR0
                cur, j = ln[at[cur]][1], j + 1
    rpos = torch.tensor([1 + 4 * i + 2 for i in range(2 * d)], device=args.device)
    keep = None
    if args.names_mask:
        cond, lay = args.names_mask.split(":")
        keep = build_keep(seqs, d, 2, cond, random.Random(4242))
        TOY_STATE["layers"] = {int(v) for v in lay.split(",")}
    recs = []
    with torch.no_grad():
        for i in range(0, n, 200):
            TOY_STATE["keep"] = None if keep is None else keep[i:i + 200].to(args.device)
            _, rec = model(x[i:i + 200], record=True)
            recs.append([r[0][:, rpos].float() for r in rec])
    TOY_STATE["keep"], TOY_STATE["layers"] = None, None
    trm = torch.zeros(n, 2 * d, dtype=torch.bool)
    trm[:ntr] = True
    trm = trm.reshape(-1)
    out = []
    for l in range(len(recs[0])):
        X = torch.cat([rc[l] for rc in recs], 0).reshape(-1, recs[0][l].shape[-1]).double()
        row = []
        for j in range(J):
            y = anc[:, :, j].reshape(-1)
            ok = y >= 0
            tr, te = (trm & ok), (~trm & ok)
            if tr.sum() < 20 or te.sum() < 20:
                row.append(None)
                continue
            trd = tr.to(X.device)
            mu = X[trd].mean(0, keepdim=True)
            Xc = X - mu
            G = Xc[trd].T @ Xc[trd]
            Yh = F.one_hot(y.clamp(min=0), V).double().to(X.device)
            W = torch.linalg.solve(G + (1e-2 * G.diagonal().mean() + 1e-6) * torch.eye(G.shape[0], device=G.device, dtype=G.dtype),
                                   Xc[trd].T @ Yh[trd])
            pred = (Xc[te.to(X.device)] @ W).argmax(-1).cpu()
            row.append((pred == y[te]).float().mean().item())
        out.append(row)
    return out


def mech():
    """relay front (chain-identity read-out at each line, per layer), commit layer (counterfactual patching of all program
    positions), radial cos at program tokens, and architecture-specific statistics"""
    model.eval()
    out = {}
    for d in [int(x) for x in args.mech_d.split(",")]:
        rng = random.Random(9000 + d)
        n = 600
        seqs, qs, chain_of_line, level_of_line, tgts, alts, root_pairs = [], [], [], [], [], [], []
        for _ in range(n):
            toks, tpos, tgt, roots = program(rng, d, 2, "level", multi=False)
            # recover chain membership / level of each line from the tokens
            lines = [(toks[1 + 4 * i], toks[1 + 4 * i + 2]) for i in range(2 * d)]
            by_lhs = {l: r for l, r in lines}
            ch, lv = [], []
            for lhs, rhs in lines:
                k, cur = 1, rhs
                while cur in by_lhs:
                    cur, k = by_lhs[cur], k + 1
                ch.append(cur - VAL0)                     # root value of the line's chain (consistent across programs)
                lv.append(k)
            seqs.append(toks)
            chain_of_line.append(ch)
            level_of_line.append(lv)
            root_pairs.append([r - VAL0 for r in roots])
            tgts.append(tgt[0])
            alts.append([r for r in roots if r != tgt[0]][0])
        x = torch.tensor(seqs, device=args.device)
        T = x.shape[1]
        ch = torch.tensor(chain_of_line)          # [n, 2d]
        lv = torch.tensor(level_of_line)
        # ---- relay front: read-out of chain identity at each line's tokens (lhs, '=', rhs, newline), per layer and role
        ntr = 450
        acc_roles = {}
        recs = []
        for i in range(0, n, 200):
            _, rec = model(x[i:i + 200], record=True)
            recs.append([r[0].float() for r in rec])
        nL = len(recs[0])
        for role, rname in enumerate(("lhs", "eq", "rhs", "nl")):
            rpos = torch.tensor([1 + 4 * i + role for i in range(2 * d)], device=args.device)
            tab = []
            rp = torch.tensor(root_pairs)                                     # [n, 2]
            Y = F.one_hot(ch.reshape(-1), U).double()
            for l in range(nL):
                X = torch.cat([rc[l][:, rpos] for rc in recs], 0).reshape(-1, recs[0][l].shape[-1]).double()
                tr = torch.zeros(n, 2 * d, dtype=torch.bool)
                tr[:ntr] = True
                tr = tr.reshape(-1).to(X.device)
                mu = X[tr].mean(0, keepdim=True)
                Xc = X - mu
                G = Xc[tr].T @ Xc[tr]
                W = torch.linalg.solve(G + (1e-2 * G.diagonal().mean() + 1e-6) * torch.eye(G.shape[0], device=G.device, dtype=G.dtype),
                                       Xc[tr].T @ Y.to(X.device)[tr])
                sc = (Xc @ W).reshape(n, 2 * d, U).cpu()
                s0 = sc.gather(2, rp[:, None, 0:1].expand(n, 2 * d, 1)).squeeze(-1)
                s1 = sc.gather(2, rp[:, None, 1:2].expand(n, 2 * d, 1)).squeeze(-1)
                pred = torch.where(s0 >= s1, rp[:, 0:1].expand(n, 2 * d), rp[:, 1:2].expand(n, 2 * d))
                ok = (pred == ch)[ntr:]
                lvt = lv[ntr:]
                tab.append([ok[lvt == k].float().mean().item() for k in range(1, d + 1)])
            acc_roles[rname] = tab
        del recs
        acc_tab = [[max(acc_roles[r][l][k] for r in acc_roles) for k in range(d)] for l in range(nL)]
        front = {k: next((l for l in range(nL) if acc_tab[l][k - 1] >= 0.75), None) for k in range(1, d + 1)}
        # ---- commit: patch all program positions entering layer l with the run on the program whose roots are swapped
        x_cf = x.clone()
        for i in range(n):
            a_, b_ = tgts[i], alts[i]
            row = x_cf[i]
            m_a, m_b = row == a_, row == b_
            row[m_a], row[m_b] = b_, a_
        prog_pos = torch.arange(1, 1 + 8 * d, device=args.device)
        flip = []
        base_ok = None
        for l in range(args.layers + (1 if args.arch == "attnres" else 0)):
            fl, ok_n = 0, 0
            for i in range(0, n, 200):
                xb, xcb = x[i:i + 200], x_cf[i:i + 200]
                _, _, st = model(xcb, save_at=l)
                lg, _ = model(xb, patch={"layer": l, "pos": prog_pos, "state": st})
                pred = lg.argmax(-1).cpu()
                tg = torch.tensor(tgts[i:i + 200])
                al = torch.tensor(alts[i:i + 200])
                fl += (pred == al).sum().item()
                ok_n += (pred == tg).sum().item()
            flip.append(fl / n)
        lg, _ = model(x)
        base_ok = (lg.argmax(-1).cpu() == torch.tensor(tgts)).float().mean().item()
        commit = max([l for l, f in enumerate(flip) if f >= 0.5], default=None)
        res = {"relay_acc": acc_tab, "relay_acc_roles": acc_roles, "front": front, "patch_flip": flip, "base_acc": base_ok, "commit_layer": commit}
        # ---- radial cos at program tokens
        _, rec = model(x[:200], record=True)
        res["cos_prog"] = [F.cosine_similarity(r[1][:, 1:1 + 8 * d].float(), r[0][:, 1:1 + 8 * d].float(), dim=-1).mean().item()
                           for r in rec]
        if args.arch == "attnres":
            res["depth_attn_prog"] = [r[2][:, 1:1 + 8 * d].float().mean((0, 1)).tolist() for r in rec]
            res["depth_attn_query"] = [r[2][:, -1].float().mean(0).tolist() for r in rec]
        if args.arch in ("mhc", "hc", "mhc2", "hc2"):
            # what each stream holds at the rhs tokens: the chain's root (relay) and the token's own variable (identity)
            rpos = torch.tensor([1 + 4 * i + 2 for i in range(2 * d)], device=args.device)
            recs = []
            for i in range(0, n, 200):
                _, rec = model(x[i:i + 200], record=True)
                recs.append([r[2][:, rpos].float() for r in rec])          # [b, 2d, nstreams, dim]
            nS = recs[0][0].shape[2]
            ident = torch.tensor([[s_[1 + 4 * j + 2] - VAR0 for j in range(2 * d)] for s_ in seqs])   # [n, 2d]
            rp = torch.tensor(root_pairs)

            def probe(X, Y, ncls, restrict=None):
                X = X.double()
                tr = torch.zeros(n, 2 * d, dtype=torch.bool)
                tr[:ntr] = True
                tr = tr.reshape(-1).to(X.device)
                mu = X[tr].mean(0, keepdim=True)
                Xc = X - mu
                G = Xc[tr].T @ Xc[tr]
                Yh = F.one_hot(Y.reshape(-1), ncls).double().to(X.device)
                W = torch.linalg.solve(G + (1e-2 * G.diagonal().mean() + 1e-6) * torch.eye(G.shape[0], device=G.device, dtype=G.dtype),
                                       Xc[tr].T @ Yh[tr])
                sc = (Xc @ W).reshape(n, 2 * d, ncls).cpu()
                if restrict is not None:
                    s0 = sc.gather(2, restrict[:, None, 0:1].expand(n, 2 * d, 1)).squeeze(-1)
                    s1 = sc.gather(2, restrict[:, None, 1:2].expand(n, 2 * d, 1)).squeeze(-1)
                    pred = torch.where(s0 >= s1, restrict[:, 0:1].expand(n, 2 * d), restrict[:, 1:2].expand(n, 2 * d))
                else:
                    pred = sc.argmax(-1)
                return (pred == Y)[ntr:]
            st_rel, st_id = [], []
            for l in range(len(recs[0])):
                rowr, rowi = [], []
                for si in range(nS):
                    X = torch.cat([rc[l][:, :, si] for rc in recs], 0).reshape(-1, recs[0][l].shape[-1])
                    okr = probe(X, ch, U, restrict=rp)
                    rowr.append([okr[lv[ntr:] == k].float().mean().item() for k in range(1, d + 1)])
                    rowi.append(probe(X, ident, V + U).float().mean().item())
                st_rel.append(rowr)
                st_id.append(rowi)
            res["stream_relay"] = st_rel
            res["stream_ident"] = st_id
            del recs
            sims = []
            for r in rec:
                S_ = r[2][:, 1:1 + 8 * d].float()          # [b, t, n, dim]
                Sn = F.normalize(S_, dim=-1)
                cs = torch.einsum("btmd,btnd->mn", Sn, Sn) / (S_.shape[0] * S_.shape[1])
                sims.append(cs.tolist())
            res["stream_cos"] = sims
        out[d] = res
        print(f"d={d}: base acc {base_ok:.2f}  front {front}  commit {commit}  flip {[round(f, 2) for f in flip]}", flush=True)
    return out


model = Net().to(args.device)
n_params = sum(p.numel() for p in model.parameters())
if args.eval_only:
    model.load_state_dict(torch.load(f"{RESULTS}/e60_model_{args.tag}.pt", map_location=args.device))
    args.steps = 0


@torch.no_grad()
def strides_analysis(d, n=300, jmax=None):
    """chain-selective attention (as paper 2, E82): at each line's right-hand-side token, attention to the rhs token of the line j
    levels up in the same chain minus that of the other chain's line at that level; per layer and head, then the best head per
    (layer, j) chosen on half of the programs and scored on the other half. Only resid / loop architectures."""
    model.eval()
    jmax = jmax or (d - 1)
    rng = random.Random(9100 + d)
    S_list, O_list = [], []
    for _ in range(n):
        toks, tpos, tgt, roots = program(rng, d, 2, "level", multi=False)
        lines = [(toks[1 + 4 * i], toks[1 + 4 * i + 2]) for i in range(2 * d)]
        by_lhs = {l: r for l, r in lines}
        info = []
        for i, (lhs, rhs) in enumerate(lines):
            k, cur = 1, rhs
            while cur in by_lhs:
                cur, k = by_lhs[cur], k + 1
            info.append((cur, k, 1 + 4 * i + 2))                  # (root token = chain id, level, rhs position)
        pos_of = {(c, k): p for c, k, p in info}
        roots_ = sorted({c for c, _, _ in info})
        x = torch.tensor([toks], device=args.device)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
        h = model.emb(x)
        Srows, Orows = [], []
        for l in range(args.layers):
            blk = model.blocks[l % len(model.blocks)] if args.arch == "loop" else model.blocks[l]
            B_, T_, D_ = h.shape
            q, k_, v = blk.qkv(blk.n1(h)).view(B_, T_, 3, blk.h, D_ // blk.h).unbind(2)
            q, k_ = [t.transpose(1, 2) for t in (q, k_)]
            q, k_ = rope(q, pos), rope(k_, pos)
            sc = (q @ k_.transpose(-1, -2)) / math.sqrt(D_ // blk.h)
            sc = sc.masked_fill(~mask, float("-inf"))
            A = sc.softmax(-1)[0]                                  # [H, T, T]
            s_ = torch.zeros(4, blk.h, jmax, device=x.device)     # [source role, head, stride]
            o_ = torch.zeros(4, blk.h, jmax, device=x.device)
            cnt = torch.zeros(jmax, device=x.device)
            for c in roots_:
                oc = [r for r in roots_ if r != c][0]
                for kk in range(2, d + 1):
                    base = pos_of[(c, kk)] - 2                        # lhs token of line (c, kk)
                    for j in range(1, min(jmax, kk - 1) + 1):
                        ts = pos_of[(c, kk - j)] - 2                  # the four tokens of the same-chain line j levels up
                        to = pos_of[(oc, kk - j)] - 2
                        for role in range(4):
                            s_[role, :, j - 1] += A[:, base + role, ts:ts + 4].sum(-1)
                            o_[role, :, j - 1] += A[:, base + role, to:to + 4].sum(-1)
                        cnt[j - 1] += 1
            Srows.append((s_ / cnt.clamp(min=1)).cpu())
            Orows.append((o_ / cnt.clamp(min=1)).cpu())
            h = h + blk(h, mask, pos)
        S_list.append(torch.stack(Srows))
        O_list.append(torch.stack(Orows))
    S_, O_ = torch.stack(S_list), torch.stack(O_list)              # [n, L, 4, H, J]
    D_ = S_ - O_
    half = n // 2
    best = D_[:half].mean(0).argmax(2)                              # [L, 4, J]
    Dt = D_[half:].mean(0)                                          # [L, 4, H, J]
    diff = Dt.gather(2, best[:, :, None, :]).squeeze(2)             # [L, 4, J] held-out, best head
    return {"roles": ["lhs", "eq", "rhs", "nl"], "diff": diff.tolist(),
            "same_full": S_[half:].mean(0).tolist(), "other_full": O_[half:].mean(0).tolist()}

if args.strides:
    model.load_state_dict(torch.load(f"{RESULTS}/e60_model_{args.tag}.pt", map_location=args.device))
    res_ = strides_analysis(args.strides)
    json.dump(res_, open(f"{RESULTS}/e60_strides_{args.tag}_d{args.strides}.json", "w"))
    for l, per_role in enumerate(res_["diff"]):
        cells = []
        for rname, row in zip(res_["roles"], per_role):
            longest = max([j + 1 for j, v in enumerate(row) if v >= 0.1], default=0)
            cells.append(f"{rname}: {longest} ({' '.join(f'{v:.2f}' for v in row[:6])})")
        print(f"layer {l}: " + " | ".join(cells), flush=True)
    raise SystemExit(0)
if args.masks:
    model.load_state_dict(torch.load(f"{RESULTS}/e60_model_{args.tag}.pt", map_location=args.device))
    res_m = {}
    for cond in ("none", "parent", "far", "same_far", "other_far", "rand_far"):
        res_m[cond] = {d_: mask_accuracy(d_, cond, c=args.mask_chains) for d_ in [int(v) for v in args.mask_depths.split(",")]}
        print(f"{cond:9s} " + " ".join(f"d{d_}:{a_:.2f}" for d_, a_ in res_m[cond].items()), flush=True)
    json.dump(res_m, open(f"{RESULTS}/e60_masks_{args.tag}{'' if args.mask_chains == 2 else f'_c{args.mask_chains}'}.json", "w"))
    raise SystemExit(0)
if args.names:
    model.load_state_dict(torch.load(f"{RESULTS}/e60_model_{args.tag}.pt", map_location=args.device))
    res_n = {}
    for d_ in [int(v) for v in args.mech_d.split(",")]:
        res_n[d_] = names_probe(d_)
        for l, row in enumerate(res_n[d_]):
            print(f"d={d_} layer {l}: " + " ".join("--" if v is None else f"{v:.2f}" for v in row), flush=True)
    sfx = "" if not args.names_mask else "_" + args.names_mask.replace(":", "_L").replace(",", "-")
    json.dump(res_n, open(f"{RESULTS}/e60_names_{args.tag}{sfx}.json", "w"))
    raise SystemExit(0)
if args.analyze:
    model.load_state_dict(torch.load(f"{RESULTS}/e60_model_{args.tag}.pt", map_location=args.device))
    json.dump(mech(), open(f"{RESULTS}/e60_mech_{args.tag}.json", "w"))
    raise SystemExit(0)
print(f"arch={args.arch} layers={args.layers} params={n_params}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
rng = random.Random(args.seed)
t0 = time.time()
log = []
for step in range(args.steps):
    if step == 0:
        dm_cur, acc_hist = 1, []
    dm = args.dmax
    if args.curriculum > 0:          # adaptive: lengthen chains once the current range is solved
        dm = dm_cur
    x, sel, y, _ = batch(rng, args.bs, 1, dm, multi=bool(args.multi))
    x, y = x.to(args.device), y.to(args.device)
    sel = (sel[0].to(args.device), sel[1].to(args.device))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, _ = model(x, sel=sel)
    loss = F.cross_entropy(lg.float(), y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    acc_hist.append((lg.argmax(-1) == y).float().mean().item())
    if args.curriculum > 0 and len(acc_hist) >= 100 and sum(acc_hist[-100:]) / 100 > 0.9 and dm_cur < args.dmax:
        dm_cur += 1
        acc_hist = []
        print(f"step {step}: curriculum -> dmax {dm_cur}", flush=True)
    if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
        torch.save(model.state_dict(), f"{RESULTS}/e60_model_{args.tag}.pt")
    if args.stop_after_solved and dm_cur >= args.dmax:
        solved_at = locals().get("solved_at") or step
        if step - solved_at >= args.stop_after_solved:
            print(f"step {step}: stopping {args.stop_after_solved} steps after reaching dmax {args.dmax}", flush=True)
            break
    if step % 500 == 0:
        acc = acc_hist[-1] if acc_hist else float("nan")
        log.append(dict(step=step, loss=loss.item(), acc=acc, dmax=dm_cur))
        print(f"step {step} loss {loss.item():.3f} acc {acc:.2f} dmax {dm_cur} t={time.time() - t0:.0f}", flush=True)


@torch.no_grad()
def evaluate():
    model.eval()
    res = {}
    erng = random.Random(12345)
    for order in ("level", "interleave"):
        for c in CHAINS:
            for d in range(1, EVAL_DMAX + 1):
                if c * d > V:
                    continue
                x, _, y, roots = batch(erng, 400, d, d, order=order, c=c)
                lg, _ = model(x.to(args.device))
                lg = lg.float().cpu()
                acc = (lg.argmax(-1) == y).float().mean().item()
                racc = sum(int(r[int(torch.stack([lg[i, t] for t in r]).argmax())] == y[i]) for i, r in enumerate(roots)) / len(roots)
                res[f"{order}_c{c}_d{d}"] = dict(acc=acc, racc=racc)
    return res


def reach(res, order, c, thr=0.8):
    best = 0
    for d in range(1, EVAL_DMAX + 1):
        if f"{order}_c{c}_d{d}" not in res:
            break
        if res[f"{order}_c{c}_d{d}"]["racc"] >= thr:
            best = d
        else:
            break
    return best


@torch.no_grad()
def diagnostics():
    """per layer: mean ||read|| relative to the embedding, radial cos(update, read) at program tokens and at the query"""
    model.eval()
    drng = random.Random(777)
    x, _, y, _ = batch(drng, 200, max(1, args.dmax // 2), max(1, args.dmax // 2), order="level", c=2)
    _, rec = model(x.to(args.device), record=True)
    out = []
    for r in rec:
        h, dl = r[0].float(), r[1].float()
        nrm = h.norm(dim=-1)
        cos = F.cosine_similarity(dl, h, dim=-1)
        prog = (x.to(args.device) != PAD)
        prog[:, -3:] = False
        out.append(dict(norm_prog=nrm[prog].mean().item(), norm_query=nrm[:, -1].mean().item(),
                        cos_prog=cos[prog].mean().item(), cos_query=cos[:, -1].mean().item(),
                        step_query=(dl.norm(dim=-1)[:, -1] / nrm[:, -1].clamp(min=1e-6)).mean().item()))
    return out


if not args.eval_only:
    torch.save(model.state_dict(), f"{RESULTS}/e60_model_{args.tag}.pt")
res = evaluate()
diag = diagnostics()
extra = {}
if args.arch == "loop":
    # more loops at inference than in training: does accuracy on long chains keep improving?
    model.eval()
    xrng = random.Random(4242)
    for mult in (1, 1.5, 2, 3):
        Lx = int(args.layers * mult)
        row = {}
        with torch.no_grad():
            for d in sorted(set([4, 8, 12, 16] + list(range(4, EVAL_DMAX + 1, 4)))):
                x, _, y, roots = batch(xrng, 300, d, d, order="interleave", c=2)
                lg, _ = model(x.to(args.device), layers=Lx)
                lg = lg.float().cpu()
                row[d] = sum(int(r[int(torch.stack([lg[i, t] for t in r]).argmax())] == y[i]) for i, r in enumerate(roots)) / len(roots)
        extra[f"loops_x{mult}"] = row
        print(f"inference with {Lx} effective layers ({mult}x): " + " ".join(f"d{d}:{v:.2f}" for d, v in row.items()), flush=True)
summary = {f"reach_{o}_c{c}": reach(res, o, c) for o in ("level", "interleave") for c in CHAINS}
print("reach", json.dumps(summary), flush=True)
for o in ("level", "interleave"):
    for c in CHAINS:
        print(f"  {o:10s} c={c}: " + " ".join(f"{res[k]['racc']:.2f}" for k in [f'{o}_c{c}_d{d}' for d in range(1, EVAL_DMAX + 1)] if k in res), flush=True)
print("diag", json.dumps([{k: round(v, 3) for k, v in d.items()} for d in diag]), flush=True)
json.dump({"args": vars(args), "params": n_params, "log": log, "eval": res, "reach": summary, "diag": diag, "extra": extra},
          open(f"{RESULTS}/e60_arch_{args.tag}.json", "w"))
