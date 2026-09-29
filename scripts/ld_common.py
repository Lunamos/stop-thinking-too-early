"""Shared utilities for the loop / depth-dynamics study (loopdyn).

Core pieces:
  * load_model(): HF causal LM in bf16 on one GPU, eval mode, no grads.
  * Runner: executes an arbitrary *layer schedule* (a list of block indices,
    repeats allowed) on a batch, with optional norm-matching when a block is
    re-entered, and records residual states after chosen schedule steps.
  * logit-lens readouts.
  * variable-binding (pointer-chasing) prompt generator with controllable
    depth, number of chains, and presentation order.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field

import torch

# LD is the repository root (the directory above scripts/); results go to LD/results. Set LOOPDYN_DIR to use another copy.
LD = os.environ.get("LOOPDYN_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(LD, "results")
FIGS = os.path.join(RESULTS, "figs")       # quick-look plots of exploratory scripts (the paper's figures come from analysis/)
if os.environ.get("LOOPDYN_HF"):
    os.environ.setdefault("HF_HOME", os.environ["LOOPDYN_HF"])
try:                                        # machine-specific defaults, not tracked (e.g. os.environ.setdefault("HF_HOME", ...))
    import local_config  # noqa: F401
except ImportError:
    pass


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------
# model loading and anatomy
# --------------------------------------------------------------------------

def load_model(name, device="cuda:0", dtype=torch.bfloat16, **kw):
    """name may carry a revision after '@' (e.g. allenai/Olmo-3-1025-7B@stage1-step10000)"""
    import transformers

    rev = None
    if "@" in name:
        name, rev = name.split("@", 1)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, revision=rev, **kw
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(name, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


@dataclass
class Anatomy:
    inner: torch.nn.Module
    layers: torch.nn.ModuleList
    norm: torch.nn.Module
    embed: torch.nn.Module
    lm_head: torch.nn.Module
    rotary: torch.nn.Module
    config: object
    layer_types: list = field(default_factory=list)
    embed_scale: float = 1.0
    softcap: float | None = None


def anatomy(model) -> Anatomy:
    inner = model.model
    if hasattr(inner, "language_model"):  # multimodal wrappers (gemma-3 it)
        inner = inner.language_model
    cfg = inner.config
    n = cfg.num_hidden_layers
    lt = list(getattr(cfg, "layer_types", None) or ["full_attention"] * n)
    scale = 1.0
    if "gemma" in cfg.model_type:
        # Gemma scales embeddings by sqrt(d_model) inside the embedding module
        # in recent transformers (Gemma3TextScaledWordEmbedding); nothing to do.
        scale = 1.0
    return Anatomy(
        inner=inner,
        layers=inner.layers,
        norm=inner.norm,
        embed=inner.embed_tokens,
        lm_head=model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings(),
        rotary=inner.rotary_emb,
        config=cfg,
        layer_types=lt,
        embed_scale=scale,
        softcap=getattr(cfg, "final_logit_softcapping", None),
    )


# --------------------------------------------------------------------------
# schedule runner
# --------------------------------------------------------------------------

class Runner:
    """Run a decoder on a batch with an arbitrary layer schedule.

    schedule: list of block indices, e.g. list(range(36)) is the normal model;
    list(range(0,20)) + list(range(14,36)) repeats blocks 14..19 once.
    reentry: 'none' | 'norm' — with 'norm', whenever the schedule jumps
    backwards (re-enters a block it has already used), the residual is
    rescaled per token to the norm it had when that block was first entered.
    """

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.A = anatomy(model)
        self.n_layers = len(self.A.layers)
        self.device = next(model.parameters()).device

    def encode(self, prompts, add_bos=None):
        tok = self.tok
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def _masks(self, h, attn_mask, position_ids):
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        kw = dict(
            config=self.A.config,
            inputs_embeds=h,
            attention_mask=attn_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        masks = {"full_attention": create_causal_mask(**kw)}
        if "sliding_attention" in self.A.layer_types:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kw)
        return masks

    def _rotary(self, h, position_ids):
        rot = self.A.rotary
        try:
            return rot(h, position_ids)
        except TypeError:
            return rot(h, position_ids, "full_attention")

    @torch.no_grad()
    def run(self, *a, **k):
        return self.run_grad(*a, **k)

    def run_grad(
        self,
        input_ids,
        attn_mask=None,
        schedule=None,
        record_steps=None,
        reentry="none",
        return_hidden_final=False,
        hook=None,
        h0=None,
        start=0,
    ):
        """Returns dict(h=final residual [B,T,d], rec={step: residual after step}).

        hook(step, layer_idx, h) -> h may modify the residual after a step.
        h0/start: resume from residual h0 (the input to schedule[start]).
        """
        A = self.A
        if schedule is None:
            schedule = list(range(self.n_layers))
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        position_ids = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)
        h = A.embed(input_ids) if h0 is None else h0
        masks = self._masks(h, attn_mask, position_ids)
        pe_cache = {}
        rec = {}
        first_norm = {}
        prev = -1
        for s, li in enumerate(schedule):
            if s < start:
                continue
            layer = A.layers[li]
            if reentry == "norm":
                if li in first_norm and li <= prev:
                    cur = h.float().norm(dim=-1, keepdim=True)
                    h = (h.float() * (first_norm[li] / cur.clamp(min=1e-6))).to(h.dtype)
                if li not in first_norm:
                    first_norm[li] = h.float().norm(dim=-1, keepdim=True)
            lt = A.layer_types[li]
            if lt not in pe_cache:
                # gemma3 uses a different rope base for local layers
                try:
                    pe_cache[lt] = A.rotary(h, position_ids, lt)
                except TypeError:
                    pe_cache[lt] = A.rotary(h, position_ids)
            out = layer(
                h,
                attention_mask=masks[lt],
                position_embeddings=pe_cache[lt],
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            h = out if torch.is_tensor(out) else out[0]
            if hook is not None:
                h = hook(s, li, h)
            if record_steps is not None and s in record_steps:
                rec[s] = h
            prev = li
        return {"h": h, "rec": rec}

    def unembed(self, h):
        A = self.A
        logits = A.lm_head(A.norm(h.to(A.lm_head.weight.dtype)))
        if A.softcap is not None:
            logits = A.softcap * torch.tanh(logits / A.softcap)
        return logits

    @torch.no_grad()
    def last_logits(self, prompts, schedule=None, reentry="none", bs=16):
        outs = []
        for i in range(0, len(prompts), bs):
            ids, am = self.encode(prompts[i : i + bs])
            r = self.run(ids, am, schedule=schedule, reentry=reentry)
            outs.append(self.unembed(r["h"][:, -1]).float().cpu())
        return torch.cat(outs)


# --------------------------------------------------------------------------
# variable binding / pointer chasing task
# --------------------------------------------------------------------------

NOUNS = (
    "apple river table chair mountain garden window horse candle rabbit tiger forest "
    "ocean bridge castle pencil guitar island rocket planet violin dragon lemon cherry "
    "tomato banana carrot pepper onion potato cookie butter cheese bread coffee pizza "
    "pasta salad honey sugar glass metal stone paper silver gold copper iron cotton wool "
    "silk leather rubber plastic wood clay sand snow rain cloud storm thunder wind fire "
    "smoke dust ice steam light shadow mirror clock lamp door wall floor roof tower "
    "church school hospital library museum market station airport harbor village city "
    "desert valley lake beach cave hill field farm road street park zoo circus theater "
    "cinema kitchen bedroom bathroom garage"
).split()
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class VBItem:
    prompt: str
    answer: str  # value word (no leading space)
    depth: int
    n_chains: int
    order: str
    query_var: str
    lines: list  # list of dict(lhs, rhs, chain, level)
    roots: list  # root value per chain
    meta: dict = field(default_factory=dict)


def make_vb(
    rng: random.Random,
    depth: int,
    n_chains: int = 3,
    order: str = "forward",
    header: str = "Each line assigns a value to a variable. Track the values.\n",
    query_fmt: str = "{q} =",
    value_pool=None,
    var_pool=None,
):
    """Build one variable-binding prompt.

    Each chain c has a root value v_c and variables x_{c,1..depth}:
        x_{c,1} = v_c ; x_{c,k} = x_{c,k-1}.
    The query asks for x_{q,depth} of one random chain q, answer v_q.
    order: 'forward'  -> lines grouped by level 1..depth, chains shuffled within level
           'reverse'  -> levels depth..1
           'shuffled' -> all lines randomly permuted
           'chain'    -> each chain written contiguously (level order), chains in random order
    """
    value_pool = value_pool or NOUNS
    var_pool = var_pool or LETTERS
    need = depth * n_chains
    assert need <= len(var_pool), "not enough variable names"
    vars_ = rng.sample(var_pool, need)
    vals = rng.sample(value_pool, n_chains)
    chains = [[vars_[c * depth + k] for k in range(depth)] for c in range(n_chains)]
    lines = []
    for c in range(n_chains):
        for k in range(depth):
            rhs = vals[c] if k == 0 else chains[c][k - 1]
            lines.append(dict(lhs=chains[c][k], rhs=rhs, chain=c, level=k + 1))
    if order == "forward":
        by_level = []
        for k in range(1, depth + 1):
            lv = [l for l in lines if l["level"] == k]
            rng.shuffle(lv)
            by_level += lv
        lines = by_level
    elif order == "reverse":
        by_level = []
        for k in range(depth, 0, -1):
            lv = [l for l in lines if l["level"] == k]
            rng.shuffle(lv)
            by_level += lv
        lines = by_level
    elif order == "shuffled":
        rng.shuffle(lines)
    elif order == "chain":
        cs = list(range(n_chains))
        rng.shuffle(cs)
        lines = [l for c in cs for l in lines if l["chain"] == c]
    elif order == "interleave":
        # uniformly random merge of the chains that keeps each chain in level order (define before use),
        # with no level structure: line position says nothing about chain or level
        rest = [[l for l in lines if l["chain"] == c] for c in range(n_chains)]
        merged = []
        while any(rest):
            c = rng.choices(range(n_chains), weights=[len(r) for r in rest])[0]
            merged.append(rest[c].pop(0))
        lines = merged
    else:
        raise ValueError(order)
    q = rng.randrange(n_chains)
    qv = chains[q][depth - 1]
    body = "".join(f"{l['lhs']} = {l['rhs']}\n" for l in lines)
    prompt = header + body + query_fmt.format(q=qv)
    return VBItem(
        prompt=prompt,
        answer=vals[q],
        depth=depth,
        n_chains=n_chains,
        order=order,
        query_var=qv,
        lines=lines,
        roots=vals,
        meta=dict(query_chain=q, chains=chains),
    )


def value_token_ids(tok, words=None):
    words = words or NOUNS
    ids = {}
    for w in words:
        t = tok.encode(" " + w, add_special_tokens=False)
        if len(t) == 1:
            ids[w] = t[0]
    return ids


def make_contain(rng, d, n_chains=3, order="inward", value_pool=None):
    """Containment chains in English: 'The coin is in the cup.' ... query the innermost object; answer = outermost
    container. order: inward (outermost statements first; each line refers to an earlier-introduced container),
    outward (innermost first), shuffled. Returns VBItem-like object (prompt, answer, roots)."""
    pool = value_pool or NOUNS
    words = rng.sample(pool, n_chains * (d + 1))
    chains = [words[c * (d + 1):(c + 1) * (d + 1)] for c in range(n_chains)]
    lines = []
    for c, ch in enumerate(chains):
        for k in range(d):
            lines.append(dict(level=k, chain=c, text=f"The {ch[k]} is in the {ch[k + 1]}."))
    if order in ("inward", "outward"):
        lv = sorted(set(l["level"] for l in lines), reverse=(order == "inward"))
        out = []
        for k in lv:
            b = [l for l in lines if l["level"] == k]
            rng.shuffle(b)
            out += b
        lines = out
    else:
        rng.shuffle(lines)
    q = rng.randrange(n_chains)
    item = chains[q][0]
    prompt = " ".join(l["text"] for l in lines) + f"\nQuestion: Where is the {item}?\nAnswer: The {item} is ultimately in the"
    return VBItem(prompt=prompt, answer=chains[q][-1], depth=d, n_chains=n_chains, order=order, query_var=item,
                  lines=lines, roots=[ch[-1] for ch in chains], meta=dict(query_chain=q, chains=chains))
