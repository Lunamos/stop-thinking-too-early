"""Runner for Ouro-style looped LMs (transformers 4.56 env, loopdyn/.venv-loop).

Ouro forward: h = embed(x); for t in range(T): for l in range(L): h = layer_l(h);
                h = norm(h)   (at every loop end); logits = lm_head(h).
Here a *step* s = t*L + l. OuroRunner.run() can start from an arbitrary step with
a supplied residual (for patching) and records the residual after each step
(post-norm for loop-final steps).
"""
import torch


class OuroRunner:
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.inner = model.model
        self.L = len(self.inner.layers)
        self.device = next(model.parameters()).device

    def _mask(self, h, am, pos):
        from transformers.masking_utils import create_causal_mask
        cache_position = torch.arange(h.shape[1], device=h.device)
        return create_causal_mask(config=self.inner.config, input_embeds=h, attention_mask=am,
                                  cache_position=cache_position, past_key_values=None,
                                  position_ids=pos)

    @torch.no_grad()
    def run(self, ids, am=None, T=4, h0=None, start=0, record=False):
        inner = self.inner
        if am is None:
            am = torch.ones_like(ids)
        pos = (am.long().cumsum(-1) - 1).clamp(min=0)
        h = inner.embed_tokens(ids) if h0 is None else h0
        mask = self._mask(h, am, pos)
        pe = inner.rotary_emb(h, pos)
        rec = {}
        S = T * self.L
        for s in range(start, S):
            t, l = divmod(s, self.L)
            h = inner.layers[l](h, attention_mask=mask, position_ids=pos, position_embeddings=pe,
                                past_key_value=None, use_cache=False)
            if isinstance(h, tuple):
                h = h[0]
            if l == self.L - 1:
                h = inner.norm(h)
            if record:
                rec[s] = h
        return {"h": h, "rec": rec}

    def logits(self, h):
        return self.model.lm_head(h)
