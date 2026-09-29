"""text loss (WikiText) of the frozen model with / without a no-loop adapter at block a."""
import sys, math, torch, torch.nn as nn
from ld_common import Runner, load_model
model_name, adapter_path, a = sys.argv[1], sys.argv[2], int(sys.argv[3])
model, tok = load_model(model_name, "cuda:0")
R = Runner(model, tok)
d_model = model.config.get_text_config().hidden_size
class ReEntry(nn.Module):
    def __init__(self, d, r=64):
        super().__init__(); self.A = nn.Linear(d, r, bias=False); self.B = nn.Linear(r, d, bias=False); self.s = nn.Parameter(torch.ones(1))
    def forward(self, h):
        x = h.float(); rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)
_sd = torch.load(adapter_path)
M = ReEntry(d_model, _sd['A.weight'].shape[0]).cuda(); M.load_state_dict(_sd)
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 800 and not r["text"].strip().startswith("=")][:64]
for use in (False, True):
    hook = (lambda s, li, h: M(h) if s == a - 1 else h) if use else None
    tot = cnt = 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.cuda()
            h = R.run(ids, torch.ones_like(ids), hook=hook)["h"]
            lg = R.unembed(h[:, :-1]).float()
            lp = torch.log_softmax(lg, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            tot += -lp.sum().item(); cnt += lp.numel()
    print("adapter" if use else "base", "wikitext loss", round(tot / cnt, 4), "ppl", round(math.exp(tot / cnt), 3))
