"""Which small change starts the relay (E91, scripts/e91_intervention_compare.py). Qwen3-8B, one intervention trained exactly like the
paper's map (two chains, level order, programs of up to 20 lines, 1,200 steps, answer cross-entropy plus the WikiText KL penalty),
evaluated on 200 fresh programs per length. Exact accuracy in percent; parameters trained; text KL: the WikiText penalty term
KL(frozen || changed) per token, averaged over the logged training steps from step 1,000 on (one batch of eight texts each). Writes
out/tables/interventions.tex and prints the table."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RES, TABLES  # noqa: E402

OUT = os.path.join(TABLES, "interventions.tex")
DS = [8, 16, 24]
ROWS = [
    ("q8_map_a14", "map (ours): $h \\leftarrow s\\,h + BAh$", "rank 8, layer 14"),
    ("q8_multimap_6_10_14", "the same map at three layers", "layers 6, 10, 14, shared"),
    ("q8_flow_N3_a14", "low-rank flow (FLAS-style), 3 steps", "rank 8, layer 14"),
    ("q8_flow_N1_a14", "low-rank flow, 1 step", "rank 8, layer 14"),
    ("q8_fb_time_N3", "FLAS flow block, 3 steps", "MLP + time, layer 14"),
    ("q8_fb_attn_time_N3", "\\quad with self-attention", "layer 14"),
    ("q8_fb_notime_N3", "\\quad without time embedding", "layer 14"),
    ("q8_fb_time_N1", "\\quad 1 step", "layer 14"),
    ("q8_lora_a14", "LoRA (all projections of one layer)", "rank 8, layer 14"),
    None,                                              # the same changes after the placement limit (maps work up to layer 20)
    ("q8_map_a26", "map", "rank 8, layer 26"),
    ("q8_flow_N3_a26", "low-rank flow, 3 steps", "rank 8, layer 26"),
    ("q8_fb_time_N1_a26", "FLAS flow block, 1 step", "MLP + time, layer 26"),
    ("q8_lora_a26", "LoRA (all projections of one layer)", "rank 8, layer 26"),
]


def fmt_params(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n:,}"


rows, frozen = [], None
for row in ROWS:
    if row is None:
        rows.append(None)
        continue
    tag, label, where = row
    p = f"{RES}/e91_compare_{tag}.json"
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    ev = r["eval"]
    if frozen is None:
        frozen = [100 * ev[f"d{d}_frozen"][0] for d in DS]
    kl = [x["kl"] for x in r["log"] if x["step"] >= 1000]
    rows.append((label, where, fmt_params(r["n_params"]), sum(kl) / len(kl), [100 * ev[f"d{d}_map"][0] for d in DS]))

lines = ["\\begin{tabular}{llrr" + "r" * len(DS) + "}", "\\toprule",
         "Change & Where & Parameters & Text KL & " + " & ".join(f"{d} lines" for d in DS) + " \\\\", "\\midrule",
         "none (frozen) & & 0 & 0 & " + " & ".join(f"{v:.1f}" for v in frozen) + " \\\\"]
if rows and rows[-1] is None:
    rows.pop()
for r in rows:
    if r is None:
        lines += ["\\midrule", f"\\multicolumn{{{4 + len(DS)}}}{{l}}{{\\emph{{at layer 26, after the placement limit (maps work up to layer 20):}}}} \\\\"]
        continue
    label, where, n, kl, acc = r
    lines.append(f"{label} & {where} & {n} & {kl:.3f} & " + " & ".join(f"{v:.1f}" for v in acc) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
os.makedirs(TABLES, exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
