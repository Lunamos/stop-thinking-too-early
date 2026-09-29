"""Standard models on MuSiQue with fixed prompts.

Every trained adapter of the MuSiQue experiments (e52_mqa_train.py) was re-evaluated with PYTHONHASHSEED=0, so that all runs see the
same paragraph order (build_prompt seeds its shuffle with Python's per-process string hash), on the same 900 development questions
with their supporting paragraphs, saving per-question predictions (results/e52_mqa_rv_*.json; commands in REPRODUCE.md).
Per question we take exact match against the answer and its aliases; arms trained with several seeds are averaged per question
first. Gains over the frozen model and differences between arms get paired 95% bootstrap intervals over questions (2,000
resamples; one generator per comparison, seeded by its name). The intervals reflect question sampling, not training seeds; seed
ranges are reported next to them.

Writes checks/musique_std.json, tables/qa_std.tex and tables/numbers_qa.tex (macros used in the text); prints a summary.
usage: python std_musique_paired.py
"""
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS, DATA, RES, TABLES  # noqa: E402
from mqa_common import em_score  # noqa: E402

VAL = json.load(open(f"{DATA}/musique_val900_seed1.json"))          # load_musique("validation", n=900, seed=1), exported
MODELS = {                                                            # tag, name, blocks, learning rate of the main LoRA arms
    "q8": ("Qwen3-8B", 36, "3e-4"),
    "o3": ("OLMo-3-7B", 32, "1e-4"),
    "l31": ("Llama-3.1-8B", 32, "1e-4"),
}
# arm -> run tags (seeds); LoRA ranges are the inclusive block ranges trained
ARMS = {
    "q8": {
        **{f"map@{a}": [f"q8_mus_map_a{a}"] for a in (6, 10, 20, 26, 30)},
        "map@14": ["q8_mus_map_a14", "q8_mus_map_a14_s2", "q8_mus_map_a14_s3"],
        "LoRA all": ["q8_mus_lora8_lr3e4", "q8_mus_lora8_lr3e4_s1"],
        "LoRA 0-20": ["q8_mus_lora8_pre21_lr3e4", "q8_mus_lora8_pre21_lr3e4_s1"],
        "LoRA 21-35": ["q8_mus_lora8_post21_lr3e4", "q8_mus_lora8_post21_lr3e4_s1"],
        "LoRA 0-8": ["q8_mus_lora8_q1_lr3e4"], "LoRA 9-17": ["q8_mus_lora8_q2_lr3e4"],
        "LoRA 18-26": ["q8_mus_lora8_q3_lr3e4"], "LoRA 27-35": ["q8_mus_lora8_q4_lr3e4"],
        "LoRA 0-17": ["q8_mus_lora8_first18_lr3e4", "q8_mus_lora8_first18_lr3e4_s1"],
        "LoRA 18-35": ["q8_mus_lora8_last18_lr3e4", "q8_mus_lora8_last18_lr3e4_s1"],
        "LoRA all (lr 1e-4)": ["q8_mus_lora8"], "LoRA 21-35 (lr 1e-4)": ["q8_mus_lora8_post21"],
        "steering vector@14": ["q8_mus_steer_a14"], "SQuAD map@14": ["q8_squad_map_a14"],
        "FLAS flow@6": ["q8_mus_flow_a6"], "FLAS block@6": ["q8_mus_flowmlp_a6"],
    },
    "o3": {
        **{f"map@{a}": [f"o3_mus_map_a{a}"] for a in (4, 8, 12, 16, 20, 24, 28)},
        "LoRA all": ["o3_mus_lora8"],
        "LoRA 0-14": ["o3_mus_lora8_pre", "o3_mus_lora8_pre_s1"],
        "LoRA 15-31": ["o3_mus_lora8_post", "o3_mus_lora8_post_s1"],
    },
    "l31": {
        **{f"map@{a}": [f"l31_mus_map_a{a}"] for a in (4, 8, 12, 16, 20, 24, 28)},
        "LoRA all": ["l31_mus_lora8"],
        "LoRA 0-14": ["l31_mus_lora8_pre15", "l31_mus_lora8_pre15_s1"],
        "LoRA 15-31": ["l31_mus_lora8_post15", "l31_mus_lora8_post15_s1"],
    },
}
CONTRASTS = {                                                         # (arm a, arm b): a - b on the same questions
    "q8": [("LoRA 0-20", "LoRA 21-35"), ("LoRA all", "LoRA 0-20"), ("LoRA 0-8", "LoRA 27-35"), ("LoRA 0-17", "LoRA 18-35"),
           ("map@6", "map@30"), ("map@14", "map@26"), ("map@14", "steering vector@14"), ("map@6", "FLAS flow@6"),
           ("map@6", "FLAS block@6")],
    "o3": [("LoRA 0-14", "LoRA 15-31"), ("LoRA all", "LoRA 0-14"), ("map@4", "map@28")],
    "l31": [("LoRA 0-14", "LoRA 15-31"), ("LoRA all", "LoRA 0-14"), ("map@8", "map@28")],
}


def per_question(tag):
    p = f"{RES}/e52_mqa_rv_{tag}.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    preds = r.get("preds", {}).get("musique_gold")
    if preds is None or len(preds) != len(VAL):
        return None
    em = np.array([max(em_score(x, a) for a in (ex.get("aliases") or [ex["answer"]])) for ex, x in zip(VAL, preds)], float)
    assert abs(em.mean() - r["res"]["musique_gold"]["all_em"]) < 1e-9, tag          # same questions, same order
    return em, r["args"]


def boot(diff, key, B=2000):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pct(x):
    return round(100 * float(x), 1)


out = {"n_questions": len(VAL), "models": {}}
macros = []
for m, (name, N, lr) in MODELS.items():
    fz = per_question(f"{m}_frozen")
    if fz is None:
        print(f"{name}: frozen re-evaluation missing")
        continue
    fz = fz[0]
    rec = {"name": name, "blocks": N, "frozen_em": pct(fz.mean()), "arms": {}, "contrasts": {}}
    arm_q = {}
    for arm, tags in ARMS[m].items():
        runs = [(t, per_question(t)) for t in tags]
        runs = [(t, r) for t, r in runs if r is not None]
        if not runs:
            continue
        q = np.mean([r[0] for _, r in runs], axis=0)
        arm_q[arm] = q
        lo, hi = boot(q - fz, f"{m}|{arm}|frozen")
        seeds = [pct(r[0].mean()) for _, r in runs]
        rec["arms"][arm] = {"runs": [t for t, _ in runs], "missing": [t for t in tags if t not in [x for x, _ in runs]],
                            "em": pct(q.mean()), "seed_em": seeds, "gain": pct(q.mean() - fz.mean()), "ci": [pct(lo), pct(hi)],
                            "lora_layers": runs[0][1][1].get("lora_layers"), "lr": runs[0][1][1].get("lr")}
    for a, b in CONTRASTS[m]:
        if a in arm_q and b in arm_q:
            d = arm_q[a] - arm_q[b]
            lo, hi = boot(d, f"{m}|{a}|{b}")
            rec["contrasts"][f"{a} - {b}"] = {"diff": pct(d.mean()), "ci": [pct(lo), pct(hi)]}
    out["models"][m] = rec
    print(f"== {name}: frozen {rec['frozen_em']}")
    for arm, v in rec["arms"].items():
        print(f"  {arm:22s} EM {v['em']:5.1f} (seeds {v['seed_em']})  gain {v['gain']:+5.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]"
              + (f"  missing {v['missing']}" if v["missing"] else ""))
    for k, v in rec["contrasts"].items():
        print(f"  {k:36s} {v['diff']:+5.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")

json.dump(out, open(f"{CHECKS}/musique_std.json", "w"), indent=1)

# table: one row per arm
rows = []
for m, rec in out["models"].items():
    first = True
    for arm, v in [("frozen", None)] + list(rec["arms"].items()):
        if v is None:
            cells = [rec["name"], "frozen", "--", f"{rec['frozen_em']:.1f}", "--"]
        else:
            seeds = "/".join(f"{s:.1f}" for s in v["seed_em"]) if len(v["seed_em"]) > 1 else "--"
            cells = ["" if not first else rec["name"], arm.replace("@", " at "), seeds, f"{v['em']:.1f}",
                     f"{v['gain']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]"]
        if not first and cells[0] == rec["name"]:
            cells[0] = ""
        rows.append(" & ".join(cells) + r" \\")
        first = False
    rows.append(r"\midrule")
rows = rows[:-1]
with open(f"{TABLES}/qa_std.tex", "w") as f:
    f.write("\n".join([r"\begin{tabular}{llrrl}", r"\toprule",
                       r"Model & Arm & Seeds (EM) & EM & Gain [95\% CI] \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


KEYS = {"q8": "Qwen", "o3": "Olmo", "l31": "Llama"}
for m, rec in out["models"].items():
    k = KEYS[m]
    mac(f"{k}FrozenEM", f"{rec['frozen_em']:.1f}")
    for arm, short in (("LoRA all", "All"), ("LoRA 0-20", "Early"), ("LoRA 21-35", "Late"), ("LoRA 0-14", "Early"),
                       ("LoRA 15-31", "Late")):
        if arm in rec["arms"]:
            v = rec["arms"][arm]
            mac(f"{k}{short}EM", f"{v['em']:.1f}")
            mac(f"{k}{short}Gain", f"{v['gain']:+.1f}".replace("+", ""))
            mac(f"{k}{short}CI", f"[{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]")
with open(f"{TABLES}/numbers_qa.tex", "w") as f:
    f.write("% generated by analysis/std_musique_paired.py\n" + "\n".join(macros) + "\n")
print("wrote", f"{CHECKS}/musique_std.json", f"{TABLES}/qa_std.tex", f"{TABLES}/numbers_qa.tex")
