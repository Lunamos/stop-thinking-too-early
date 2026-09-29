"""Multi-hop QA utilities: HotpotQA loading, prompts, answer normalization (official HotpotQA EM/F1), greedy generation with an
optional single-layer map (rank-r map or steering vector) applied at the input of block `a` via a forward hook."""
import collections
import random
import re
import string

import torch
import torch.nn as nn

INSTR = "Answer the question using the passages. Give only the answer, as a short phrase.\n\n"
DEMO = ("[Eiffel Tower]\nThe Eiffel Tower is a wrought-iron lattice tower in Paris, designed by the company of Gustave Eiffel.\n\n"
        "[Gustave Eiffel]\nGustave Eiffel was a French civil engineer, born in Dijon in 1832.\n\n"
        "Question: In which city was the engineer whose company designed the Eiffel Tower born?\nAnswer: Dijon\n\n")


def load_hotpot(split, n=None, seed=0, types=("bridge", "comparison")):
    import datasets
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        if ex["type"] not in types:
            continue
        gold_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        paras = {t: "".join(s).strip() for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])}
        if not all(t in paras for t in gold_titles):
            continue
        out.append(dict(id=ex["id"], question=ex["question"], answer=ex["answer"], type=ex["type"], level=ex["level"],
                        gold=[(t, paras[t]) for t in gold_titles],
                        distractors=[(t, p) for t, p in paras.items() if t not in gold_titles]))
        if n and len(out) >= n:
            break
    return out


def build_prompt(ex, setting="gold", n_distract=2, demo=True, seed=0, max_para_chars=900):
    rng = random.Random(hash(ex["id"]) % (2 ** 31) + seed)
    if setting == "closed":
        paras = []
    elif setting in ("gold_fwd", "gold_rev"):
        # gold paragraphs in the order of the reasoning chain (first hop first), or reversed
        paras = list(ex["gold"]) if setting == "gold_fwd" else list(ex["gold"])[::-1]
    else:
        paras = list(ex["gold"])
        if setting == "distractor":
            paras += rng.sample(ex["distractors"], min(n_distract, len(ex["distractors"])))
        rng.shuffle(paras)
    body = "".join(f"[{t}]\n{p[:max_para_chars]}\n\n" for t, p in paras)
    head = INSTR + (DEMO if demo else "")
    return head + body + f"Question: {ex['question']}\nAnswer:"


# ---- official HotpotQA normalization
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    np_, ng = normalize_answer(prediction), normalize_answer(ground_truth)
    if np_ in ("yes", "no", "noanswer") and np_ != ng:
        return 0.0
    if ng in ("yes", "no", "noanswer") and np_ != ng:
        return 0.0
    pt, gt = np_.split(), ng.split()
    common = collections.Counter(pt) & collections.Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return 2 * p * r / (p + r)


def em_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ---- maps
class Map(nn.Module):
    """rank-r map h <- s h + rms(h) B A (h / rms(h)), or steering vector h <- s h + rms(h) v"""
    def __init__(self, d, rank=8, steer=False):
        super().__init__()
        if steer:
            self.v = nn.Parameter(torch.zeros(d))
        else:
            self.A = nn.Linear(d, rank, bias=False)
            self.B = nn.Linear(rank, d, bias=False)
            nn.init.normal_(self.A.weight, std=d ** -0.5)
            nn.init.zeros_(self.B.weight)
        self.s = nn.Parameter(torch.ones(1))

    @classmethod
    def from_state(cls, sd, d):
        if "v" in sd:
            m = cls(d, steer=True)
        else:
            m = cls(d, rank=sd["A.weight"].shape[0])
        m.load_state_dict(sd)
        return m

    def forward(self, h):
        x = h.float()
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        if hasattr(self, "v"):
            return (self.s * x + rms * self.v).to(h.dtype)
        return (self.s * x + rms * self.B(self.A(x / rms))).to(h.dtype)


def decoder_layers(model):
    m = model.model
    if hasattr(m, "language_model"):
        m = m.language_model
    return m.layers


def attach_map(model, M, a):
    """apply M to the output of block a-1 (= input of block a); returns the hook handle"""
    layer = decoder_layers(model)[a - 1]

    def fwd_hook(mod, inp, out):
        if torch.is_tensor(out):
            return M(out)
        return (M(out[0]),) + tuple(out[1:])
    return layer.register_forward_hook(fwd_hook)


@torch.no_grad()
def generate_answers(model, tok, prompts, bs=8, max_new_tokens=16):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    nl = tok.encode("\n", add_special_tokens=False)
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                           eos_token_id=[tok.eos_token_id] + nl)
        for row in g[:, enc.input_ids.shape[1]:]:
            txt = tok.decode(row, skip_special_tokens=True)
            outs.append(txt.split("\n")[0].strip())
    return outs


def score(exs, preds):
    res = collections.defaultdict(list)
    for ex, p in zip(exs, preds):
        for key in ("all", ex["type"]):
            res[key + "_em"].append(em_score(p, ex["answer"]))
            res[key + "_f1"].append(f1_score(p, ex["answer"]))
    return {k: sum(v) / len(v) for k, v in res.items()} | {"n": len(preds)}


def load_musique(split, n=None, seed=0, hops=(2, 3, 4)):
    """MuSiQue-Ans; gold = supporting paragraphs of the decomposition (order shuffled at prompt time)"""
    import datasets
    ds = datasets.load_dataset("dgslibisey/MuSiQue", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        if not ex["answerable"]:
            continue
        h = int(ex["id"][0])
        if h not in hops:
            continue
        paras = {p["idx"]: (p["title"], p["paragraph_text"].strip()) for p in ex["paragraphs"]}
        sup = [s["paragraph_support_idx"] for s in ex["question_decomposition"]]
        if any(s not in paras for s in sup):
            continue
        out.append(dict(id=ex["id"], question=ex["question"], answer=ex["answer"],
                        aliases=[ex["answer"]] + list(ex["answer_aliases"] or []), type=f"{h}hop", level=str(h),
                        gold=[paras[s] for s in dict.fromkeys(sup)],
                        distractors=[v for k, v in paras.items() if k not in sup],
                        decomposition=[(s["question"], s["answer"]) for s in ex["question_decomposition"]]))
        if n and len(out) >= n:
            break
    return out


def load_2wiki(split="dev", n=None, seed=0, types=("compositional", "inference")):
    """2WikiMultihopQA (kamelliao mirror, with Wikidata answer aliases); gold = supporting paragraphs in reasoning order"""
    import ast
    import json as _json
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("kamelliao/2wikimultihopqa", f"data/{split}.json", repo_type="dataset")
    apath = hf_hub_download("kamelliao/2wikimultihopqa", "data/id_aliases.json", repo_type="dataset")
    alias = {}
    for ln in open(apath):
        r = _json.loads(ln)
        alias[r["Q_id"]] = list(r.get("aliases") or []) + list(r.get("demonyms") or [])
    data = _json.load(open(path))
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = data[i]
        if ex["type"] not in types:
            continue
        ctx = ast.literal_eval(ex["context"]) if isinstance(ex["context"], str) else ex["context"]
        sup = ast.literal_eval(ex["supporting_facts"]) if isinstance(ex["supporting_facts"], str) else ex["supporting_facts"]
        paras = {t: " ".join(ss).strip() for t, ss in ctx}
        sents = {t: ss for t, ss in ctx}
        last = {}
        for t, sid in sup:
            last[t] = max(last.get(t, 0), sid)
        # gold paragraphs end one sentence after their last supporting sentence (keeps the needed fact inside the prompt)
        paras.update({t: " ".join(sents[t][:last[t] + 2]).strip() for t in last if t in sents})
        gold_titles = list(dict.fromkeys(t for t, _ in sup))
        if not gold_titles or not all(t in paras for t in gold_titles):
            continue
        al = [ex["answer"]] + [a for a in alias.get(ex.get("answer_id", ""), []) if a and len(a) < 80]
        out.append(dict(id=ex["_id"], question=ex["question"], answer=ex["answer"], aliases=al, type=ex["type"],
                        level=ex["type"], gold=[(t, paras[t]) for t in gold_titles],
                        distractors=[(t, p) for t, p in paras.items() if t not in gold_titles]))
        if n and len(out) >= n:
            break
    return out


def score_aliases(exs, preds):
    res = collections.defaultdict(list)
    for ex, p in zip(exs, preds):
        al = ex.get("aliases") or [ex["answer"]]
        em = max(em_score(p, a) for a in al)
        f1 = max(f1_score(p, a) for a in al)
        for key in ("all", ex["type"]):
            res[key + "_em"].append(em)
            res[key + "_f1"].append(f1)
    return {k: sum(v) / len(v) for k, v in res.items()} | {"n": len(preds)}


def load_squad(split, n=None, seed=0):
    """SQuAD v1.1: single-paragraph, single-hop extractive QA in the same prompt format (format control)"""
    import datasets
    ds = datasets.load_dataset("rajpurkar/squad", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        ans = ex["answers"]["text"]
        if not ans:
            continue
        out.append(dict(id=ex["id"], question=ex["question"], answer=ans[0], aliases=list(dict.fromkeys(ans)),
                        type="1hop", level="1", gold=[(ex["title"].replace("_", " "), ex["context"])], distractors=[]))
        if n and len(out) >= n:
            break
    return out
