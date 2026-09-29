"""Weight-verified parents for the post-training zoo. For every descendant, the relative Frobenius distance of q_proj and down_proj at
three depths (1/4, 1/2, 3/4) to each candidate parent (the base and the official post-trained model of the family); the parent is
the nearest. Then each descendant's change in default reach and in base-map reach relative to its parent. CPU only, reads safetensors
from the local Hugging Face cache. Writes results/zoo_parents.json and prints a table."""
import glob
import json
import os

import torch
from safetensors import safe_open

HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FAMILIES = {
    "q17": (28, ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"]),
    "q4": (36, ["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B"]),
    "q8": (36, ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"]),
    "o3": (32, ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Think-SFT", "allenai/Olmo-3-7B-Think-DPO", "allenai/Olmo-3-7B-Instruct"]),
    "l31": (32, ["NousResearch/Meta-Llama-3.1-8B"]),
}


def snapshot(mid):
    snaps = glob.glob(f"{HUB}/models--{mid.replace('/', '--')}/snapshots/*")
    return snaps[0] if snaps else None


def tensors(mid, names):
    d = snapshot(mid)
    out = {}
    if d is None:
        return out
    for f in glob.glob(f"{d}/*.safetensors"):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                for n in names:
                    if k.endswith(n):
                        out[n] = sf.get_tensor(k).float()
    return out


rows = {}
for tag, (N, parents) in FAMILIES.items():
    layers = [N // 4, N // 2, 3 * N // 4]
    names = [f"layers.{l}.self_attn.q_proj.weight" for l in layers] + [f"layers.{l}.mlp.down_proj.weight" for l in layers]
    ref = {p: tensors(p, names) for p in parents}
    for f in sorted(glob.glob(f"{RES}/e75_zoo_{tag}_*.json")):
        r = json.load(open(f))
        mid = r["model"]
        W = tensors(mid, names)
        if len(W) < len(names):
            continue
        dist = {}
        for p, P in ref.items():
            if len(P) < len(names) or p == mid:
                continue
            num = sum(((W[n] - P[n]) ** 2).sum().item() for n in names)
            den = sum((P[n] ** 2).sum().item() for n in names)
            dist[p] = (num / den) ** 0.5
        if not dist:
            rows[mid] = dict(family=tag, reachA=r["reachA"], reach_map=r["reach_map2"], dist={}, parent=mid, dist_parent=0.0,
                             one_line=r["setupA"]["1"][1] if "1" in r["setupA"] else r["setupA"][1][1])
            continue
        parent = min(dist, key=dist.get)
        rows[mid] = dict(family=tag, reachA=r["reachA"], reach_map=r["reach_map2"], dist=dist, parent=parent,
                         dist_parent=dist[parent], one_line=r["setupA"]["1"][1] if "1" in r["setupA"] else r["setupA"][1][1])
# changes relative to the parent (the parent's own zoo entry)
for mid, v in rows.items():
    p = rows.get(v["parent"])
    if p is not None and v["parent"] != mid:
        v["d_reachA"] = v["reachA"] - p["reachA"]
        v["d_map"] = v["reach_map"] - p["reach_map"]
json.dump(rows, open(f"{RES}/zoo_parents.json", "w"), indent=1)
for mid, v in sorted(rows.items(), key=lambda x: (x[1]["family"], x[1]["dist_parent"])):
    print(f"{v['family']:4s} {mid[:52]:52s} parent {v['parent'].split('/')[-1]:22s} dist {100 * v['dist_parent']:6.2f}% "
          f"dA {v.get('d_reachA', 0):+.2f} dMap {v.get('d_map', 0):+.1f}  one-line {v['one_line']:.2f}")
