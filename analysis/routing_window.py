"""Routing-window model of latent pointer chasing: estimate every quantity per model from existing result files, test the
model's reach predictions across the 13-model panel, and draw out/notes/routing_window.png.

No GPU, no new runs: reads only results/*.json (results/prereg_edges.json from prereg_score.py). Writes
out/notes/routing_window_table.json and out/notes/routing_window.png.

Definitions (fixed from the Qwen3-8B mechanism before looking at the other models' fits):
  a    walk start: first layer at which the queried line's pointer (three-line chains, pointer on line 3, e10 panel) keeps
       less than 0.75 of its effect on its own token.            (Qwen3-8B: 17 on three-line and on five-line chains)
  b    window end: first layer at which the final token holds at least 0.9 of both pointers' effects (lines 2 and 3).
                                                                   (Qwen3-8B: 24; 'by layer 24 no program position holds any')
  D    window width b - a (layers).
  H    value handoff: first layer at which the final token holds half of the root value's effect (mean over 1-3 lines).
  c    commit (paper): mean first layer at which the pointers on lines 2 and 3 keep less than half of their effect.
  rho  default relay (paper): sum over lines 3-8 of clip((best read-out - 0.6)/0.4), two chains of eight lines (e32b).
  tau  layers per walk read, 2.5, from Qwen3-8B five-line chains answered correctly (reads leave at layers 17, 20, 22).
  R    reach: longest chain answered with 80% choice accuracy among three chains (setup A), interpolated (tab_panel).

Model:   R = (2 + rho) + D / tau - delta      [relay lines + extra walk reads that fit in the window - reliability margin]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "notes")
os.makedirs(OUT, exist_ok=True)
TAU = 2.5

# name, e10 tag, e32b relay tag, total parameters (billions, model cards), family
PANEL = [("Llama-3.2-1B", "llama32_1b", "relay_llama32_1b", 1.24, "Llama"),
         ("Qwen3-0.6B", "q06base", "relay_q06", 0.60, "Qwen"),
         ("Qwen3-1.7B", "q17base", "relay_q17", 1.7, "Qwen"),
         ("Llama-3.2-3B", "llama32_3b", "relay_llama32_3b", 3.2, "Llama"),
         ("Qwen3-4B", "q4base", "relay_q4", 4.0, "Qwen"),
         ("Gemma-3-4B", "gemma3_4b", "relay_gemma3_4b", 4.3, "Gemma"),
         ("Llama-3.1-8B", "llama31_8b", "relay_llama31_8b", 8.0, "Llama"),
         ("OLMo-3-7B", "olmo3_7b", "relay_olmo3_7b", 7.3, "OLMo"),
         ("Qwen3-8B", "q8base", "relay_q8", 8.2, "Qwen"),
         ("Gemma-3-12B", "gemma3_12b", "relay_gemma3_12b", 12.2, "Gemma"),
         ("Qwen3-14B", "q14base", "relay_q14", 14.8, "Qwen"),
         ("Gemma-3-27B", "gemma3_27b", "relay_gemma3_27b", 27.4, "Gemma"),
         ("OLMo-3-32B", "olmo3_32b", "relay_olmo3_32b", 32.2, "OLMo")]


# ------------------------------------------------------------------ helpers
def first_below(y, thr):
    return next((i for i, x in enumerate(y) if x < thr), float("nan"))


def first_above(y, thr):
    return next((i for i, x in enumerate(y) if x >= thr), float("nan"))


def reach(acc, thr=0.8):
    """longest chain answered with accuracy >= thr, linear interpolation (same rule as tab_panel_v5.py)"""
    ds = sorted(acc)
    best = 0.0
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= thr > acc[d1]:
            return d0 + (acc[d0] - thr) / (acc[d0] - acc[d1]) * (d1 - d0)
        if acc[d1] >= thr:
            best = d1
    return best if acc[ds[0]] >= thr else 0.0


def band80(own):
    """layers holding the central 80% of the decline of a pointer's own share (interpolated 10% and 90% crossings)"""
    own = np.asarray(own, float)
    s0, e0 = own[0], own[-1]

    def cross(f):
        thr = s0 - f * (s0 - e0)
        for i in range(1, len(own)):
            if own[i] <= thr < own[i - 1]:
                return i - 1 + (own[i - 1] - thr) / (own[i - 1] - own[i])
        return float("nan")
    return cross(0.1), cross(0.9)


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):          # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(u, v):
    return float(np.corrcoef(ranks(u), ranks(v))[0, 1])


def pearson(u, v):
    return float(np.corrcoef(u, v)[0, 1])


def partial_spearman(x, y, z):
    """rank correlation of x and y after removing a linear effect of rank(z) from both ranks"""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


# ------------------------------------------------------------------ 1. per-model quantities
rows = []
for name, tag, rtag, params, fam in PANEL:
    r = json.load(open(f"{RES}/e10_panel_{tag}.json"))
    N = r["N"]
    racc = {int(d): v["racc"] for d, v in r["acc"].items()}
    S = {d: (racc[d] - 1 / 3) / (2 / 3) for d in racc}                  # chance-corrected success
    R = reach(racc)
    H = float(np.mean([first_above(r["val"][d]["agg"]["query|last"], 0.5) for d in ("1", "2", "3")]))
    p3, p2 = r["ptr"]["3"]["agg"], r["ptr"]["2"]["agg"]
    own3, own2 = np.array(p3["q|3|rhs"]), np.array(p2["q|2|rhs"])
    fin3, fin2 = np.array(p3["query|last"]), np.array(p2["query|last"])
    a = first_below(own3, 0.75)
    a50 = first_below(own3, 0.5)
    commit = (first_below(own3, 0.5) + first_below(own2, 0.5)) / 2
    b = max(first_above(fin3, 0.9), first_above(fin2, 0.9))
    g = json.load(open(f"{RES}/e32b_wave_{rtag}.json"))["frozen"]["grids"]["rhs_slot"]
    rho = sum(min(1.0, max(0.0, (max(g[k]) - 0.6) / 0.4)) for k in range(3, min(len(g), 9)))
    rho_b = sum(min(1.0, max(0.0, (max(g[k][:int(b) + 1]) - 0.6) / 0.4)) for k in range(3, min(len(g), 9)))
    r3 = first_above(g[3], 0.75)
    r4 = first_above(g[4], 0.7)                      # line-4 label onset (0.7: several models peak at 0.73-0.79)
    lo80, hi80 = band80(own3)
    rows.append(dict(name=name, fam=fam, N=N, P=params, R=R, S={int(k): v for k, v in S.items()}, band80=(lo80, hi80),
                     W80=hi80 - lo80,
                     racc={int(k): v for k, v in racc.items()}, a=a, a50=a50, commit=commit, b=b, D=b - a, H=H,
                     gapHb=H - b, rho=rho, rho_b=rho_b, r3=r3, r4=r4, line4=max(g[4])))

print("Per-model quantities (layers unless stated)")
hdr = ["N", "P", "R", "rho", "r3", "r4", "a", "commit", "b", "D", "H", "gapHb"]
print(f"  {'model':13s} " + " ".join(f"{h:>6s}" for h in hdr))
for x in rows:
    print(f"  {x['name']:13s} " + " ".join(f"{x[h]:6.1f}" if not isinstance(x[h], int) else f"{x[h]:6d}" for h in hdr))

N_ = np.array([x["N"] for x in rows], float)
D_ = np.array([x["D"] for x in rows], float)
R_ = np.array([x["R"] for x in rows])
rho_ = np.array([x["rho"] for x in rows])
logP = np.log10([x["P"] for x in rows])

# ------------------------------------------------------------------ 2. structure of the window
rng = np.random.default_rng(0)
slope = np.polyfit(N_, D_, 1)[0]
boots = []
for _ in range(10000):
    i = rng.integers(0, len(N_), len(N_))
    if len(set(N_[i])) > 2:
        boots.append(np.polyfit(N_[i], D_[i], 1)[0])
lo, hi = np.percentile(boots, [2.5, 97.5])
prop = float(np.mean(D_ / N_))
print(f"\nWindow width D: mean {D_.mean():.1f} layers, sd {D_.std(ddof=1):.1f}, range {D_.min():.0f}-{D_.max():.0f}; "
      f"N range {N_.min():.0f}-{N_.max():.0f}")
print(f"  slope dD/dN = {slope:.3f} (95% bootstrap CI {lo:.3f} to {hi:.3f}); a width proportional to depth would have slope "
      f"{D_.mean() / N_.mean():.3f} (mean D/N {prop:.2f})")
W80 = np.array([x["W80"] for x in rows])
print(f"  second definition, layers holding the central 80% of the queried pointer's departure: mean {W80.mean():.1f}, "
      f"sd {W80.std(ddof=1):.1f}, range {W80.min():.1f}-{W80.max():.1f}; slope {np.polyfit(N_, W80, 1)[0]:+.3f} (proportional "
      f"{W80.mean() / N_.mean():.3f}): " + ", ".join(f"{x['name']} {x['W80']:.1f}" for x in rows))
print(f"  relative position of the window start a/N: mean {np.mean([x['a'] / x['N'] for x in rows]):.2f}, sd "
      f"{np.std([x['a'] / x['N'] for x in rows], ddof=1):.2f}")
both = [x for x in rows if not np.isnan(x["r4"])]
print(f"  line-4 relay label onset (read-out >= 0.7) vs walk start a, {len(both)} models with a line-4 label: "
      + ", ".join(f"{x['name']} {x['r4']:.0f}/{x['a']:.0f}" for x in both)
      + f"; mean r4 - a = {np.mean([x['r4'] - x['a'] for x in both]):+.1f}, median |r4 - a| = "
        f"{np.median([abs(x['r4'] - x['a']) for x in both]):.1f}")
print(f"  idle gap H - b: " + ", ".join(f"{x['name']} {x['gapHb']:.0f}" for x in rows))


# ------------------------------------------------------------------ 3. reach models, leave-one-out
def fit_eval(name, X, y, free):
    """least squares on design X (columns), LOO predictions; X may be (n,0) for a fixed-prediction model with offset only"""
    n = len(y)
    pred = np.empty(n)
    loo = np.empty(n)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    for i in range(n):
        m = np.arange(n) != i
        c = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        loo[i] = X[i] @ c
    return dict(name=name, free=free, coef=coef.tolist(), pred=pred.tolist(), loo=loo.tolist(),
                spearman=spearman(pred, y), pearson=pearson(pred, y), mae=float(np.mean(np.abs(pred - y))),
                loo_mae=float(np.mean(np.abs(loo - y))), loo_max=float(np.max(np.abs(loo - y))),
                loo_rmse=float(np.sqrt(np.mean((loo - y) ** 2))))


one = np.ones(len(rows))
win = np.array([x["D"] for x in rows]) / TAU
win_commit = np.array([x["commit"] - x["a"] for x in rows]) / TAU
win_H = np.array([x["H"] - x["a"] for x in rows]) / TAU
models = []
# the routing-window model: slope on relay and window fixed at 1 (lines), only the offset delta is fitted
models.append(fit_eval("window: R = 2 + rho + (b-a)/tau - delta", one[:, None], R_ - (2 + rho_ + win), 1))
models.append(fit_eval("window end = commit: R = 2 + rho + (c-a)/tau - delta", one[:, None], R_ - (2 + rho_ + win_commit), 1))
models.append(fit_eval("window end = handoff: R = 2 + rho + (H-a)/tau - delta", one[:, None], R_ - (2 + rho_ + win_H), 1))
models.append(fit_eval("relay only: R = rho + c0", one[:, None], R_ - rho_, 1))
models.append(fit_eval("window only: R = (b-a)/tau + c0", one[:, None], R_ - win, 1))
models.append(fit_eval("constant: R = c0", one[:, None], R_, 1))
models.append(fit_eval("scale: R = b0 + b1 log10(params)", np.c_[one, logP], R_, 2))
models.append(fit_eval("depth: R = b0 + b1 N", np.c_[one, N_], R_, 2))
models.append(fit_eval("free: R = b0 + b1 rho + b2 (b-a)", np.c_[one, rho_, D_], R_, 3))
# add the offset back for the offset-only models so that pred/loo are reaches
base = {0: 2 + rho_ + win, 1: 2 + rho_ + win_commit, 2: 2 + rho_ + win_H, 3: rho_, 4: win, 5: 0 * rho_}
for k, off in base.items():
    m = models[k]
    m["pred"] = (np.array(m["pred"]) + off).tolist()
    m["loo"] = (np.array(m["loo"]) + off).tolist()
    p, l = np.array(m["pred"]), np.array(m["loo"])
    m.update(spearman=spearman(p, R_) if k != 5 else float("nan"), pearson=pearson(p, R_) if k != 5 else float("nan"),
             mae=float(np.mean(np.abs(p - R_))), loo_mae=float(np.mean(np.abs(l - R_))),
             loo_max=float(np.max(np.abs(l - R_))), loo_rmse=float(np.sqrt(np.mean((l - R_) ** 2))))

print("\nReach models (13 models; reach range {:.1f}-{:.1f}, sd {:.2f})".format(R_.min(), R_.max(), R_.std(ddof=1)))
print(f"  {'model':52s} {'free':>4s} {'spear':>6s} {'pears':>6s} {'MAE':>5s} {'LOO-MAE':>7s} {'LOO-max':>7s}")
for m in models:
    print(f"  {m['name']:52s} {m['free']:4d} {m['spearman']:6.2f} {m['pearson']:6.2f} {m['mae']:5.2f} {m['loo_mae']:7.2f} "
          f"{m['loo_max']:7.2f}")
W = models[0]
print(f"\n  window model: delta = {-W['coef'][0]:.2f} lines; residuals (measured - predicted, LOO):")
for x, p in zip(rows, W["loo"]):
    print(f"    {x['name']:13s} R {x['R']:.2f}  pred {p:.2f}  resid {x['R'] - p:+.2f}")
print(f"  partial Spearman(reach, rho | log params) = {partial_spearman(rho_, R_, logP):+.2f}; "
      f"(reach, D | log params) = {partial_spearman(D_, R_, logP):+.2f}; (reach, N | log params) = "
      f"{partial_spearman(N_, R_, logP):+.2f}; Spearman(reach, log params) = {spearman(logP, R_):+.2f}")
o7 = next(x for x in rows if x["name"] == "OLMo-3-7B")
o32 = next(x for x in rows if x["name"] == "OLMo-3-32B")
i7, i32 = rows.index(o7), rows.index(o32)
print("  OLMo-3-32B minus OLMo-3-7B (measured {:+.2f} lines):".format(o32["R"] - o7["R"]))
for m in models:
    print(f"    {m['name']:52s} predicted {m['pred'][i32] - m['pred'][i7]:+.2f}")

# ------------------------------------------------------------------ 4. where maps help: three regimes
pe = json.load(open(f"{RES}/prereg_edges.json"))
win_of = {x["name"].replace("-Base", ""): x for x in rows}
alias = {"Qwen3-8B": "Qwen3-8B", "OLMo-3-7B": "OLMo-3-7B", "Llama-3.1-8B": "Llama-3.1-8B", "Qwen3-1.7B": "Qwen3-1.7B",
         "Qwen3-0.6B": "Qwen3-0.6B", "Llama-3.2-1B": "Llama-3.2-1B", "Qwen3-4B": "Qwen3-4B"}
place = []
print("\nMap placement against the frozen model's window (rank-8 map, exact accuracy, reach in lines)")
for k, v in pe.items():
    x = win_of[alias.get(k, k)]
    pts = sorted((int(p), rr) for p, rr in v["reach"].items())
    edge = v["edge"]
    after_b = [rr - v["R0"] for p, rr in pts if p > x["b"]]
    between = [(p, rr - v["R0"]) for p, rr in pts if edge < p <= x["b"]]
    before = [rr for p, rr in pts if p < edge]
    place.append(dict(model=k, N=v["N"], a=x["a"], b=x["b"], edge=edge, bracket=v["bracket"], R0=v["R0"],
                      pts=pts, max=v["max_reach"]))
    print(f"  {k:13s} a {x['a']:.0f}  edge {edge:.1f} (edge-a {edge - x['a']:+.1f})  b {x['b']:.0f} (b-edge {x['b'] - edge:.1f}) "
          f" gain after b: {', '.join(f'{g:+.1f}' for g in after_b) or '--'};  between edge and b: "
          f"{', '.join(f'L{p} {g:+.1f}' for p, g in between)}")
ea = np.array([p["edge"] - p["a"] for p in place])
eb = np.array([p["b"] - p["edge"] for p in place])
allafter = [rr - p["R0"] for p in place for q, rr in p["pts"] if q > p["b"]]
print(f"  edge - a: mean {ea.mean():+.1f}, sd {ea.std(ddof=1):.1f}; b - edge: mean {eb.mean():.1f}, sd {eb.std(ddof=1):.1f}; "
      f"gain after b over {len(allafter)} placements: mean {np.mean(allafter):+.2f}, max {np.max(allafter):+.2f} lines")
# rule 'edge = a + mean offset', leave-one-out, against the pre-registered rules
errs = []
for i, p in enumerate(place):
    off = np.mean([q["edge"] - q["a"] for j, q in enumerate(place) if j != i])
    errs.append(abs(p["a"] + off - p["edge"]))
print(f"  rule 'edge = a + offset' (offset fitted leave-one-out): MAE {np.mean(errs):.2f} layers; inside bracket +-1: "
      f"{sum(p['bracket'][0] - 1 <= p['a'] + np.mean([q['edge'] - q['a'] for j, q in enumerate(place) if j != i]) <= p['bracket'][1] + 1 for i, p in enumerate(place))}/{len(place)}")
for rule in ("R1", "R2", "R3", "R4"):
    e = [v["errors"][rule] for v in pe.values() if rule in v["errors"]]
    ins = [v["inside"][rule] for v in pe.values() if rule in v["inside"]]
    print(f"  pre-registered {rule}: MAE {np.mean(e):.2f} layers, inside {sum(ins)}/{len(ins)}")

# ------------------------------------------------------------------ 5. Ouro: one window per loop
L = 24
ouro = {}
for f, key in [("e8_ouro_o14_ptr_d2_T4.json", "q|2|rhs"), ("e8_ouro_o14_ptr_d3_T4_l3.json", "q|3|rhs"),
               ("e8_ouro_o14_ptr_d3_T4_l2.json", "q|2|rhs")]:
    r = json.load(open(f"{RES}/{f}"))
    own = np.array(r["agg"][key])
    per = []
    for t in range(4):
        seg = own[t * L:(t + 1) * L + 1]
        d = -np.diff(seg)
        lay = [i for i, x in enumerate(d) if x > 0.03]
        per.append(dict(loop=t + 1, drop=float(seg[0] - seg[-1]), layers=lay))
    for t in (2, 3):
        lo_, hi_ = band80(own[t * L:(t + 1) * L + 1])
        per[t]["band80"] = (lo_, hi_)
    ouro[f] = dict(own=own.tolist(), per_loop=per)
print("\nOuro-1.4B (24 layers x 4 loops): share of a pointer's effect that leaves its token in each loop, and the layers where it"
      " leaves")
for f, v in ouro.items():
    print(f"  {f:34s} " + "; ".join(f"loop {p['loop']} {p['drop']:.2f} @{p['layers']}" for p in v["per_loop"])
          + "; central-80% band " + ", ".join(f"loop {p['loop']} {p['band80'][0]:.1f}-{p['band80'][1]:.1f} "
                                             f"({p['band80'][1] - p['band80'][0]:.1f})" for p in v["per_loop"][2:]))
ouro_w80 = [p["band80"][1] - p["band80"][0] for v in ouro.values() for p in v["per_loop"][2:]]
print(f"  Ouro-1.4B central-80% band per loop (loops 3-4, three traces): mean {np.mean(ouro_w80):.1f}, range "
      f"{min(ouro_w80):.1f}-{max(ouro_w80):.1f} layers")
e6 = {}
for tag, f in [("Ouro-1.4B", "e6_ouro_ouro14_nl_n300.json"), ("Ouro-2.6B", "e6_ouro_ouro26_nl.json")]:
    cells = json.load(open(f"{RES}/{f}"))["cells"]
    tab = {}
    for c in cells:
        tab.setdefault(c["T"], {})[c["depth"]] = c["racc"]
    e6[tag] = {T: reach(tab[T]) for T in sorted(tab)}
    print(f"  {tag} reach by loops: " + ", ".join(f"T{T} {e6[tag][T]:.2f}" for T in sorted(e6[tag])))

# ------------------------------------------------------------------ 6. robustness of the window width
print("\nRobustness: window width D = b - a under other thresholds (a: queried pointer's own share < a_thr; b: final token holds"
      " >= b_thr of both pointers)")
robust = []
for a_thr in (0.9, 0.75, 0.5):
    for b_thr in (0.5, 0.75, 0.9):
        Ds = []
        for name, tag, rtag, params, fam in PANEL:
            r = json.load(open(f"{RES}/e10_panel_{tag}.json"))
            p3, p2 = r["ptr"]["3"]["agg"], r["ptr"]["2"]["agg"]
            Ds.append(max(first_above(p3["query|last"], b_thr), first_above(p2["query|last"], b_thr))
                      - first_below(p3["q|3|rhs"], a_thr))
        Ds = np.array(Ds, float)
        sl = np.polyfit(N_, Ds, 1)[0]
        robust.append(dict(a_thr=a_thr, b_thr=b_thr, mean=float(Ds.mean()), sd=float(Ds.std(ddof=1)), slope=float(sl),
                           prop=float(Ds.mean() / N_.mean())))
        print(f"  a<{a_thr:.2f} b>={b_thr:.2f}: mean {Ds.mean():4.1f} sd {Ds.std(ddof=1):3.1f}  slope {sl:+.3f}  "
              f"(proportional {Ds.mean() / N_.mean():.3f})")
print("Replicate on two-chain traces (e10_panel_c2, 48 pairs) vs three-chain traces (24 pairs):")
replicate = []
for name, tag in [("Qwen3-0.6B", "q06base"), ("Qwen3-1.7B", "q17base"), ("Llama-3.1-8B", "llama31_8b"),
                  ("OLMo-3-7B", "olmo3_7b"), ("Qwen3-8B", "q8base")]:
    ab = []
    for pre in ("e10_panel_", "e10_panel_c2_"):
        r = json.load(open(f"{RES}/{pre}{tag}.json"))
        p3, p2 = r["ptr"]["3"]["agg"], r["ptr"]["2"]["agg"]
        a_ = first_below(p3["q|3|rhs"], 0.75)
        b_ = max(first_above(p3["query|last"], 0.9), first_above(p2["query|last"], 0.9))
        ab.append((a_, b_))
    replicate.append(dict(model=name, three=ab[0], two=ab[1]))
    print(f"  {name:13s} three chains [{ab[0][0]}, {ab[0][1]}] D={ab[0][1] - ab[0][0]}   two chains [{ab[1][0]}, {ab[1][1]}] "
          f"D={ab[1][1] - ab[1][0]}")

# ------------------------------------------------------------------ 7. capacity form and implied read time per model
print("\nCapacity form: d_max = 3 + D/tau (three lines labelled before the window in all 13 models), reach R = d_max - delta")
for x in rows:
    S = x["S"]
    ds = sorted(S)
    dm = float(ds[-1])
    for d0, d1 in zip(ds, ds[1:]):                       # last crossing of S = 0.1 (chance-corrected success 10%)
        if S[d0] >= 0.1 > S[d1]:
            dm = d0 + (S[d0] - 0.1) / (S[d0] - S[d1])
    x["dmax"] = dm
q8 = next(x for x in rows if x["name"] == "Qwen3-8B")
DELTA = 3 + q8["D"] / TAU - q8["R"]                      # calibrated on Qwen3-8B only
for x in rows:
    x["tau_implied"] = x["D"] / (x["R"] - 3 + DELTA)
print(f"  delta (Qwen3-8B: 3 + {q8['D']:.0f}/{TAU} - {q8['R']:.2f}) = {DELTA:.2f} lines; measured d_max - R per model: "
      + ", ".join(f"{x['name']} {x['dmax'] - x['R']:.1f}" for x in rows))
print("  read time tau each model would need for the model to hold (layers per read): "
      + ", ".join(f"{x['name']} {x['tau_implied']:.1f}" for x in rows))
print(f"  Spearman(tau_implied, log params) = {spearman([x['tau_implied'] for x in rows], logP):+.2f}")

# ------------------------------------------------------------------ 8. cross-task routing (E72, in progress when read)
print("\nOther tasks (e72_clock_*.json, runs in progress): layer at which the final token holds half of a routing / content"
      " counterfactual's effect, against the pointer window end b")
e72 = {}
for tag, name in [("q8", "Qwen3-8B"), ("l31", "Llama-3.1-8B"), ("o3", "OLMo-3-7B")]:
    res = {}
    for suf in ("", "_b"):
        p = f"{RES}/e72_clock_{tag}{suf}.json"
        if os.path.exists(p):
            res.update(json.load(open(p))["res"])
    x = next(r_ for r_ in rows if r_["name"] == name)
    e72[name] = {k: v["final_above_half"] for k, v in res.items()}
    rt = {k.replace("_routing", ""): v["final_above_half"] for k, v in res.items() if k.endswith("routing")}
    ct = {k.replace("_content", ""): v["final_above_half"] for k, v in res.items() if k.endswith("content")}
    print(f"  {name:13s} b={x['b']:.0f} H={x['H']:.0f}  routing: " + ", ".join(f"{k} {v}" for k, v in rt.items())
          + "   content: " + ", ".join(f"{k} {v}" for k, v in ct.items()))

# ------------------------------------------------------------------ 9. predictions for runs whose results were not yet read
print("\nPredictions (recorded before any evaluation result of these runs existed in results/):")
off = float(np.mean([p["edge"] - p["a"] for p in place]))
offb = float(np.mean([p["b"] - p["edge"] for p in place]))
for name, tested in [("OLMo-3-32B", [14, 18, 22, 26, 30, 34]), ("Gemma-3-12B", [14, 18, 21, 24, 27, 30, 34])]:
    x = next(r_ for r_ in rows if r_["name"] == name)
    e_a, e_b = x["a"] + off, x["b"] - offb
    lab = {p: ("unlock" if p < min(e_a, e_b) - 1 else "after window: gain <= 0.6 lines" if p > x["b"] else
               "near edge" if p <= max(e_a, e_b) + 1 else "inside window: walk-only gain, falls toward 0 at b")
           for p in tested}
    print(f"  {name}: window [{x['a']:.0f}, {x['b']:.0f}], edge predicted {e_a:.1f} (a + {off:.1f}) or {e_b:.1f} (b - {offb:.1f}); "
          + "; ".join(f"L{p}: {v}" for p, v in lab.items()))
sc = next(m for m in models if m["name"].startswith("scale"))["coef"]
dp = next(m for m in models if m["name"].startswith("depth"))["coef"]
print(f"  Qwen3-32B-Base (64 layers, width 5120 like Qwen3-14B, 32.8B): scale fit {sc[0] + sc[1] * np.log10(32.8):.2f} lines, "
      f"depth fit {dp[0] + dp[1] * 64:.2f}; window model: window 6-9 layers wide, reach within 0.5 of Qwen3-14B (3.55)")

# ------------------------------------------------------------------ 10. save table
json.dump(dict(tau=TAU, delta=DELTA, rows=rows, models=models, placement=place, robustness=robust, replicate=replicate,
               ouro_ptr=ouro, ouro_reach=e6, e72=e72),
          open(f"{OUT}/routing_window_table.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ 11. figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from figstyle import C, setup  # noqa: E402

setup()
# categorical hues in the fixed order of figstyle.CAT: blue, orange, aqua, yellow, magenta, green
FAMC = {"Qwen": C["blue"], "Llama": C["orange"], "Gemma": C["aqua"], "OLMo": C["yellow"]}
OURO = {"Ouro-1.4B": C["magenta"], "Ouro-2.6B": C["green"]}
fig, axs = plt.subplots(1, 4, figsize=(16, 4.1), gridspec_kw=dict(width_ratios=[1.3, 0.95, 1.05, 0.95], wspace=0.36))

# (a) window, value copy, relay and map edge in absolute layers
ax = axs[0]
edge_of = {p["model"]: p["edge"] for p in place}
order = sorted(range(len(rows)), key=lambda i: (rows[i]["N"], rows[i]["R"]))
for y, i in enumerate(order):
    x = rows[i]
    ax.plot([0, x["N"]], [y, y], color=C["grid"], lw=6, solid_capstyle="butt", zorder=1)
    ax.plot([x["a"], x["b"]], [y, y], color=FAMC[x["fam"]], lw=6, solid_capstyle="butt", zorder=2)
    ax.plot([x["H"]], [y], marker="|", ms=12, mew=2.2, color=C["ink"], zorder=3)
    ax.plot([x["r3"]], [y], marker="o", ms=4.2, color=C["ink2"], mfc="white", mew=1.1, zorder=3)
    if x["name"] in edge_of:
        ax.plot([edge_of[x["name"]]], [y + 0.40], marker="v", ms=4.5, color=C["ink"], zorder=4)
    ax.text(x["N"] + 1.2, y, f"{x['name']}  {x['R']:.1f}", va="center", fontsize=6.9, color=C["ink2"])
ax.set_yticks([])
ax.set_ylim(-0.8, len(rows) - 0.1)
ax.set_xlim(0, 100)
ax.set_xlabel("layer")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], color=C["ink2"], lw=6, label="walk window [a, b]"),
                   Line2D([], [], color=C["ink"], marker="|", ls="", ms=10, mew=2, label="value copied to output"),
                   Line2D([], [], color=C["ink2"], marker="o", mfc="white", ls="", ms=4, label="relay labels line 3"),
                   Line2D([], [], color=C["ink"], marker="v", ls="", ms=5, label="map edge (7 models)")],
          loc="lower right", fontsize=6.6, handlelength=1.6, framealpha=1, facecolor="white", edgecolor="white")
ax.set_title("(a) where the pointer walk happens (reach in lines)", fontsize=8.4, loc="left")

# (b) window width against depth
ax = axs[1]
for x in rows:
    ax.scatter(x["N"], x["D"], s=26, facecolor="white", edgecolor=FAMC[x["fam"]], linewidth=1.1, zorder=3)
    ax.scatter(x["N"], x["W80"], s=30, color=FAMC[x["fam"]], edgecolor="white", linewidth=0.6, zorder=4)
ax.errorbar([24], [np.mean(ouro_w80)], yerr=[[np.mean(ouro_w80) - min(ouro_w80)], [max(ouro_w80) - np.mean(ouro_w80)]],
            fmt="D", ms=5.5, color=OURO["Ouro-1.4B"], mec="white", mew=0.6, elinewidth=1, capsize=0, zorder=4)
ax.annotate("Ouro-1.4B, per loop", (24, np.mean(ouro_w80)), (25.5, 1.5), fontsize=6.8, color=C["ink2"],
            arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.6))
xs = np.linspace(12, 66, 10)
ax.plot(xs, W80.mean() * xs / N_.mean(), ls="--", color=C["muted"], lw=1)
ax.text(65, W80.mean() * 65 / N_.mean() - 1.1, "if width ∝ depth", fontsize=6.8, color=C["muted"], ha="right")
ax.axhline(W80.mean(), color=C["ink2"], lw=0.8)
ax.text(13, W80.mean() + 0.35, f"mean {W80.mean():.1f} layers (sd {W80.std(ddof=1):.1f})", fontsize=6.8, color=C["ink2"])
ax.scatter([], [], s=26, facecolor="white", edgecolor=C["ink2"], linewidth=1.1, label="b − a")
ax.scatter([], [], s=30, color=C["ink2"], label="central 80% of departure")
ax.set_xlabel("layers N")
ax.set_ylabel("width of the routing window (layers)")
ax.set_ylim(0, 16)
ax.set_xlim(10, 68)
for fam, col in FAMC.items():
    ax.scatter([], [], color=col, s=22, label=fam)
ax.legend(loc="upper left", fontsize=6.8, ncol=2)
ax.set_title("(b) the window does not grow with depth", fontsize=8.4, loc="left")

# (c) map reach against placement relative to the window
ax = axs[2]
fam_of = {x["name"]: x["fam"] for x in rows}
ax.axvspan(0, 1, color=C["band2"], lw=0, zorder=0)
for p in place:
    xs_ = [(q - p["a"]) / (p["b"] - p["a"]) for q, _ in p["pts"]]
    ys_ = [rr / p["R0"] for _, rr in p["pts"]]
    ax.plot(xs_, ys_, marker="o", ms=2.8, lw=1.1, color=FAMC[fam_of[p["model"]]], alpha=0.9, zorder=2)
ax.axhline(1, color=C["muted"], lw=0.8, ls="--")
ax.axvline(np.mean([(p["edge"] - p["a"]) / (p["b"] - p["a"]) for p in place]), color=C["ink2"], lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_ylim(0.85, 20)
ax.set_yticks([1, 2, 5, 10])
ax.set_yticklabels(["1", "2", "5", "10"])
ax.set_xlabel("map layer, (p − a) / (b − a)")
ax.set_ylabel("reach with map / frozen reach")
ax.text(0.62, 16.5, "window [a, b]", ha="center", fontsize=6.8, color=C["ink2"])
ax.text(1.62, 2.6, f"after b: +{np.mean(allafter):.2f} lines\n(mean of {len(allafter)} maps)", fontsize=6.6,
        color=C["ink2"], ha="center")
ax.legend(handles=[Line2D([], [], color=C["ink2"], lw=0.8, ls=":", label="mean map edge"),
                   Line2D([], [], color=C["muted"], lw=0.8, ls="--", label="frozen reach")],
          loc="lower left", fontsize=6.6, handlelength=2)
ax.set_title("(c) where a one-layer map helps (7 models)", fontsize=8.4, loc="left")

# (d) Ouro loops
ax = axs[3]
for tag in ("Ouro-1.4B", "Ouro-2.6B"):
    Ts = sorted(e6[tag])
    ax.plot(Ts, [e6[tag][T] for T in Ts], marker="o", ms=3.4, color=OURO[tag],
            label=tag + (" (24 layers/loop)" if "1.4" in tag else " (48 layers/loop)"))
ax.plot([2, 3, 4], [1.62, 2.62, 3.62], ls=":", color=C["muted"], lw=1)
ax.text(4.15, 3.55, "+1 line per loop", fontsize=6.8, color=C["muted"], ha="left")
ax.set_xlabel("loops at inference (trained with 4)")
ax.set_ylabel("reach (lines, 80% choice accuracy)")
ax.set_ylim(0, 4)
ax.set_xticks(range(1, 11))
ax.legend(loc="lower right", fontsize=6.6)
ax.set_title("(d) a loop adds a window, not layers", fontsize=8.4, loc="left")
fig.savefig(f"{OUT}/routing_window.png")
print("\nwrote", f"{OUT}/routing_window.png", "and", f"{OUT}/routing_window_table.json")
