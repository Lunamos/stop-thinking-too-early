"""Standard models: recompute tables, macros and the placement analysis from results/ (CPU only).

Reads results/ directly. Outputs: out/tables/{panel,edges,flanks,accuracy}.tex and out/checks/analysis_std.json (+ the
list of files read).
"""
from pathlib import Path
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CHECKS as _C, RES as _R, TABLES as _T  # noqa: E402

DATA, TABLES, CHECKS = Path(_R), Path(_T), Path(_C)
USED = set()

def read(name):
    name = name if name.endswith('.json') else name + '.json'
    USED.add(name)
    return json.loads((DATA / name).read_text())

def reach(acc, assume_one=False):
    acc = {int(k): float(v) for k, v in acc.items()}
    if assume_one:
        acc.setdefault(1, 1.)
    ds = sorted(acc)
    if acc[ds[0]] < .8:
        return 1. if assume_one else 0.
    best = ds[0]
    for d0, d1 in zip(ds, ds[1:]):
        if acc[d0] >= .8 > acc[d1]:
            return d0 + (acc[d0] - .8) / (acc[d0] - acc[d1]) * (d1 - d0)
        if acc[d1] >= .8:
            best = d1
    return float(best)

def evreach(d, k=1):
    return reach({int(x.split('_')[0][1:]): y for x, y in d['eval'].items() if x.endswith(f'_K{k}')}, True)

def wilson(p, n):
    z=1.959963984540054
    c=(p+z*z/(2*n))/(1+z*z/n)
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(c-h), float(c+h)]

PANEL = [('llama32_1b','Llama-3.2-1B'), ('q06base','Qwen3-0.6B'), ('q17base','Qwen3-1.7B'),
         ('llama32_3b','Llama-3.2-3B'), ('q4base','Qwen3-4B'), ('gemma3_4b','Gemma-3-4B'),
         ('llama31_8b','Llama-3.1-8B'), ('olmo3_7b','OLMo-3-7B'), ('q8base','Qwen3-8B'),
         ('gemma3_12b','Gemma-3-12B'), ('q14base','Qwen3-14B'), ('gemma3_27b','Gemma-3-27B'), ('olmo3_32b','OLMo-3-32B')]
MODELS = [('q8','Qwen3-8B',36,20.5,20.2,19,'development'), ('olmo','OLMo-3-7B',32,15,12.7,13,'development'),
          ('l31','Llama-3.1-8B',32,12.5,13.4,9,'development'), ('q17','Qwen3-1.7B',28,18,13.5,12,'development'),
          ('q06','Qwen3-0.6B',28,10,13.7,12,'development'), ('l1b','Llama-3.2-1B',16,8,2.8,5,'held-out'),
          ('q4','Qwen3-4B',36,22,20.6,16,'held-out'), ('g12','Gemma-3-12B',48,23,27.5,17,'held-out'),
          ('o32','OLMo-3-32B',64,19,30.4,15,'held-out')]

def tab(name, spec, header, rows):
    (TABLES / (name+'.tex')).write_text('\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule'] +
      [r' & '.join(map(str,row))+r' \\' for row in rows] + [r'\bottomrule',r'\end{tabular}'])+'\n')

def main():
    out={'panel':{},'placement':{},'headline':{}}
    for tag,name in PANEL:
        d=read('e10_panel_'+tag)
        cs=[]
        for line,v in d['ptr'].items():
            cs.append(next(i for i,x in enumerate(v['agg'][f'q|{line}|rhs']) if x<.5))
        out['panel'][name]={'N':d['N'], 'accuracy':{k:v['racc'] for k,v in d['acc'].items()},
                            'reach':reach({k:v['racc'] for k,v in d['acc'].items()}), 'commit':float(np.mean(cs))}
    for tag,name,N,commit,copy,r4,kind in MODELS:
        runs={}
        seeds={}
        for p in sorted(DATA.glob(f'e19_reentry_place_{tag}_a*.json')):
            m=re.search(r'_a(\d+)(_s(\d+))?\.json$',p.name)
            if not m: continue
            d=read(p.name)
            if 'eval' not in d: continue
            a=int(m[1]); seed=int(m[3] or 0)
            seeds.setdefault(a,{})[seed]=evreach(d)
            if seed==0: runs[a]=d
        R={a:evreach(d) for a,d in runs.items()}
        R0=float(np.median([evreach(d,0) for d in runs.values()]))
        threshold=R0+.5*(max(R.values())-R0)
        work=[a for a in sorted(R) if R[a]>=threshold]
        last=max(work); later=[a for a in sorted(R) if a>last]
        bracket=[last,min(later)] if later else None
        midpoint=float(np.mean(bracket)) if bracket else None
        preds={'commit':commit,'45%':.45*N,'copy-11.5':copy}
        # Score the original seed-0 sweep. Flank replications are reported separately.
        nonmonotone=any(R[a]<threshold and any(R[b]>=threshold for b in R if b>a) for a in R)
        rec={'N':N,'kind':kind,'reach':R,'frozen':R0,'threshold':threshold,'bracket':bracket,'midpoint':midpoint,
             'predictions':preds,'errors':{k:abs(v-midpoint) for k,v in preds.items()},
             'inside':{k:bracket[0]-1<=v<=bracket[1]+1 for k,v in preds.items()},'seeds':seeds,
             'nonmonotone_work_status':nonmonotone,'layers':sorted(R)}
        # Parametric binomial resampling of aggregate accuracies, not prompt or training bootstrap.
        rng=np.random.default_rng(0); brackets={}; vals={a:[] for a in bracket}
        for _ in range(2000):
            sample={}; base=[]
            for a,d in runs.items():
                n=d['args']['neval']
                ev={k:float(rng.binomial(n,p)/n) for k,p in d['eval'].items()}
                v={'eval':ev}; sample[a]=evreach(v); base.append(evreach(v,0))
                if a in vals: vals[a].append(sample[a])
            b=float(np.median(base)); th=b+.5*(max(sample.values())-b)
            w=max(a for a in sample if sample[a]>=th); nxt=[a for a in sample if a>w]
            key=f'{w}|{min(nxt)}' if nxt else 'unbracketed'
            brackets[key]=brackets.get(key,0)+1
        rec['sampling']={'B':2000,'bracket_frequencies':{k:v/2000 for k,v in brackets.items()},
                         'reach95':{a:np.percentile(v,[2.5,97.5]).tolist() for a,v in vals.items()}}
        out['placement'][name]=rec
    rr=list(out['placement'].values()); held=[r for r in rr if r['kind']=='held-out']
    out['placement_summary']={'heldout_mae':{k:float(np.mean([r['errors'][k] for r in held])) for k in held[0]['errors']},
      'heldout_pass':{k:sum(r['inside'][k] for r in held) for k in held[0]['errors']},
      'all_commit_mae':float(np.mean([r['errors']['commit'] for r in rr])),
      'all_48pct_mae':float(np.mean([abs(r['midpoint']-.48*r['N']) for r in rr])),
      'relative_correlation':float(np.corrcoef([r['midpoint']/r['N'] for r in rr],[r['predictions']['commit']/r['N'] for r in rr])[0,1])}
    mat=read('e33_order_r8kl_matrix')['res']; long=read('e19_reentry_long_q8_r8_kl1')
    out['headline']={'matrix24':mat['c2_forward_d24'],'matrix24_n':200,'long48':long['eval']['d48_K1'],
      'long48_frozen':long['eval']['d48_K0'],'long_n':long['args']['neval'],
      'map24_wilson95':wilson(mat['c2_forward_d24']['map']['acc'],200),
      'map48_wilson95':wilson(long['eval']['d48_K1'],long['args']['neval'])}
    tab('panel','lrrrrrrrr',r'Model & Blocks & 1 line & 2 & 3 & 4 & 5 & 6 & Reach',
        [[name,v['N']]+[f'{100*v["accuracy"][str(k)]:.1f}' for k in range(1,7)]+[f'{v["reach"]:.2f}'] for name,v in out['panel'].items()])
    tab('edges','lrrrrr',r'Model & Blocks & Edge bracket & Commit & 45\% & Copy$-11.5$',
        [[name,v['N'],f'{v["bracket"][0]}--{v["bracket"][1]}']+
         [f'{v["predictions"][k]:.1f}'+(r'$^\checkmark$' if v['inside'][k] else '') for k in ['commit','45%','copy-11.5']]
          for name,v in out['placement'].items()])
    tab('flanks','lrrrrrr',r'Model & Before / after & Seed 0 (before) & Seed 0 (after) & Seed 1 (before) & Seed 1 (after) & Bracket stability',
        [[name,f'{v["bracket"][0]} / {v["bracket"][1]}']+
         [f'{v["seeds"][a][s]:.2f}' if s in v['seeds'][a] else '--' for s in [0,1] for a in v['bracket']]+
         [f'{100*v["sampling"]["bracket_frequencies"].get("|".join(map(str,v["bracket"])),0):.1f}\\%']
         for name,v in out['placement'].items()])
    rows=[]
    for cond,label in [('c2_forward','2, level'),('c2_interleave','2, interleaved'),('c3_forward','3, level')]:
        for method in ['frozen','map']:
            rows.append([label,method]+[f'{100*mat[f"{cond}_d{d}"][method]["acc"]:.1f}' if f'{cond}_d{d}' in mat else '--' for d in [4,8,12,16,20,24]])
    tab('accuracy','llrrrrrr',r'Chains, order & Model & 4 lines & 8 & 12 & 16 & 20 & 24',rows)
    (CHECKS/'analysis_std.json').write_text(json.dumps(out,indent=2))
    (CHECKS/'sources_std_analysis.json').write_text(json.dumps(sorted(USED),indent=2))
    print(json.dumps({'headline':out['headline'],'placement':out['placement_summary'],
      'sweeps':{k:{x:v[x] for x in ['layers','bracket','nonmonotone_work_status']} for k,v in out['placement'].items()}},indent=2))

if __name__=='__main__': main()
