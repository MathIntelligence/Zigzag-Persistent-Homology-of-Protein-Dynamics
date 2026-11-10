"""Confirmatory analysis exactly as PROTOCOL.md. Usage: python analyze.py OUT_DIR EXPLORATORY_MERGED_CSV EXPLORATORY_GATED_CSV"""
import sys, glob, json, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
OUT, EXM, EXG = sys.argv[1:4]
rows = [w for f in glob.glob(OUT + '/*.json') for w in json.load(open(f))['windows']]
d = pd.DataFrame(rows).reset_index(drop=True)
lines = []
def P(*a): s = ' '.join(map(str, a)); print(s, flush=True); lines.append(s)

BETA = ['beta_min', 'beta_mean', 'beta_max', 'beta_sd', 'beta_ac1', 'beta_absdiff']
ZZ = ['zz_spanning', 'zz_retained_frac', 'zz_max_overlap', 'zz_mean_overlap', 'zz_mean_lifetime', 'zz_step_survival',
      'zz_turnover', 'zz_classes']
TRK = sorted(c for c in d.columns if c.startswith('trk_'))
assert len(TRK) == 40, len(TRK)

def zscore(s, cols):
    return s.groupby('protein')[cols].transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)).fillna(0).values
def lodo_pred(s, y, cols):
    X = zscore(s, cols); Y = s[y].values.astype(int); Pp = s.protein.values; pred = np.zeros(len(Y))
    for p in np.unique(Pp):
        tr = Pp != p
        pred[~tr] = LogisticRegression(max_iter=5000).fit(X[tr], Y[tr]).predict_proba(X[~tr])[:, 1]
    return pred
def transfer_pred(train, s, y, cols):
    m = LogisticRegression(max_iter=5000).fit(zscore(train, cols), train[y].values.astype(int))
    return m.predict_proba(zscore(s, cols))[:, 1]
def dom_aucs(s, y, pred):
    return {p: roc_auc_score(g[y], pred[g.index]) for p, g in s.groupby('protein') if g[y].nunique() == 2}
def summ(a): v = np.array(list(a.values())); return f'median {np.median(v):.3f} worst {v.min():.2f} (n={len(v)})'
def boot(s, y, p1, p2, n=2000, seed=20261009):
    rng = np.random.default_rng(seed); s = s.assign(_a=p1, _b=p2)
    runs = {p: [(g[y].values, g._a.values, g._b.values) for _, g in gg.groupby(['T', 'rep'])] for p, gg in s.groupby('protein')}
    doms = list(runs); diffs = []
    for _ in range(n):
        dd = []
        for p in rng.choice(doms, len(doms)):
            R = runs[p]; idx = rng.integers(len(R), size=len(R)); yy = np.concatenate([R[i][0] for i in idx])
            if len(set(yy)) < 2: continue
            dd.append(roc_auc_score(yy, np.concatenate([R[i][1] for i in idx])) - roc_auc_score(yy, np.concatenate([R[i][2] for i in idx])))
        if dd: diffs.append(np.median(dd))
    x = np.array(diffs); return np.percentile(x, [2.5, 97.5]), (x <= 0).mean()
def paired(s, y, A, B, pa, pb, name):
    ks = [k for k in A if k in B]; dd = np.array([A[k] - B[k] for k in ks]); w = int((dd > 1e-12).sum()); l = int((dd < -1e-12).sum())
    ci, pl = boot(s, y, pa, pb)
    P(f'  {name}: median diff {np.median(dd):+.3f}, wins {w}/{len(ks)} (ties {len(ks)-w-l}), 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], P(<=0) {pl:.3f}')
    return np.median(dd), w, len(ks), ci, pl

nruns = d.groupby(['protein', 'T', 'rep']).ngroups
P(f'cohort: {d.protein.nunique()} domains, {nruns} qualifying runs, {len(d)} windows, {int(d.occupied.sum())} occupied')

# ---------------- PRIMARY
o = d[d.occupied].copy(); o['lab'] = np.where(o.Q >= 0.6, 1, np.where(o.Q < 0.3, 0, -1)); s = o[o.lab >= 0]
cnt = s.groupby('protein').lab.agg(lambda x: min((x == 1).sum(), (x == 0).sum()))
s = s[s.protein.isin(cnt[cnt >= 5].index)].reset_index(drop=True)
P(f'\n== PRIMARY native-like (Q>=0.6) vs unfolded (Q<0.3), occupied windows: eligible domains {s.protein.nunique()}, windows {len(s)}')
sets = [('beta trace', BETA), ('tracker', TRK), ('zigzag', ZZ), ('tracker + zigzag', TRK + ZZ),
        ('beta + tracker', BETA + TRK), ('beta + tracker + zigzag', BETA + TRK + ZZ)]
pr, A = {}, {}
for n, c in sets:
    pr[n] = lodo_pred(s, 'lab', c); A[n] = dom_aucs(s, 'lab', pr[n]); P(f'  LODO {n:24s} {summ(A[n])}')
if s.protein.nunique() < 8:
    P('  PRIMARY RESULT: INCONCLUSIVE (fewer than 8 eligible domains)'); success = None
else:
    md, w, k, ci, pl = paired(s, 'lab', A['tracker + zigzag'], A['tracker'], pr['tracker + zigzag'], pr['tracker'], 'PRIMARY tracker + zigzag - tracker')
    success = bool(ci[0] > 0 and w > k / 2)
    P(f'  PRIMARY RESULT: {"SUCCESS" if success else "FAIL"} (needs CI lower bound > 0 and wins > {k/2:g})')

P('\n== S1 secondary comparisons (same task)')
for a, b in [('zigzag', 'tracker'), ('beta + tracker + zigzag', 'beta + tracker'), ('tracker', 'beta trace'), ('zigzag', 'beta trace')]:
    paired(s, 'lab', A[a], A[b], pr[a], pr[b], f'{a} - {b}')

P('\n== S2 hidden replacement vs tracker (occupied windows)')
h = d[d.occupied].copy(); hr = h.zz_spanning == 0
P(f'  occupied {len(h)}, zigzag-continuous {int((~hr).sum())}, hidden replacement {int(hr.sum())} ({hr.mean():.1%})')
variants = sorted({c[len('trk_spanning_'):] for c in TRK if c.startswith('trk_spanning_')})
for v in variants:
    t = h[f'trk_spanning_{v}'] > 0
    P(f'  {v:10s} labels continuous: zz-continuous {t[~hr].mean():6.1%} | hidden replacement {t[hr].mean():6.1%} | balanced acc {(t[~hr].mean() + (~t[hr]).mean())/2:.3f}')

P('\n== S3 predicting zigzag continuity among occupied windows')
c3 = d[d.occupied].copy(); c3['cont'] = (c3.zz_spanning > 0).astype(int)
c3 = c3[c3.groupby('protein').cont.transform('nunique') == 2].reset_index(drop=True)
p3, A3 = {}, {}
for n, c in [('beta trace', BETA), ('tracker', TRK), ('beta + tracker', BETA + TRK)]:
    p3[n] = lodo_pred(c3, 'cont', c); A3[n] = dom_aucs(c3, 'cont', p3[n]); P(f'  LODO {n:24s} {summ(A3[n])}')
paired(c3, 'cont', A3['beta + tracker'], A3['beta trace'], p3['beta + tracker'], p3['beta trace'], 'beta + tracker - beta trace')
paired(c3, 'cont', A3['beta trace'], A3['tracker'], p3['beta trace'], p3['tracker'], 'beta trace - tracker')

P('\n== S4 transfer: trained on the 14 exploratory domains, tested on each eligible new domain')
ex = pd.read_csv(EXM).merge(pd.read_csv(EXG), on=['protein', 'T', 'rep', 'start'])
ex = ex[ex.occupied].copy(); ex['lab'] = np.where(ex.Q >= 0.6, 1, np.where(ex.Q < 0.3, 0, -1)); ex = ex[ex.lab >= 0].reset_index(drop=True)
pt, At = {}, {}
for n, c in [('tracker', TRK), ('zigzag', ZZ), ('tracker + zigzag', TRK + ZZ)]:
    pt[n] = transfer_pred(ex, s, 'lab', c); At[n] = dom_aucs(s, 'lab', pt[n]); P(f'  transfer {n:20s} {summ(At[n])}')
paired(s, 'lab', At['tracker + zigzag'], At['tracker'], pt['tracker + zigzag'], pt['tracker'], 'transfer tracker + zigzag - tracker')
paired(s, 'lab', At['zigzag'], At['tracker'], pt['zigzag'], pt['tracker'], 'transfer zigzag - tracker')

d.to_csv('window_features_confirm.csv', index=False)
open('analysis_output.txt', 'w').write('\n'.join(lines) + '\n')
json.dump(dict(primary_success=success), open('decision.json', 'w'))
