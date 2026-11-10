import json, glob, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, binomtest
warnings.filterwarnings('ignore')
B = 'mdcath_order_added_value/'
trk = pd.DataFrame([r for f in glob.glob('tracker_runs/*.json') for r in json.load(open(f))])
d = pd.read_csv(B+'baseline_test/window_features.csv').merge(pd.read_csv(B+'baseline_test/window_zigzag_features.csv'),
                                                             on=['protein', 'T', 'rep', 'start'])
d = d.merge(trk, on=['protein', 'T', 'rep', 'start'], how='inner')
# raw (no debounce) zigzag windows
raw = []
for f in glob.glob(B+'global/*/eps6_deb0/*.json'):
    p = f.split('/')[-3]; T, rep = map(int, Path(f).stem.split('_'))
    if T == 320: continue
    r = json.load(open(f))
    if not r['qualifying']: continue
    for w in r['windows']:
        raw.append(dict(protein=p, T=T, rep=rep, start=w['start'], raw_occupied=w['occupancy_continuous'],
                        raw_spanning=w['spanning'], raw_minbeta=w['min_beta']))
d = d.merge(pd.DataFrame(raw), on=['protein', 'T', 'rep', 'start'], how='left')
print('windows', len(d), 'domains', d.protein.nunique(), 'with raw zigzag', d.raw_spanning.notna().sum())
print('sanity: framewise min beta at eps (Rips) == raw zigzag min beta:', (d.fw_beta_eps_min == d.raw_minbeta).mean().round(3))

def dom_auc(s, y, f, sg=1):
    v = {p: roc_auc_score(g[y], sg*g[f]) for p, g in s.groupby('protein') if g[y].nunique() == 2}
    return v
def summ(v): a = np.array(list(v.values())); return f'median {np.median(a):.3f} worst {a.min():.2f} (n={len(a)})'
def lodo(s, y, cols):
    X = s.groupby('protein')[cols].transform(lambda x: (x-x.mean())/(x.std() if x.std() > 0 else 1)).fillna(0).values
    Y = s[y].values.astype(int); P = s.protein.values; pred = np.zeros(len(Y))
    for p in np.unique(P):
        tr = P != p
        if len(set(Y[tr])) < 2: continue
        pred[~tr] = LogisticRegression(max_iter=5000).fit(X[tr], Y[tr]).predict_proba(X[~tr])[:, 1]
    return {p: roc_auc_score(Y[P == p], pred[P == p]) for p in np.unique(P) if len(set(Y[P == p])) == 2}

TRK = {v: [f'trk_spanning_{v}', f'trk_step_survival_{v}', f'trk_mean_overlap_{v}', f'trk_mean_life_{v}', f'trk_max_overlap_{v}']
       for v in ('full', 'eps')}
TRK_ALL = TRK['full'] + TRK['eps']
BETA = ['beta_min', 'beta_mean', 'beta_max', 'beta_sd', 'beta_ac1', 'beta_absdiff']
ZZ = ['zz_spanning', 'zz_retained_frac', 'zz_max_overlap', 'zz_mean_overlap', 'zz_mean_lifetime', 'zz_step_survival',
      'zz_turnover', 'zz_classes']

out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# ---------------------------------------------------------------- TEST 1: does the tracker recover continuity?
for tag, occ, span in [('DEBOUNCED zigzag (paper setting, eps6 deb3)', 'occupied', 'spanning'),
                       ('RAW zigzag (eps6 deb0; same input as the framewise diagrams)', 'raw_occupied', 'raw_spanning')]:
    s = d[d[occ] == True].copy(); s['cont'] = (s[span] > 0).astype(int)
    P(f'\n== TEST 1 [{tag}] occupied windows {len(s)}, continuous {s.cont.sum()}, hidden replacement {(1-s.cont).sum()}')
    for v in ('full', 'eps'):
        t = s[f'trk_spanning_{v}'] > 0
        hr = s.cont == 0
        P(f'  tracker[{v}] says continuous: in zigzag-continuous {t[~hr].mean():.1%} | in hidden-replacement {t[hr].mean():.1%}'
          f' | agreement {(t == (s.cont == 1)).mean():.1%} | Spearman(trk spanning, zz spanning) '
          f'{spearmanr(s[f"trk_spanning_{v}"], s[span]).correlation:.2f}')
        P(f'  hidden-replacement windows the tracker labels hidden replacement: {(~t[hr]).mean():.1%}')
    for f in TRK_ALL + ['beta_min']:
        P(f'  AUC predicting continuity  {f:26s} {summ(dom_auc(s, "cont", f))}')
    for n, c in [('beta trace', BETA), ('tracker', TRK_ALL), ('beta trace + tracker', BETA+TRK_ALL)]:
        P(f'  LODO predicting continuity {n:26s} {summ(lodo(s, "cont", c))}')

# ---------------------------------------------------------------- TEST 2: native-like vs unfolded (paper table task)
o = d[d.occupied].copy(); o['lab'] = np.where(o.Q >= 0.6, 1, np.where(o.Q < 0.3, 0, -1)); s = o[o.lab >= 0].copy()
P(f'\n== TEST 2 native-like (Q>=.6) vs unfolded (Q<.3), occupied windows {len(s)}')
A = {}
for f in ['beta_min'] + TRK_ALL + ['zz_spanning', 'zz_mean_overlap', 'zz_step_survival']:
    A[f] = dom_auc(s, 'lab', f); P(f'  {f:26s} {summ(A[f])}')
L = {}
for n, c in [('framewise beta trace', BETA), ('tracker', TRK_ALL), ('zigzag', ZZ), ('tracker + zigzag', TRK_ALL+ZZ),
             ('beta + tracker', BETA+TRK_ALL), ('beta + tracker + zigzag', BETA+TRK_ALL+ZZ)]:
    L[n] = lodo(s, 'lab', c); P(f'  LODO {n:26s} {summ(L[n])}')

def paired(a, b, name):
    ks = [k for k in a if k in b]; dd = np.array([a[k]-b[k] for k in ks]); w = (dd > 1e-9).sum(); l = (dd < -1e-9).sum()
    p = binomtest(w, w+l, 0.5, alternative='greater').pvalue if w+l else 1
    P(f'  {name:50s} median diff {np.median(dd):+.3f}, wins {w}/{len(ks)} (ties {len(ks)-w-l}), sign p={p:.3g}')
P('  -- paired, within-domain')
best_trk = max(TRK_ALL, key=lambda f: np.median(list(A[f].values())))
P(f'  best single tracker feature: {best_trk}')
paired(A['zz_mean_overlap'], A[best_trk], 'zz mean overlap vs best tracker feature')
paired(A[best_trk], A['beta_min'], 'best tracker feature vs beta_min')
paired(L['zigzag'], L['tracker'], 'LODO zigzag vs tracker')
paired(L['tracker + zigzag'], L['tracker'], 'LODO tracker+zigzag vs tracker')
paired(L['tracker + zigzag'], L['zigzag'], 'LODO tracker+zigzag vs zigzag')
paired(L['beta + tracker + zigzag'], L['beta + tracker'], 'LODO beta+tracker+zigzag vs beta+tracker')

# ---------------------------------------------------------------- TEST 3: compact windows (harder task)
q = o[o.Q >= 0.6].groupby('protein').rg.quantile(.95); c = o[o.rg <= o.protein.map(q)].copy()
c['lab'] = np.where(c.Q >= 0.6, 1, np.where(c.Q < 0.45, 0, -1)); c = c[c.lab >= 0]
c = c[c.groupby('protein').lab.transform(lambda x: min((x == 1).sum(), (x == 0).sum())) >= 5]
P(f'\n== TEST 3 compact windows, Q>=.6 vs Q<.45, windows {len(c)}, domains {c.protein.nunique()} (exploratory cohort)')
A3 = {}
for f in ['beta_min'] + TRK_ALL + ['zz_mean_overlap', 'zz_step_survival']:
    A3[f] = dom_auc(c, 'lab', f); P(f'  {f:26s} {summ(A3[f])}')
L3 = {}
for n, cc in [('framewise beta trace', BETA), ('tracker', TRK_ALL), ('zigzag', ZZ), ('tracker + zigzag', TRK_ALL+ZZ),
              ('beta + tracker + zigzag', BETA+TRK_ALL+ZZ)]:
    L3[n] = lodo(c, 'lab', cc); P(f'  LODO {n:26s} {summ(L3[n])}')
paired(L3['tracker + zigzag'], L3['tracker'], 'LODO tracker+zigzag vs tracker')
paired(L3['zigzag'], L3['tracker'], 'LODO zigzag vs tracker')

d.to_csv('window_tracker_merged.csv', index=False)
Path('analysis_output.txt').write_text('\n'.join(out))
