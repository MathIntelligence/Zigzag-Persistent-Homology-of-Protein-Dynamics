import json, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
d=pd.read_csv('window_tracker_merged.csv').merge(pd.read_csv('window_tracker_gated.csv'),on=['protein','T','rep','start'])
out=[]
def P(*a): s=' '.join(map(str,a)); print(s); out.append(s)
def dom_auc(s,y,f): return {p:roc_auc_score(g[y],g[f]) for p,g in s.groupby('protein') if g[y].nunique()==2}
def summ(v): a=np.array(list(v.values())); return f'median {np.median(a):.3f} worst {a.min():.2f}'
variants=[('full',''),('eps','')]+[(v,f'_g{g}') for g in (0.25,0.5,1.0) for v in ('full','eps')]
TRK=[c for c in d.columns if c.startswith('trk_')]
BETA=['beta_min','beta_mean','beta_max','beta_sd','beta_ac1','beta_absdiff']
ZZ=['zz_spanning','zz_retained_frac','zz_max_overlap','zz_mean_overlap','zz_mean_lifetime','zz_step_survival','zz_turnover','zz_classes']
def lodo_pred(s,y,cols):
    X=s.groupby('protein')[cols].transform(lambda x:(x-x.mean())/(x.std() if x.std()>0 else 1)).fillna(0).values
    Y=s[y].values.astype(int);Pp=s.protein.values;pred=np.zeros(len(Y))
    for p in np.unique(Pp):
        tr=Pp!=p;pred[~tr]=LogisticRegression(max_iter=5000,C=1.0).fit(X[tr],Y[tr]).predict_proba(X[~tr])[:,1]
    return pred
rng=np.random.default_rng(20260924)
def boot(s,y,p1,p2,n=2000):
    s=s.assign(_a=p1,_b=p2); doms=s.protein.unique(); G={p:s[s.protein==p] for p in doms}
    R={p:G[p][['T','rep']].drop_duplicates().values for p in doms}; diffs=[]
    for _ in range(n):
        dd=[]
        for p in rng.choice(doms,len(doms)):
            g=G[p]; pick=R[p][rng.integers(len(R[p]),size=len(R[p]))]
            b=pd.concat([g[(g['T']==a)&(g.rep==r)] for a,r in pick])
            if b[y].nunique()==2: dd.append(roc_auc_score(b[y],b._a)-roc_auc_score(b[y],b._b))
        if dd: diffs.append(np.median(dd))
    x=np.array(diffs); return np.percentile(x,[2.5,97.5]),(x<=0).mean()
def compare(s,y,title):
    P(f'\n== {title}: windows {len(s)}, domains {s.groupby("protein")[y].nunique().eq(2).sum()}')
    sets=[('beta trace',BETA),('tracker (all 16 variants x 5 feats)',TRK),('zigzag',ZZ),('tracker + zigzag',TRK+ZZ),
          ('beta + tracker',BETA+TRK),('beta + tracker + zigzag',BETA+TRK+ZZ)]
    pr={}
    for n,c in sets:
        pr[n]=lodo_pred(s,y,c); a={p:roc_auc_score(g[y],pr[n][g.index.map(s.index.get_loc)]) for p,g in s.groupby('protein') if g[y].nunique()==2}
        P(f'  LODO {n:38s} {summ(a)}')
    for a,b in [('zigzag','tracker (all 16 variants x 5 feats)'),('tracker + zigzag','tracker (all 16 variants x 5 feats)'),
                ('tracker + zigzag','zigzag'),('beta + tracker + zigzag','beta + tracker')]:
        ci,pl=boot(s,y,pr[a],pr[b]); P(f'  bootstrap {a} - {b}: 95% CI [{ci[0]:+.3f},{ci[1]:+.3f}], P(<=0)={pl:.3f}')
