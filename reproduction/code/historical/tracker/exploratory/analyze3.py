import warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
import analyze2_lib as L
warnings.filterwarnings('ignore')
from analyze2_lib import *
def fast_boot(s,y,p1,p2,n=2000,seed=20260924):
    rng=np.random.default_rng(seed); s=s.assign(_a=p1,_b=p2)
    runs={p:[(g[y].values,g._a.values,g._b.values) for _,g in gg.groupby(['T','rep'])] for p,gg in s.groupby('protein')}
    doms=list(runs); diffs=[]
    for _ in range(n):
        dd=[]
        for p in rng.choice(doms,len(doms)):
            R=runs[p]; idx=rng.integers(len(R),size=len(R))
            yy=np.concatenate([R[i][0] for i in idx])
            if len(set(yy))<2: continue
            a=np.concatenate([R[i][1] for i in idx]); b=np.concatenate([R[i][2] for i in idx])
            dd.append(roc_auc_score(yy,a)-roc_auc_score(yy,b))
        if dd: diffs.append(np.median(dd))
    x=np.array(diffs); return np.percentile(x,[2.5,97.5]),(x<=0).mean()
def run(s,y,title,sets,pairs):
    P(f'\n== {title}: windows {len(s)}, domains {s.groupby("protein")[y].nunique().eq(2).sum()}')
    pr={};A={}
    for n,c in sets:
        pr[n]=lodo_pred(s,y,c); A[n]={p:roc_auc_score(g[y],pr[n][g.index]) for p,g in s.groupby('protein') if g[y].nunique()==2}
        P(f'  LODO {n:24s} {summ(A[n])}')
    for a,b in pairs:
        ks=[k for k in A[a] if k in A[b]]; dd=np.array([A[a][k]-A[b][k] for k in ks]); w=(dd>1e-9).sum(); l=(dd<-1e-9).sum()
        ci,pl=fast_boot(s,y,pr[a],pr[b])
        P(f'  {a} - {b}: median {np.median(dd):+.3f}, wins {w}/{len(ks)} (ties {len(ks)-w-l}), bootstrap 95% CI [{ci[0]:+.3f},{ci[1]:+.3f}], P(<=0)={pl:.3f}')
    return pr
sets=[('beta trace',BETA),('tracker',TRK),('zigzag',ZZ),('tracker + zigzag',TRK+ZZ),('beta + tracker',BETA+TRK),('beta + tracker + zigzag',BETA+TRK+ZZ)]
pairs=[('zigzag','tracker'),('tracker + zigzag','tracker'),('tracker + zigzag','zigzag'),('beta + tracker + zigzag','beta + tracker'),('tracker','beta trace')]
o=d[d.occupied].copy().reset_index(drop=True); o['lab']=np.where(o.Q>=0.6,1,np.where(o.Q<0.3,0,-1)); s=o[o.lab>=0].reset_index(drop=True)
run(s,'lab','TEST 2b native-like vs unfolded (Q>=.6 vs <.3), all tracker variants incl. gated',sets,pairs)
A={c:dom_auc(s,'lab',c) for c in TRK}; b=sorted(A,key=lambda c:-np.median(list(A[c].values())))[:5]
P('  best single tracker features:',[(c,summ(A[c])) for c in b])
P('  zz_mean_overlap',summ(dom_auc(s,'lab','zz_mean_overlap')))
s2=o.copy(); s2['cont']=(s2.spanning>0).astype(int)
run(s2,'cont','TEST 1c predicting zigzag continuity (deb3) among occupied windows',
    [('beta trace',BETA),('tracker',TRK),('beta + tracker',BETA+TRK)],[('beta + tracker','beta trace'),('beta trace','tracker')])
s3=d[d.raw_occupied==True].copy().reset_index(drop=True); s3['cont']=(s3.raw_spanning>0).astype(int)
run(s3,'cont','TEST 1d predicting RAW zigzag continuity among raw-occupied windows',
    [('beta trace',BETA),('tracker',TRK),('beta + tracker',BETA+TRK)],[('beta + tracker','beta trace'),('beta trace','tracker')])
open('analysis3_output.txt','w').write('\n'.join(out))
