import pandas as pd, numpy as np, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/zigzagv3VS/mdcath_order_added_value/baseline_test/'
d=pd.read_csv(U+'window_features.csv').merge(pd.read_csv(U+'window_zigzag_features.csv'),on=['protein','T','rep','start'])
o=d[d.occupied].copy(); o['lab']=np.where(o.Q>=0.6,1,np.where(o.Q<0.3,0,-1)); s=o[o.lab>=0].copy()
rng=np.random.default_rng(20260924)
def aucs(df,f):
    return {p:roc_auc_score(g.lab,g[f]) for p,g in df.groupby('protein') if g.lab.nunique()==2}
def boot(f1,f2,n=2000):
    doms=s.protein.unique(); diffs=[]
    for _ in range(n):
        parts=[]
        for k,p in enumerate(rng.choice(doms,len(doms))):
            g=s[s.protein==p]; runs=g[['T','rep']].drop_duplicates().values
            pick=runs[rng.integers(len(runs),size=len(runs))]
            parts.append(pd.concat([g[(g['T']==t)&(g.rep==r)] for t,r in pick]).assign(protein=f'{p}#{k}'))
        b=pd.concat(parts);a1=aucs(b,f1);a2=aucs(b,f2);ks=[k for k in a1 if k in a2]
        if ks: diffs.append(np.median([a1[k]-a2[k] for k in ks]))
    x=np.array(diffs); return np.median(x),np.percentile(x,[2.5,97.5]),(x<=0).mean()
full={f:aucs(s,f) for f in ['zz_step_survival','zz_mean_overlap','zz_spanning','beta_min']}
for f in ['zz_step_survival','zz_mean_overlap','zz_spanning']:
    wins=sum(full[f][p]>full['beta_min'][p] for p in full[f]); ties=sum(full[f][p]==full['beta_min'][p] for p in full[f])
    m,ci,pl=boot(f,'beta_min'); print(f'{f} vs beta_min: wins {wins}/13 ties {ties}; boot median diff {m:.3f} 95% [{ci[0]:.3f},{ci[1]:.3f}] P<=0 {pl:.3f}')
for p in ['2ivxB01','3hwcA01','2np9A01']: print(p,{f:round(full[f][p],2) for f in full})
