import csv,json,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'data/baseline_test'
key=lambda r:tuple(r[k] for k in ['protein','T','rep','start'])
z={key(r):r for r in csv.DictReader(open(p/'window_zigzag_features.csv'))};d=[]
for r in csv.DictReader(open(p/'window_features.csv')):
 if r['occupied'].lower() in ('true','1') and (float(r['Q'])>=.6 or float(r['Q'])<.3):d.append(dict(r,**{k:v for k,v in z[key(r)].items() if k not in r}))
y=np.array([float(r['Q'])>=.6 for r in d],int);P=np.array([r['protein'] for r in d]);Z=['zz_spanning','zz_retained_frac','zz_max_overlap','zz_mean_overlap','zz_mean_lifetime','zz_step_survival'];B=['beta_min','beta_mean','beta_max','beta_sd','beta_ac1','beta_absdiff'];res={}
for n,c in [('zigzag_six',Z),('framewise_six',B),('both',Z+B)]:
 X=np.array([[float(r[f]) for f in c] for r in d]);pred=np.zeros(len(y))
 for p0 in np.unique(P):
  ix=P==p0;sd=np.nanstd(X[ix],axis=0,ddof=1);sd[sd==0]=1;X[ix]=(X[ix]-np.nanmean(X[ix],axis=0))/sd
 X=np.nan_to_num(X)
 for p0 in np.unique(P):
  tr=P!=p0;pred[~tr]=LogisticRegression(max_iter=3000).fit(X[tr],y[tr]).predict_proba(X[~tr])[:,1]
 v=[roc_auc_score(y[P==p0],pred[P==p0]) for p0 in np.unique(P) if len(set(y[P==p0]))==2];res[n]={'median':float(np.median(v)),'min':float(min(v)),'domains':len(v)}
print(json.dumps(res,indent=2));out=Path(__file__).resolve().parents[1]/'reproduced';out.mkdir(exist_ok=True);(out/'combined_check.json').write_text(json.dumps(res,indent=2));print('windows',len(d),'domains',len(set(P)))
