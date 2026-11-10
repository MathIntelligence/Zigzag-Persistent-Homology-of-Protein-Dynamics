"""Per-window zigzag-barcode features from the saved eps6/debounce3 whole-complex results (no recomputation)."""
import json, csv, numpy as np
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
rows=[]
for f in sorted(BASE.glob('global/*/eps6_deb3/*.json')):
    p=f.parts[-3];T,rep=map(int,f.stem.split('_'))
    if T==320: continue
    r=json.loads(f.read_text())
    if not r['qualifying']: continue
    lo=np.array(r['lo'],float);hi=np.array(r['hi'],float);beta=np.array(r['beta'])
    assert np.array_equal(beta,((lo[:,None]<=np.arange(500))&(hi[:,None]>np.arange(500))).sum(0))
    for w in r['windows']:
        a,b=w['start'],w['stop'];L=b-a
        inter=(lo<b)&(hi>a)
        ov=(np.minimum(hi,b)-np.maximum(lo,a))[inter]/L
        alive=lambda t:(lo<=t)&(hi>t)
        surv=[ (alive(t)&alive(t+1)).sum()/max(alive(t).sum(),1) for t in range(a,b-1)]
        births=int(((lo>a)&(lo<b)).sum()); deaths=int(((hi>a)&(hi<b)).sum())
        rows.append(dict(protein=p,T=T,rep=rep,start=a,
          zz_spanning=int(((lo<=a)&(hi>=b)).sum()),
          zz_retained_frac=((lo<=a)&(hi>=b)).sum()/max(alive(a).sum(),1),
          zz_turnover=(births+deaths)/L, zz_births=births, zz_deaths=deaths,
          zz_classes=int(inter.sum()),
          zz_max_overlap=float(ov.max()) if len(ov) else 0.0,
          zz_mean_overlap=float(ov.mean()) if len(ov) else 0.0,
          zz_mean_lifetime=float((hi-lo)[inter].mean()) if inter.any() else 0.0,
          zz_step_survival=float(np.mean(surv))))
out=Path(__file__).parent/'window_zigzag_features.csv'
with open(out,'w',newline='') as fh:
    wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('WROTE',len(rows))
