"""Gated variants: a match is kept only if the L-inf distance between matched points is <= delta (A).
Recomputed from the saved per-frame diagrams; same windows."""
import json, glob, numpy as np
from pathlib import Path
from multiprocessing import Pool
import tracker as T
BASE_MATCH=T.match
def match_gated(A,B,delta):
    m=BASE_MATCH(A,B)
    return {i:j for i,j in m.items() if np.max(np.abs(A[i]-B[j]))<=delta}
def feats(Ds,variant,delta):
    orig=BASE_MATCH
    T.match=lambda A,B: match_gated(A,B,delta)
    try: return T.window_features(Ds,variant)
    finally: T.match=orig
def run(f):
    p,Tm,rep=Path(f).stem.split('_'); r=json.load(open(f'tracker_runs/{p}_{Tm}_{rep}.json'))
    z=np.load(f'tracker_runs/{p}_{Tm}_{rep}_diagrams.npz'); Ds=[z[f'arr_{i}'] for i in range(len(z.files))]
    rows=[]
    for w in r:
        a=w['start']; row=dict(protein=p,T=int(Tm),rep=int(rep),start=a)
        for dl in (0.25,0.5,1.0):
            for v in ('full','eps'):
                for k,val in feats(Ds[a:a+20],v,dl).items():
                    if k!='fw_beta_eps_min': row[f'{k}_{v}_g{dl}']=val
        rows.append(row)
    return rows
if __name__=='__main__':
    fs=sorted(glob.glob('tracker_runs/*_diagrams.npz')); fs=[f.replace('_diagrams.npz','.json') for f in fs]
    with Pool(2) as pool: R=[x for rr in pool.map(run,fs) for x in rr]
    import pandas as pd; pd.DataFrame(R).to_csv('window_tracker_gated.csv',index=False); print(len(R))
