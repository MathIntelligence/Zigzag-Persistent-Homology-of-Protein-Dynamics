"""Continuity null models on 20-frame windows (validation domains, eps 6 A, debounce 3).
observed: zigzag of the window's own debounced contact complexes (equals restriction of the full-run barcode).
frame-order null: frames of the window permuted (every framewise complex, hence framewise PH and occupancy, unchanged).
contact-timing null: each contact's presence series circularly shifted within the window by an independent random
offset (each contact keeps its number of frames present and, up to wrap-around, its dwell times; coordination between
contacts is destroyed)."""
import json,glob,numpy as np,sys,csv
sys.path.insert(0,'/home/claude/r2'); from zzlib import *
rng=np.random.default_rng(20260930); K=10; L=20
rows=[]
import sys as _s
PART=int(_s.argv[1]);FILES=sorted(glob.glob('data/global/*/eps6_deb3/*.json'))[PART::2]
rng=np.random.default_rng(20260930+PART)
for f in FILES:
    r=json.load(open(f)); p=r['protein'];T=r['temperature'];rep=r['replica']
    if T==413 or (T!=320 and not r['qualifying']): continue
    import os
    tag=f'{p}_{T}_{rep}'
    if os.path.exists(f'nullout/{tag}.csv'): continue
    rows=[]
    X=np.load(f'data/validation/input/{p}_{T}_{rep}_ca.npy').astype(float); N=X.shape[1]
    pairs,pres=presence(X)
    for w in r['windows']:
        a=w['start'];P=pres[:,a:a+L]
        def stats(Pw):
            lo,hi,beta=bars_from_presence(pairs,Pw,N,L)
            ww=windows(lo,hi,beta,L)[0]; return ww
        ob=stats(P); assert ob['spanning']==w['spanning'] and ob['occupied']==w['occupancy_continuous']
        fo=[stats(P[:,rng.permutation(L)]) for _ in range(K)]
        sh=rng.integers(0,L,size=(K,P.shape[0]))
        idx=(np.arange(L)[None,:]-sh[:,:,None])%L
        ct=[stats(np.take_along_axis(P,idx[k],axis=1)) for k in range(K)]
        rows.append(dict(protein=p,T=T,rep=rep,start=a,Q=w['Q_mean'],occupied=ob['occupied'],
          obs_span=ob['spanning'],obs_overlap=ob['mean_overlap'],obs_surv=ob['step_survival'],
          fo_span=np.mean([x['spanning'] for x in fo]),fo_cont=np.mean([x['spanning']>0 for x in fo]),
          fo_surv=np.mean([x['step_survival'] for x in fo]),fo_overlap=np.mean([x['mean_overlap'] for x in fo]),
          ct_span=np.mean([x['spanning'] for x in ct]),ct_cont=np.mean([x['spanning']>0 for x in ct]),
          ct_occ=np.mean([x['occupied'] for x in ct]),ct_hidden=np.mean([x['occupied'] and x['spanning']==0 for x in ct]),
          ct_surv=np.mean([x['step_survival'] for x in ct]),ct_overlap=np.mean([x['mean_overlap'] for x in ct])))
    pd_rows=rows
    with open(f'nullout/{tag}.csv','w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(p,T,rep,flush=True)
print('done')
