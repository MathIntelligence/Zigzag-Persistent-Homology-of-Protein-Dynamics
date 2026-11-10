"""Per-window framewise and contact-persistence baselines for the zigzag continuity comparisons.
Reads the saved eps6/debounce3 whole-complex results (global/) and the same C-alpha inputs."""
import json, sys, numpy as np, csv
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(BASE))
from zigzag_dynamics.filtration import _edge_presence,_debounce_rows
from zigzag_dynamics.loopmap import native_contacts
SOURCES={p:ROOT/s for s,ps in (('mdcath_order_extension_5',['2d1pA00','2d4uB00','2dawA01']),
  ('mdcath_order_search_batch_01',['2ervA00','2fd5A02']),
  ('mdcath_order_targeted_10',['4a4aA02','2ivxB01','3hwcA01','3hbvP01'])) for p in ps}
CROSS={'2ervA00':ROOT/'mdcath_order_cross_temperature_all15','3hwcA01':ROOT/'mdcath_order_targeted_cross_temperature',
       '3hbvP01':ROOT/'mdcath_order_targeted_cross_temperature'}
VAL=BASE/'validation'
def src(p,T):
    if p in SOURCES: return (CROSS[p] if T==413 else SOURCES[p]), SOURCES[p]
    return VAL,VAL
def kabsch_rmsd(A,B):
    A=A-A.mean(0);B=B-B.mean(0);U,S,Vt=np.linalg.svd(A.T@B);d=np.sign(np.linalg.det(U@Vt))
    S[-1]*=d;return np.sqrt(max((A**2).sum()+(B**2).sum()-2*S.sum(),0)/len(A))
def jac(a,b):
    u=(a|b).sum();return 1.0 if u==0 else (a&b).sum()/u
rows=[]
for f in sorted(BASE.glob('global/*/eps6_deb3/*.json')):
    p=f.parts[-3];T,rep=map(int,f.stem.split('_'))
    if T==320: continue            # comparisons use 413-450 K qualifying runs
    r=json.loads(f.read_text())
    if not r['qualifying']: continue
    s,rs=src(p,T)
    X=np.load(s/'input'/f'{p}_{T}_{rep}_ca.npy').astype(float)
    ref=np.load(rs/'input'/f'{p}_320_0_ca.npy')[0].astype(float)
    nat=native_contacts(ref); N=X.shape[1]
    natset=set(map(tuple,nat.tolist()))
    pairs,pres=_edge_presence(X,6.0,hysteresis=None);pres=_debounce_rows(pres,3)   # (E,T)
    I,J=np.triu_indices(N,k=4)
    beta=np.array(r['beta'])
    rg=np.sqrt(((X-X.mean(1,keepdims=True))**2).sum(2).mean(1))
    for w in r['windows']:
        a,b=w['start'],w['stop'];bt=beta[a:b].astype(float);P=pres[:,a:b]
        ac=0.0
        if bt.std()>0: ac=float(np.corrcoef(bt[:-1],bt[1:])[0,1]) if bt[:-1].std()>0 and bt[1:].std()>0 else 0.0
        D8=np.linalg.norm(X[a:b][:,I]-X[a:b][:,J],axis=2)<8
        ret8=D8.all(0);ret_pairs=[(i,j) for i,j,k in zip(I,J,ret8) if k]
        nn=sum(1 for q in ret_pairs if q not in natset)
        rows.append(dict(protein=p,T=T,rep=rep,start=a,Q=w['Q_mean'],spanning=w['spanning'],
          occupied=w['occupancy_continuous'],hidden=w['hidden_replacement'],
          beta_min=bt.min(),beta_max=bt.max(),beta_mean=bt.mean(),beta_sd=bt.std(),beta_ac1=ac,
          beta_absdiff=np.abs(np.diff(bt)).mean(),
          n_edges=P.sum(0).mean(),retained6=int(P.all(1).sum()),
          jacc_consec=np.mean([jac(P[:,t],P[:,t+1]) for t in range(b-a-1)]),
          jacc_ends=jac(P[:,0],P[:,-1]),
          flux=int(sum((P[:,t]^P[:,t+1]).sum() for t in range(b-a-1))),
          retained8=len(ret_pairs),retained8_nonnative=nn,
          rg=rg[a:b].mean(),rmsd=np.mean([kabsch_rmsd(X[t],ref) for t in range(a,b)])))
    print(p,T,rep,flush=True)
out=Path(__file__).parent/'window_features.csv'
with open(out,'w',newline='') as fh:
    wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('WROTE',len(rows))
