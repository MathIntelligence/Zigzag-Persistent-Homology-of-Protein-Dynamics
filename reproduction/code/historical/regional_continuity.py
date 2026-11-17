"""Basis-independent regional temporal continuity and its Betti-count ablation."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import audit_order as a
from zigzag_dynamics.filtration import build_filtration,_edge_presence,_debounce_rows
from zigzag_dynamics.zigzag import compute_bars

BASE=a.BASE;ROOT=a.ROOT;old=a.old


def intervals_from_barcode(bars):
    mask=bars.dim==1
    return bars.alive_lo[mask],bars.alive_hi[mask]


def coverage(lo,hi,start,stop):
    return int(((lo<=start)&(hi>=stop)).sum())


def describe(lo,hi,beta,q):
    windows=[];bouts=[]
    for start in range(0,len(beta)-19,20):
        stop=start+20;v=beta[start:stop]
        overlap=np.maximum(0,np.minimum(hi,stop)-np.maximum(lo,start))
        spanning=coverage(lo,hi,start,stop)
        windows.append(dict(start=start,stop=stop,min_beta=int(v.min()),max_beta=int(v.max()),
            spanning=spanning,occupancy_continuous=bool(v.min()>0),
            hidden_replacement=bool(v.min()>0 and spanning==0),
            constant_one=bool(np.all(v==1)),
            longest_overlap=float(overlap.max()/20) if len(overlap) else 0.,Q_mean=float(q[start:stop].mean())))
    one=(beta==1);changes=np.flatnonzero(np.diff(np.r_[False,one,False].astype(int)))
    for start,stop in zip(changes[::2],changes[1::2]):
        if stop-start>=20:
            bouts.append(dict(start=int(start),stop=int(stop),length=int(stop-start),spanning=coverage(lo,hi,start,stop)))
    return windows,bouts


def run(p,temp,rep,source,eps=6.,debounce=3):
    path=source/'input'/f'{p}_{temp}_{rep}_ca.npy'
    dest=BASE/'continuity'/p/f'eps{eps:g}_deb{debounce}';dest.mkdir(parents=True,exist_ok=True)
    target=dest/f'{temp}_{rep}.json'
    families_path=a.SOURCES[p]/'results'/p/'training_families.json'
    fingerprint=dict(code=old.sha(__file__),protocol=old.sha(BASE/'PROTOCOL_CONTINUITY.md'),
                     input=old.sha(path),families=old.sha(families_path),eps=eps,debounce=debounce)
    if target.exists():
        result=json.loads(target.read_text());assert result['fingerprint']==fingerprint;return result
    X=np.load(path).astype(float)
    ref=np.load(a.SOURCES[p]/'input'/f'{p}_320_0_ca.npy')[0].astype(float)
    native=old.native_contacts(ref)
    q=(np.linalg.norm(X[:,native[:,0]]-X[:,native[:,1]],axis=-1)<8).mean(axis=1)
    starts=np.flatnonzero(np.convolve((q<.2).astype(int),np.ones(20,int),'valid')==20)
    confirm=int(starts[0]+19) if len(starts) and q[0]>=.6 else None
    families=[f for f in json.loads(families_path.read_text()) if f['core']]
    regions=[]
    for f in families:
        Y=X[:,f['support']]
        sim,tm=build_filtration(Y,eps=eps,max_dim=2,debounce=debounce)
        bars=compute_bars(sim,tm,len(Y),1.)
        lo,hi=intervals_from_barcode(bars)
        beta=((lo[:,None]<=np.arange(500))&(hi[:,None]>np.arange(500))).sum(axis=0)
        # Independent clique-boundary calculation at fixed deterministic frames.
        pairs,pres=_edge_presence(Y,eps,hysteresis=None)
        pres=_debounce_rows(pres,debounce)
        for t in [0,1,19,100,249,499]:
            assert a.h1_basis(len(f['support']),pairs[pres[:,t]])==beta[t]
        # At the main scale also verify every frame against the separately saved baseline.
        baseline=BASE/'results'/p/f'{temp}_traces.npz'
        if eps==6 and debounce==3 and baseline.exists():
            b=np.load(baseline);k=list(b['replica_ids']).index(rep);j=list(b['family_ids']).index(f['fid'])
            np.testing.assert_array_equal(beta,b['frame_h1_debounce_counts'][k,j])
        windows,bouts=describe(lo,hi,beta,q)
        regions.append(dict(family=f['fid'],support=f['support'],lo=lo,hi=hi,beta=beta,windows=windows,bouts=bouts))
    result=dict(fingerprint=fingerprint,protein=p,temperature=temp,replica=rep,eps=eps,debounce=debounce,
                confirmation_frame=confirm,regions=regions)
    old.write_json(target,result)
    windows=[w for r in regions for w in r['windows']]
    print(f'{p} {temp}/{rep} eps{eps:g} deb{debounce}: '+
          f'{sum(w["hidden_replacement"] for w in windows)}/{sum(w["occupancy_continuous"] for w in windows)} hidden-replacement windows',flush=True)
    return result


def main():
    for p,source in a.SOURCES.items():
        result=json.loads((source/'results'/p/'results.json').read_text())
        configs=[(320,source,list(range(5))),(450,source,result['eligible_replicas'])]
        if p in a.CROSS:
            other=a.CROSS[p];cr=json.loads((other/'results'/p/'413/results.json').read_text())
            configs.append((413,other,cr['eligible_replicas']))
        for temp,src,reps in configs:
            for rep in reps:run(p,temp,rep,src)
    for p,other in a.CROSS.items():
        for temp,src in [(450,a.SOURCES[p]),(413,other)]:
            path=src/'results'/p/('results.json' if temp==450 else '413/results.json')
            for eps,debounce in [(5.,3),(7.,3),(6.,0)]:
                for rep in json.loads(path.read_text())['eligible_replicas']:run(p,temp,rep,src,eps,debounce)
    print('REGIONAL CONTINUITY ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':main()
