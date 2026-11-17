"""Whole-complex continuity, avoiding induced-region and family-match artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import audit_order as a
from regional_continuity import intervals_from_barcode,describe
from zigzag_dynamics.filtration import build_filtration,_edge_presence,_debounce_rows
from zigzag_dynamics.zigzag import compute_bars

BASE=a.BASE;ROOT=a.ROOT;old=a.old
VAL=BASE/'validation'


def analyze(p,temp,rep,src,refsrc,eps=6.,debounce=3):
    path=src/'input'/f'{p}_{temp}_{rep}_ca.npy'
    dest=BASE/'global'/p/f'eps{eps:g}_deb{debounce}';dest.mkdir(parents=True,exist_ok=True)
    target=dest/f'{temp}_{rep}.json'
    fp=dict(code=old.sha(__file__),protocol=old.sha(BASE/'PROTOCOL_GLOBAL_VALIDATION.md'),input=old.sha(path),
            ref=old.sha(refsrc/'input'/f'{p}_320_0_ca.npy'),eps=eps,debounce=debounce)
    if target.exists():
        r=json.loads(target.read_text());assert r['fingerprint']==fp;return r
    X=np.load(path).astype(float);ref=np.load(refsrc/'input'/f'{p}_320_0_ca.npy')[0].astype(float)
    native=old.native_contacts(ref)
    q=(np.linalg.norm(X[:,native[:,0]]-X[:,native[:,1]],axis=-1)<8).mean(axis=1)
    starts=np.flatnonzero(np.convolve((q<.2).astype(int),np.ones(20,int),'valid')==20)
    confirm=int(starts[0]+19) if len(starts) and q[0]>=.6 else None
    sim,tm=build_filtration(X,eps=eps,max_dim=2,debounce=debounce)
    bars=compute_bars(sim,tm,len(X),1.)
    lo,hi=intervals_from_barcode(bars)
    beta=((lo[:,None]<=np.arange(500))&(hi[:,None]>np.arange(500))).sum(0)
    pairs,pres=_edge_presence(X,eps,hysteresis=None);pres=_debounce_rows(pres,debounce)
    # Every-frame independent boundary-matrix rank check, not just sparse examples.
    independent=np.array([a.h1_basis(X.shape[1],pairs[pres[:,t]]) for t in range(500)])
    np.testing.assert_array_equal(beta,independent)
    windows,bouts=describe(lo,hi,beta,q)
    for w in windows:
        w['after_confirmation']=confirm is not None and w['start']>=confirm
        w['all_low_Q']=bool(np.all(q[w['start']:w['stop']]<.2))
        w['all_high_Q']=bool(np.all(q[w['start']:w['stop']]>=.6))
    r=dict(fingerprint=fp,protein=p,temperature=temp,replica=rep,eps=eps,debounce=debounce,
           qualifying=confirm is not None,confirmation_frame=confirm,initial_Q=float(q[0]),
           lo=lo,hi=hi,beta=beta,Q=q,windows=windows,bouts=bouts,
           every_frame_independent_betti_verified=True)
    old.write_json(target,r)
    print(f'{p} {temp}/{rep} eps{eps:g} deb{debounce}: '+
          f'{sum(w["hidden_replacement"] for w in windows)}/{sum(w["occupancy_continuous"] for w in windows)} hidden; '+
          f'{sum(w["hidden_replacement"] and w["constant_one"] for w in windows)} constant-one; Q eligible={r["qualifying"]}',flush=True)
    return r


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--validation',action='store_true');args=arg.parse_args()
    jobs=[]
    if args.validation:
        sel=json.loads((VAL/'selection.json').read_text())
        for p in sel['selected']:
            for temp in [320,413,450]:
                for rep in range(5):jobs.append((p,temp,rep,VAL,VAL))
    else:
        for p,src in a.SOURCES.items():
            for rep in range(5):jobs.append((p,320,rep,src,src))
            r=json.loads((src/'results'/p/'results.json').read_text())
            for rep in r['eligible_replicas']:jobs.append((p,450,rep,src,src))
            if p in a.CROSS:
                cr=a.CROSS[p];r=json.loads((cr/'results'/p/'413/results.json').read_text())
                for rep in r['eligible_replicas']:jobs.append((p,413,rep,cr,src))
    for j in jobs:analyze(*j)
    for j in jobs:
        p,temp,rep,src,refsrc=j
        if temp==320:continue
        r=analyze(*j)
        if r['qualifying']:
            for eps,debounce in [(5.,3),(7.,3),(6.,0)]:analyze(*j,eps=eps,debounce=debounce)
    print('GLOBAL CONTINUITY COMPLETE',flush=True)


if __name__=='__main__':main()
