"""Matched descriptive baselines and robustness audit; never edits old results."""
from pathlib import Path
from types import SimpleNamespace
import itertools
import json
import sys
import time

import numpy as np
from scipy.stats import kendalltau

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / 'mdcath_large_domains'))
import heldout_order as old
from zigzag_dynamics.filtration import _edge_presence, _debounce_rows

SOURCES = {p: ROOT / s for s, ps in (
    ('mdcath_order_extension_5', ['2d1pA00','2d4uB00','2dawA01']),
    ('mdcath_order_search_batch_01', ['2ervA00','2fd5A02']),
    ('mdcath_order_targeted_10', ['4a4aA02','2ivxB01','3hwcA01','3hbvP01'])
) for p in ps}
CROSS = {p: ROOT / s for p,s in (
    ('2ervA00','mdcath_order_cross_temperature_all15'),
    ('3hwcA01','mdcath_order_targeted_cross_temperature'),
    ('3hbvP01','mdcath_order_targeted_cross_temperature'))}


def binary_rank(columns):
    pivots = {}
    for c in columns:
        c = int(c)
        while c:
            k = c.bit_length()-1
            if k in pivots:
                c ^= pivots[k]
            else:
                pivots[k] = c
                break
    return len(pivots)


def h1_basis(n, edges, return_basis=False):
    """Exact H1 of a clique complex through dimension 2, over F2.

    Fundamental graph cycles modulo triangle boundaries. Returned representatives
    are deterministic graph cycles, not volume-optimal or biologically unique.
    """
    edges = [tuple(map(int,e)) for e in edges]
    adj = [set() for _ in range(n)]
    lookup = {e:k for k,e in enumerate(edges)}
    for i,j in edges: adj[i].add(j); adj[j].add(i)
    triangles=[]
    for i,j in edges:
        for k in sorted(adj[i]&adj[j]):
            if k>j:
                triangles.append((1<<lookup[(i,j)]) ^ (1<<lookup[(i,k)]) ^ (1<<lookup[(j,k)]))
    parent=list(range(n)); forest=[[] for _ in range(n)]; nonforest=[]
    def find(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]];v=parent[v]
        return v
    for idx,(i,j) in enumerate(edges):
        a,b=find(i),find(j)
        if a!=b:
            parent[a]=b;forest[i].append((j,idx));forest[j].append((i,idx))
        else: nonforest.append((idx,i,j))
    rank=binary_rank(triangles)
    beta=len(nonforest)-rank
    assert beta>=0
    if not return_basis: return beta
    pivots={}
    def add(c):
        while c:
            k=c.bit_length()-1
            if k in pivots: c^=pivots[k]
            else: pivots[k]=c;return True
        return False
    for c in triangles:add(c)
    reps=[]
    for idx,i,j in nonforest:
        stack=[(i,-1,0)]
        while stack:
            v,prev,chain=stack.pop()
            if v==j:break
            stack.extend((w,v,chain^(1<<e)) for w,e in forest[v] if w!=prev)
        chain^=1<<idx
        if add(chain): reps.append([edges[k] for k in range(len(edges)) if chain>>k&1])
    assert len(reps)==beta
    return beta,reps


def drop_short(a,k=3):
    a=np.asarray(a,bool).copy()
    for row in a.reshape(-1,a.shape[-1]):
        changes=np.flatnonzero(np.diff(np.r_[False,row,False].astype(int)))
        for lo,hi in zip(changes[::2],changes[1::2]):
            if hi-lo<k: row[lo:hi]=False
    return a


def static_traces(X,supports):
    pairs,raw=_edge_presence(X,6.,hysteresis=None)
    smoothed=_debounce_rows(raw,3)
    all_out={}
    for name,edge_presence in [('frame_h1_raw',raw),('frame_h1_debounce',smoothed)]:
        out=np.zeros((len(supports),len(X)),int)
        for j,support in enumerate(supports):
            mapping={v:i for i,v in enumerate(support)}
            idx=[k for k,(u,v) in enumerate(pairs) if u in mapping and v in mapping]
            localpairs=np.asarray([(mapping[int(pairs[k,0])],mapping[int(pairs[k,1])]) for k in idx],int).reshape(-1,2)
            pres=edge_presence[idx].T
            cache={}
            for t,row in enumerate(pres):
                key=np.packbits(row).tobytes()
                if key not in cache:cache[key]=h1_basis(len(support),localpairs[row])
                out[j,t]=cache[key]
        all_out[name]=out
    return all_out


def pair_matrix(a,b=None,observed_only=False):
    a=np.asarray(a,float); b=a if b is None else np.asarray(b,float)
    out=np.full((len(a),len(b)),np.nan);counts=np.zeros_like(out,int)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            ok=np.isfinite(x)&np.isfinite(y)
            if observed_only:ok&=(x>=0)&(x<499)&(y>=0)&(y<499)
            counts[i,j]=ok.sum()
            if ok.sum()>=4:out[i,j]=kendalltau(x[ok],y[ok]).statistic
    return out,counts


def finite_mean(x):
    x=np.asarray(x);return float(np.nanmean(x)) if np.isfinite(x).any() else None


def summarize_pair(a,b=None,observed_only=False):
    m,c=pair_matrix(a,b,observed_only)
    if b is None:
        idx=np.triu_indices(len(a),1); vals=m[idx]; count=c[idx]
    else: vals=m.ravel();count=c.ravel()
    result=dict(tau=finite_mean(vals),defined=int(np.isfinite(vals).sum()),total=len(vals),
                min_common=int(count.min()),max_common=int(count.max()),pair_taus=vals)
    if b is not None:
        rng=np.random.default_rng(20260910)
        aa=rng.integers(len(a),size=(5000,len(a)));bb=rng.integers(len(b),size=(5000,len(b)))
        draws=m[aa[:,:,None],bb[:,None,:]].reshape(5000,-1)
        n=np.isfinite(draws).sum(axis=1)
        v=np.divide(np.nansum(draws,axis=1),n,out=np.full(5000,np.nan),where=n>0)
        result['run_bootstrap_95']=np.quantile(v[np.isfinite(v)],[.025,.975]) if np.isfinite(v).any() else []
        result['bootstrap_undefined']=int((n==0).sum())
    return result


def no_ambiguous(p,source,temp,replicas,core):
    prior=SOURCES[p]/'results'/p
    families=json.loads((prior/'training_families.json').read_text())
    fs=[SimpleNamespace(**{k:f[k] for k in ['fid','support','members']}) for f in families]
    train={f'320_{r}':json.loads((prior/f'320_{r}_episodes.json').read_text())['episodes'] for r in range(5)}
    out=np.zeros((len(replicas),len(core),500),bool)
    index={fid:j for j,fid in enumerate(core)}
    match_info=[]
    for k,r in enumerate(replicas):
        eps=json.loads((source/'results'/p/f'{temp}_{r}_episodes.json').read_text())['episodes']
        assigned,dist,amb=old.assign_heldout(eps,fs,train)
        match_info.append(dict(replica=int(r),episodes=len(eps),ambiguous=sum(amb),matched=sum(x>=0 for x in assigned)))
        for ep,fid,a in zip(eps,assigned,amb):
            if fid in index and not a:out[k,index[fid],ep['lo']:ep['hi']]=True
    return out,match_info


def run_group(p,temp,source,occupancy_path):
    dest=BASE/'results'/p;dest.mkdir(exist_ok=True,parents=True)
    target=dest/f'{temp}_traces.npz'
    family_path=SOURCES[p]/'results'/p/'training_families.json'
    families=[f for f in json.loads(family_path.read_text()) if f['core']]
    supports=[f['support'] for f in families]
    occ=np.load(occupancy_path);reps=occ['replica_ids'];core=occ['family_ids']
    assert [f['fid'] for f in families]==core.tolist()
    inputs=[source/'input'/f'{p}_{temp}_{r}_ca.npy' for r in reps]
    fingerprint=dict(code=old.sha(__file__),protocol=old.sha(BASE/'PROTOCOL.md'),
        inputs={str(x.relative_to(ROOT)):old.sha(x) for x in inputs},
        families=old.sha(family_path),occupancy=old.sha(occupancy_path))
    meta=dest/f'{temp}_meta.json'
    if target.exists():
        assert json.loads(meta.read_text())['fingerprint']==fingerprint
        return dict(np.load(target))
    native=old.native_contacts(np.load(SOURCES[p]/'input'/f'{p}_320_0_ca.npy')[0].astype(float))
    arrays={'zigzag':occ['alive'],'replica_ids':reps,'family_ids':core}
    q=[];static={'frame_h1_raw':[],'frame_h1_debounce':[]}
    for r,path in zip(reps,inputs):
        X=np.load(path).astype(float)
        q.append([old.contact_trace(X,s,native) for s in supports])
        t0=time.time();traces=static_traces(X,supports)
        for key in static:static[key].append(traces[key])
        print(f'{p} {temp}/{r}: localized framewise homology complete ({time.time()-t0:.1f}s)',flush=True)
    arrays['q']=np.asarray(q);arrays['contact']=arrays['q']>=.5
    for key,val in static.items():
        arrays[key+'_counts']=np.asarray(val)
        arrays[key]=drop_short(arrays[key+'_counts']>0)
    arrays['zigzag_unambiguous'],info=no_ambiguous(p,source,temp,reps,core)
    np.savez_compressed(target,**arrays)
    old.write_json(meta,dict(fingerprint=fingerprint,matching=info))
    return arrays


METHODS=['zigzag','zigzag_unambiguous','contact','frame_h1_raw','frame_h1_debounce']


def main():
    (BASE/'results').mkdir(exist_ok=True)
    rows=[];crossrows=[];associations=[]
    for p,src in SOURCES.items():
        a=run_group(p,450,src,src/'results'/p/'occupancy_traces.npz')
        groups={450:a}
        if p in CROSS:
            src413=CROSS[p]
            b=run_group(p,413,src413,src413/'results'/p/'413/occupancy_traces.npz')
            groups[413]=b
        for temp,g in groups.items():
            z=old.last_present(g['zigzag'])
            for method in METHODS:
                last=old.last_present(g[method])
                for observed in [False,True]:
                    stats=summarize_pair(last,observed_only=observed)
                    rows.append(dict(protein=p,temperature=temp,method=method,observed_only=observed,**stats,
                        absent=float((last<0).mean()),censored=float((last==499).mean())))
                    if method!='zigzag':
                        vals=[]
                        for x,y in zip(z,last):
                            mask=(x>=0)&(x<499)&(y>=0)&(y<499) if observed else np.ones(len(x),bool)
                            vals.append(kendalltau(x[mask],y[mask]).statistic if mask.sum()>=4 else np.nan)
                        associations.append(dict(protein=p,temperature=temp,method=method,observed_only=observed,
                            mean_tau=finite_mean(vals),replica_taus=vals))
            if temp==413:
                for method in METHODS:
                    for observed in [False,True]:
                        stats=summarize_pair(old.last_present(g[method]),old.last_present(a[method]),observed)
                        crossrows.append(dict(protein=p,method=method,observed_only=observed,**stats))
                print(p,'cross audit:',[(r['method'],r['observed_only'],r['tau']) for r in crossrows if r['protein']==p],flush=True)
        old.write_json(BASE/'results/within.json',rows)
        old.write_json(BASE/'results/cross.json',crossrows)
        old.write_json(BASE/'results/associations.json',associations)
    print('BASELINE AUDIT COMPUTATION COMPLETE',flush=True)


if __name__=='__main__':main()
