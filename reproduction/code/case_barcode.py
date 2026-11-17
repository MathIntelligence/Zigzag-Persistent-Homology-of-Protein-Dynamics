import numpy as np
import dionysus as d, sys, bisect
import itertools
def rips(P,eps):
    n=len(P);a=np.linalg.norm(P[:,None]-P[None,:],axis=-1)<=eps
    out={(i,) for i in range(n)}
    out.update((i,j) for i in range(n) for j in range(i+1,n) if a[i,j])
    out.update((i,j,k) for i in range(n) for j in range(i+1,n) if a[i,j] for k in range(j+1,n) if a[i,k] and a[j,k])
    return out
from collections import defaultdict
def run(rep,f0,f1,eps=8.0,X=None):
    if X is None: raise ValueError('Coordinates X required')
    def L(s,f):
        if len(s)==1: return 0
        P=X[f][list(s)]; return max(np.linalg.norm(P[a]-P[b]) for a in range(len(s)) for b in range(a+1,len(s)))
    times=defaultdict(list); op={}; c=0
    prev=rips(X[f0],eps)
    for s in prev: times[s].append(0)
    c=1; cps=[(c,f0,'K')]; c+=1
    for f in range(f0+1,f1+1):
        cur=rips(X[f],eps)
        for s in sorted(cur-prev,key=lambda s:(len(s),L(s,f))):
            times[s].append(c); op[c]=('add',s,f); c+=1
        cps.append((c,f,'U')); c+=1
        for s in sorted(prev-cur,key=lambda s:(-len(s),-L(s,f-1))):
            times[s].append(c); op[c]=('rem',s,f); c+=1
        cps.append((c,f,'K')); c+=1
        prev=cur
    assert c<16_000_000
    order=sorted(times,key=lambda s:(len(s),s))
    F=d.Filtration([d.Simplex(list(s)) for s in order])
    zz,dgms,cells=d.zigzag_homology_persistence(F,[times[s] for s in order])
    ct=np.array([x[0] for x in cps]); out=[]
    for p in dgms[1]:
        b=int(round(p.birth)); dd=p.death; dd=int(round(dd)) if np.isfinite(dd) else 10**9
        i0=np.searchsorted(ct,b); i1=np.searchsorted(ct,dd)-1   # checkpoints in [b,dd)
        if i1<i0: continue
        first,last=cps[i0],cps[i1]
        bf=None if i0==0 else first[1]
        df=None if (dd==10**9 or i1==len(cps)-1) else (last[1]+1 if last[2]=='K' else last[1])
        out.append(dict(bf=bf,df=df,bop=op.get(b),dop=op.get(dd),life_frames=(last[1]-first[1]) ))
    return out
