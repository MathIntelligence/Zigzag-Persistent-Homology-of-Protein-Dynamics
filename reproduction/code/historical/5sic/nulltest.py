import numpy as np, pandas as pd, dionysus as d, sys
from collections import defaultdict
n=107; iu=np.triu_indices(n,1)
def flag(edgemask):
    E=set(); adj=defaultdict(set)
    for k in np.where(edgemask)[0]:
        i,j=iu[0][k],iu[1][k]; E.add((i,j)); adj[i].add(j); adj[j].add(i)
    S={(i,) for i in range(n)}|E
    for i,j in E:
        for k in adj[i]&adj[j]:
            if k>j: S.add((i,j,k))
    return S
def E_step(m0,m1):
    K0,K1=flag(m0),flag(m1); times={}
    for s in K0: times[s]=[0]
    c=1; cps=[(c,'K0')]; c+=1
    add=sorted(K1-K0,key=len)
    for s in add: times[s]=[c]; c+=1
    cps.append((c,'U')); c+=1
    for s in sorted(K0-K1,key=lambda s:-len(s)): times[s].append(c); c+=1
    cps.append((c,'K1')); c+=1
    order=sorted(times,key=lambda s:(len(s),s))
    F=d.Filtration([d.Simplex(list(s)) for s in order])
    zz,dg,_=d.zigzag_homology_persistence(F,[times[s] for s in order])
    ct=np.array([x[0] for x in cps]); ev=0
    for p in dg[1]:
        b=p.birth; de=p.death if np.isfinite(p.death) else 1e12
        ins=[cps[i][1] for i in range(3) if b<=ct[i]<de]
        if not ins: continue
        if 'K0' in ins and 'K1' not in ins: ev+=1   # death
        if 'K1' in ins and 'K0' not in ins: ev+=1   # birth
        if ins==['U']: ev+=2
    return ev
