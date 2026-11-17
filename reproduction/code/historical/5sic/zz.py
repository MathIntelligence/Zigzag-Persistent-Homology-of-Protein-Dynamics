from load import *
import dionysus as d, sys, pickle
from scipy.spatial import cKDTree
from collections import defaultdict
def rips(pts,eps=8.0):
    pairs=cKDTree(pts).query_pairs(r=eps); adj=defaultdict(set)
    for i,j in pairs: adj[i].add(j); adj[j].add(i)
    S={(i,) for i in range(len(pts))}; S|=set(pairs)
    for i,j in pairs:
        for k in adj[i]&adj[j]:
            if k>j: S.add((i,j,k))
    return S
def run(rep,f0,f1,eps=8.0,tag=''):
    t=load(rep); X=t.xyz[:,t.top.select('name CA')]*10
    def L(s,f):  # max edge length of simplex in frame f
        if len(s)==1: return 0
        P=X[f][list(s)]; return max(np.linalg.norm(P[a]-P[b]) for a in range(len(s)) for b in range(a+1,len(s)))
    times=defaultdict(list); op={}
    prev=rips(X[f0],eps)
    for s in prev: times[s].append(0.0)
    op[0.0]=('init',None,f0)
    for f in range(f0+1,f1+1):
        cur=rips(X[f],eps)
        add=sorted(cur-prev,key=lambda s:(len(s),L(s,f)))
        rem=sorted(prev-cur,key=lambda s:(-len(s),-L(s,f-1)))
        k=0
        for s in add:
            tt=(f-f0)-0.5+k*1e-5; times[s].append(tt); op[round(tt,6)]=('add',s,f); k+=1
        k=0
        for s in rem:
            tt=(f-f0)+k*1e-5; times[s].append(tt); op[round(tt,6)]=('rem',s,f); k+=1
        prev=cur
    order=sorted(times,key=lambda s:(len(s),s))
    F=d.Filtration([d.Simplex(list(s)) for s in order])
    TT=[times[s] for s in order]
    zz,dgms,cells=d.zigzag_homology_persistence(F,TT)
    ev=[]
    for p in dgms[1]:
        b=round(p.birth,6); de=p.death
        ev.append(dict(birth_t=b,death_t=(round(de,6) if np.isfinite(de) else np.inf)))
    return ev,op,X
if __name__=='__main__':
    rep=int(sys.argv[1]); f0=int(sys.argv[2]); f1=int(sys.argv[3])
    ev,op,X=run(rep,f0,f1)
    pickle.dump((ev,op,f0,f1),open(f'zz_{rep}_{f0}_{f1}.pkl','wb'))
    # per-frame counts
    from math import ceil
    B=defaultdict(int);Dd=defaultdict(int)
    for e in ev:
        if e['birth_t']>0: B[f0+ceil(e['birth_t'])]+=1
        if np.isfinite(e['death_t']): Dd[f0+ceil(e['death_t'])]+=1
    for f in range(f0+1,f1+1): print(f,B[f],Dd[f],B[f]+Dd[f])
