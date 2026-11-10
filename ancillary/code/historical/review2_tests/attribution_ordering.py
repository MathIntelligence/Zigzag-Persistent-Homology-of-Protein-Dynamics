"""Critical-simplex attribution with alternative valid within-step orderings (5sicI00, 450 K, eps 8 A, no debounce).
mode 'diameter': insertions by (dimension, diameter ascending), deletions by (dimension descending, diameter descending) -- manuscript convention.
mode 'random': insertions by dimension then random; deletions by dimension descending then random (still face-before-coface
for insertions and coface-before-face for deletions)."""
import mdtraj as md, numpy as np, dionysus as d, random
from scipy.spatial import cKDTree
from collections import defaultdict
U='/mnt/user-data/uploads/zigzagv3VS/mdcath/extracted/5sicI00/'
def load(rep):
    t=md.load(U+f'5sicI00_450_{rep}.xtc',top=U+'5sicI00.pdb'); return t.xyz[:,t.top.select('name CA')]*10,[r.resSeq for r in t.top.residues]
def rips(pts,eps=8.0):
    pairs=cKDTree(pts).query_pairs(r=eps); adj=defaultdict(set)
    for i,j in pairs: adj[i].add(j); adj[j].add(i)
    S={(i,) for i in range(len(pts))}; S|=set(pairs)
    for i,j in pairs:
        for k in adj[i]&adj[j]:
            if k>j: S.add((i,j,k))
    return S
def run(X,f0,f1,mode='diameter',seed=0,eps=8.0):
    rnd=random.Random(seed)
    def L(s,f):
        if len(s)==1: return 0
        P=X[f][list(s)]; return max(np.linalg.norm(P[a]-P[b]) for a in range(len(s)) for b in range(a+1,len(s)))
    times=defaultdict(list); op={}; c=0; prev=rips(X[f0],eps)
    for s in prev: times[s].append(0)
    c=1; cps=[(c,f0,'K')]; c+=1
    for f in range(f0+1,f1+1):
        cur=rips(X[f],eps)
        if mode=='diameter':
            add=sorted(cur-prev,key=lambda s:(len(s),L(s,f))); rem=sorted(prev-cur,key=lambda s:(-len(s),-L(s,f-1)))
        else:
            add=sorted(cur-prev,key=lambda s:(len(s),rnd.random())); rem=sorted(prev-cur,key=lambda s:(-len(s),rnd.random()))
        for s in add: times[s].append(c); op[c]=('add',s,f); c+=1
        cps.append((c,f,'U')); c+=1
        for s in rem: times[s].append(c); op[c]=('rem',s,f); c+=1
        cps.append((c,f,'K')); c+=1; prev=cur
    order=sorted(times,key=lambda s:(len(s),s))
    F=d.Filtration([d.Simplex(list(s)) for s in order])
    zz,dgms,cells=d.zigzag_homology_persistence(F,[times[s] for s in order])
    ct=np.array([x[0] for x in cps]); out=[]
    for p in dgms[1]:
        b=int(round(p.birth)); dd=p.death; dd=int(round(dd)) if np.isfinite(dd) else 10**9
        i0=np.searchsorted(ct,b); i1=np.searchsorted(ct,dd)-1
        if i1<i0: continue
        first,last=cps[i0],cps[i1]
        bf=None if i0==0 else first[1]
        df=None if (dd==10**9 or i1==len(cps)-1) else (last[1]+1 if last[2]=='K' else last[1])
        for kind,fr,o in [('birth',bf,op.get(b)),('death',df,op.get(dd))]:
            if fr is None or o is None: continue
            out.append((kind,fr,o[1]))
    return out
