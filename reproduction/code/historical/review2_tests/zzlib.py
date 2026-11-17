import numpy as np, sys
sys.path.insert(0,'/home/claude/r2')
from zd import filtration as F
from zd.zigzag import compute_bars
def build_from_presence(pairs,pres_e,N,T):
    keep=pres_e.any(axis=1); pairs=pairs[keep]; pres_e=np.ascontiguousarray(pres_e[keep])
    packed_e=np.packbits(pres_e,axis=1)
    blocks=[(np.arange(N,dtype=np.int64).reshape(-1,1),None),(pairs,packed_e)]
    adj=F._adjacency(pairs,N); eidx=np.full((N,N),-1,dtype=np.int64)
    eidx[pairs[:,0],pairs[:,1]]=np.arange(len(pairs)); eidx[pairs[:,1],pairs[:,0]]=np.arange(len(pairs))
    tris=F._candidate_triangles(pairs,adj)
    if len(tris):
        rows=np.stack([eidx[tris[:,0],tris[:,1]],eidx[tris[:,0],tris[:,2]],eidx[tris[:,1],tris[:,2]]],axis=1)
        packed_t,k=F._and_rows(packed_e,rows,200000); blocks.append((tris[k],np.ascontiguousarray(packed_t[k])))
    S=[];TT=[]
    for v,packed in blocks:
        o=np.lexsort(v.T[::-1]);v=v[o]
        tl=[[0,T]]*len(v) if packed is None else F._toggles_from_packed(packed[o],T)
        S.extend(v.tolist());TT.extend(tl)
    return S,TT
def presence(X,eps=6.0,deb=3):
    pairs,pres=F._edge_presence(X,eps,hysteresis=None)
    if deb>1: pres=F._debounce_rows(pres,deb)
    return pairs,pres
def bars_from_presence(pairs,pres,N,T):
    S,TT=build_from_presence(pairs,pres,N,T); b=compute_bars(S,TT,T,1.)
    m=b.dim==1; lo=b.alive_lo[m].astype(float); hi=b.alive_hi[m].astype(float)
    beta=((lo[:,None]<=np.arange(T))&(hi[:,None]>np.arange(T))).sum(0)
    return lo,hi,beta
def windows(lo,hi,beta,L=20):
    out=[]
    for a in range(0,len(beta)-L+1,L):
        b=a+L; span=int(((lo<=a)&(hi>=b)).sum())
        ov=np.maximum(0,np.minimum(hi,b)-np.maximum(lo,a)); ov=ov[ov>0]/L
        alive=lambda t:(lo<=t)&(hi>t)
        surv=np.mean([(alive(t)&alive(t+1)).sum()/max(alive(t).sum(),1) for t in range(a,b-1)])
        out.append(dict(start=a,occupied=bool(beta[a:b].min()>0),spanning=span,mean_overlap=float(ov.mean()) if len(ov) else 0.,step_survival=float(surv)))
    return out
