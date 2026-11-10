import numpy as np, sys
sys.path.insert(0,'/home/claude/conf')
from zzd.filtration import build_filtration
from zzd.zigzag import compute_bars
def native_contacts(ref,cutoff=8.0,min_seq_sep=4):
    N=ref.shape[0]; D=np.linalg.norm(ref[:,None]-ref[None],axis=2); I,J=np.triu_indices(N,k=min_seq_sep); m=D[I,J]<cutoff
    return np.stack([I[m],J[m]],axis=1)
def analyze(X,ref,eps,debounce=3):
    X=X.astype(float); native=native_contacts(ref.astype(float))
    q=(np.linalg.norm(X[:,native[:,0]]-X[:,native[:,1]],axis=-1)<8).mean(axis=1)
    T=len(X)
    starts=np.flatnonzero(np.convolve((q<.2).astype(int),np.ones(20,int),'valid')==20)
    confirm=int(starts[0]+19) if len(starts) and q[0]>=.6 else None
    sim,tm=build_filtration(X,eps=eps,max_dim=2,debounce=debounce)
    bars=compute_bars(sim,tm,T,1.)
    mask=bars.dim==1; lo,hi=bars.alive_lo[mask],bars.alive_hi[mask]
    beta=((lo[:,None]<=np.arange(T))&(hi[:,None]>np.arange(T))).sum(0)
    W=[]
    for start in range(0,T-19,20):
        stop=start+20; v=beta[start:stop]; sp=int(((lo<=start)&(hi>=stop)).sum())
        W.append(dict(start=start,stop=stop,min_beta=int(v.min()),max_beta=int(v.max()),spanning=sp,
                      occupancy_continuous=bool(v.min()>0),Q_mean=float(q[start:stop].mean())))
    return dict(qualifying=confirm is not None,windows=W,q=q,beta=beta)
