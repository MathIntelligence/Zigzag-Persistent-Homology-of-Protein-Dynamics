import numpy as np, sys
sys.path.insert(0,'/home/claude/conf')
from zzd.filtration import build_filtration
from zzd.zigzag import compute_bars
from cont import native_contacts
def windows(X,ref,eps,L=20,debounce=3):
    X=X.astype(float); nat=native_contacts(ref.astype(float))
    q=(np.linalg.norm(X[:,nat[:,0]]-X[:,nat[:,1]],axis=-1)<8).mean(1); T=len(X)
    sim,tm=build_filtration(X,eps=eps,max_dim=2,debounce=debounce); bars=compute_bars(sim,tm,T,1.)
    m=bars.dim==1; lo,hi=bars.alive_lo[m],bars.alive_hi[m]
    beta=((lo[:,None]<=np.arange(T))&(hi[:,None]>np.arange(T))).sum(0)
    W=[]
    for s in range(0,T-L+1,L):
        v=beta[s:s+L]; sp=int(((lo<=s)&(hi>=s+L)).sum())
        W.append(dict(start=s,min_beta=int(v.min()),mean_beta=float(v.mean()),spanning=sp,Q=float(q[s:s+L].mean())))
    return W,q,beta
