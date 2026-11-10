from load import *
import os
b=pd.read_csv(U+'manuscript/temporal_topology_2026-09-16/data/5sicI00_eps8_H1H2_bars_all_T.csv')
w=pd.read_csv(U+'mdcath_consecutive_wasserstein_3x3/w1_sensitivity/frame_pair_distances.csv')
ref=md.load(U+'mdcath/extracted/5sicI00/5sicI00.pdb'); R=ref.xyz[0][ref.top.select('name CA')]*10
n=107; DR=np.linalg.norm(R[:,None]-R[None],axis=-1); sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None])
nat=(DR<8)&(sep>3)
rows=[]
for T in [320,348,379,413,450]:
  for rep in range(5):
    t=md.load(U+f'mdcath/extracted/5sicI00/5sicI00_{T}_{rep}.xtc',top=U+'mdcath/extracted/5sicI00/5sicI00.pdb')
    X=t.xyz[:,t.top.select('name CA')]*10; F=len(X)
    D=np.linalg.norm(X[:,:,None]-X[:,None],axis=-1)
    iu=np.triu_indices(n,1); A=(D<8)[:,iu[0],iu[1]]   # full Rips edge set (as in zigzag)
    Q=((D<8)&nat).sum((1,2))/nat.sum()
    s=b[(b.temperature_K==T)&(b.replica==rep)&(b.dimension==1)]
    bi=np.bincount(s.birth_ns.astype(int),minlength=F+1); de=np.bincount(s.death_ns.astype(int),minlength=F+1)
    ww=w[(w.protein=='5sicI00')&(w.temperature_K==T)&(w.replica==rep)].sort_values('frame_i')
    W1=dict(zip(ww.frame_i,ww.wasserstein_A))
    for f in range(F-1):
      gain=(A[f+1]&~A[f]).sum(); loss=(A[f]&~A[f+1]).sum()
      rows.append(dict(T=T,rep=rep,f=f,E=bi[f+1]+de[f+1],gain=gain,loss=loss,flux=gain+loss,exch=min(gain,loss),edges=A[f].sum(),Q=Q[f],W1=W1.get(f,np.nan)))
    np.save(f'Afull_{T}_{rep}.npy',A)
    print(T,rep,F)
pd.DataFrame(rows).to_csv('transitions_allT.csv',index=False)
