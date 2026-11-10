import pandas as pd, numpy as np, glob, os
U='/mnt/user-data/uploads/zigzagv3VS/'
W=pd.read_csv(U+'mdcath_order_added_value/window_summary.csv')
val=['3vskA02','3k1aA02','2vsyA01','2np9A01','3j7ye00']
src={'2d1pA00':'mdcath_order_extension_5','2d4uB00':'mdcath_order_extension_5','2dawA01':'mdcath_order_extension_5',
'2ervA00':'mdcath_order_search_batch_01','2fd5A02':'mdcath_order_search_batch_01','4a4aA02':'mdcath_order_targeted_10','2ivxB01':'mdcath_order_targeted_10','3hwcA01':'mdcath_order_targeted_10','3hbvP01':'mdcath_order_targeted_10'}
for v in val: src[v]='mdcath_order_added_value/validation'
def path(p,T,r):
    b=src[p]
    if p=='2ervA00' and T==413: b='mdcath_order_cross_temperature_all15'
    return U+f'{b}/input/{p}_{T}_{r}_ca.npy'
s=W[(W.eps==6)&(W.debounce==3)&(W.temperature>=413)&W.qualifying&W.occupancy_continuous].copy()
s['frac']=s.spanning/s.min_beta
s['cls']=np.where(s.Q_mean>=0.6,'folded',np.where(s.Q_mean<0.3,np.where(s.frac>=0.5,'discordant',np.where(s.spanning==0,'replaced','other_unf')),'mid'))
runs=s[s.cls=='discordant'][['protein','temperature','replica']].drop_duplicates()
out=[]
for _,r in runs.iterrows():
    p,T,rep=r.protein,r.temperature,r.replica
    X=np.load(path(p,T,rep)).astype(float); ref=np.load(U+f'{src[p]}/input/{p}_320_0_ca.npy')[0].astype(float)
    n=X.shape[1]; I,J=np.triu_indices(n,4)
    DR=np.linalg.norm(ref[I]-ref[J],axis=1); nat=DR<8
    D=np.linalg.norm(X[:,I]-X[:,J],axis=2); C=D<8
    rg=np.sqrt(((X-X.mean(1,keepdims=True))**2).sum(2).mean(1))
    rgref=np.sqrt(((ref-ref.mean(0))**2).sum(1).mean())
    g=s[(s.protein==p)&(s.temperature==T)&(s.replica==rep)]
    for _,w in g.iterrows():
        a,b=int(w.start),int(w.stop)
        ret=C[a:b].all(0); any_=C[a:b].any(0)
        nxt=C[b:b+20].all(0) if b+20<=len(X) else None
        out.append(dict(protein=p,T=T,rep=rep,start=a,cls=w.cls,Q=w.Q_mean,frac=w.frac,spanning=w.spanning,min_beta=w.min_beta,
            rg_ratio=rg[a:b].mean()/rgref, n_retained=int(ret.sum()), retained_frac=ret.sum()/max(1,any_.sum()),
            nonnative_retained=int((ret&~nat).sum()), frac_retained_nonnative=(ret&~nat).sum()/max(1,ret.sum()),
            kept_next=(ret&nxt).sum()/max(1,ret.sum()) if nxt is not None else np.nan))
R=pd.DataFrame(out); R.to_csv('disc_features.csv',index=False)
cols=['Q','frac','rg_ratio','n_retained','retained_frac','nonnative_retained','frac_retained_nonnative','kept_next']
print(R.groupby('cls')[cols].median().round(3).to_string())
print(R.groupby('cls').size())
