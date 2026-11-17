from nulltest import *
from load import *
import time
rng=np.random.default_rng(7)
d_=pd.read_csv('transitions_allT.csv'); out=[]
t0=time.time()
for T in [320,348,379,413,450]:
    sub=d_[d_['T']==T].sample(30,random_state=2)
    cache={}
    for _,r in sub.iterrows():
        key=(T,int(r.rep))
        if key not in cache:
            A=np.load(f'Afull_{T}_{int(r.rep)}.npy')
            t=md.load(U+f'mdcath/extracted/5sicI00/5sicI00_{T}_{int(r.rep)}.xtc',top=U+'mdcath/extracted/5sicI00/5sicI00.pdb')
            X=t.xyz[:,t.top.select('name CA')]*10
            cache[key]=(A,X)
        A,X=cache[key]; f=int(r.f)
        m0,m1=A[f],A[f+1]; real=E_step(m0,m1)
        Df=np.linalg.norm(X[f][:,None]-X[f][None],axis=-1)[iu]
        on=np.where(m0&(Df>7))[0]; cand=np.where((~m0)&(Df<9))[0]
        g,l=int((m1&~m0).sum()),int((m0&~m1).sum())
        nulls=[]
        for k in range(3):
            mm=m0.copy(); mm[rng.choice(on,min(l,len(on)),replace=False)]=False; mm[rng.choice(cand,min(g,len(cand)),replace=False)]=True
            nulls.append(E_step(m0,mm))
        out.append(dict(T=T,rep=r.rep,f=f,flux=g+l,Q=r.Q,E=real,null_mean=np.mean(nulls),null_sd=np.std(nulls)))
    print(T,time.time()-t0,flush=True)
pd.DataFrame(out).to_csv('null_results_jitter.csv',index=False)
