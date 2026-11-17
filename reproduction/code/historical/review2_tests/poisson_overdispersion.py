"""Residue-level association of critical simplices with contact-partner exchange (5sicI00, 450 K, 5 replicas):
original Poisson fit, overdispersion check, cluster-robust SEs (by residue; by replica-frame), negative binomial."""
import numpy as np, pandas as pd, mdtraj as md, statsmodels.formula.api as smf, statsmodels.api as sm, sys
U='/mnt/user-data/uploads/zigzagv3VS/'
def contacts(rep):
    t=md.load(U+f'mdcath/extracted/5sicI00/5sicI00_450_{rep}.xtc',top=U+'mdcath/extracted/5sicI00/5sicI00.pdb')
    X=t.xyz[:,t.top.select('name CA')]*10; n=X.shape[1]
    D=np.linalg.norm(X[:,:,None]-X[:,None],axis=-1); seq=np.abs(np.arange(n)[:,None]-np.arange(n)[None])
    return (D<8)&(seq>4),n
def resframe(events_by_rep):
    out=[]
    for rep,E in events_by_rep.items():
        A,n=contacts(rep); C=np.zeros((500,n))
        for fr,idx in E:
            for x in idx: C[fr,x]+=1
        cnt=A.sum(2)
        for f in range(1,500):
            flux=(A[f]^A[f-1]).sum(1)
            for i in range(n): out.append((rep,f,i,C[f,i],flux[i],abs(cnt[f,i]-cnt[f-1,i]),cnt[f-1,i]))
    R=pd.DataFrame(out,columns=['rep','f','i','crit','flux','adc','count']); R['rf']=R.rep*1000+R.f; return R
def fits(R):
    m=smf.poisson('crit~flux+adc+count',R).fit(disp=0)
    disp=float((m.resid_pearson**2).sum()/m.df_resid)
    cr=smf.poisson('crit~flux+adc+count',R).fit(disp=0,cov_type='cluster',cov_kwds={'groups':R.i})
    cf=smf.poisson('crit~flux+adc+count',R).fit(disp=0,cov_type='cluster',cov_kwds={'groups':R.rf})
    nb=smf.negativebinomial('crit~flux+adc+count',R).fit(disp=0,maxiter=200)
    nbc=smf.negativebinomial('crit~flux+adc+count',R).fit(disp=0,maxiter=200,cov_type='cluster',cov_kwds={'groups':R.i})
    return dict(dispersion=disp,
      poisson_z=m.tvalues[['flux','adc']].round(1).to_dict(),
      poisson_cluster_residue_z=cr.tvalues[['flux','adc']].round(1).to_dict(),
      poisson_cluster_frame_z=cf.tvalues[['flux','adc']].round(1).to_dict(),
      negbin_z=nb.tvalues[['flux','adc']].round(1).to_dict(),
      negbin_cluster_residue_z=nbc.tvalues[['flux','adc']].round(1).to_dict(),
      negbin_alpha=float(nb.params['alpha']),
      irr_flux=float(np.exp(nb.params['flux'])),irr_adc=float(np.exp(nb.params['adc'])))
if __name__=='__main__':
    ev={}
    for rep in range(5):
        E=pd.read_csv(U+f'mdcath_5sicI00_450K_r0_frame240_250_phenomenon/events_full_{rep}.csv')
        ev[rep]=[(r.frame,[r.i,r.j]+([r.k] if r.k>=0 else [])) for r in E.itertuples()]
    R=resframe(ev); print(len(R)); import json; print(json.dumps(fits(R),indent=1))
