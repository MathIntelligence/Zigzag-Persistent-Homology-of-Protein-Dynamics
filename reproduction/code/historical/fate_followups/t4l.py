import numpy as np, pandas as pd, sys, mdtraj as md
from itertools import permutations
sys.path.insert(0,'/home/claude/conf'); from cont2 import windows
U='/mnt/user-data/uploads/zigzagv3VS/fireprotdb_md_validation/'
ref=md.load(U+'2LZM.pdb'); ref=ref.atom_slice(ref.top.select('name CA and protein')); R=ref.xyz[0]*10
print('ref CA',R.shape)
ddg={'wild_type':0.0,'stabilizing_S117V':-2.0,'destabilizing_G30F':1.5}
rows=[]
for tag in ddg:
  for r in range(3):
    X=np.load(U+f'ca_coords/{tag}_r{r}_ca.npy')
    for eps,stride,L in [(5.0,20,20),(6.0,20,20),(5.0,5,80)]:
      W,q,beta=windows(X[::stride],R[:X.shape[1]],eps,L)
      occ=[w for w in W if w['min_beta']>0]
      C=np.mean([w['spanning']/w['min_beta'] for w in occ]) if occ else np.nan
      rep=np.mean([w['spanning']==0 for w in occ]) if occ else np.nan
      rows.append(dict(variant=tag,run=r,ddg=ddg[tag],eps=eps,stride=stride,L=L,C=C,replacement=rep,n_occ=len(occ),n_win=len(W),
                       mean_beta=float(beta.mean()),Q=float(q.mean())))
      print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('t4l_runs.csv',index=False)
