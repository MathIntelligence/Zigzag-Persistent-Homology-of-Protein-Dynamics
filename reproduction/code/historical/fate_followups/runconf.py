import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'/home/claude/conf'); from cont import analyze
I='/mnt/user-data/uploads/zigzagv3VS/mdcath_order_targeted_10/input/'
p=sys.argv[1]; rows=[]
ref=np.load(I+f'{p}_320_0_ca.npy')[0]
for T in [413,450]:
  for rep in range(5):
    X=np.load(I+f'{p}_{T}_{rep}_ca.npy')
    for eps in [5.0,6.0]:
      r=analyze(X,ref,eps)
      for w in r['windows']: rows.append(dict(protein=p,temperature=T,replica=rep,eps=eps,debounce=3,qualifying=r['qualifying'],**w))
pd.DataFrame(rows).to_csv(f'win_{p}.csv',index=False); print(p,'done',flush=True)
