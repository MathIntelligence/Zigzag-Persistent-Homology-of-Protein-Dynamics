import sys; from zz2 import run; import pandas as pd
from load import *
rep=int(sys.argv[1]); t=load(rep); res=[r.resSeq for r in t.top.residues]
C=run(rep,0,499); rows=[]
for c in C:
  for kind,fr,o in [('birth',c['bf'],c['bop']),('death',c['df'],c['dop'])]:
    if fr is None or o is None: continue
    a,s,ff=o
    rows.append(dict(rep=rep,frame=fr,kind=kind,op=a,dim=len(s)-1,i=s[0],j=s[1],k=(s[2] if len(s)>2 else -1),res=tuple(res[x] for x in s),life=c['life_frames']))
pd.DataFrame(rows).to_csv(f'events_full_{rep}.csv',index=False); print(rep,len(rows))
