"""Independent F2 limit-to-colimit verification of the worked full-complex window."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'historical'))
from independent_rank import rank_from_edge_history
from zigzag_dynamics.filtration import _edge_presence,_debounce_rows
A=Path(__file__).resolve().parents[1];X=np.load(A/'data/worked_example_ca.npy').astype(float)
pairs,pres=_edge_presence(X,6.,hysteresis=None);pres=_debounce_rows(pres,3)
result={}
for bridge in ['union','intersection']:
    started=time.perf_counter();rank,dims=rank_from_edge_history(X.shape[1],pairs,pres[:,380:400],bridge)
    assert rank==0 and min(dims[::2])==1 and max(dims[::2])==3
    result[bridge]={'spanning_rank':rank,'frame_dimensions':dims[::2],'verification_seconds':time.perf_counter()-started}
(A/'reproduced/independent_worked_example.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
