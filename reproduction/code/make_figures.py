"""Regenerate the corrected screen and a deterministic full-complex example."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/zigzag-arxiv-mpl')
from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from zigzag_dynamics.filtration import _edge_presence, _debounce_rows

ANC=Path(__file__).resolve().parents[1]
OUT=ANC.parents[0]/'figures'
DATA=ANC/'data'
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
blue='#267ca5';orange='#d86f13'
import runpy
runpy.run_path(str(Path(__file__).with_name('make_screen.py')),run_name='__main__')

W=pd.read_csv(DATA/'window_summary.csv')
candidates=W[(W.eps==6)&(W.debounce==3)&(W.temperature>=413)&W.qualifying&W.hidden_replacement]
r=candidates.sort_values(['max_beta','min_beta','protein','temperature','replica','start']).iloc[0]
assert (r.protein,int(r.temperature),int(r.replica),int(r.start))==('2fd5A02',450,1,380)
z=json.loads((DATA/'global/2fd5A02/eps6_deb3/450_1.json').read_text())
X=np.load(DATA/'worked_example_ca.npy').astype(float)
pairs,pres=_edge_presence(X,6.,hysteresis=None);pres=_debounce_rows(pres,3)
lo=np.array(z['lo']);hi=np.array(z['hi']);a,b=380,400
assert np.min(z['beta'][a:b])==1 and np.max(z['beta'][a:b])==3
assert not ((lo<=a)&(hi>=b)).any()

def insert(v,basis):
    while v:
        k=v.bit_length()-1
        if k in basis:v^=basis[k]
        else:basis[k]=v;return True
    return False

def nonzero_cycle(t):
    g=nx.Graph();g.add_nodes_from(range(X.shape[1]));g.add_edges_from(pairs[pres[:,t]])
    edges=sorted(tuple(sorted(e)) for e in g.edges);index={e:i for i,e in enumerate(edges)}
    boundary={}
    for i in g:
        for j in sorted(v for v in g[i] if v>i):
            for k in sorted(v for v in set(g[i])&set(g[j]) if v>j):
                v=sum(1<<index[e] for e in [(i,j),(i,k),(j,k)]);insert(v,boundary)
    cycles=sorted(nx.cycle_basis(g),key=lambda c:(len(c),sorted(c)))
    for c in cycles:
        es=[tuple(sorted((c[i],c[(i+1)%len(c)]))) for i in range(len(c))]
        v=sum(1<<index[e] for e in es)
        if insert(v,dict(boundary)):
            assert all(sum(i in e for e in es)%2==0 for i in c)
            return c
    raise AssertionError('No nonzero class')

fig=plt.figure(figsize=(7,5.5),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[.65,1,1.6])
ax=fig.add_subplot(gs[0,:]);ax.step(np.arange(a,b),z['beta'][a:b],where='mid',color=blue)
ax.set(xlim=(a-.5,b-.5),ylim=(0,3.7),yticks=[0,1,2,3],xticks=[380,385,390,395,399],ylabel=r'$\beta_1$',title='A  Occupancy throughout; no class spans frames 380–399')
ax=fig.add_subplot(gs[1,:]);inds=np.flatnonzero((lo<b)&(hi>a));inds=sorted(inds,key=lambda j:(lo[j],hi[j]))
for y,j in enumerate(inds):
    ax.plot([max(lo[j],a)-.45,min(hi[j],b)-.55],[y,y],lw=2,color=blue,solid_capstyle='butt')
ax.axvline(a-.5,color='0.5',ls='--',lw=.7);ax.axvline(b-.5,color='0.5',ls='--',lw=.7)
ax.set(xlim=(a-.5,b-.5),xlabel='Saved frame index (1 ns spacing)',ylabel='Interval',yticks=[],xticks=[380,385,390,395,399])
ax.text(.01,.96,'B  All intervals intersecting this window',transform=ax.transAxes,va='top',fontsize=8)
cycles={}
center=X[a].mean(0);_,_,v=np.linalg.svd(X[a]-center,full_matrices=False)
for col,t in enumerate([380,389,399]):
    cycle=nonzero_cycle(t);cycles[str(t)]=[int(i+1) for i in cycle]
    # Rigidly align each frame to the first before using its common principal plane.
    A=X[t]-X[t].mean(0);B=X[a]-center;u,_,vt=np.linalg.svd(A.T@B)
    rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt;xy=A@rot@v[:2].T
    ax=fig.add_subplot(gs[2,col]);ax.plot(xy[:,0],xy[:,1],color='.8',lw=.7)
    c=cycle+[cycle[0]];ax.plot(xy[c,0],xy[c,1],'-o',color=orange,lw=1.5,ms=3)
    import textwrap
    names=', '.join(str(i+1) for i in sorted(cycle))
    ax.text(.5,-.10,'Cycle indices:\n'+'\n'.join(textwrap.wrap(names,24)),transform=ax.transAxes,ha='center',va='top',fontsize=7)
    ax.set_title(f'{"CDE"[col]}  Frame {t}',fontsize=9);ax.set_aspect('equal');ax.axis('off')
fig.savefig(OUT/'worked_replacement.pdf');fig.savefig(OUT/'worked_replacement.png',dpi=180);plt.close(fig)
(ANC/'reproduced/worked_example.json').write_text(json.dumps(dict(protein='2fd5A02',temperature=450,replica=1,start=a,stop=b,
    min_beta=1,max_beta=3,spanning=0,mean_Q=float(r.Q_mean),cycles_one_based_indices=cycles,
    cycle_scope='Independently chosen non-boundary cycles; not compatible barcode representatives.'),indent=2)+'\n')
print('Both figures regenerated; full-complex witness checked over F2.')
