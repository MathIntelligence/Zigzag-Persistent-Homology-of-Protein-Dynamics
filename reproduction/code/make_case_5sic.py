from pathlib import Path
import numpy as np, matplotlib, dionysus as d, pickle
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'figures'
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from case_barcode import run
X=np.load(ROOT/'data/5sic_450_rep0_ca.npy'); res=np.arange(7,114); n=107
# ---------- framewise diagrams ----------
def dgm(P):
    f=d.fill_rips(P.astype(float),2,14.)
    h=d.homology_persistence(f)
    z=d.init_diagrams(h,f)[1]
    return np.array([[p.birth,p.death] for p in z])
frames=[244,245,246,247]
D={f:dgm(X[f]) for f in frames}
# ---------- zigzag bars on window 236-256 (eps 8, as in case study) ----------
f0,f1=236,256
bars=run(0,f0,f1,eps=8.0,X=X)
# convert to frame intervals: first/last checkpoint frames
iv=[]
for b in bars:
    s=b['bf'] if b['bf'] is not None else f0
    e=b['df'] if b['df'] is not None else f1+1
    iv.append((s,e))
iv=sorted(iv,key=lambda x:(x[0],x[1]))
pickle.dump((D,iv),open(ROOT/'reproduced/case_pics_data.pkl','wb'))
# ---------- snapshots: align on core (res>=23) to frame 240 ----------
core=res>=23
def kabsch(P,Q):
    Pc=P-P[core].mean(0); Qc=Q-Q[core].mean(0)
    H=Pc[core].T@Qc[core]; U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T; return (R@Pc.T).T
ref=X[240]-X[240][core].mean(0)
w,v=np.linalg.eigh(np.cov(ref.T)); axes=v[:,::-1][:,:2]
snaps=[241,243,246,249,251]
all_xy=np.concatenate([kabsch(X[f],X[240])@axes for f in snaps])
xy_min=all_xy.min(axis=0)-3;xy_max=all_xy.max(axis=0)+3
BLUE,ORANGE,GRAY,YEL='#2a78d6','#eb6834','#b9b8b2','#eda100'

# Use three independently laid-out panels so labels remain legible at manuscript width.
plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
fig=plt.figure(figsize=(7.2,6.8),layout='constrained')
sub=fig.subfigures(3,1,height_ratios=[1.1,1,1.3])
sub[0].suptitle('A  N-terminal segment (orange); contact partners numbered ≥27 are blue',
                 x=.01,ha='left',fontsize=9)
N=res<=22
for ax,f in zip(sub[0].subplots(1,5),snaps):
    P=kabsch(X[f],X[240])@axes
    Dm=np.linalg.norm(X[f][:,None]-X[f][None],axis=-1)
    partners=(Dm[N]<8).any(0)&(res>26)
    seg=np.stack([P[:-1],P[1:]],1)
    cols=[ORANGE if N[i] and N[i+1] else GRAY for i in range(n-1)]
    ax.add_collection(LineCollection(seg,colors=cols,linewidths=1))
    ax.scatter(P[partners,0],P[partners,1],s=10,color=BLUE,zorder=3)
    ax.scatter(P[N,0],P[N,1],s=5,color=ORANGE,zorder=4)
    ax.set_aspect('equal');ax.set_xlim(xy_min[0],xy_max[0]);ax.set_ylim(xy_min[1],xy_max[1]);ax.axis('off')
    pr=res[partners];chunks=[]
    for v in pr:
        if not chunks or v-chunks[-1][-1]>3:chunks.append([int(v)])
        else:chunks[-1].append(int(v))
    lab=', '.join(str(c[0]) if len(c)==1 else f'{c[0]}–{c[-1]}' for c in chunks)
    if len(lab)>18:lab=lab.replace(', ', ',\n',1)
    ax.set_title(f'Frame {f}\n'+('docked on\n'+lab if lab else 'released'),fontsize=7.5)

def wasserstein(A,B):
    assert np.isfinite(A).all() and np.isfinite(B).all()
    n,m=len(A),len(B);cost=np.full((n+m,n+m),np.inf)
    cost[:n,:m]=np.max(np.abs(A[:,None,:]-B[None,:,:]),axis=2)
    cost[np.arange(n),m+np.arange(n)]=(A[:,1]-A[:,0])/2
    cost[n+np.arange(m),np.arange(m)]=(B[:,1]-B[:,0])/2
    cost[n:,m:]=0
    ii,jj=linear_sum_assignment(cost);return cost[ii,jj].sum()
sub[1].suptitle('B  Framewise diagrams; window-mean distance is at the 45th percentile',x=.01,ha='left',fontsize=9)
for k,ax in enumerate(sub[1].subplots(1,4)):
    f=frames[k];v=D[f]
    ax.scatter(v[:,0],v[:,1],s=8,color=['#2a78d6','#1baf7a','#eb6834','#4a3aa7'][k],alpha=.85)
    ax.plot([3.5,11.5],[3.5,11.5],'--',color='.4',lw=.6)
    ax.set(xlim=(3.5,11.5),ylim=(3.5,11.5),xticks=[4,6,8,10],yticks=[4,6,8,10])
    ax.set_aspect('equal');ax.tick_params(labelsize=7)
    ax.set_xlabel('Birth (Å)',fontsize=8)
    if k==0:ax.set_ylabel('Death (Å)',fontsize=8)
    else:ax.set_yticklabels([])
    title=f'Frame {f}'
    if k<3:title+=f'\n$W_1$ to next = {wasserstein(v,D[frames[k+1]]):.1f} Å'
    else:title+='\n '
    ax.set_title(title,fontsize=7.5)
    for name in ['top','right']:ax.spines[name].set_visible(False)
sub[2].suptitle('C  Zigzag barcode, frames 236–256 (ε = 8 Å)\nWindow 240–250: 39 births, 39 deaths; one spanning class',x=.01,ha='left',fontsize=9)
ax=sub[2].subplots()
for k,(st,en) in enumerate(iv):
    long=st<=240 and en>=251
    color=BLUE if long else ('#7fb0e8' if en-st>=3 else '#a8a7a1')
    ax.plot([st-.45,en-.55],[k,k],color=color,lw=2 if long else .8,solid_capstyle='butt')
ax.axvspan(239.5,250.5,color=YEL,alpha=.15,lw=0)
ax.set_xlim(f0-.5,f1+.5);ax.set_yticks([])
ax.set_xlabel('Frame index (1 ns spacing)',fontsize=8);ax.tick_params(labelsize=7)
for name in ['top','right','left']:ax.spines[name].set_visible(False)
nspan=sum(st<=240 and en>=251 for st,en in iv)
births=sum(240<st<=250 for st,en in iv);deaths=sum(240<en<=250 for st,en in iv)
assert (nspan,births,deaths)==(1,39,39)
fig.savefig(OUT/'case_5sic_pictures.pdf');fig.savefig(OUT/'case_5sic_pictures.png',dpi=180)
print(len(iv),nspan,births,deaths)
