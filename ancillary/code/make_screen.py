"""Retain the author's original two-row screening design; separate overlapping labels."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/zigzag-arxiv-mpl')
from pathlib import Path
import csv,math
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'figures';DATA=ROOT/'data/screen_5sic'
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
summary={int(float(r['temperature_K'])):r for r in csv.DictReader((DATA/'condition_summary.csv').open())}
fig=plt.figure(figsize=(12,5.2),layout='constrained');outer=fig.add_gridspec(1,2,wspace=.05)
labels=[]
for col,T in enumerate([348,450]):
 g=outer[col].subgridspec(2,1,height_ratios=[1,1.35],hspace=.08)
 top=fig.add_subplot(g[0]);bottom=fig.add_subplot(g[1],sharex=top)
 color={348:'#1f77b4',450:'#d95f02'}[T]
 w=list(csv.DictReader((DATA/f'{T}K_framewise_W1.csv').open()));bars=list(csv.DictReader((DATA/f'{T}K_zigzag_H1_bars.csv').open()))
 x=np.array([int(r['transition']) for r in w]);W=np.array([float(r['W1_H1']) for r in w])
 b=np.array([float(r['birth']) for r in bars]);d=np.array([float(r['death']) for r in bars]);steps=np.array([int(r['from_frame']) for r in w])
 E=np.array([((b>i)&(b<=i+1)).sum()+((d>i)&(d<=i+1)).sum() for i in steps])
 cut=float(summary[T]['low_W1_threshold']);mask=(W<=cut)&(E>0)
 top.plot(x,W,color=color,lw=1.35);top.axhline(cut,color='#666666',ls='--',lw=.9)
 for idx,k in enumerate(np.flatnonzero(mask)):
  top.scatter(x[k],W[k],s=24+7*E[k],color='#b2182b',edgecolors='white',lw=.45,zorder=4)
  dy=8 if idx%2==0 else -11
  offsets={348:{3:(-6,8),4:(6,8),9:(0,8),13:(-5,8),14:(5,-11),25:(0,-11),27:(-6,8),28:(6,-11),31:(0,8),34:(0,-11),38:(-5,-11),39:(5,8)},450:{3:(0,8),5:(-5,-11),6:(5,8),7:(0,-11),8:(5,8),13:(0,-11),21:(0,8),25:(-3,-11),26:(5,8),30:(0,-11),32:(0,8),35:(0,-11),38:(0,8)}}
  dx,dy=offsets[T].get(int(x[k]),(0,dy))
  a=top.annotate(f'E={E[k]}',(x[k],W[k]),xytext=(dx,dy),textcoords='offset points',ha='center',
                  va='bottom' if dy>0 else 'top',fontsize=8.5,color='#8b0015',
                  bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.15},
                  arrowprops={'arrowstyle':'-','color':'#a65a63','lw':.45,'shrinkA':2,'shrinkB':4})
  labels.append((top,a,dy))
 top.set(xlim=(0,50),ylim=(max(0,W.min()-.28*np.ptp(W)),W.max()+.18*np.ptp(W)),ylabel=r'$W_1$ (Å)')
 top.set_title(f'5sicI00 — {T} K  |  cutoff={cut:.2f}, candidates={mask.sum()}',fontsize=10)
 top.grid(alpha=.18,lw=.6);top.tick_params(labelbottom=False,labelsize=8)
 vals=[((50 if not np.isfinite(dd) else dd)-bb,bb,50 if not np.isfinite(dd) else dd) for bb,dd in zip(b,d)]
 vals.sort(key=lambda v:(-v[0],v[1]))
 for y,(_,bb,dd) in enumerate(vals):bottom.hlines(y,bb,dd,color=color,lw=1)
 for xx in x[mask]:bottom.axvline(xx,color='#b2182b',alpha=.24,lw=.8,ls='--')
 bottom.set(xlim=(0,50),ylim=(-1,len(vals)),xlabel='Downsampled frame / transition',ylabel=f'Zigzag H1 bars ({len(vals)})')
 bottom.grid(axis='x',alpha=.18,lw=.6);bottom.tick_params(labelsize=8)
fig.savefig(OUT/'5sic_w1_vs_zigzag.pdf');fig.savefig(OUT/'5sic_w1_vs_zigzag.png',dpi=180)
print('Original colours, scale ranges, marker sizes and duration-sorted barcodes retained; label overlap resolved.')
