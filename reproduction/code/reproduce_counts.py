"""Reproduce primary counts and run/window correlations from saved barcodes.
No coordinates are used and no molecular dynamics are rerun.
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'data/run_summary.csv').open()))
cohort={r['protein']:r['cohort'] for r in rows}
counts=defaultdict(lambda:dict(runs=0,occupied=0,hidden=0,runs_with_hidden=0))
allruns=[];allwindows=[];constant_one_hidden=0
for path in sorted((ROOT/'data/global').glob('*/eps6_deb3/*.json')):
 r=json.loads(path.read_text()); key=(cohort[r['protein']],r['temperature'])
 occ=[w for w in r['windows'] if w['occupancy_continuous']]
 allruns.append((r['protein'],np.mean([w['spanning'] for w in occ]) if occ else 0,np.mean(r['Q'])))
 if r['temperature']>320 and r['qualifying']:
  allwindows.extend((w['spanning'],w['Q_mean']) for w in occ)
 if r['temperature']!=320 and not r['qualifying']:continue
 c=counts[key];c['runs']+=1
 c['occupied']+=sum(w['occupancy_continuous'] for w in r['windows'])
 c['hidden']+=sum(w['hidden_replacement'] for w in r['windows'])
 c['runs_with_hidden']+=any(w['hidden_replacement'] for w in r['windows'])
 constant_one_hidden+=sum(w['hidden_replacement'] and w['constant_one'] for w in r['windows'])
# The archived cohort labels are inspected rather than guessed.
for k,v in counts.items(): print(k,v)
print('cohorts',sorted(set(cohort.values())))
for t,hidden,occupied in [(450,742,1269),(320,4,1750),(413,156,663)]:
 c=[v for k,v in counts.items() if k[1]==t]
 assert sum(x['hidden'] for x in c)==hidden and sum(x['occupied'] for x in c)==occupied
assert constant_one_hidden==0
runrho=float(spearmanr([r[1] for r in allruns],[r[2] for r in allruns]).statistic)
wrho=float(spearmanr(*zip(*allwindows)).statistic)
by=[]
for protein in sorted({r[0] for r in allruns}):
 rr=[r for r in allruns if r[0]==protein]
 by.append(float(spearmanr([r[1] for r in rr],[r[2] for r in rr]).statistic))
report={'counts':{str(k):v for k,v in counts.items()},'runs_in_correlation':len(allruns),'run_spearman':runrho,'median_within_domain_run_spearman':float(np.median(by)),'all_window_spearman':wrho,'constant_one_hidden_windows':constant_one_hidden}
assert len(allruns)==175 and len(allwindows)==1932 and round(runrho,2)==.79 and round(wrho,2)==.58
(ROOT/'reproduced/primary_counts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
