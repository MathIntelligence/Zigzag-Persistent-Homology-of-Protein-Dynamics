"""Independent source, rank, and comparator checks for the added-value study."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import sys
import h5py
import numpy as np
from scipy.stats import kendalltau
import audit_order as a
from independent_rank import rank_from_edge_history
from test_audit import dionysus_h1
from zigzag_dynamics.filtration import _edge_presence,_debounce_rows

BASE=a.BASE;ROOT=a.ROOT;old=a.old
sys.path.insert(0,str(ROOT))
from extract_mdcath_ca_for_wasserstein import ca_indices


def source_for(p,temp):
    if p not in a.SOURCES:return BASE/'validation'
    return a.CROSS[p] if temp==413 else a.SOURCES[p]


def main():
    checks={}
    val=BASE/'validation'
    sel=json.loads((val/'selection.json').read_text())
    assert sel['protocol_sha256']==old.sha(BASE/'PROTOCOL_GLOBAL_VALIDATION.md')
    assert all(sel['raw_files_absent'].values())
    rank=ROOT/'mdcath_order_targeted_10/ranked_candidates.json'
    assert sel['ranking_sha256']==old.sha(rank)
    assert sel['selected']==[r['protein'] for r in json.loads(rank.read_text())[10:15]]
    manifest=list(csv.DictReader((val/'input/trajectory_manifest.csv').open()))
    assert len(manifest)==75
    for p in sel['selected']:
        with h5py.File(ROOT/'mdcath/data'/f'mdcath_dataset_{p}.h5') as f:
            idx=ca_indices(f,p)
            for row in [x for x in manifest if x['protein']==p]:
                path=val/'input'/row['array_file'];X=np.load(path)
                assert old.sha(path)==row['array_sha256']
                assert X.shape==(500,len(idx),3) and np.isfinite(X).all()
                np.testing.assert_array_equal(X,np.asarray(f[row['source_dataset']][:,idx,:],np.float32))
    checks['validation_source_arrays']=75
    # Recompute all original cross statistics directly using SciPy.
    cross=json.loads((BASE/'results/cross.json').read_text())
    for r in cross:
        p=r['protein'];method=r['method'];obs=r['observed_only']
        x=old.last_present(np.load(BASE/'results'/p/'413_traces.npz')[method])
        y=old.last_present(np.load(BASE/'results'/p/'450_traces.npz')[method])
        vals=[]
        for xx in x:
            for yy in y:
                good=np.ones(len(xx),bool)
                if obs:good=(xx>=0)&(xx<499)&(yy>=0)&(yy<499)
                vals.append(kendalltau(xx[good],yy[good]).statistic if good.sum()>=4 else np.nan)
        assert sum(np.isfinite(vals))==r['defined']
        if r['tau'] is None:assert not np.isfinite(vals).any()
        else:assert np.isclose(np.nanmean(vals),r['tau'])
    checks['cross_statistics_recomputed']=len(cross)
    rows=json.loads((BASE/'intersection/summary.json').read_text())
    # Pick deterministic examples per domain and parameter set, preserving all categories.
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['protein'],r['eps'],r['debounce'])].append(r)
    selected=[]
    for key,rs in sorted(grouped.items()):
        rs=sorted(rs,key=lambda r:(r['temperature'],r['replica'],r['start']))
        for name,predicate in [('replacement',lambda r:r['hidden_replacement']),
                                ('transport',lambda r:r['transport_without_fixed_cycle']),
                                ('survivor',lambda r:r['spanning']>0)]:
            candidate=next((r for r in rs if predicate(r)),None)
            if candidate:selected.append((name,candidate))
    cache={};verified=[]
    for kind,row in selected:
        p,temp,rep=row['protein'],row['temperature'],row['replica'];eps,deb=row['eps'],row['debounce']
        key=(p,temp,rep,eps,deb)
        if key not in cache:
            X=np.load(source_for(p,temp)/'input'/f'{p}_{temp}_{rep}_ca.npy').astype(float)
            pairs,pres=_edge_presence(X,eps,hysteresis=None);pres=_debounce_rows(pres,deb)
            cache={key:(len(X[0]),pairs,pres)}
        n,pairs,pres=cache[key];lo,hi=row['start'],row['stop']
        span,dims=rank_from_edge_history(n,pairs,pres[:,lo:hi])
        assert span==row['spanning'],(kind,key,lo,span,row['spanning'])
        assert min(dims[::2])==row['min_beta'] and max(dims[::2])==row['max_beta']
        intersection=pairs[pres[:,lo:hi].all(1)]
        assert dionysus_h1(n,intersection)==row['intersection_beta']
        verified.append(dict(kind=kind,protein=p,temperature=temp,replica=rep,eps=eps,debounce=deb,
            start=lo,stop=hi,barcode_spanning=row['spanning'],independent_limit_colimit_rank=span,
            independently_verified_intersection_beta=row['intersection_beta']))
        print('rank verified',kind,p,temp,rep,eps,deb,lo,span,flush=True)
    checks['independent_generalized_ranks']=len(verified)
    # Verify all saved source fingerprints and every-frame Betti check flags.
    globalfiles=list((BASE/'global').glob('*/*/*_*.json'))
    for path in globalfiles:
        r=json.loads(path.read_text());fp=r['fingerprint']
        assert fp['code']==old.sha(BASE/'global_continuity.py')
        assert fp['protocol']==old.sha(BASE/'PROTOCOL_GLOBAL_VALIDATION.md')
        assert fp['input']==old.sha(source_for(r['protein'],r['temperature'])/'input'/f'{r["protein"]}_{r["temperature"]}_{r["replica"]}_ca.npy')
        assert r['every_frame_independent_betti_verified']
    checks['global_trajectory_parameter_analyses']=len(globalfiles)
    checks['full_complex_frames_independently_checked']=len(globalfiles)*500
    checks['main_validation_runs']=sum(1 for path in globalfiles if path.parts[-3] in sel['selected'] and path.parent.name=='eps6_deb3')
    assert checks['main_validation_runs']==75
    old.write_json(BASE/'verification.json',dict(status='PASS',checks=checks,rank_examples=verified,
        code_sha256={name:old.sha(BASE/name) for name in ['verify_results.py','independent_rank.py','audit_order.py','global_continuity.py']}))
    print('INDEPENDENT AUDIT PASS',checks,flush=True)


if __name__=='__main__':main()
