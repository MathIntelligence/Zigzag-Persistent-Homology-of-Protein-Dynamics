"""Per-window features for the confirmatory cohort (see PROTOCOL.md). Usage: python compute.py CA_DIR OUT_DIR [NPROC]"""
import sys, os, glob, json, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import zzlib, trk
CA, OUT = sys.argv[1], sys.argv[2]
L = 20

def job(args):
    dom, key = args
    f = os.path.join(OUT, f'{dom}_{key}.json')
    if os.path.exists(f): return f
    z = np.load(os.path.join(CA, dom + '.npz')); X = z[key].astype(float); ref = z['ref'].astype(float)
    T, rep = map(int, key.split('_'))
    D = np.linalg.norm(ref[:, None] - ref[None], axis=-1); i, j = np.where(np.triu(D < 8, 4))
    q = (np.linalg.norm(X[:, i] - X[:, j], axis=-1) < 8).mean(1)
    st = np.flatnonzero(np.convolve((q < .2).astype(int), np.ones(20, int), 'valid') == 20)
    qual = bool(len(st) and q[0] >= .6)
    out = dict(dom=dom, T=T, rep=rep, frames=len(X), N=X.shape[1], qualifying=qual, windows=[])
    if qual:
        pa, pr = zzlib.presence(X); lo, hi, beta = zzlib.bars_from_presence(pa, pr, X.shape[1], X.shape[0])
        Ds = [trk.diagram(x) for x in X]
        alive = lambda t: (lo <= t) & (hi > t)
        for a in range(0, len(X) - L + 1, L):
            b = a + L; bt = beta[a:b].astype(float)
            inter = (lo < b) & (hi > a); ov = (np.minimum(hi, b) - np.maximum(lo, a))[inter] / L
            surv = [(alive(t) & alive(t+1)).sum() / max(alive(t).sum(), 1) for t in range(a, b-1)]
            births = int(((lo > a) & (lo < b)).sum()); deaths = int(((hi > a) & (hi < b)).sum())
            ac = 0.0
            if bt.std() > 0 and bt[:-1].std() > 0 and bt[1:].std() > 0: ac = float(np.corrcoef(bt[:-1], bt[1:])[0, 1])
            row = dict(protein=dom, T=T, rep=rep, start=a, Q=float(q[a:b].mean()),
                       occupied=bool(bt.min() > 0),
                       beta_min=bt.min(), beta_max=bt.max(), beta_mean=bt.mean(), beta_sd=bt.std(), beta_ac1=ac,
                       beta_absdiff=float(np.abs(np.diff(bt)).mean()),
                       zz_spanning=int(((lo <= a) & (hi >= b)).sum()),
                       zz_retained_frac=float(((lo <= a) & (hi >= b)).sum() / max(alive(a).sum(), 1)),
                       zz_turnover=(births + deaths) / L, zz_classes=int(inter.sum()),
                       zz_max_overlap=float(ov.max()) if len(ov) else 0.0,
                       zz_mean_overlap=float(ov.mean()) if len(ov) else 0.0,
                       zz_mean_lifetime=float((hi - lo)[inter].mean()) if inter.any() else 0.0,
                       zz_step_survival=float(np.mean(surv)))
            row['spanning'] = row['zz_spanning']
            row.update(trk.all_features(Ds[a:b]))
            out['windows'].append(row)
    json.dump(out, open(f, 'w'))
    return f

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for f in sorted(glob.glob(os.path.join(CA, '*.npz'))):
        dom = os.path.basename(f)[:-4]; jobs += [(dom, k) for k in np.load(f).files if k != 'ref']
    if len(sys.argv) > 4: jobs = jobs[:int(sys.argv[4])]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    print(len(jobs), 'runs on', n, 'processes', flush=True)
    with Pool(n) as p:
        for k, r in enumerate(p.imap_unordered(job, jobs, chunksize=1)): print(k + 1, r, flush=True)
    print('DONE', flush=True)
