"""Diagram-only tracker: a framewise-PH imitation of zigzag continuity.

For every frame, compute the Rips H1 persistence diagram of the C-alpha cloud (gudhi, max edge 10 A, max dim 2).
Consecutive diagrams are matched by the optimal Wasserstein-1 matching (L-inf ground metric, diagonal allowed).
Matched points are chained. No residue labels are used: only (birth, death) coordinates.
A chain is "present at eps" in a frame when its point satisfies birth <= eps < death (eps = 6 A), i.e. it is one
of the beta_1 classes of the eps-complex in that frame.

Two variants:
  full : match whole diagrams, then read presence at eps (the tracker may follow a point through frames where it
         is not alive at eps, which breaks the run).
  eps  : match only the sub-diagrams of points alive at eps.

Window features (non-overlapping 20-frame windows, same windows as the zigzag analysis):
  trk_spanning      number of chains present at eps in every frame of the window   (analogue of zz spanning)
  trk_step_survival mean over steps of fraction of eps-present points matched to an eps-present point
  trk_mean_overlap  mean over eps-present chain runs of (run length / 20)          (analogue of zz mean overlap)
  trk_mean_life     mean eps-present run length (frames)
  trk_max_overlap   longest run / 20
"""
import json, sys, numpy as np, gudhi
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool

EPS = 6.0; MAXE = 10.0; WIN = 20
BASE = Path('mdcath_order_added_value'); ROOT = Path('.')
SOURCES = {p: ROOT/s for s, ps in (('mdcath_order_extension_5', ['2d1pA00', '2d4uB00', '2dawA01']),
          ('mdcath_order_search_batch_01', ['2ervA00', '2fd5A02']),
          ('mdcath_order_targeted_10', ['4a4aA02', '2ivxB01', '3hwcA01', '3hbvP01'])) for p in ps}
CROSS = {'2ervA00': ROOT/'mdcath_order_cross_temperature_all15', '3hwcA01': ROOT/'mdcath_order_targeted_cross_temperature',
         '3hbvP01': ROOT/'mdcath_order_targeted_cross_temperature'}
VAL = BASE/'validation'
def src(p, T):
    if p in SOURCES: return CROSS[p] if T == 413 else SOURCES[p]
    return VAL

def diagram(X):
    st = gudhi.RipsComplex(points=X, max_edge_length=MAXE).create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2, min_persistence=0)
    D = st.persistence_intervals_in_dimension(1)
    if len(D) == 0: return np.zeros((0, 2))
    D = np.array(D, float); D[~np.isfinite(D[:, 1]), 1] = MAXE
    return D

def match(A, B):
    """Optimal W1 matching with diagonal (L-inf ground). Returns dict i->j for off-diagonal matches."""
    n, m = len(A), len(B)
    if n == 0 or m == 0: return {}
    C = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    dA = (A[:, 1] - A[:, 0]) / 2; dB = (B[:, 1] - B[:, 0]) / 2
    big = 1e9
    M = np.zeros((n + m, m + n))
    M[:n, :m] = C
    M[:n, m:] = big; M[np.arange(n), m + np.arange(n)] = dA          # A_i -> diagonal
    M[n:, :m] = big; M[n + np.arange(m), np.arange(m)] = dB          # diagonal -> B_j
    r, c = linear_sum_assignment(M)
    return {i: j for i, j in zip(r, c) if i < n and j < m}

def alive(D): return (D[:, 0] <= EPS) & (D[:, 1] > EPS) if len(D) else np.zeros(0, bool)

def window_features(Ds, variant):
    L = len(Ds)
    if variant == 'eps':
        Ds = [D[alive(D)] for D in Ds]
    al = [alive(D) for D in Ds]
    maps = [match(Ds[t], Ds[t+1]) for t in range(L-1)]
    # chains of eps-present runs: follow i at t -> maps[t][i] while both alive
    runs = []; started = [set() for _ in range(L)]
    for t in range(L):
        for i in np.flatnonzero(al[t]):
            if i in started[t]: continue
            k, tt = i, t; length = 1; started[t].add(i)
            while tt < L-1 and k in maps[tt] and al[tt+1][maps[tt][k]]:
                k = maps[tt][k]; tt += 1; length += 1; started[tt].add(k)
            runs.append((t, length))
    surv = []
    for t in range(L-1):
        idx = np.flatnonzero(al[t])
        if len(idx): surv.append(np.mean([(i in maps[t]) and al[t+1][maps[t][i]] for i in idx]))
    lens = np.array([l for _, l in runs]) if runs else np.zeros(0)
    return dict(trk_spanning=int(sum(1 for s, l in runs if s == 0 and l == L)),
                trk_step_survival=float(np.mean(surv)) if surv else 0.0,
                trk_mean_overlap=float(lens.mean()/L) if len(lens) else 0.0,
                trk_mean_life=float(lens.mean()) if len(lens) else 0.0,
                trk_max_overlap=float(lens.max()/L) if len(lens) else 0.0,
                fw_beta_eps_min=int(min(a.sum() for a in al)))

def run(job):
    f = Path(job); p = f.parts[-3]; T, rep = map(int, f.stem.split('_'))
    r = json.loads(f.read_text())
    X = np.load(src(p, T)/'input'/f'{p}_{T}_{rep}_ca.npy').astype(float)
    Ds = [diagram(x) for x in X]
    rows = []
    for w in r['windows']:
        a, b = w['start'], w['stop']
        row = dict(protein=p, T=T, rep=rep, start=a)
        for v in ('full', 'eps'):
            for k, val in window_features(Ds[a:b], v).items():
                row[f'{k}_{v}' if k != 'fw_beta_eps_min' else k] = val
        rows.append(row)
    out = Path('tracker_runs')/f'{p}_{T}_{rep}.json'; out.write_text(json.dumps(rows))
    # also keep diagrams for reuse
    np.savez_compressed(Path('tracker_runs')/f'{p}_{T}_{rep}_diagrams.npz', *Ds)
    return job

if __name__ == '__main__':
    Path('tracker_runs').mkdir(exist_ok=True)
    jobs = []
    for f in sorted(BASE.glob('global/*/eps6_deb3/*.json')):
        T = int(f.stem.split('_')[0])
        if T == 320: continue
        r = json.loads(f.read_text())
        if not r['qualifying']: continue
        if (Path('tracker_runs')/f'{f.parts[-3]}_{f.stem}.json').exists(): continue
        jobs.append(str(f))
    if len(sys.argv) > 1: jobs = jobs[:int(sys.argv[1])]
    print('jobs', len(jobs), flush=True)
    with Pool(2) as pool:
        for j in pool.imap_unordered(run, jobs): print('done', j, flush=True)
