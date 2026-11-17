"""Diagram-only tracker (identical logic to framewise_tracker/tracker.py + gated.py, without monkey-patching)."""
import numpy as np, gudhi
from scipy.optimize import linear_sum_assignment
EPS = 6.0; MAXE = 10.0

def diagram(X):
    st = gudhi.RipsComplex(points=X, max_edge_length=MAXE).create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2, min_persistence=0)
    D = st.persistence_intervals_in_dimension(1)
    if len(D) == 0: return np.zeros((0, 2))
    D = np.array(D, float); D[~np.isfinite(D[:, 1]), 1] = MAXE
    return D

def match(A, B, gate=None):
    n, m = len(A), len(B)
    if n == 0 or m == 0: return {}
    C = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    dA = (A[:, 1] - A[:, 0]) / 2; dB = (B[:, 1] - B[:, 0]) / 2
    big = 1e9; M = np.zeros((n + m, m + n))
    M[:n, :m] = C
    M[:n, m:] = big; M[np.arange(n), m + np.arange(n)] = dA
    M[n:, :m] = big; M[n + np.arange(m), np.arange(m)] = dB
    r, c = linear_sum_assignment(M)
    out = {i: j for i, j in zip(r, c) if i < n and j < m}
    if gate is not None: out = {i: j for i, j in out.items() if C[i, j] <= gate}
    return out

def alive(D): return (D[:, 0] <= EPS) & (D[:, 1] > EPS) if len(D) else np.zeros(0, bool)

def window_features(Ds, variant, gate=None):
    L = len(Ds)
    if variant == 'eps': Ds = [D[alive(D)] for D in Ds]
    al = [alive(D) for D in Ds]
    maps = [match(Ds[t], Ds[t+1], gate) for t in range(L-1)]
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
                trk_max_overlap=float(lens.max()/L) if len(lens) else 0.0)

VARIANTS = [('full', None, ''), ('eps', None, '')] + [(v, g, f'_g{g}') for g in (0.25, 0.5, 1.0) for v in ('full', 'eps')]
def all_features(Ds):
    row = {}
    for v, g, suf in VARIANTS:
        for k, val in window_features(Ds, v, g).items(): row[f'{k}_{v}{suf}'] = val
    return row
