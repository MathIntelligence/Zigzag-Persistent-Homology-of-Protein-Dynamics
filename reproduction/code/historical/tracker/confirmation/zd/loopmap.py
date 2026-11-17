"""Loop-lifetime map: residue-resolved kinetics from zigzag H1 episodes.

An *episode* is an H1 bar with lifetime >= ``min_episode_ns`` together with
the residue support of one live representative cycle.  Per residue we report

* occupancy O(r): fraction of frames in which r belongs to an alive episode;
* lifetime  L(r): sum l^2 / sum l over episodes containing r (ns);
* n_episodes(r).

Episodes are additionally chained into *lineages* (a re-formed loop linked to
its predecessor when their supports overlap by Jaccard >= ``lineage_jaccard``
and the successor is born after the predecessor dies).  Lineage-level totals
are reported alongside but the per-residue statistics use episodes.

Controls computed on the same residue axis: C-alpha RMSF after superposition
to the trajectory mean, and native-contact occupancy Q(r).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filtration import build_filtration
from .representatives import recover_representatives
from .zigzag import Bars, compute_bars


# ----------------------------------------------------------------------- #
# geometry controls
# ----------------------------------------------------------------------- #
def kabsch_superpose(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Superpose every frame of X (T,N,3) onto ref (N,3); returns aligned copy."""
    out = np.empty_like(X, dtype=float)
    ref_c = ref - ref.mean(axis=0)
    for t in range(X.shape[0]):
        P = X[t] - X[t].mean(axis=0)
        H = P.T @ ref_c
        U, _S, Vt = np.linalg.svd(H)
        dsign = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, dsign])
        R = Vt.T @ D @ U.T
        out[t] = P @ R.T + ref.mean(axis=0)
    return out


def rmsf(X: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """Per-residue RMSF (Å) after iterative superposition to the mean."""
    ref = X[0].astype(float)
    for _ in range(n_iter):
        A = kabsch_superpose(X, ref)
        ref = A.mean(axis=0)
    A = kabsch_superpose(X, ref)
    return np.sqrt(((A - ref) ** 2).sum(axis=2).mean(axis=0))


def native_contacts(ref: np.ndarray, cutoff: float = 8.0, min_seq_sep: int = 4) -> np.ndarray:
    N = ref.shape[0]
    D = np.linalg.norm(ref[:, None] - ref[None], axis=2)
    I, J = np.triu_indices(N, k=min_seq_sep)
    m = D[I, J] < cutoff
    return np.stack([I[m], J[m]], axis=1)


def contact_occupancy(X: np.ndarray, pairs: np.ndarray, cutoff: float = 8.0, tol: float = 1.0) -> np.ndarray:
    """Q(r): mean over frames of the fraction of r's native contacts present
    (present = distance < tol * cutoff; tol = 1.0 keeps the cutoff strict so that
    fluctuations of a folded protein register)."""
    N = X.shape[1]
    d = np.linalg.norm(X[:, pairs[:, 0]] - X[:, pairs[:, 1]], axis=2)     # (T, P)
    present = (d < tol * cutoff).astype(float)
    per_res = np.zeros(N)
    counts = np.bincount(np.concatenate([pairs[:, 0], pairs[:, 1]]), minlength=N)
    frac = present.mean(axis=0)                                             # per pair
    np.add.at(per_res, pairs[:, 0], frac)
    np.add.at(per_res, pairs[:, 1], frac)
    return np.divide(per_res, counts, out=np.full(N, np.nan), where=counts > 0)


# ----------------------------------------------------------------------- #
# episodes and lineages
# ----------------------------------------------------------------------- #
@dataclass
class Episode:
    bar: int
    lo: int
    hi: int
    lifetime_ns: float
    support: tuple[int, ...]
    lineage: int = -1


def geometric_support(X: np.ndarray, snapshots: list[dict], radius: float = 5.0) -> set[int]:
    """Residues whose C-alpha lies within `radius` of any representative edge
    midpoint, in the snapshot's own frame — a path-independent notion of
    'residues on the loop' that removes the every-other-residue artefact."""
    out: set[int] = set()
    for snap in snapshots:
        P = X[snap["frame"]]
        for i, j in snap["edges"]:
            mid = 0.5 * (P[i] + P[j])
            d = np.linalg.norm(P - mid, axis=1)
            out.update(np.nonzero(d <= radius)[0].tolist())
    return out


def build_episodes(X: np.ndarray, dt_ns: float, *, eps=8.0, max_dim=2, debounce=3,
                   min_episode_ns=1.0, support="vertices", n_snapshots=1, geo_radius=5.0,
                   verbose=False) -> tuple[list[Episode], Bars]:
    """support: 'vertices' (cycle vertex set) or 'geometric' (residues within
    geo_radius of any cycle edge midpoint); n_snapshots > 1 takes the union over
    equally spaced frames of the bar's life (robust to representative choice)."""
    T = X.shape[0]
    s, tm = build_filtration(X, eps=eps, max_dim=max_dim, debounce=debounce)
    bars = compute_bars(s, tm, T, dt_ns)
    idx = np.nonzero((bars.dim == 1) & (bars.lifetime_ns >= min_episode_ns))[0]

    def frames_of(i):
        lo, hi = bars.alive_lo[i], bars.alive_hi[i]
        return [int(lo + (hi - lo) * (k + 1) / (n_snapshots + 1)) for k in range(n_snapshots)]

    reps = recover_representatives(s, tm, bars, idx, frames_of=frames_of, verbose=verbose)
    eps_list = []
    for i in idx:
        if int(i) not in reps:
            continue
        r = reps[int(i)]
        if support == "geometric":
            sup = tuple(sorted(geometric_support(X, r["snapshots"], geo_radius)))
        else:
            sup = tuple(r["vertices"])
        eps_list.append(Episode(int(i), int(bars.alive_lo[i]), int(bars.alive_hi[i]),
                                float(bars.lifetime_ns[i]), sup))
    eps_list.sort(key=lambda e: (e.lo, e.hi))
    return eps_list, bars


def assign_lineages(episodes: list[Episode], jaccard: float = 0.5) -> int:
    """Greedy chaining by support overlap; returns number of lineages."""
    last_of: list[Episode] = []          # last episode of each lineage
    for e in episodes:
        best, best_j = -1, 0.0
        se = set(e.support)
        for L, prev in enumerate(last_of):
            if prev.hi > e.lo:
                continue                 # must not overlap in time
            sp = set(prev.support)
            j = len(se & sp) / len(se | sp)
            if j >= jaccard and j > best_j:
                best, best_j = L, j
        if best < 0:
            e.lineage = len(last_of)
            last_of.append(e)
        else:
            e.lineage = best
            last_of[best] = e
    return len(last_of)


def per_residue_maps(episodes: list[Episode], n_res: int, n_frames: int):
    occ = np.zeros(n_res)
    sum_l = np.zeros(n_res)
    sum_l2 = np.zeros(n_res)
    n_ep = np.zeros(n_res, dtype=int)
    alive = np.zeros((n_frames, n_res), dtype=bool)
    for e in episodes:
        r = list(e.support)
        alive[e.lo:e.hi, r] = True
        sum_l[r] += e.lifetime_ns
        sum_l2[r] += e.lifetime_ns ** 2
        n_ep[r] += 1
    occ = alive.mean(axis=0)
    L = np.divide(sum_l2, sum_l, out=np.zeros(n_res), where=sum_l > 0)
    return dict(occupancy=occ, lifetime=L, n_episodes=n_ep, total_ns=sum_l)


def analyze_replica(X: np.ndarray, dt_ns: float, ref: np.ndarray, *, eps=8.0, debounce=3,
                    min_episode_ns=1.0, lineage_jaccard=0.5, verbose=False) -> dict:
    episodes, bars = build_episodes(X, dt_ns, eps=eps, debounce=debounce,
                                    min_episode_ns=min_episode_ns, verbose=verbose)
    n_lin = assign_lineages(episodes, lineage_jaccard)
    maps = per_residue_maps(episodes, X.shape[1], X.shape[0])
    maps["rmsf"] = rmsf(X)
    maps["q"] = contact_occupancy(X, native_contacts(ref, eps))
    maps["n_episodes_total"] = len(episodes)
    maps["n_lineages"] = n_lin
    maps["tau_top_ns"] = float(np.square(bars.lifetime_ns[bars.dim == 1]).sum() /
                               max(bars.lifetime_ns[bars.dim == 1].sum(), 1e-12))
    maps["episodes"] = [dict(lo=e.lo, hi=e.hi, lifetime_ns=e.lifetime_ns,
                             support=list(e.support), lineage=e.lineage) for e in episodes]
    return maps


# ----------------------------------------------------------------------- #
# near/far statistic with pseudo-site null
# ----------------------------------------------------------------------- #
def near_mask(ref: np.ndarray, site: int, radius: float = 8.0) -> np.ndarray:
    d = np.linalg.norm(ref - ref[site], axis=1)
    return d <= radius


def near_far_statistic(delta: np.ndarray, near: np.ndarray) -> float:
    d = delta[np.isfinite(delta)]
    n = near[np.isfinite(delta)]
    if n.sum() == 0 or (~n).sum() == 0:
        return float("nan")
    return float(d[n].mean() - d[~n].mean())


def pseudo_site_null(delta: np.ndarray, ref: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """S for the near set centred on every residue."""
    return np.array([near_far_statistic(delta, near_mask(ref, s, radius)) for s in range(ref.shape[0])])


def pseudo_site_p(delta: np.ndarray, ref: np.ndarray, site: int, direction: str, radius: float = 8.0):
    null = pseudo_site_null(delta, ref, radius)
    s = null[site]
    ok = np.isfinite(null)
    if direction == "negative":
        p = float((null[ok] <= s).mean())
    else:
        p = float((null[ok] >= s).mean())
    return s, p, null


def label_permutation_p(delta: np.ndarray, near: np.ndarray, direction: str, n=10000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    s = near_far_statistic(delta, near)
    k = int(near.sum())
    fin = np.isfinite(delta)
    idx = np.nonzero(fin)[0]
    cnt = 0
    for _ in range(n):
        perm = np.zeros_like(near)
        perm[rng.choice(idx, k, replace=False)] = True
        sp = near_far_statistic(delta, perm)
        cnt += (sp <= s) if direction == "negative" else (sp >= s)
    return cnt / n
