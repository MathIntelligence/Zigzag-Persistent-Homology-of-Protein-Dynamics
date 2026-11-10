"""Topological state decomposition of a trajectory.

Frames are grouped into discrete *topological states* from a per-frame
feature matrix ``X`` (T, D) — by default the persistence images of
:mod:`descriptors`, which are recurrence-aware; a boolean zigzag occupancy
matrix also works but treats every re-formation of a loop as a new feature.

From the state trajectory ``s(t)`` we obtain the ensemble quantities that a
single structure cannot provide:

* state populations                        pi_i
* effective number of states               exp(H),  H = -sum pi_i ln pi_i
* mean dwell time per state                [ns]
* transition matrix at lag tau             T_ij(tau)   (symmetrised counts)
* implied timescales                       t_k = -tau / ln lambda_k   [ns]
* per-state zigzag signatures              which long-lived bars live in a state

This is the topological analogue of a Markov state model with deterministic,
parameter-light state construction (Ward linkage, one state count).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist


@dataclass
class StateDecomposition:
    labels: np.ndarray                 # (T,) state index 0..S-1, ordered by population
    n_states: int
    dt_ns: float
    lag_frames: int
    populations: np.ndarray            # (S,)
    mean_dwell_ns: np.ndarray          # (S,)
    n_visits: np.ndarray               # (S,)
    transition_counts: np.ndarray      # (S, S)
    transition_matrix: np.ndarray      # (S, S) row-stochastic
    implied_timescales_ns: np.ndarray  # (S-1,) descending
    signatures: np.ndarray | None = None   # (S, K) mean bar occupancy per state
    method: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def entropy_nats(self) -> float:
        p = self.populations[self.populations > 0]
        return float(-(p * np.log(p)).sum())

    @property
    def effective_states(self) -> float:
        return float(np.exp(self.entropy_nats))

    @property
    def n_switches(self) -> int:
        return int((self.labels[1:] != self.labels[:-1]).sum())

    def switch_frames(self) -> np.ndarray:
        """Frames t at which s(t) != s(t-1)."""
        return np.nonzero(self.labels[1:] != self.labels[:-1])[0] + 1

    def defining_features(self, state: int, threshold: float = 0.5) -> np.ndarray:
        if self.signatures is None:
            return np.array([], dtype=int)
        return np.nonzero(self.signatures[state] >= threshold)[0]

    def summary(self) -> dict:
        return {
            "method": self.method,
            "n_states": int(self.n_states),
            "populations": self.populations.round(4).tolist(),
            "entropy_nats": round(self.entropy_nats, 4),
            "effective_states": round(self.effective_states, 3),
            "mean_dwell_ns": self.mean_dwell_ns.round(3).tolist(),
            "n_visits": self.n_visits.tolist(),
            "n_switches": self.n_switches,
            "lag_ns": self.lag_frames * self.dt_ns,
            "implied_timescales_ns": [None if not np.isfinite(x) else round(float(x), 3)
                                      for x in self.implied_timescales_ns],
            "transition_matrix": self.transition_matrix.round(4).tolist(),
            **{k: v for k, v in self.extras.items()},
        }


# --------------------------------------------------------------------------- #
def _runs(labels: np.ndarray):
    T = labels.size
    start = 0
    for t in range(1, T + 1):
        if t == T or labels[t] != labels[start]:
            yield int(labels[start]), start, t
            start = t


def kinetics(labels: np.ndarray, n_states: int, dt_ns: float, lag_frames: int = 1):
    """Populations, dwell times, transition counts/matrix, implied timescales."""
    T = labels.size
    pops = np.bincount(labels, minlength=n_states) / max(T, 1)

    dwell_sum = np.zeros(n_states)
    visits = np.zeros(n_states, dtype=int)
    for s, a, b in _runs(labels):
        dwell_sum[s] += (b - a)
        visits[s] += 1
    mean_dwell = np.where(visits > 0, dwell_sum / np.maximum(visits, 1), 0.0) * dt_ns

    counts = np.zeros((n_states, n_states))
    if T > lag_frames:
        np.add.at(counts, (labels[:-lag_frames], labels[lag_frames:]), 1.0)
    sym = counts + counts.T                       # detailed balance (standard MSM practice)
    row = sym.sum(axis=1, keepdims=True)
    Tm = np.divide(sym, row, out=np.zeros_like(sym), where=row > 0)
    Tm[row[:, 0] == 0] = np.eye(n_states)[row[:, 0] == 0]

    pi = np.maximum(sym.sum(axis=1) / max(sym.sum(), 1.0), 1e-300)
    D = np.sqrt(pi)
    S = (D[:, None] * Tm) / D[None, :]
    S = 0.5 * (S + S.T)
    lam = np.sort(np.linalg.eigvalsh(S))[::-1][1:] if n_states > 1 else np.array([])
    its = np.full(lam.shape, np.inf)
    ok = (lam > 0) & (lam < 1)
    its[ok] = -lag_frames * dt_ns / np.log(lam[ok])
    its[lam <= 0] = 0.0
    return pops, mean_dwell, visits, counts, Tm, its


def cluster_frames(X: np.ndarray, n_states: int, *, method: str = "ward", max_pairwise: int = 6000,
                   seed: int = 0) -> np.ndarray:
    """Deterministic frame clustering -> labels (0..n-1, unordered)."""
    T = X.shape[0]
    n_states = max(1, min(int(n_states), T))
    if n_states == 1:
        return np.zeros(T, dtype=int)
    if T <= max_pairwise:
        Z = linkage(pdist(X, metric="euclidean"), method=method)
        return fcluster(Z, t=n_states, criterion="maxclust") - 1
    # large T: k-means with fixed seed (Ward's O(T^2) memory is prohibitive)
    _, lab = kmeans2(X, n_states, minit="++", seed=np.random.default_rng(seed))
    return lab.astype(int)


def decompose_states(
    X: np.ndarray,
    dt_ns: float = 1.0,
    *,
    n_states: int = 4,
    lag_frames: int = 1,
    occupancy: np.ndarray | None = None,
    method: str = "ward",
) -> StateDecomposition:
    """Cluster frames by topological feature vectors and compute kinetics.

    Parameters
    ----------
    X : (T, D) per-frame features (persistence images, or a boolean occupancy
        matrix cast to float).
    dt_ns : physical time per frame.
    n_states : number of states (Ward dendrogram cut / k-means for large T).
    lag_frames : lag for the transition matrix and implied timescales.
    occupancy : optional (T, K) zigzag bar occupancy used for state signatures.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    if T == 0:
        raise ValueError("empty feature matrix")
    if X.shape[1] == 0 or np.allclose(X, X[0]):
        labels = np.zeros(T, dtype=int)
    else:
        labels = cluster_frames(X, n_states, method=method)
    S = int(labels.max()) + 1

    pops = np.bincount(labels, minlength=S)
    order = np.argsort(-pops, kind="stable")
    remap = np.empty(S, dtype=int)
    remap[order] = np.arange(S)
    labels = remap[labels]

    pops, dwell, visits, counts, Tm, its = kinetics(labels, S, dt_ns, lag_frames)
    sig = None
    if occupancy is not None and occupancy.shape[0] == T:
        sig = state_signatures(labels, S, occupancy)
    return StateDecomposition(labels, S, dt_ns, lag_frames, pops, dwell, visits,
                              counts, Tm, its, sig, method=method)


def state_signatures(labels: np.ndarray, n_states: int, occupancy: np.ndarray) -> np.ndarray:
    """(S, K): fraction of a state's frames in which zigzag bar k is alive."""
    B = np.asarray(occupancy, dtype=float)
    sig = np.zeros((n_states, B.shape[1]))
    for s in range(n_states):
        m = labels == s
        if m.any():
            sig[s] = B[m].mean(axis=0)
    return sig


def timescale_spectrum(X: np.ndarray, dt_ns: float, state_counts=range(2, 9), lag_frames: int = 1,
                       method: str = "ward") -> dict[int, np.ndarray]:
    """Slowest implied timescales for several state counts.

    A state count after which the slowest timescale stops growing (a 'gap')
    is the natural resolution of the topological state space.
    """
    out = {}
    for k in state_counts:
        dec = decompose_states(X, dt_ns, n_states=k, lag_frames=lag_frames, method=method)
        out[int(k)] = dec.implied_timescales_ns[: min(3, dec.implied_timescales_ns.size)]
    return out


def switch_alignment(labels: np.ndarray, rate: np.ndarray) -> float:
    """Mean zigzag event rate at state-switch frames divided by the overall mean.

    > 1 means state changes coincide with bursts of topological births/deaths
    in the zigzag — an internal consistency check between the two views.
    """
    sw = np.nonzero(labels[1:] != labels[:-1])[0] + 1
    if sw.size == 0 or rate.mean() <= 0:
        return float("nan")
    return float(rate[sw].mean() / rate.mean())


def select_n_states(
    X: np.ndarray,
    dt_ns: float,
    *,
    max_states: int = 6,
    min_timescale_frames: float = 15.0,
    min_population: float = 0.02,
    lag_frames: int = 1,
    method: str = "ward",
) -> tuple[int, dict[int, np.ndarray]]:
    """Pick the number of states by kinetic validity.

    The largest ``k`` is kept for which every one of the ``k - 1`` implied
    timescales is at least ``min_timescale_frames`` and every state holds at
    least ``min_population`` of the frames.  If no ``k >= 2`` qualifies the
    trajectory is a single topological basin at this resolution (``k = 1``).
    Returns ``(k, spectrum)`` where ``spectrum[k]`` are the implied timescales.
    """
    thr = min_timescale_frames * dt_ns
    best, spectrum = 1, {}
    for k in range(2, max_states + 1):
        dec = decompose_states(X, dt_ns, n_states=k, lag_frames=lag_frames, method=method)
        its = dec.implied_timescales_ns
        spectrum[k] = its
        if dec.n_states == k and its.size == k - 1 and np.all(its >= thr) \
                and dec.populations.min() >= min_population:
            best = k
    return best, spectrum
