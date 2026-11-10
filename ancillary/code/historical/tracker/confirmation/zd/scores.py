"""Stability scalars and per-frame signals derived from zigzag bars.

All time-valued outputs are in nanoseconds so that a bar lasting 3 ms reads
as more stable than one lasting 3 us regardless of how the trajectory was
sampled; ``Bars.dt_ns`` supplies the fixed physical time reference.

Functions
---------
tau_top(bars)                  persistence-weighted mean lifetime  sum l^2 / sum l   [ns]
per_frame_tss(bars)            TSS(t) = sum of lifetimes of bars alive at t         [ns]
event_rate(bars, window)       births + deaths per ns in a sliding window           [1/ns]
occupancy_matrix(bars)         B[t, k] = 1 if bar k alive at frame t
occupancy_autocorrelation(B)   C(tau) = mean_t Jaccard(B_t, B_{t+tau})
relaxation_time(C, dt_ns)      first tau with C(tau) <= 1/e (linear interpolation)  [ns]
"""

from __future__ import annotations

import numpy as np

from .zigzag import Bars


def _lifetimes(bars: Bars, dims=None) -> np.ndarray:
    b = bars if dims is None else bars.in_dims(dims)
    l = b.lifetime_ns
    return l[l > 0]


def tau_top(bars: Bars, dims=(1,)) -> float:
    """Persistence-weighted mean lifetime, in ns (``nan`` if no bars)."""
    l = _lifetimes(bars, dims)
    if l.size == 0:
        return float("nan")
    return float(np.square(l).sum() / l.sum())


def total_persistence(bars: Bars, dims=(1,)) -> float:
    return float(_lifetimes(bars, dims).sum())


def per_frame_tss(bars: Bars, dims=(1,)) -> np.ndarray:
    """TSS(t): summed lifetimes (ns) of the bars alive at each frame."""
    b = bars.in_dims(dims)
    out = np.zeros(bars.n_frames)
    for lo, hi, lf in zip(b.alive_lo, b.alive_hi, b.lifetime_ns):
        if hi > lo:
            out[lo:hi] += lf
    return out


def event_rate(bars: Bars, window_frames: int = 10, dims=(1,)) -> np.ndarray:
    """Topological event rate: (births + deaths) per ns in a centred window.

    High rates mark frames where the contact topology is being remodelled
    (transitions); near-zero rates mark metastable stretches.
    """
    b = bars.in_dims(dims)
    T = bars.n_frames
    events = np.zeros(T)
    lo, hi = b.alive_lo, b.alive_hi
    np.add.at(events, np.clip(lo[~b.touches_start], 0, T - 1), 1.0)
    np.add.at(events, np.clip(hi[~b.touches_end] - 1, 0, T - 1), 1.0)
    w = max(1, int(window_frames))
    kernel = np.ones(w) / (w * bars.dt_ns)
    padded = np.pad(events, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:T]


def occupancy_matrix(bars: Bars, dims=(1, 2), min_lifetime_frames: float = 1.0) -> np.ndarray:
    """Boolean (T, K) matrix: which persistent features are alive at each frame."""
    b = bars.in_dims(dims).with_min_lifetime(min_lifetime_frames)
    B = np.zeros((bars.n_frames, len(b)), dtype=bool)
    for k, (lo, hi) in enumerate(zip(b.alive_lo, b.alive_hi)):
        B[lo:hi, k] = True
    return B


def occupancy_autocorrelation(B: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Jaccard similarity of the alive-feature set at lag tau, averaged over t.

    C(0) = 1 and C decays towards the similarity of unrelated frames; its
    1/e crossing is a *topological relaxation time*, the characteristic time
    over which the protein's contact topology is remembered.
    """
    T, K = B.shape
    if max_lag is None:
        max_lag = T // 2
    max_lag = min(max_lag, T - 1)
    Bi = B.astype(np.int32)
    C = np.ones(max_lag + 1)
    for tau in range(1, max_lag + 1):
        a, c = Bi[:-tau], Bi[tau:]
        inter = (a & c).sum(axis=1)
        union = (a | c).sum(axis=1)
        valid = union > 0
        C[tau] = float(np.mean(inter[valid] / union[valid])) if valid.any() else 1.0
    return C


def relaxation_time(C: np.ndarray, dt_ns: float, level: float = np.exp(-1.0)) -> float:
    """First lag (ns) at which C drops to ``level``; ``inf`` if never."""
    below = np.nonzero(C <= level)[0]
    if below.size == 0:
        return float("inf")
    i = int(below[0])
    if i == 0:
        return 0.0
    c0, c1 = C[i - 1], C[i]
    frac = (c0 - level) / (c0 - c1) if c0 != c1 else 0.0
    return float((i - 1 + frac) * dt_ns)
