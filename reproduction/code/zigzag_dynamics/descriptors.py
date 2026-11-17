"""Per-frame topological descriptors that are recurrence-aware.

A zigzag bar is an *episode* of a feature: a loop that breaks and re-forms
gives two bars, so bar occupancy alone cannot tell that the protein has
*returned* to an earlier topological state.  For state populations and
kinetics we therefore describe each frame by its own persistence diagram
(static Vietoris–Rips H1 of the C-alpha cloud, via ripser) vectorised as a
persistence image [Adams et al., JMLR 2017].  Identical topologies give
identical vectors regardless of when they occur, which is exactly what a
state decomposition needs; the zigzag bars then supply the feature-level
kinetics inside and across those states.
"""

from __future__ import annotations

import numpy as np


def frame_diagrams(coords: np.ndarray, thresh: float = 12.0, maxdim: int = 1) -> list[np.ndarray]:
    """Finite H1 (and lower) Rips diagrams for every frame, via ripser."""
    from ripser import ripser

    out = []
    for pts in coords:
        dg = ripser(np.asarray(pts, dtype=float), maxdim=maxdim, thresh=thresh)["dgms"][maxdim]
        dg = np.asarray(dg, dtype=float).reshape(-1, 2)
        dg = dg[np.isfinite(dg).all(axis=1)]
        out.append(dg)
    return out


def persistence_images(
    diagrams: list[np.ndarray],
    *,
    resolution: int = 16,
    sigma: float | None = None,
    birth_range: tuple[float, float] | None = None,
    pers_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """(T, resolution**2) persistence images with linear persistence weighting.

    Ranges default to the global extent over all frames so every frame is
    embedded in the same coordinate system.
    """
    pts = [np.column_stack([d[:, 0], d[:, 1] - d[:, 0]]) for d in diagrams]
    allp = np.concatenate([p for p in pts if len(p)] or [np.zeros((0, 2))])
    if birth_range is None:
        birth_range = (0.0, float(allp[:, 0].max()) if len(allp) else 1.0)
    if pers_range is None:
        pers_range = (0.0, float(allp[:, 1].max()) if len(allp) else 1.0)
    bx = np.linspace(*birth_range, resolution)
    py = np.linspace(*pers_range, resolution)
    if sigma is None:
        sigma = max((birth_range[1] - birth_range[0]), (pers_range[1] - pers_range[0])) / resolution
    BX, PY = np.meshgrid(bx, py, indexing="ij")
    grid = np.stack([BX.ravel(), PY.ravel()], axis=1)                # (R*R, 2)
    pmax = pers_range[1] if pers_range[1] > 0 else 1.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    X = np.zeros((len(pts), grid.shape[0]))
    for t, p in enumerate(pts):
        if len(p) == 0:
            continue
        w = p[:, 1] / pmax                                           # linear weight
        d2 = ((grid[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)   # (G, n)
        X[t] = (np.exp(-d2 * inv2s2) * w[None, :]).sum(axis=1)
    return X


def frame_features(coords: np.ndarray, *, thresh: float = 12.0, resolution: int = 16) -> np.ndarray:
    """Convenience: coords -> per-frame persistence-image feature matrix."""
    return persistence_images(frame_diagrams(coords, thresh=thresh), resolution=resolution)


def smooth_features(X: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over frames (edge-padded).

    At 1 ns sampling the frame-to-frame change of a persistence image is as
    large as its spread over the whole trajectory (thermal noise), so states
    are resolved on a slightly coarser clock.  Dwell times shorter than the
    window are not resolvable afterwards; :func:`states.select_n_states`
    accounts for this through ``min_timescale_frames``.
    """
    w = int(window)
    if w <= 1:
        return np.asarray(X, dtype=float)
    X = np.asarray(X, dtype=float)
    pad = (w // 2, w - 1 - w // 2)
    Xp = np.pad(X, (pad, (0, 0)), mode="edge")
    cs = np.cumsum(Xp, axis=0)
    cs = np.vstack([np.zeros((1, X.shape[1])), cs])
    return (cs[w:] - cs[:-w]) / w
