"""Coordinates -> zigzag filtration toggles, vectorised.

Output contract (identical to ``coords_to_fzz.coords_to_fzz``)
--------------------------------------------------------------
``simplices`` is a list of sorted vertex lists ordered by (dimension, tuple),
``times`` is the parallel list of alternating toggle frames
``[appear, disappear, appear, ...]``: a simplex disappears at the first frame
in which it is absent, and one still present in the last frame disappears at
``T`` (= number of frames).  Vertices are present for the whole trajectory.

Rules
-----
* an edge (i, j) is present in a frame when ``dist(i, j) <= eps``
  (cKDTree.query_pairs semantics, as in the reference implementation);
* a triangle / tetrahedron is present when all of its edges are present
  (Vietoris–Rips clique rule).

Why this is faster
------------------
The reference walks every frame, rebuilds the whole complex as a Python set
of tuples, and diffs consecutive sets.  Here every simplex's presence over
time is one *bit-packed row* (``T/8`` bytes); higher simplices are bitwise
ANDs of edge rows; toggles are change points of each row.  Python-level work
is proportional to the number of candidate simplices, not frames x simplices.

Dynamics-aware options
----------------------
``hysteresis=(eps_on, eps_off)``
    A contact forms when ``d <= eps_on`` and breaks only when
    ``d >= eps_off`` (Schmitt trigger).  Contacts hovering near a single
    cutoff otherwise flicker and generate many short, uninformative bars and
    a large number of toggles for ``fast_zigzag`` (whose cost grows with the
    number of toggles, not the number of frames).
``debounce``
    Remove presence runs *and* absence gaps shorter than ``debounce`` frames
    (morphological opening followed by closing on each edge row).

Both act on edges only; higher simplices follow from the clique rule, so the
result is still a Vietoris–Rips-type zigzag of a (time-smoothed) contact map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class FiltrationStats:
    n_frames: int
    n_vertices: int
    n_edges: int
    n_triangles: int
    n_tetrahedra: int
    n_toggles: int

    @property
    def n_simplices(self) -> int:
        return self.n_vertices + self.n_edges + self.n_triangles + self.n_tetrahedra


# --------------------------------------------------------------------------- #
# edge presence  (rows = edges, columns = frames)
# --------------------------------------------------------------------------- #
def _edge_presence(coords, eps, *, hysteresis):
    """Return (pairs (E,2) int, presence (E,T) bool) for edges ever present."""
    T, N, _ = coords.shape
    iu = np.triu_indices(N, k=1)
    n_pairs = iu[0].size

    if hysteresis is None:
        eps_on = eps_off = float(eps)
    else:
        eps_on, eps_off = map(float, hysteresis)
        if eps_on > eps_off:
            raise ValueError("hysteresis must be (eps_on, eps_off) with eps_on <= eps_off")

    # condensed pairwise distances per frame (double precision, like cKDTree);
    # only boolean masks are kept so memory is T x n_pairs bits, not doubles
    present = np.empty((T, n_pairs), dtype=bool)
    if hysteresis is None:
        for t in range(T):
            present[t] = pdist(coords[t].astype(np.float64, copy=False)) <= eps_on
    else:
        d0 = pdist(coords[0].astype(np.float64, copy=False))
        present[0] = d0 <= 0.5 * (eps_on + eps_off)           # no history at t=0
        for t in range(1, T):
            d = pdist(coords[t].astype(np.float64, copy=False))
            present[t] = (d <= eps_on) | (present[t - 1] & ~(d >= eps_off))

    ever = present.any(axis=0)
    pairs = np.stack([iu[0][ever], iu[1][ever]], axis=1).astype(np.int64)
    return pairs, np.ascontiguousarray(present[:, ever].T)


def _debounce_rows(present: np.ndarray, k: int) -> np.ndarray:
    """Remove presence runs shorter than k, then absence gaps shorter than k."""
    if k <= 1:
        return present
    out = present.copy()
    E, T = out.shape
    for negate in (False, True):                  # opening (runs) then closing (gaps)
        rows = ~out if negate else out
        padded = np.zeros((E, T + 2), dtype=bool)
        padded[:, 1:T + 1] = rows
        change = padded[:, 1:] != padded[:, :-1]  # (E, T+1), row-major -> grouped by row
        r_idx, t_idx = np.nonzero(change)
        starts, ends, rows_of = t_idx[0::2], t_idx[1::2], r_idx[0::2]
        short = (ends - starts) < k
        for s, e, r in zip(starts[short], ends[short], rows_of[short]):
            out[r, s:e] = bool(negate)            # fill short gap / drop short run
    return out


# --------------------------------------------------------------------------- #
# higher simplices
# --------------------------------------------------------------------------- #
def _adjacency(pairs: np.ndarray, n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in pairs.tolist():
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _candidate_triangles(pairs: np.ndarray, adj) -> np.ndarray:
    tris = []
    for i, j in pairs.tolist():
        for k in adj[i] & adj[j]:
            if k > j:
                tris.append((i, j, k))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _candidate_tetrahedra(tris: np.ndarray, adj) -> np.ndarray:
    tets = []
    for i, j, k in tris.tolist():
        for l in adj[i] & adj[j] & adj[k]:
            if l > k:
                tets.append((i, j, k, l))
    return np.asarray(tets, dtype=np.int64).reshape(-1, 4)


def _key_lookup(keys_sorted: np.ndarray, order: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(keys_sorted, query)
    pos_c = np.minimum(pos, len(keys_sorted) - 1)
    found = keys_sorted[pos_c] == query
    return np.where(found, order[pos_c], -1)


def _and_rows(packed: np.ndarray, rows: np.ndarray, chunk: int):
    """AND of bit-packed rows -> (packed result, keep mask of ever-present)."""
    out = np.empty((rows.shape[0], packed.shape[1]), dtype=np.uint8)
    for start in range(0, rows.shape[0], chunk):
        sl = slice(start, start + chunk)
        acc = packed[rows[sl, 0]].copy()
        for c in range(1, rows.shape[1]):
            acc &= packed[rows[sl, c]]
        out[sl] = acc
    return out, out.any(axis=1)


# --------------------------------------------------------------------------- #
# toggles
# --------------------------------------------------------------------------- #
def _toggles_from_packed(packed: np.ndarray, T: int) -> list[list[int]]:
    rows = np.unpackbits(packed, axis=1, count=T).astype(bool)
    E = rows.shape[0]
    padded = np.zeros((E, T + 2), dtype=bool)
    padded[:, 1:T + 1] = rows
    change = padded[:, 1:] != padded[:, :-1]                  # (E, T+1)
    r_idx, t_idx = np.nonzero(change)                         # grouped by row, time ascending
    counts = np.bincount(r_idx, minlength=E)
    return [arr.tolist() for arr in np.split(t_idx, np.cumsum(counts)[:-1])]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_filtration(
    coords: np.ndarray,
    eps: float = 8.0,
    max_dim: int = 3,
    *,
    hysteresis: tuple[float, float] | None = None,
    debounce: int = 0,
    row_chunk: int = 200_000,
    return_stats: bool = False,
):
    """Vectorised Vietoris–Rips zigzag toggles.

    Parameters
    ----------
    coords : (T, N, 3) array in the unit of ``eps``.
    eps : contact cutoff (used when ``hysteresis`` is None).
    max_dim : highest simplex dimension (0..3).  For H1 use 2; tetrahedra are
        only needed for H2 and roughly double the work of ``fast_zigzag``.
    hysteresis : optional (eps_on, eps_off) Schmitt-trigger cutoffs.
    debounce : drop presence runs and absence gaps shorter than this many frames.
    return_stats : also return a :class:`FiltrationStats`.

    Returns
    -------
    simplices, times[, stats]
    """
    coords = np.asarray(coords)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (frames, atoms, 3)")
    if not 0 <= max_dim <= 3:
        raise ValueError("max_dim must be 0..3")
    T, N, _ = coords.shape
    if T == 0:
        raise ValueError("need at least one frame")

    pairs, pres_e = _edge_presence(coords, eps, hysteresis=hysteresis)
    if debounce > 1:
        pres_e = _debounce_rows(pres_e, int(debounce))
        keep = pres_e.any(axis=1)
        pairs, pres_e = pairs[keep], np.ascontiguousarray(pres_e[keep])
    packed_e = np.packbits(pres_e, axis=1)                    # (E, ceil(T/8))

    blocks: list[tuple[np.ndarray, np.ndarray | None]] = [
        (np.arange(N, dtype=np.int64).reshape(-1, 1), None)]
    if max_dim >= 1:
        blocks.append((pairs, packed_e))

    tris = packed_t = None
    if max_dim >= 2 and len(pairs):
        adj = _adjacency(pairs, N)
        eidx = np.full((N, N), -1, dtype=np.int64)
        eidx[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
        eidx[pairs[:, 1], pairs[:, 0]] = np.arange(len(pairs))
        tris = _candidate_triangles(pairs, adj)
        if len(tris):
            rows = np.stack([eidx[tris[:, 0], tris[:, 1]],
                             eidx[tris[:, 0], tris[:, 2]],
                             eidx[tris[:, 1], tris[:, 2]]], axis=1)
            packed_t, keep = _and_rows(packed_e, rows, row_chunk)
            tris, packed_t = tris[keep], np.ascontiguousarray(packed_t[keep])
            blocks.append((tris, packed_t))

    if max_dim >= 3 and tris is not None and len(tris):
        tets = _candidate_tetrahedra(tris, adj)
        if len(tets):
            NN = np.int64(N)
            tri_keys = (tris[:, 0] * NN + tris[:, 1]) * NN + tris[:, 2]
            order = np.argsort(tri_keys)
            keys_sorted = tri_keys[order]

            def col(a, b, c):
                return _key_lookup(keys_sorted, order, (a * NN + b) * NN + c)

            # three triangles cover all six edges: ijk, ijl, ikl
            c0 = col(tets[:, 0], tets[:, 1], tets[:, 2])
            c1 = col(tets[:, 0], tets[:, 1], tets[:, 3])
            c2 = col(tets[:, 0], tets[:, 2], tets[:, 3])
            ok = (c0 >= 0) & (c1 >= 0) & (c2 >= 0)
            tets = tets[ok]
            rows = np.stack([c0[ok], c1[ok], c2[ok]], axis=1)
            packed_q, keep = _and_rows(packed_t, rows, row_chunk)
            tets, packed_q = tets[keep], np.ascontiguousarray(packed_q[keep])
            if len(tets):
                blocks.append((tets, packed_q))

    simplices: list[list[int]] = []
    times: list[list[int]] = []
    n_toggles = 0
    counts = [0, 0, 0, 0]
    for vertices, packed in blocks:
        d = vertices.shape[1] - 1
        counts[d] = len(vertices)
        order = np.lexsort(vertices.T[::-1])                  # lexicographic, as sorted()
        vertices = vertices[order]
        if packed is None:
            tl = [[0, T]] * len(vertices)
        else:
            tl = _toggles_from_packed(packed[order], T)
        simplices.extend(vertices.tolist())
        times.extend(tl)
        n_toggles += sum(len(x) for x in tl)

    if return_stats:
        return simplices, times, FiltrationStats(T, counts[0], counts[1], counts[2], counts[3], n_toggles)
    return simplices, times
