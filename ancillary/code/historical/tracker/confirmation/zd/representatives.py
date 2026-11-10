"""Residue support for zigzag H1 bars via the direct live homology basis.

Adapted from ``topomol_workbench.live_zigzag_representatives`` (validated
against synthetic cc/co/oc/oo loops and three mdCATH domains).  Requires the
minimally patched Dionysus that exposes ``ZigzagPersistence.alive_births``
(``vendor_patches/dionysus-2.2.3-alive-births.patch``); set
``ZIGZAG_DIONYSUS_LIVE`` to its install directory.

Differences from the workbench version: one representative per bar, taken
at a chosen frame (default: the bar's mid-life), for an arbitrary number of
bars — the live basis is only read at the operations where a snapshot is
needed, so the cost is dominated by the zigzag itself, not by the snapshots.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np

from .zigzag import Bars


def load_live_dionysus():
    path = os.environ.get("ZIGZAG_DIONYSUS_LIVE")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    import dionysus as d
    if not hasattr(d.ZigzagPersistence, "alive_births"):
        raise RuntimeError("patched Dionysus with ZigzagPersistence.alive_births is required "
                           "(set ZIGZAG_DIONYSUS_LIVE to its install directory)")
    return d


def _event_table(filtration, times):
    events = []
    for simplex_index, simplex_times in enumerate(times):
        direction = True
        dimension = filtration[simplex_index].dimension()
        for time in simplex_times:
            events.append(dict(time=float(time), simplex_index=simplex_index,
                               dimension=dimension, direction=direction))
            direction = not direction
    events.sort(key=lambda e: (e["time"], 0 if e["direction"] else 1,
                               e["dimension"] if e["direction"] else -e["dimension"],
                               e["simplex_index"]))
    return events


def _same(a, b):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)


def _match(bars: Bars, idx: np.ndarray, diagrams, events) -> dict[int, int]:
    """bar index -> birth operation, matched on (dim, endpoint, birth, death)."""
    cands = defaultdict(list)
    for dim, dg in enumerate(diagrams):
        for pt in dg:
            b, dth = float(pt.birth), float(pt.death)
            if not (math.isfinite(b) and math.isfinite(dth) and dth > b):
                continue
            op = int(pt.data)
            cands[dim].append(dict(b=b, d=dth, op=op, end="c" if events[op]["direction"] else "o", used=False))
    out = {}
    for i in idx:
        dim, it = int(bars.dim[i]), str(bars.itype[i])
        b, dth = float(bars.birth[i]), float(bars.death[i])
        if not math.isfinite(dth):
            continue
        ok = [c for c in cands[dim] if not c["used"] and c["end"] == it[0] and _same(c["b"], b) and _same(c["d"], dth)]
        if not ok:
            continue
        c = min(ok, key=lambda c: c["op"])
        c["used"] = True
        out[int(i)] = c["op"]
    return out


def _chain_edges(chain, cells, filtration) -> list[tuple[int, int]]:
    support = set()
    for entry in chain:
        if int(entry.element) % 2 == 0:
            continue
        s = filtration[int(cells[entry.index])]
        key = tuple(sorted(int(v) for v in s))
        if len(key) != 2:
            raise ValueError("H1 live cycle contains a non-edge simplex")
        support.symmetric_difference_update({key})
    return sorted(support)


def recover_representatives(
    simplices, times, bars: Bars, bar_indices, *, frame_of=None, frames_of=None, verbose=False,
) -> dict[int, dict]:
    """Return {bar index: {"frame": f, "edges": [(i,j),...], "vertices": [...]}}.

    ``frame_of(i)`` picks the snapshot frame for bar ``i`` (default mid-life).
    ``frames_of(i)`` may instead return several frames; the result then also
    carries ``"snapshots": [{"frame", "edges", "vertices"}, ...]`` and the
    top-level edges/vertices are the union over snapshots.
    Bars that cannot be matched to the direct basis are omitted.
    """
    d = load_live_dionysus()
    bar_indices = np.asarray(bar_indices, dtype=int)
    filt = d.Filtration(simplices)
    events = _event_table(filt, times)
    _p, diagrams, _c = d.zigzag_homology_persistence(filt, times, prime=2)
    op_of_bar = _match(bars, bar_indices, diagrams, events)
    if verbose:
        print(f"[repr] matched {len(op_of_bar)}/{len(bar_indices)} bars to birth operations")
    if not op_of_bar:
        return {}

    if frame_of is None:
        def frame_of(i):
            return int((bars.alive_lo[i] + bars.alive_hi[i]) // 2)
    if frames_of is None:
        def frames_of(i):
            return [frame_of(i)]
    want = defaultdict(list)            # frame -> [(bar index, birth op)]
    for i, op in op_of_bar.items():
        for f in frames_of(i):
            f = min(max(int(f), bars.alive_lo[i]), bars.alive_hi[i] - 1)
            if (i, op) not in want[int(f)]:
                want[int(f)].append((i, op))
    pending = sorted(want)
    targets_by_op = {op: i for i, op in op_of_bar.items()}

    state = dict(operation=0, current_time=None, snapshot=None)
    recovered: dict[int, dict] = {}

    def take_snapshot(zz, cells, frame):
        births = [int(v) for v in zz.alive_births()]
        cycles = list(zz)
        if len(births) != len(cycles):
            raise ValueError("live birth labels and cycles disagree")
        lookup = dict(zip(births, cycles))
        for i, op in want[frame]:
            if op in lookup:
                edges = _chain_edges(lookup[op], cells, filt)
                verts = sorted({v for e in edges for v in e})
                snap = dict(frame=frame, edges=edges, vertices=verts)
                if i in recovered:
                    r = recovered[i]
                    r["snapshots"].append(snap)
                    r["edges"] = sorted(set(r["edges"]) | set(edges))
                    r["vertices"] = sorted(set(r["vertices"]) | set(verts))
                else:
                    recovered[i] = dict(frame=frame, edges=list(edges), vertices=list(verts), snapshots=[snap])

    n_events = len(events)

    def callback(simplex_index, time, direction, zz, cells):
        # Dionysus invokes the callback AFTER applying operation k.  The complex
        # for frames t <= f < t_next is final once the last operation at time t
        # has been applied, i.e. when the next event has a strictly later time.
        k = state["operation"]
        ev = events[k]
        if int(simplex_index) != ev["simplex_index"] or bool(direction) != ev["direction"] \
                or not _same(float(time), ev["time"]):
            raise ValueError("Python and Dionysus zigzag event order diverged")
        t = ev["time"]
        t_next = events[k + 1]["time"] if k + 1 < n_events else math.inf
        if t_next > t:
            while pending and pending[0] < t_next:
                f = pending.pop(0)
                if f >= t:
                    take_snapshot(zz, cells, f)
        state["operation"] += 1

    d.zigzag_homology_persistence(filt, times, prime=2, callback=callback)
    if verbose:
        print(f"[repr] recovered {len(recovered)} representatives")
    return recovered
