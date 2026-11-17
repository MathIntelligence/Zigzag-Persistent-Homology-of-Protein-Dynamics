"""Run dionysus fast_zigzag and return bars in frames *and* absolute time."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INTERVAL_TYPES = ("cc", "co", "oc", "oo")


@dataclass
class Bars:
    """Zigzag bars as parallel arrays.

    ``birth``/``death`` are sampled-frame coordinates as reported by dionysus
    (death = +inf for a bar alive at the end).  ``alive_lo``/``alive_hi`` are
    the integer half-open frame range on which the feature is present, with
    infinite deaths clipped to ``n_frames``.  ``lifetime_ns`` uses
    ``dt_ns`` (physical time per sampled frame).
    """

    dim: np.ndarray
    itype: np.ndarray            # interval type code, one of INTERVAL_TYPES
    birth: np.ndarray
    death: np.ndarray
    n_frames: int
    dt_ns: float
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.dim.size)

    @property
    def is_inf(self) -> np.ndarray:
        return ~np.isfinite(self.death)

    @property
    def alive_lo(self) -> np.ndarray:
        return np.clip(np.ceil(self.birth), 0, self.n_frames).astype(int)

    @property
    def alive_hi(self) -> np.ndarray:
        d = np.where(self.is_inf, self.n_frames, self.death)
        return np.clip(np.ceil(d), 0, self.n_frames).astype(int)

    @property
    def lifetime_frames(self) -> np.ndarray:
        return (self.alive_hi - self.alive_lo).astype(float)

    @property
    def lifetime_ns(self) -> np.ndarray:
        return self.lifetime_frames * self.dt_ns

    @property
    def touches_start(self) -> np.ndarray:
        return self.alive_lo <= 0

    @property
    def touches_end(self) -> np.ndarray:
        return self.alive_hi >= self.n_frames

    def select(self, mask: np.ndarray) -> "Bars":
        return Bars(self.dim[mask], self.itype[mask], self.birth[mask],
                    self.death[mask], self.n_frames, self.dt_ns, dict(self.meta))

    def in_dims(self, dims) -> "Bars":
        return self.select(np.isin(self.dim, list(dims)))

    def with_min_lifetime(self, frames: float) -> "Bars":
        return self.select(self.lifetime_frames >= frames)

    def interior(self) -> "Bars":
        """Bars that neither touch the first nor the last frame."""
        return self.select(~self.touches_start & ~self.touches_end)

    def sorted_by_lifetime(self) -> "Bars":
        order = np.argsort(-self.lifetime_frames, kind="stable")
        return self.select(order)

    def to_records(self) -> list[dict]:
        lo, hi, lf = self.alive_lo, self.alive_hi, self.lifetime_ns
        return [
            dict(dim=int(self.dim[i]), itype=str(self.itype[i]),
                 birth=float(self.birth[i]), death=float(self.death[i]),
                 alive_lo=int(lo[i]), alive_hi=int(hi[i]),
                 lifetime_frames=int(hi[i] - lo[i]), lifetime_ns=float(lf[i]))
            for i in range(len(self))
        ]

    @staticmethod
    def from_arrays(dim, itype, birth, death, n_frames, dt_ns, meta=None) -> "Bars":
        return Bars(np.asarray(dim, int), np.asarray(itype, dtype="<U2"),
                    np.asarray(birth, float), np.asarray(death, float),
                    int(n_frames), float(dt_ns), dict(meta or {}))


def diagrams_to_bars(dgms, n_frames: int, dt_ns: float, meta=None) -> Bars:
    """Flatten ``dionysus.init_zigzag_diagrams`` output into :class:`Bars`."""
    dims, types, births, deaths = [], [], [], []
    for dim, typed in enumerate(dgms):
        for itype, dgm in typed.items():
            for pt in dgm:
                dims.append(dim)
                types.append(str(itype))
                births.append(float(pt.birth))
                deaths.append(float(pt.death) if math.isfinite(pt.death) else math.inf)
    return Bars.from_arrays(dims, types, births, deaths, n_frames, dt_ns, meta)


def compute_bars(simplices, times, n_frames: int, dt_ns: float = 1.0, meta=None) -> Bars:
    """fast_zigzag -> homology_persistence -> Bars (imports dionysus lazily)."""
    import dionysus as d

    cone = d.fast_zigzag(simplices, times)
    r, _v = d.homology_persistence(cone, method="matrix_v")
    dgms = d.init_zigzag_diagrams(r, cone)
    return diagrams_to_bars(dgms, n_frames, dt_ns, meta)
