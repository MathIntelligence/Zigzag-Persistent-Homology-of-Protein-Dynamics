"""zigzag_dynamics — from static topology to protein dynamics.

A compact, tested package that turns a C-alpha trajectory into

1. a zigzag filtration (vectorised, bit-exact with ``coords_to_fzz``), with
   optional contact hysteresis / debouncing to suppress cutoff flicker;
2. zigzag persistence bars in absolute physical time (ns);
3. per-trajectory stability scalars (``tau_top``, TSS(t), event rate,
   occupancy autocorrelation and its relaxation time);
4. a *topological state decomposition*: frames are grouped by which persistent
   features are alive, giving discrete topological states with populations,
   dwell times, a transition matrix and implied timescales — the ensemble
   quantities that a single static structure cannot provide.

Modules
-------
filtration   coordinates -> (simplices, times) for dionysus.fast_zigzag
zigzag       fast_zigzag wrapper -> Bars in frames and ns
scores       tau_top, TSS(t), event rate, occupancy autocorrelation
states       occupancy matrix -> topological states + kinetics
pipeline     one-call driver + CLI (python -m zigzag_dynamics)
"""

from .filtration import build_filtration, FiltrationStats
from .zigzag import Bars, compute_bars
from .scores import (
    tau_top,
    per_frame_tss,
    event_rate,
    occupancy_matrix,
    occupancy_autocorrelation,
    relaxation_time,
)
from .states import decompose_states, StateDecomposition
from .pipeline import run_trajectory, TrajectoryResult

__all__ = [
    "build_filtration",
    "FiltrationStats",
    "Bars",
    "compute_bars",
    "tau_top",
    "per_frame_tss",
    "event_rate",
    "occupancy_matrix",
    "occupancy_autocorrelation",
    "relaxation_time",
    "decompose_states",
    "StateDecomposition",
    "run_trajectory",
    "TrajectoryResult",
]

__version__ = "0.1.0"
