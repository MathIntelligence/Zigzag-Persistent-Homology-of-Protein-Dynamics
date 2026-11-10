"""One-call driver and command-line interface.

    python -m zigzag_dynamics run coords.npy --dt-ns 1.0 --eps 8 --out results/
    python -m zigzag_dynamics run topology.pdb trajectory.xtc --dt-ns 0.1 --stride 10 --out results/
    python -m zigzag_dynamics bench coords.npy            # reference vs vectorised filtration

Outputs in ``--out``:
    bars.csv               every zigzag bar (frames and ns)
    per_frame.csv          frame, time_ns, tss_ns, event_rate_per_ns, state
    states.json            populations, dwell times, transition matrix, timescales
    summary.json           scalars: tau_top, relaxation time, effective states, ...
    occupancy.npy          (T, K) boolean occupancy matrix
    overview.png           TSS(t), event rate, state trajectory, autocorrelation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .filtration import build_filtration, FiltrationStats
from .scores import (
    event_rate,
    occupancy_autocorrelation,
    occupancy_matrix,
    per_frame_tss,
    relaxation_time,
    tau_top,
    total_persistence,
)
from .descriptors import frame_features, smooth_features
from .states import StateDecomposition, decompose_states, select_n_states, switch_alignment
from .zigzag import Bars, compute_bars


@dataclass
class TrajectoryResult:
    bars: Bars
    stats: FiltrationStats
    tss: np.ndarray
    rate: np.ndarray
    occupancy: np.ndarray
    autocorr: np.ndarray
    states: StateDecomposition
    scalars: dict
    frame_features: np.ndarray | None = None
    spectrum: dict | None = None
    timings: dict = field(default_factory=dict)

    def write(self, out: Path, *, plot: bool = True) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        recs = self.bars.to_records()
        if recs:
            with (out / "bars.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0]))
                w.writeheader()
                w.writerows(recs)
        T = self.bars.n_frames
        with (out / "per_frame.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_ns", "tss_ns", "event_rate_per_ns", "state"])
            for t in range(T):
                w.writerow([t, t * self.bars.dt_ns, f"{self.tss[t]:.6g}",
                            f"{self.rate[t]:.6g}", int(self.states.labels[t])])
        summary = self.states.summary()
        if self.spectrum:
            summary["timescale_spectrum_ns"] = {str(k): [round(float(x), 3) if np.isfinite(x) else None for x in v]
                                               for k, v in self.spectrum.items()}
        if self.states.signatures is not None:
            summary["state_signatures"] = self.states.signatures.round(3).tolist()
        (out / "states.json").write_text(json.dumps(summary, indent=2) + "\n")
        if self.frame_features is not None:
            np.save(out / "frame_features.npy", self.frame_features)
        (out / "summary.json").write_text(json.dumps(
            {**self.scalars, "filtration": self.stats.__dict__, "timings_s": self.timings},
            indent=2, default=float) + "\n")
        np.save(out / "occupancy.npy", self.occupancy)
        np.save(out / "autocorrelation.npy", self.autocorr)
        if plot:
            plot_overview(self, out / "overview.png")


# --------------------------------------------------------------------------- #
def run_trajectory(
    coords: np.ndarray,
    dt_ns: float,
    *,
    eps: float = 8.0,
    max_dim: int = 2,
    hysteresis: tuple[float, float] | None = None,
    debounce: int = 0,
    score_dims=(1,),
    state_dims=(1,),
    min_lifetime_frames: float = 2.0,
    n_states: int | str = "auto",
    max_states: int = 6,
    state_features: str = "persistence_image",
    pi_thresh: float = 12.0,
    pi_resolution: int = 16,
    smooth_frames: int = 5,
    min_timescale_frames: float | None = None,
    lag_frames: int = 1,
    rate_window_frames: int = 10,
    meta: dict | None = None,
) -> TrajectoryResult:
    """Coordinates -> bars -> scores -> topological states."""
    coords = np.asarray(coords)
    T = coords.shape[0]
    timings = {}

    t0 = time.perf_counter()
    simplices, times, stats = build_filtration(
        coords, eps=eps, max_dim=max_dim, hysteresis=hysteresis,
        debounce=debounce, return_stats=True)
    timings["filtration"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bars = compute_bars(simplices, times, T, dt_ns, meta=dict(
        eps=eps, max_dim=max_dim, hysteresis=hysteresis, debounce=debounce, **(meta or {})))
    timings["zigzag"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tss = per_frame_tss(bars, score_dims)
    rate = event_rate(bars, rate_window_frames, score_dims)
    B = occupancy_matrix(bars, state_dims, min_lifetime_frames)
    C = occupancy_autocorrelation(B)
    timings["scores"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if state_features == "persistence_image":
        X = frame_features(coords, thresh=pi_thresh, resolution=pi_resolution)
    elif state_features == "occupancy":
        X = B.astype(float)
    else:
        raise ValueError("state_features must be 'persistence_image' or 'occupancy'")
    X = smooth_features(X, smooth_frames)
    if min_timescale_frames is None:
        min_timescale_frames = max(3.0 * smooth_frames, 5.0)
    if n_states == "auto":
        k, spectrum = select_n_states(X, dt_ns, max_states=max_states, lag_frames=lag_frames,
                                      min_timescale_frames=min_timescale_frames)
    else:
        k, spectrum = int(n_states), None
    states = decompose_states(X, dt_ns, n_states=k, lag_frames=lag_frames, occupancy=B)
    states.extras.update(features=state_features, smooth_frames=smooth_frames,
                         min_timescale_frames=min_timescale_frames, selection="auto" if n_states == "auto" else "fixed")
    timings["states"] = time.perf_counter() - t0

    scalars = {
        "n_frames": int(T),
        "n_atoms": int(coords.shape[1]),
        "dt_ns": float(dt_ns),
        "duration_ns": float(T * dt_ns),
        "n_bars_total": int(len(bars)),
        "n_bars_scored": int(len(bars.in_dims(score_dims))),
        "tau_top_ns": tau_top(bars, score_dims),
        "total_persistence_ns": total_persistence(bars, score_dims),
        "mean_tss_ns": float(tss.mean()),
        "mean_event_rate_per_ns": float(rate.mean()),
        "n_features_in_occupancy": int(B.shape[1]),
        "topological_relaxation_time_ns": relaxation_time(C, dt_ns),
        "n_states": int(states.n_states),
        "effective_states": states.effective_states,
        "state_entropy_nats": states.entropy_nats,
        "slowest_implied_timescale_ns": (float(states.implied_timescales_ns[0])
                                         if states.implied_timescales_ns.size else None),
        "n_state_switches": states.n_switches,
        "event_rate_at_switches_ratio": switch_alignment(states.labels, rate),
    }
    return TrajectoryResult(bars, stats, tss, rate, B, C, states, scalars, X, spectrum, timings)


# --------------------------------------------------------------------------- #
def plot_overview(res: TrajectoryResult, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, dt = res.bars.n_frames, res.bars.dt_ns
    t = np.arange(T) * dt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False,
                             gridspec_kw=dict(height_ratios=[2, 1.2, 1, 1.4]))
    ax = axes[0]
    b = res.bars.in_dims((1,)).sorted_by_lifetime()
    for y, (lo, hi) in enumerate(zip(b.alive_lo, b.alive_hi)):
        ax.plot([lo * dt, hi * dt], [y, y], color="#1f77b4", lw=1.0)
    ax.set_ylabel("H1 bars")
    ax.set_title("zigzag barcode (H1)")
    ax.set_xlim(0, T * dt)

    ax = axes[1]
    ax.plot(t, res.tss, color="#1f77b4", lw=1.2)
    ax.set_ylabel("TSS(t) [ns]")
    ax2 = ax.twinx()
    ax2.plot(t, res.rate, color="#d62728", lw=1.0, alpha=0.8)
    ax2.set_ylabel("events / ns", color="#d62728")
    ax.set_xlim(0, T * dt)

    ax = axes[2]
    S = res.states.n_states
    ax.step(t, res.states.labels, where="post", color="#2ca02c", lw=1.2)
    ax.set_yticks(range(S))
    ax.set_ylabel("topological\nstate")
    ax.set_xlabel("time [ns]")
    ax.set_xlim(0, T * dt)

    ax = axes[3]
    lags = np.arange(res.autocorr.size) * dt
    ax.plot(lags, res.autocorr, color="#9467bd", lw=1.5)
    ax.axhline(np.exp(-1), color="grey", ls=":")
    tr = res.scalars["topological_relaxation_time_ns"]
    if np.isfinite(tr):
        ax.axvline(tr, color="grey", ls="--", label=f"relaxation {tr:.1f} ns")
        ax.legend(loc="upper right")
    ax.set_xlabel("lag [ns]")
    ax.set_ylabel("occupancy\nautocorrelation")
    fig.suptitle(
        f"tau_top={res.scalars['tau_top_ns']:.2f} ns  |  "
        f"states={S} (eff. {res.scalars['effective_states']:.2f})  |  "
        f"slowest timescale={res.scalars['slowest_implied_timescale_ns']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def load_coords(args) -> tuple[np.ndarray, dict]:
    inputs = args.inputs
    if len(inputs) == 1 and inputs[0].endswith(".npy"):
        return np.load(inputs[0]), {"source": inputs[0]}
    if len(inputs) == 2:
        import MDAnalysis as mda
        u = mda.Universe(inputs[0], inputs[1])
        atoms = u.select_atoms(args.selection)
        coords = np.array([atoms.positions.copy() for _ in u.trajectory[::args.stride]])
        return coords, {"topology": inputs[0], "trajectory": inputs[1],
                        "selection": args.selection, "stride": args.stride}
    raise SystemExit("give either one .npy (T,N,3) file or a topology + trajectory pair")


def _cmd_run(args) -> int:
    coords, meta = load_coords(args)
    hyst = tuple(args.hysteresis) if args.hysteresis else None
    res = run_trajectory(
        coords, args.dt_ns * (args.stride if len(args.inputs) == 2 else 1),
        eps=args.eps, max_dim=args.max_dim, hysteresis=hyst, debounce=args.debounce,
        score_dims=tuple(args.dims), state_dims=tuple(args.dims),
        min_lifetime_frames=args.min_lifetime,
        n_states="auto" if args.n_states == "auto" else int(args.n_states), max_states=args.max_states,
        state_features=args.state_features, pi_thresh=args.pi_thresh, smooth_frames=args.smooth,
        lag_frames=args.lag, meta=meta)
    res.write(Path(args.out), plot=not args.no_plot)
    print(json.dumps(res.scalars, indent=2, default=float))
    print("timings:", {k: round(v, 2) for k, v in res.timings.items()})
    print(f"wrote {Path(args.out).resolve()}")
    return 0


def _cmd_bench(args) -> int:
    coords, _ = load_coords(args)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from coords_to_fzz import coords_to_fzz
    except ImportError:
        coords_to_fzz = None
    for md in (2, 3):
        t0 = time.perf_counter()
        s2, t2, st = build_filtration(coords, eps=args.eps, max_dim=md, return_stats=True)
        new = time.perf_counter() - t0
        line = f"max_dim={md}: vectorised {new:.2f}s  simplices={st.n_simplices} toggles={st.n_toggles}"
        if coords_to_fzz is not None:
            t0 = time.perf_counter()
            s1, t1 = coords_to_fzz(coords, eps=args.eps, max_dim=md)
            ref = time.perf_counter() - t0
            line += f"  | reference {ref:.2f}s  speedup {ref / new:.1f}x  identical={(s1 == s2) and (t1 == t2)}"
        print(line)
    if args.hysteresis:
        s3, t3, st = build_filtration(coords, eps=args.eps, max_dim=2,
                                      hysteresis=tuple(args.hysteresis), return_stats=True)
        print(f"hysteresis {tuple(args.hysteresis)}: toggles={st.n_toggles}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="zigzag_dynamics")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        q = sub.add_parser(name)
        q.add_argument("inputs", nargs="+")
        q.add_argument("--eps", type=float, default=8.0)
        q.add_argument("--selection", default="name CA")
        q.add_argument("--stride", type=int, default=1)
        q.add_argument("--hysteresis", type=float, nargs=2, metavar=("EPS_ON", "EPS_OFF"))
    r = sub.choices["run"]
    r.add_argument("--dt-ns", type=float, required=True, help="physical time per saved frame (before --stride)")
    r.add_argument("--out", required=True)
    r.add_argument("--max-dim", type=int, default=2)
    r.add_argument("--dims", type=int, nargs="+", default=[1])
    r.add_argument("--debounce", type=int, default=0)
    r.add_argument("--min-lifetime", type=float, default=2.0, help="frames; bars shorter are ignored for states")
    r.add_argument("--n-states", default="auto", help="'auto' (kinetic selection) or an integer")
    r.add_argument("--max-states", type=int, default=6)
    r.add_argument("--smooth", type=int, default=5, help="moving-average window (frames) for state features")
    r.add_argument("--state-features", choices=["persistence_image", "occupancy"], default="persistence_image")
    r.add_argument("--pi-thresh", type=float, default=12.0, help="ripser threshold (A) for per-frame diagrams")
    r.add_argument("--lag", type=int, default=1)
    r.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    return _cmd_run(args) if args.cmd == "run" else _cmd_bench(args)
