"""Run zigzag_dynamics over a manifest of mdCATH C-alpha arrays -> summary CSV.

    python zigzag_dynamics/run_mdcath_demo.py mdcath_consecutive_wasserstein_3x3/input \
        --out zigzag_dynamics_results --debounce 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zigzag_dynamics import run_trajectory  # noqa: E402

FIELDS = ["protein", "temperature_K", "replica", "n_frames", "tau_top_ns",
          "topological_relaxation_time_ns", "mean_event_rate_per_ns", "n_states",
          "effective_states", "slowest_implied_timescale_ns", "n_state_switches",
          "event_rate_at_switches_ratio", "populations", "mean_dwell_ns", "toggles", "zigzag_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--debounce", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader((args.input_dir / "trajectory_manifest.csv").open())
            if (args.input_dir / r["array_file"]).exists()]
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in rows:
        coords = np.load(args.input_dir / r["array_file"])
        tag = f'{r["protein"]}_{r["temperature_K"]}_{r["replica"]}'
        res = run_trajectory(coords, float(r["frame_interval_ns"]), eps=args.eps,
                             max_dim=2, debounce=args.debounce, meta=dict(source=tag))
        res.write(args.out / tag)
        s = res.scalars
        out_rows.append({
            "protein": r["protein"], "temperature_K": r["temperature_K"], "replica": r["replica"],
            "n_frames": s["n_frames"], "tau_top_ns": round(s["tau_top_ns"], 3),
            "topological_relaxation_time_ns": round(s["topological_relaxation_time_ns"], 3),
            "mean_event_rate_per_ns": round(s["mean_event_rate_per_ns"], 4),
            "n_states": s["n_states"], "effective_states": round(s["effective_states"], 3),
            "slowest_implied_timescale_ns": (None if s["slowest_implied_timescale_ns"] is None
                                             else round(s["slowest_implied_timescale_ns"], 3)),
            "n_state_switches": s["n_state_switches"],
            "event_rate_at_switches_ratio": round(s["event_rate_at_switches_ratio"], 3),
            "populations": " ".join(f"{p:.3f}" for p in res.states.populations),
            "mean_dwell_ns": " ".join(f"{d:.1f}" for d in res.states.mean_dwell_ns),
            "toggles": res.stats.n_toggles, "zigzag_s": round(res.timings["zigzag"], 2),
        })
        print(tag, {k: out_rows[-1][k] for k in ("tau_top_ns", "n_states", "slowest_implied_timescale_ns")}, flush=True)
    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
