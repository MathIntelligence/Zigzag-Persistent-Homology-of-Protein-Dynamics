#!/usr/bin/env python3
"""Localize persistent H1 representatives across an mdCATH temperature series.

The analysis is deliberately fixed-formula and does not fit states, classifiers,
weights, or temperature-specific thresholds.  It compares residue participation in
the five longest non-boundary H1 bars with loss of native C-alpha contacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import spearmanr

from topomol_workbench.analysis import run_zigzag_analysis
from topomol_workbench.loaders import load_h5


TEMPERATURES = (320, 348, 379, 413, 450)
REPLICAS = (0, 1, 2, 3, 4)
REPRESENTATIVE_PERCENTAGES = (10.0, 25.0, 50.0, 75.0, 90.0)


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_contact_pairs(
    reference_coordinates: np.ndarray,
    cutoff_angstrom: float,
    minimum_sequence_separation: int,
) -> np.ndarray:
    """Return local C-alpha index pairs defining reference native contacts."""
    count = reference_coordinates.shape[0]
    pairs = []
    for left in range(count):
        for right in range(left + minimum_sequence_separation, count):
            distance = np.linalg.norm(
                reference_coordinates[left] - reference_coordinates[right]
            )
            if distance < cutoff_angstrom:
                pairs.append((left, right))
    if not pairs:
        raise ValueError("The reference structure produced no native contacts.")
    return np.asarray(pairs, dtype=int)


def contact_observables(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    cutoff_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame Q and per-residue incident-contact occupancy."""
    displacements = coordinates[:, pairs[:, 0], :] - coordinates[:, pairs[:, 1], :]
    present = np.linalg.norm(displacements, axis=2) < cutoff_angstrom
    q_by_frame = present.mean(axis=1)
    pair_occupancy = present.mean(axis=0)
    residue_values: list[list[float]] = [
        [] for _ in range(coordinates.shape[1])
    ]
    for (left, right), occupancy in zip(pairs, pair_occupancy, strict=True):
        residue_values[int(left)].append(float(occupancy))
        residue_values[int(right)].append(float(occupancy))
    residue_occupancy = np.asarray(
        [np.mean(values) if values else np.nan for values in residue_values],
        dtype=float,
    )
    return q_by_frame, residue_occupancy


def kabsch_rmsd_angstrom(
    coordinates: np.ndarray, reference_coordinates: np.ndarray
) -> np.ndarray:
    """Align every C-alpha frame to the reference and return RMSD in Angstrom."""
    reference = reference_coordinates - reference_coordinates.mean(axis=0)
    output = np.empty(coordinates.shape[0], dtype=float)
    for frame_index, frame in enumerate(coordinates):
        mobile = frame - frame.mean(axis=0)
        covariance = mobile.T @ reference
        left, _singular_values, right_t = np.linalg.svd(covariance)
        correction = np.eye(3)
        if np.linalg.det(left @ right_t) < 0:
            correction[-1, -1] = -1
        rotation = left @ correction @ right_t
        difference = mobile @ rotation - reference
        output[frame_index] = math.sqrt(float(np.square(difference).sum()) / len(frame))
    return output


def mean_and_sample_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def independent_bootstrap_difference(
    high: np.ndarray,
    low: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    """Percentile CI for difference in independent five-replica group means."""
    high_indices = rng.integers(0, len(high), size=(draws, len(high)))
    low_indices = rng.integers(0, len(low), size=(draws, len(low)))
    differences = high[high_indices].mean(axis=1) - low[low_indices].mean(axis=1)
    low_ci, high_ci = np.quantile(differences, (0.025, 0.975))
    return float(low_ci), float(high_ci)


def contiguous_segments(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    segments = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("mdcath/data/mdcath_dataset_5sicI00.h5"),
    )
    parser.add_argument("--protein", default="5sicI00")
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--top-bars", type=int, default=5)
    parser.add_argument("--frame-interval-ns", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mdcath_5sic_h1_residue_localization_eps8"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epsilon <= 0 or args.top_bars < 1 or args.frame_interval_ns <= 0:
        raise ValueError("Epsilon, top-bar count, and frame interval must be positive.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.h5.resolve()

    reference_trajectory = load_h5(
        h5_path, args.protein, "320", "0", stride=1, max_frames=None
    )
    reference = reference_trajectory.coordinates_angstrom[0]
    residue_labels = reference_trajectory.residue_labels
    contact_pairs = native_contact_pairs(reference, args.epsilon, 4)
    native_degree = np.bincount(contact_pairs.ravel(), minlength=len(residue_labels))

    trajectory_rows: list[dict] = []
    bar_rows: list[dict] = []
    representative_rows: list[dict] = []
    replica_residue_rows: list[dict] = []
    per_residue_data: dict[tuple[int, int, int], dict] = {}

    for temperature in TEMPERATURES:
        for replica in REPLICAS:
            print(f"Analyzing {args.protein} {temperature} K replica {replica}", flush=True)
            trajectory = load_h5(
                h5_path,
                args.protein,
                str(temperature),
                str(replica),
                stride=1,
                max_frames=None,
            )
            if trajectory.residue_labels != residue_labels:
                raise ValueError("C-alpha residue labels changed between trajectories.")
            result = run_zigzag_analysis(
                trajectory=trajectory,
                epsilon_angstrom=args.epsilon,
                max_simplex_dimension=2,
                homology_dimension=1,
                frame_interval_ns=args.frame_interval_ns,
                stride=1,
                top_bars=args.top_bars,
                exclude_boundary_bars=True,
            )
            q_by_frame, residue_contact_occupancy = contact_observables(
                trajectory.coordinates_angstrom, contact_pairs, args.epsilon
            )
            rmsd = kabsch_rmsd_angstrom(trajectory.coordinates_angstrom, reference)
            finite_h1 = [
                row
                for row in result.barcode_rows
                if row["dimension"] == 1 and row["finite_positive"]
            ]
            trajectory_rows.append(
                {
                    "protein": args.protein,
                    "temperature_K": temperature,
                    "replica": replica,
                    "frames": len(trajectory.source_frames),
                    "ca_atoms": len(residue_labels),
                    "native_contacts": len(contact_pairs),
                    "positive_finite_H1_bars": len(finite_h1),
                    "selected_nonboundary_H1_bars": len(result.top_bars),
                    "H1_tau_top_ns": result.tau_top_by_dimension[1],
                    "mean_rmsd_A": float(np.mean(rmsd)),
                    "median_rmsd_A": float(np.median(rmsd)),
                    "mean_native_contact_Q": float(np.mean(q_by_frame)),
                    "median_native_contact_Q": float(np.median(q_by_frame)),
                }
            )
            lifetimes = {bar.rank: bar.lifetime_ns for bar in result.top_bars}
            for bar in result.top_bars:
                bar_rows.append(
                    {
                        "protein": args.protein,
                        "temperature_K": temperature,
                        "replica": replica,
                        **asdict(bar),
                    }
                )
            views_by_residue: dict[int, set[tuple[int, float]]] = defaultdict(set)
            for row in result.representatives:
                residue_index = int(row["local_atom_index"])
                view = (int(row["rank"]), float(row["percentage"]))
                views_by_residue[residue_index].add(view)
                representative_rows.append(
                    {
                        "protein": args.protein,
                        "temperature_K": temperature,
                        "replica": replica,
                        **row,
                        "bar_lifetime_ns": lifetimes[int(row["rank"])],
                    }
                )
            expected_views = len(result.top_bars) * len(REPRESENTATIVE_PERCENTAGES)
            recovered_views = {
                (int(row["rank"]), float(row["percentage"]))
                for row in result.representatives
            }
            selected_type_counts = {
                interval_type: sum(
                    bar.interval_type == interval_type for bar in result.top_bars
                )
                for interval_type in ("cc", "co", "oc", "oo")
            }
            trajectory_rows[-1].update(
                {
                    "representative_views_requested": expected_views,
                    "representative_views_with_nonzero_support": len(recovered_views),
                    "representative_view_recovery_fraction": len(recovered_views)
                    / expected_views,
                    **{
                        f"selected_{interval_type}_bars": count
                        for interval_type, count in selected_type_counts.items()
                    },
                }
            )
            total_lifetime_view_weight = (
                len(REPRESENTATIVE_PERCENTAGES) * sum(lifetimes.values())
            )
            for residue_index, residue_label in enumerate(residue_labels):
                views = views_by_residue.get(residue_index, set())
                weighted = sum(lifetimes[rank] for rank, _percentage in views)
                values = {
                    "protein": args.protein,
                    "temperature_K": temperature,
                    "replica": replica,
                    "local_ca_index": residue_index,
                    "residue_label": residue_label,
                    "native_contact_degree": int(native_degree[residue_index]),
                    "representative_views_present": len(views),
                    "representative_views_total": expected_views,
                    "view_frequency": len(views) / expected_views,
                    "lifetime_weighted_view_frequency": weighted
                    / total_lifetime_view_weight,
                    "native_contact_occupancy": float(
                        residue_contact_occupancy[residue_index]
                    ),
                }
                replica_residue_rows.append(values)
                per_residue_data[(temperature, replica, residue_index)] = values
            replica_profile = np.asarray(
                [
                    per_residue_data[(temperature, replica, residue_index)][
                        "lifetime_weighted_view_frequency"
                    ]
                    for residue_index in range(len(residue_labels))
                ],
                dtype=float,
            )
            replica_mass = float(replica_profile.sum())
            replica_probabilities = (
                replica_profile[replica_profile > 0] / replica_mass
            )
            replica_effective_count = float(
                np.exp(
                    -np.sum(
                        replica_probabilities * np.log(replica_probabilities)
                    )
                )
            )
            trajectory_rows[-1].update(
                {
                    "within_replica_effective_localized_residue_count": (
                        replica_effective_count
                    ),
                    "within_replica_effective_localized_fraction": (
                        replica_effective_count / len(residue_labels)
                    ),
                    "within_replica_top15_localization_mass_fraction": float(
                        np.sort(replica_profile)[
                            -min(15, len(replica_profile)) :
                        ].sum()
                        / replica_mass
                    ),
                }
            )

    aggregate_rows: list[dict] = []
    for temperature in TEMPERATURES:
        for residue_index, residue_label in enumerate(residue_labels):
            rows = [
                per_residue_data[(temperature, replica, residue_index)]
                for replica in REPLICAS
            ]
            view_mean, view_sd = mean_and_sample_sd([row["view_frequency"] for row in rows])
            weighted_mean, weighted_sd = mean_and_sample_sd(
                [row["lifetime_weighted_view_frequency"] for row in rows]
            )
            contact_mean, contact_sd = mean_and_sample_sd(
                [row["native_contact_occupancy"] for row in rows]
            )
            aggregate_rows.append(
                {
                    "protein": args.protein,
                    "temperature_K": temperature,
                    "local_ca_index": residue_index,
                    "residue_label": residue_label,
                    "native_contact_degree": int(native_degree[residue_index]),
                    "mean_view_frequency": view_mean,
                    "sd_view_frequency_across_replicas": view_sd,
                    "mean_lifetime_weighted_view_frequency": weighted_mean,
                    "sd_lifetime_weighted_view_frequency_across_replicas": weighted_sd,
                    "mean_native_contact_occupancy": contact_mean,
                    "sd_native_contact_occupancy_across_replicas": contact_sd,
                    "replicas_with_any_representative_participation": sum(
                        row["view_frequency"] > 0 for row in rows
                    ),
                }
            )

    aggregate_lookup = {
        (int(row["temperature_K"]), int(row["local_ca_index"])): row
        for row in aggregate_rows
    }
    rng = np.random.default_rng(args.seed)
    endpoint_rows: list[dict] = []
    for residue_index, residue_label in enumerate(residue_labels):
        low = np.asarray(
            [
                per_residue_data[(320, replica, residue_index)][
                    "lifetime_weighted_view_frequency"
                ]
                for replica in REPLICAS
            ]
        )
        high = np.asarray(
            [
                per_residue_data[(450, replica, residue_index)][
                    "lifetime_weighted_view_frequency"
                ]
                for replica in REPLICAS
            ]
        )
        low_contact = np.asarray(
            [
                per_residue_data[(320, replica, residue_index)][
                    "native_contact_occupancy"
                ]
                for replica in REPLICAS
            ]
        )
        high_contact = np.asarray(
            [
                per_residue_data[(450, replica, residue_index)][
                    "native_contact_occupancy"
                ]
                for replica in REPLICAS
            ]
        )
        topo_ci = independent_bootstrap_difference(
            high, low, rng, args.bootstrap_draws
        )
        contact_ci = independent_bootstrap_difference(
            high_contact, low_contact, rng, args.bootstrap_draws
        )
        endpoint_rows.append(
            {
                "protein": args.protein,
                "local_ca_index": residue_index,
                "residue_label": residue_label,
                "native_contact_degree": int(native_degree[residue_index]),
                "mean_weighted_frequency_320K": float(low.mean()),
                "mean_weighted_frequency_450K": float(high.mean()),
                "delta_weighted_frequency_450_minus_320": float(
                    high.mean() - low.mean()
                ),
                "delta_weighted_frequency_bootstrap_ci_low": topo_ci[0],
                "delta_weighted_frequency_bootstrap_ci_high": topo_ci[1],
                "mean_contact_occupancy_320K": float(low_contact.mean()),
                "mean_contact_occupancy_450K": float(high_contact.mean()),
                "delta_contact_occupancy_450_minus_320": float(
                    high_contact.mean() - low_contact.mean()
                ),
                "delta_contact_occupancy_bootstrap_ci_low": contact_ci[0],
                "delta_contact_occupancy_bootstrap_ci_high": contact_ci[1],
            }
        )

    # Temperature-level summaries use replicate means, never pooled-frame means.
    temperature_rows = []
    for temperature in TEMPERATURES:
        rows = [row for row in trajectory_rows if row["temperature_K"] == temperature]
        summary = {"protein": args.protein, "temperature_K": temperature, "replicas": len(rows)}
        for field in ("H1_tau_top_ns", "mean_rmsd_A", "mean_native_contact_Q"):
            mean, sd = mean_and_sample_sd([float(row[field]) for row in rows])
            summary[f"mean_{field}"] = mean
            summary[f"sd_{field}_across_replicas"] = sd
        for field in (
            "within_replica_effective_localized_residue_count",
            "within_replica_effective_localized_fraction",
            "within_replica_top15_localization_mass_fraction",
        ):
            mean, sd = mean_and_sample_sd([float(row[field]) for row in rows])
            summary[f"mean_{field}"] = mean
            summary[f"sd_{field}_across_replicas"] = sd
        recovery_mean, recovery_sd = mean_and_sample_sd(
            [float(row["representative_view_recovery_fraction"]) for row in rows]
        )
        summary["mean_representative_view_recovery_fraction"] = recovery_mean
        summary["sd_representative_view_recovery_fraction_across_replicas"] = recovery_sd
        for interval_type in ("cc", "co", "oc", "oo"):
            summary[f"selected_{interval_type}_bar_count"] = sum(
                int(row[f"selected_{interval_type}_bars"]) for row in rows
            )
        summary["median_selected_bar_lifetime_ns"] = float(
            np.median(
                [
                    row["lifetime_ns"]
                    for row in bar_rows
                    if row["temperature_K"] == temperature
                ]
            )
        )
        localization_values = np.asarray(
            [
                aggregate_lookup[(temperature, residue_index)][
                    "mean_lifetime_weighted_view_frequency"
                ]
                for residue_index in range(len(residue_labels))
            ]
        )
        localization_mass = float(localization_values.sum())
        positive_probabilities = (
            localization_values[localization_values > 0] / localization_mass
        )
        summary["mean_representative_residue_count"] = localization_mass
        summary["effective_localized_residue_count"] = float(
            np.exp(-np.sum(positive_probabilities * np.log(positive_probabilities)))
        )
        summary["top15_localization_mass_fraction"] = float(
            np.sort(localization_values)[-min(15, len(localization_values)) :].sum()
            / localization_mass
        )
        summary["residues_with_nonzero_mean_localization"] = int(
            np.count_nonzero(localization_values)
        )
        temperature_rows.append(summary)

    top_count = min(15, len(residue_labels))
    top_indices_by_temperature: dict[int, list[int]] = {}
    segment_rows = []
    for temperature in TEMPERATURES:
        ranked = sorted(
            (
                aggregate_lookup[(temperature, residue_index)]
                for residue_index in range(len(residue_labels))
            ),
            key=lambda row: (
                -float(row["mean_lifetime_weighted_view_frequency"]),
                int(row["local_ca_index"]),
            ),
        )
        top_indices = sorted(int(row["local_ca_index"]) for row in ranked[:top_count])
        top_indices_by_temperature[temperature] = top_indices
        for start, stop in contiguous_segments(top_indices):
            segment_rows.append(
                {
                    "temperature_K": temperature,
                    "start_local_ca_index": start,
                    "stop_local_ca_index": stop,
                    "start_residue_label": residue_labels[start],
                    "stop_residue_label": residue_labels[stop],
                    "residue_count": stop - start + 1,
                }
            )

    delta_topology = np.asarray(
        [row["delta_weighted_frequency_450_minus_320"] for row in endpoint_rows]
    )
    delta_contacts = np.asarray(
        [row["delta_contact_occupancy_450_minus_320"] for row in endpoint_rows]
    )
    contact_defined = np.isfinite(delta_contacts)
    delta_rho = float(
        spearmanr(
            delta_topology[contact_defined], delta_contacts[contact_defined]
        ).statistic
    )
    overlap = sorted(
        set(top_indices_by_temperature[320]) & set(top_indices_by_temperature[450])
    )
    union = sorted(
        set(top_indices_by_temperature[320]) | set(top_indices_by_temperature[450])
    )
    global_summary = {
        "spearman_rho_residue_delta_topology_vs_delta_contact_occupancy": delta_rho,
        "residues_with_defined_native_contact_occupancy": int(contact_defined.sum()),
        "residues_without_reference_native_contacts": int((~contact_defined).sum()),
        "top_residue_count_per_endpoint": top_count,
        "top_residue_endpoint_overlap_count": len(overlap),
        "top_residue_endpoint_jaccard": len(overlap) / len(union),
        "overlapping_residue_labels": [residue_labels[index] for index in overlap],
        "interpretation": (
            "Descriptive residue-level association only; adjacent residues and multiple "
            "contacts are not statistically independent. Representatives are non-unique."
        ),
    }
    expected_full_frames = max(int(row["frames"]) for row in trajectory_rows)
    full_length_keys = {
        (int(row["temperature_K"]), int(row["replica"]))
        for row in trajectory_rows
        if int(row["frames"]) == expected_full_frames
    }
    full_length_profiles: dict[int, list[float]] = {}
    full_length_temperature_summary: dict[str, dict] = {}
    for temperature in TEMPERATURES:
        included_replicas = sorted(
            replica
            for candidate_temperature, replica in full_length_keys
            if candidate_temperature == temperature
        )
        profile = [
            float(
                np.mean(
                    [
                        per_residue_data[(temperature, replica, residue_index)][
                            "lifetime_weighted_view_frequency"
                        ]
                        for replica in included_replicas
                    ]
                )
            )
            for residue_index in range(len(residue_labels))
        ]
        full_length_profiles[temperature] = profile
        total = float(sum(profile))
        probabilities = np.asarray([value / total for value in profile if value > 0])
        full_length_temperature_summary[str(temperature)] = {
            "replicas": included_replicas,
            "replica_count": len(included_replicas),
            "effective_localized_residue_count": float(
                np.exp(-np.sum(probabilities * np.log(probabilities)))
            ),
            "top15_localization_mass_fraction": float(
                sum(sorted(profile, reverse=True)[: min(15, len(profile))]) / total
            ),
            "mean_representative_residue_count": total,
            "residues_with_nonzero_mean_localization": sum(value > 0 for value in profile),
        }
    full_low = set(
        sorted(
            range(len(residue_labels)),
            key=lambda index: full_length_profiles[320][index],
            reverse=True,
        )[:top_count]
    )
    full_high = set(
        sorted(
            range(len(residue_labels)),
            key=lambda index: full_length_profiles[450][index],
            reverse=True,
        )[:top_count]
    )
    full_length_sensitivity = {
        "inclusion_rule": f"{expected_full_frames} saved frames only",
        "by_temperature": full_length_temperature_summary,
        "endpoint_top15_overlap_count": len(full_low & full_high),
        "endpoint_top15_jaccard": len(full_low & full_high) / len(full_low | full_high),
    }
    global_summary["full_length_sensitivity"] = full_length_sensitivity

    selected_bar_type = {
        (int(row["temperature_K"]), int(row["replica"]), int(row["rank"])): row[
            "interval_type"
        ]
        for row in bar_rows
    }
    recovery_views_by_type: dict[str, set[tuple[int, int, int, float]]] = defaultdict(
        set
    )
    for row in representative_rows:
        key = (
            int(row["temperature_K"]),
            int(row["replica"]),
            int(row["rank"]),
        )
        recovery_views_by_type[selected_bar_type[key]].add(
            (*key, float(row["percentage"]))
        )
    recovery_by_type_rows = []
    for interval_type in ("cc", "co", "oc", "oo"):
        selected_bars = sum(
            row["interval_type"] == interval_type for row in bar_rows
        )
        requested_views = selected_bars * len(REPRESENTATIVE_PERCENTAGES)
        recovered_views = len(recovery_views_by_type[interval_type])
        recovery_by_type_rows.append(
            {
                "protein": args.protein,
                "interval_type": interval_type,
                "selected_bars": selected_bars,
                "requested_views": requested_views,
                "recovered_views": recovered_views,
                "recovery_fraction": (
                    recovered_views / requested_views if requested_views else ""
                ),
            }
        )

    write_csv(
        output_dir / "trajectory_summary.csv",
        trajectory_rows,
        list(trajectory_rows[0]),
    )
    write_csv(output_dir / "selected_H1_bars.csv", bar_rows, list(bar_rows[0]))
    write_csv(
        output_dir / "representative_residue_views.csv",
        representative_rows,
        list(representative_rows[0]),
    )
    write_csv(
        output_dir / "residue_localization_by_replica.csv",
        replica_residue_rows,
        list(replica_residue_rows[0]),
    )
    write_csv(
        output_dir / "residue_localization_by_temperature.csv",
        aggregate_rows,
        list(aggregate_rows[0]),
    )
    write_csv(
        output_dir / "residue_endpoint_changes.csv",
        endpoint_rows,
        list(endpoint_rows[0]),
    )
    write_csv(
        output_dir / "temperature_summary.csv",
        temperature_rows,
        list(temperature_rows[0]),
    )
    write_csv(
        output_dir / "top15_contiguous_segments_by_temperature.csv",
        segment_rows,
        list(segment_rows[0]),
    )
    write_csv(
        output_dir / "representative_recovery_by_interval_type.csv",
        recovery_by_type_rows,
        list(recovery_by_type_rows[0]),
    )
    (output_dir / "full_length_sensitivity_summary.json").write_text(
        json.dumps(full_length_sensitivity, indent=2) + "\n", encoding="utf-8"
    )

    temperatures = np.asarray(TEMPERATURES)
    indices = np.arange(len(residue_labels))
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    for temperature in TEMPERATURES:
        values = np.asarray(
            [
                aggregate_lookup[(temperature, index)][
                    "mean_lifetime_weighted_view_frequency"
                ]
                for index in indices
            ]
        )
        axes[0].plot(indices, values, label=f"{temperature} K", linewidth=1.5)
    axes[0].set_ylabel("weighted H1 representative frequency")
    axes[0].set_title(f"{args.protein} persistent H1 representative localization")
    axes[0].legend(ncol=5, fontsize=9)
    heatmap = np.asarray(
        [
            [
                aggregate_lookup[(temperature, index)][
                    "mean_lifetime_weighted_view_frequency"
                ]
                for index in indices
            ]
            for temperature in TEMPERATURES
        ]
    )
    image = axes[1].imshow(heatmap, aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_yticks(np.arange(len(temperatures)), labels=[f"{value} K" for value in temperatures])
    axes[1].set_ylabel("temperature")
    axes[1].set_title("Replica-mean residue localization")
    fig.colorbar(image, ax=axes[1], label="weighted frequency", pad=0.01)
    axes[2].plot(indices, delta_topology, color="#7b2cbf", label="H1 localization change")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("450 K minus 320 K")
    axes[2].set_xlabel("local C-alpha index (see CSV for residue labels)")
    axes[2].set_title("Endpoint redistribution of persistent H1 representatives")
    fig.savefig(output_dir / "h1_residue_localization_temperature_series.png", dpi=220)
    fig.savefig(output_dir / "h1_residue_localization_temperature_series.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = axis.scatter(
        delta_contacts[contact_defined],
        delta_topology[contact_defined],
        c=native_degree[contact_defined],
        cmap="viridis",
        s=35,
        alpha=0.85,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("native-contact occupancy change (450 K minus 320 K)")
    axis.set_ylabel("weighted H1 representative-frequency change")
    axis.set_title(f"Residue redistribution versus native-contact loss (Spearman rho={delta_rho:.3f})")
    fig.colorbar(scatter, ax=axis, label="reference native-contact degree")
    fig.savefig(output_dir / "h1_localization_vs_contact_loss.png", dpi=220)
    fig.savefig(output_dir / "h1_localization_vs_contact_loss.pdf")
    plt.close(fig)

    endpoint_ranked = sorted(
        endpoint_rows,
        key=lambda row: abs(float(row["delta_weighted_frequency_450_minus_320"])),
        reverse=True,
    )
    strongest = endpoint_ranked[:10]
    low_temp = temperature_rows[0]
    high_temp = temperature_rows[-1]
    full_low_summary = full_length_temperature_summary["320"]
    full_high_summary = full_length_temperature_summary["450"]
    total_requested_views = sum(
        int(row["representative_views_requested"]) for row in trajectory_rows
    )
    total_recovered_views = sum(
        int(row["representative_views_with_nonzero_support"])
        for row in trajectory_rows
    )
    complete_recovery = total_recovered_views == total_requested_views
    if complete_recovery:
        validation_heading = "representative recovery passed"
        validation_paragraph = (
            f"The direct live zigzag basis recovered nonzero molecular H1 support for all "
            f"{total_recovered_views}/{total_requested_views} requested top-bar views "
            f"across all temperatures and replicas (320 K: "
            f"{low_temp['mean_representative_view_recovery_fraction']:.1%}; 450 K: "
            f"{high_temp['mean_representative_view_recovery_fraction']:.1%}). The same "
            "implementation recovers the exact known loop for synthetic cc, co, oc, "
            "and oo bars, and every molecular chain is checked for zero boundary and "
            "current-frame edge membership."
        )
        localization_observation = (
            f"On the ensemble-mean localization profile, the effective localized-residue count changed from "
            f"{low_temp['effective_localized_residue_count']:.1f} at 320 K to "
            f"{high_temp['effective_localized_residue_count']:.1f} at 450 K, while "
            f"the top-15 localization-mass share changed from "
            f"{low_temp['top15_localization_mass_fraction']:.1%} to "
            f"{high_temp['top15_localization_mass_fraction']:.1%}."
        )
    else:
        validation_heading = "residue localization remains incomplete"
        validation_paragraph = (
            f"The direct live zigzag basis recovered nonzero molecular H1 support for "
            f"{total_recovered_views}/{total_requested_views} requested top-bar views. "
            "The incomplete views make residue-frequency and concentration estimates "
            "coverage-biased diagnostics; barcode and H1 tau_top are unaffected."
        )
        localization_observation = (
            f"Among recovered supports, the ensemble-mean profile's effective localized-residue count changed "
            f"from {low_temp['effective_localized_residue_count']:.1f} at 320 K to "
            f"{high_temp['effective_localized_residue_count']:.1f} at 450 K, while "
            f"the top-15 share changed from "
            f"{low_temp['top15_localization_mass_fraction']:.1%} to "
            f"{high_temp['top15_localization_mass_fraction']:.1%}; this remains "
            "coverage-biased and diagnostic only."
        )
    report_lines = [
        f"# {args.protein} H1 representative-residue localization",
        "",
        "## Fixed analysis design",
        "",
        f"- Protein: `{args.protein}`; {len(residue_labels)} C-alpha atoms from `pdbProteinAtoms`.",
        f"- Temperatures: {', '.join(map(str, TEMPERATURES))} K; five replicas per temperature.",
        f"- Zigzag topology: H1, {args.epsilon:g} Angstrom Rips cutoff, simplex dimension through 2, all saved frames at {args.frame_interval_ns:g} ns spacing.",
        f"- Localization set: five longest positive finite H1 bars per trajectory after excluding bars touching either trajectory boundary; representatives sampled at {', '.join(f'{value:g}%' for value in REPRESENTATIVE_PERCENTAGES)} of each bar.",
        f"- Native contacts: reference C-alpha pairs with sequence separation at least 4 and reference distance below {args.epsilon:g} Angstrom; {len(contact_pairs)} pairs.",
        "- No learned states, classifiers, fitted weights, or test-trajectory normalization were used.",
        "",
        f"## Validation status: {validation_heading}",
        "",
        validation_paragraph,
        "",
        "## Main observations",
        "",
        f"- Replica-mean H1 tau_top changed from {low_temp['mean_H1_tau_top_ns']:.3f} +/- {low_temp['sd_H1_tau_top_ns_across_replicas']:.3f} ns at 320 K to {high_temp['mean_H1_tau_top_ns']:.3f} +/- {high_temp['sd_H1_tau_top_ns_across_replicas']:.3f} ns at 450 K.",
        f"- Mean native-contact Q changed from {low_temp['mean_mean_native_contact_Q']:.3f} +/- {low_temp['sd_mean_native_contact_Q_across_replicas']:.3f} to {high_temp['mean_mean_native_contact_Q']:.3f} +/- {high_temp['sd_mean_native_contact_Q_across_replicas']:.3f}.",
        f"- The top-{top_count} localized-residue sets at 320 and 450 K shared {len(overlap)} residues (Jaccard {global_summary['top_residue_endpoint_jaccard']:.3f}).",
        f"- {localization_observation} The primary within-replica mean N_effective changed from {low_temp['mean_within_replica_effective_localized_residue_count']:.1f} +/- {low_temp['sd_within_replica_effective_localized_residue_count_across_replicas']:.1f} to {high_temp['mean_within_replica_effective_localized_residue_count']:.1f} +/- {high_temp['sd_within_replica_effective_localized_residue_count_across_replicas']:.1f} residues; divided by protein length, it changed from {low_temp['mean_within_replica_effective_localized_fraction']:.3f} to {high_temp['mean_within_replica_effective_localized_fraction']:.3f}.",
        f"- Counting missing views as empty, the mean lifetime-weighted recovered support contained {low_temp['mean_representative_residue_count']:.2f} residues at 320 K and {high_temp['mean_representative_residue_count']:.2f} at 450 K.",
        f"- Across the {int(contact_defined.sum())} residues with at least one reference native contact, the descriptive Spearman association between change in H1 representative localization and change in native-contact occupancy was rho={delta_rho:.3f}; the other {int((~contact_defined).sum())} residues have undefined incident-contact occupancy.",
        f"- Full-length sensitivity: after excluding every trajectory shorter than {expected_full_frames} frames, the effective localized-residue count was {full_low_summary['effective_localized_residue_count']:.1f} at 320 K ({full_low_summary['replica_count']} replicas) and {full_high_summary['effective_localized_residue_count']:.1f} at 450 K ({full_high_summary['replica_count']} replicas); the corresponding top-15 shares were {full_low_summary['top15_localization_mass_fraction']:.1%} and {full_high_summary['top15_localization_mass_fraction']:.1%}.",
        "",
        "## Largest localization changes",
        "",
        "| Residue | Delta weighted frequency | 95% replica-bootstrap interval | Delta contact occupancy |",
        "|---|---:|---:|---:|",
    ]
    for row in strongest:
        report_lines.append(
            f"| {row['residue_label']} | {row['delta_weighted_frequency_450_minus_320']:.4f} | "
            f"[{row['delta_weighted_frequency_bootstrap_ci_low']:.4f}, {row['delta_weighted_frequency_bootstrap_ci_high']:.4f}] | "
            f"{row['delta_contact_occupancy_450_minus_320']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This analysis localizes one compatible representative for each selected H1 class; a homology class does not have a unique residue set. The top-five selection is an equal-size prespecified summary, not a claim that all H1 information is carried by those bars. N_effective measures dispersion of the normalized localization mass, not thermodynamic stability by itself. Cross-protein comparison should use the within-replica N_effective divided by residue count; the N_effective of the ensemble-mean profile also includes between-replica heterogeneity. Residues and incident native contacts are spatially correlated, so the residue-level Spearman value is descriptive rather than a significance test. No output here identifies a thermodynamic transition state or proves that a residue causes stability.",
            "",
            "## Reproducibility",
            "",
            "All trajectory-, bar-, representative-, residue-, endpoint-, and temperature-level tables are saved beside this report. Bootstrap intervals resample the five 320 K and five 450 K replicas independently with a fixed seed.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    metadata = {
        "analysis": f"{args.protein} persistent H1 representative-residue localization",
        "validation_status": (
            "VALIDATED_REPRESENTATIVE_RECOVERY_COMPLETE"
            if complete_recovery
            else "NEEDS_REVISION_REPRESENTATIVE_RECOVERY_INCOMPLETE"
        ),
        "source_h5": str(h5_path),
        "source_h5_sha256": file_sha256(h5_path),
        "protein": args.protein,
        "temperatures_K": list(TEMPERATURES),
        "replicas": list(REPLICAS),
        "atom_selection": "C-alpha atoms parsed from pdbProteinAtoms",
        "coordinate_unit": "angstrom",
        "epsilon_angstrom": args.epsilon,
        "homology_dimension": 1,
        "maximum_simplex_dimension": 2,
        "stride": 1,
        "saved_frame_interval_ns": args.frame_interval_ns,
        "top_bars_per_trajectory": args.top_bars,
        "exclude_boundary_bars": True,
        "representative_percentages": list(REPRESENTATIVE_PERCENTAGES),
        "representative_algorithm": "direct_live_zigzag_basis_v1",
        "representative_validation": {
            "synthetic_interval_types": ["cc", "co", "oc", "oo"],
            "runtime_checks": [
                "mod-2 zero boundary",
                "current-frame Rips edge membership",
                "non-null support after filled-boundary removal",
            ],
            "requested_views": total_requested_views,
            "recovered_views": total_recovered_views,
            "complete": complete_recovery,
            "by_interval_type": recovery_by_type_rows,
        },
        "tau_top_definition": "sum(L_i^2) / sum(L_i) over positive finite H1 lifetimes",
        "native_contact_definition": {
            "reference": "320 K replica 0 source frame 0",
            "minimum_sequence_separation": 4,
            "reference_distance_less_than_angstrom": args.epsilon,
            "frame_contact_distance_less_than_angstrom": args.epsilon,
            "pair_count": len(contact_pairs),
        },
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "scheme": "independent replica resampling within 320 K and 450 K",
        },
        "representative_nonuniqueness_caveat": (
            "Each residue set is one compatible direct live-basis representative after "
            "disconnected filled-boundary H1 components are removed."
        ),
        "representative_recovery_caveat": (
            "Any missing view after corrected reconstruction makes localization "
            "coverage-biased. Barcode and tau_top are unaffected."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "dionysus": importlib.metadata.version("dionysus"),
        },
        "global_summary": global_summary,
        "output_files": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved validated analysis inputs and outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
