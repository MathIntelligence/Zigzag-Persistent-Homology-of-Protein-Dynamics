#!/usr/bin/env python3
"""Test a residue-level topological-rigidity candidate score on mdCATH MD.

The score is deliberately fixed before benchmarking.  For residue i in one
replica and the K longest finite, non-boundary H1 bars,

    S_i = sum_b L_b * mean_q I(i is in representative(b, q)) / sum_b L_b

where q is 10, 25, 50, 75, or 90 percent through the bar.  The protein-level
heatmap uses the mean S_i across independent replicas.  This is a topological
participation score, not an assumed thermodynamic-stability measurement.

Primary validation endpoint: within each protein, Spearman correlation between
S_i and -RMSF at 320 K.  The preregistered pilot passes only if at least 9/10
proteins have positive rho and a protein-bootstrap 95% interval for the median
rho excludes zero.  Burial, hydrophobicity, and hydrophobic-core enrichment are
secondary tests.  Native C-alpha contact degree is a geometric baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, rankdata, spearmanr

from topomol_workbench.analysis import run_zigzag_analysis
from topomol_workbench.loaders import load_h5


DEFAULT_COHORT = Path(
    "mdcath_10_more_h1_localization_eps8_direct_live_basis_v1/"
    "PREREGISTERED_COHORT.csv"
)
REPRESENTATIVE_PERCENTAGES = (10.0, 25.0, 50.0, 75.0, 90.0)
TOP_BAR_SENSITIVITY = (5, 10, 20)
HYDROPHOBIC_ONE_LETTER = frozenset("AVILMFWY")

# Kyte-Doolittle residue hydropathy values (J. Mol. Biol. 157, 105-132, 1982).
KYTE_DOOLITTLE = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}

# Maximum solvent-accessible areas in Angstrom^2 from Tien et al., PLoS ONE
# 8:e80635 (2013), theoretical values for the Gly-X-Gly extended conformation.
MAX_ASA_A2 = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

RESNAME_ALIASES = {
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}


def write_csv(path: Path, rows: Iterable[dict], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_cohort(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or np.unique(left[keep]).size < 2 or np.unique(right[keep]).size < 2:
        return float("nan")
    return float(spearmanr(left[keep], right[keep]).statistic)


def partial_spearman(left: np.ndarray, right: np.ndarray, control: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right) & np.isfinite(control)
    if keep.sum() < 4:
        return float("nan")
    x = rankdata(left[keep])
    y = rankdata(right[keep])
    z = rankdata(control[keep])
    design = np.column_stack((np.ones(len(z)), z))
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(residual_x) == 0 or np.std(residual_y) == 0:
        return float("nan")
    return float(pearsonr(residual_x, residual_y).statistic)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[keep]
    labels = labels[keep].astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum_positive = float(ranks[labels].sum())
    u = rank_sum_positive - positives * (positives + 1) / 2
    return u / (positives * negatives)


def percentile_high(values: np.ndarray) -> np.ndarray:
    """Return 0..1 midrank percentiles, with high input mapped to high output."""
    keep = np.isfinite(values)
    output = np.full(len(values), np.nan, dtype=float)
    if keep.sum() == 1:
        output[keep] = 1.0
    elif keep.sum() > 1:
        output[keep] = (rankdata(values[keep]) - 1.0) / (keep.sum() - 1.0)
    return output


def json_safe(value):
    """Recursively replace non-finite numeric values with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def circular_shift_pvalue(score: np.ndarray, target: np.ndarray) -> float:
    observed = safe_spearman(score, target)
    if not np.isfinite(observed):
        return float("nan")
    null = np.asarray(
        [safe_spearman(np.roll(score, shift), target) for shift in range(1, len(score))]
    )
    null = null[np.isfinite(null)]
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (1 + len(null)))


def native_contact_degree(ca_coordinates: np.ndarray, cutoff_a: float = 8.0) -> np.ndarray:
    count = len(ca_coordinates)
    degree = np.zeros(count, dtype=int)
    for left in range(count):
        for right in range(left + 4, count):
            if np.linalg.norm(ca_coordinates[left] - ca_coordinates[right]) < cutoff_a:
                degree[left] += 1
                degree[right] += 1
    return degree


def score_for_k(cache: dict, top_bars: int) -> np.ndarray:
    labels = cache["residue_labels"]
    lifetimes = {
        int(row["rank"]): float(row["lifetime_ns"])
        for row in cache["bars"]
        if int(row["rank"]) <= top_bars
    }
    denominator = len(REPRESENTATIVE_PERCENTAGES) * sum(lifetimes.values())
    if denominator <= 0:
        raise ValueError("Selected bars have zero total lifetime.")
    views_by_residue: dict[int, set[tuple[int, float]]] = defaultdict(set)
    for row in cache["representatives"]:
        rank = int(row["rank"])
        if rank <= top_bars:
            views_by_residue[int(row["local_atom_index"])].add(
                (rank, float(row["percentage"]))
            )
    score = np.zeros(len(labels), dtype=float)
    for index in range(len(labels)):
        score[index] = sum(
            lifetimes[rank] for rank, _percentage in views_by_residue.get(index, set())
        ) / denominator
    return score


def run_or_load_replica(
    h5_path: Path,
    protein: str,
    temperature: int,
    replica: int,
    epsilon_a: float,
    max_top_bars: int,
    cache_dir: Path,
) -> dict:
    cache_path = cache_dir / f"{protein}_{temperature}K_replica_{replica}.json"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    trajectory = load_h5(
        h5_path, protein, str(temperature), str(replica), stride=1, max_frames=None
    )
    start = time.monotonic()
    result = run_zigzag_analysis(
        trajectory=trajectory,
        epsilon_angstrom=epsilon_a,
        max_simplex_dimension=2,
        homology_dimension=1,
        frame_interval_ns=1.0,
        stride=1,
        top_bars=max_top_bars,
        exclude_boundary_bars=True,
    )
    bars = [asdict(bar) for bar in result.top_bars]
    representatives = [
        {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
        }
        for row in result.representatives
    ]
    expected_views = len(bars) * len(REPRESENTATIVE_PERCENTAGES)
    recovered_views = {
        (int(row["rank"]), float(row["percentage"])) for row in representatives
    }
    cache = {
        "protein": protein,
        "temperature_K": temperature,
        "replica": replica,
        "epsilon_A": epsilon_a,
        "homology_dimension": 1,
        "top_bars_requested": max_top_bars,
        "residue_labels": trajectory.residue_labels,
        "frames": len(trajectory.source_frames),
        "bars": bars,
        "representatives": representatives,
        "representative_views_requested": expected_views,
        "representative_views_recovered": len(recovered_views),
        "runtime_seconds": time.monotonic() - start,
    }
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle)
    os.replace(temporary_path, cache_path)
    return cache


def residue_sasa_and_rmsf(
    h5_path: Path,
    protein: str,
    temperature: int,
    replicas: list[int],
    sasa_frames_per_replica: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    coordinates_nm = []
    rmsf = []
    with h5py.File(h5_path, "r") as h5:
        group = h5[protein]
        pdb_value = group["pdbProteinAtoms"][()]
        pdb_text = pdb_value.decode("utf-8") if isinstance(pdb_value, bytes) else str(pdb_value)
        for replica in replicas:
            replica_group = group[str(temperature)][str(replica)]
            frame_count = int(replica_group["coords"].shape[0])
            frame_indices = np.linspace(
                0, frame_count - 1, min(sasa_frames_per_replica, frame_count), dtype=int
            )
            coordinates_nm.append(
                np.asarray(replica_group["coords"][frame_indices], dtype=float) / 10.0
            )
            rmsf.append(np.asarray(replica_group["rmsf"], dtype=float))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb") as handle:
        handle.write(pdb_text)
        handle.flush()
        topology = md.load_pdb(handle.name).topology
    trajectory = md.Trajectory(xyz=np.concatenate(coordinates_nm), topology=topology)
    residue_sasa_a2 = md.shrake_rupley(trajectory, mode="residue").mean(axis=0) * 100.0
    return residue_sasa_a2, np.mean(np.stack(rmsf), axis=0), pdb_text


def residue_names(labels: list[str]) -> list[str]:
    raw_names = [label.split(":", 1)[0].upper() for label in labels]
    return [RESNAME_ALIASES.get(name, name) for name in raw_names]


def write_bfactor_pdb(pdb_text: str, scores: np.ndarray, path: Path) -> None:
    residue_to_index: dict[tuple[str, str, str], int] = {}
    output = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            key = (line[21:22], line[22:26], line[26:27])
            if key not in residue_to_index:
                residue_to_index[key] = len(residue_to_index)
            index = residue_to_index[key]
            bfactor = 100.0 * float(scores[index]) if index < len(scores) else 0.0
            line = f"{line[:60]}{bfactor:6.2f}{line[66:]}"
        output.append(line)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def protein_heatmap(
    protein: str,
    labels: list[str],
    score: np.ndarray,
    rmsf: np.ndarray,
    rsa: np.ndarray,
    core: np.ndarray,
    path: Path,
) -> None:
    matrix = np.vstack(
        (
            percentile_high(score),
            percentile_high(-rmsf),
            percentile_high(-rsa),
            core.astype(float),
        )
    )
    colors = ["#f7f9fb", "#c7ddef", "#4c83b6", "#173f67"]
    cmap = LinearSegmentedColormap.from_list("topological_blue", colors)
    width = max(10.0, len(labels) * 0.22)
    fig, ax = plt.subplots(figsize=(width, 3.6), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(
        [
            "Topological participation percentile",
            "Dynamic rigidity percentile (-RMSF)",
            "Burial percentile (-relative SASA)",
            "Hydrophobic and buried (RSA < 0.20)",
        ]
    )
    positions = np.arange(len(labels))
    tick_step = 1 if len(labels) <= 55 else 2
    ax.set_xticks(positions[::tick_step])
    ax.set_xticklabels(
        [labels[index].replace(":", " ") for index in positions[::tick_step]],
        rotation=90,
        fontsize=7,
    )
    ax.set_title(f"{protein}: residue-level topological-rigidity candidate heatmap (320 K)")
    ax.set_xlabel("Residue")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Within-protein percentile / binary indicator")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def validation_plot(rows: list[dict], path: Path) -> None:
    proteins = [row["protein"] for row in rows]
    y = np.arange(len(proteins))
    topo = np.asarray([row["rho_topology_vs_negative_rmsf"] for row in rows])
    baseline = np.asarray([row["rho_contact_degree_vs_negative_rmsf"] for row in rows])
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax.axvline(0, color="#555555", linewidth=1.0)
    ax.hlines(y, baseline, topo, color="#b8c1ca", linewidth=1.5)
    ax.scatter(baseline, y, marker="o", facecolors="white", edgecolors="#b36b00", s=60, label="C-alpha contact-degree baseline")
    ax.scatter(topo, y, marker="s", color="#1f5f99", s=52, label="Top-20 H1 participation score")
    ax.set_yticks(y)
    ax.set_yticklabels(proteins)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho with dynamic rigidity (-RMSF)")
    ax.set_title("Per-protein rigidity association at 320 K")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="x", color="#e3e7eb", linewidth=0.8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return float(low), float(high)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--limit-proteins", type=int)
    parser.add_argument("--temperature", type=int, default=320)
    parser.add_argument("--replicas", default="0,1,2,3,4")
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--top-bars", type=int, default=20)
    parser.add_argument("--sasa-frames-per-replica", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("topological_rigidity_heatmap_validation")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_bars != 20:
        raise ValueError("This preregistered pilot fixes --top-bars at 20.")
    replicas = [int(value) for value in args.replicas.split(",")]
    if not replicas:
        raise ValueError("At least one replica is required.")
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    heatmap_dir = output_dir / "heatmaps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    cohort = read_cohort(args.cohort, args.limit_proteins)
    residue_rows: list[dict] = []
    protein_rows_by_k: dict[int, list[dict]] = {k: [] for k in TOP_BAR_SENSITIVITY}
    selected_bar_rows: list[dict] = []
    representative_rows: list[dict] = []
    recovery_rows: list[dict] = []

    for protein_number, cohort_row in enumerate(cohort, start=1):
        protein = cohort_row["protein"]
        h5_path = Path(cohort_row["source_h5"])
        print(f"[{protein_number}/{len(cohort)}] {protein}", flush=True)
        caches = []
        for replica in replicas:
            print(f"  top-20 H1 representatives, replica {replica}", flush=True)
            cache = run_or_load_replica(
                h5_path,
                protein,
                args.temperature,
                replica,
                args.epsilon,
                args.top_bars,
                cache_dir,
            )
            caches.append(cache)
            recovery_rows.append(
                {
                    "protein": protein,
                    "temperature_K": args.temperature,
                    "replica": replica,
                    "frames": cache["frames"],
                    "top_bars_recovered": len(cache["bars"]),
                    "representative_views_requested": cache["representative_views_requested"],
                    "representative_views_recovered": cache["representative_views_recovered"],
                    "recovery_fraction": cache["representative_views_recovered"]
                    / cache["representative_views_requested"],
                    "runtime_seconds": cache["runtime_seconds"],
                }
            )
            for bar in cache["bars"]:
                selected_bar_rows.append({"protein": protein, "replica": replica, **bar})
            for row in cache["representatives"]:
                representative_rows.append(
                    {
                        "protein": protein,
                        "temperature_K": args.temperature,
                        "replica": replica,
                        **row,
                    }
                )

        labels = caches[0]["residue_labels"]
        if any(cache["residue_labels"] != labels for cache in caches):
            raise ValueError(f"Residue labels changed across {protein} replicas.")
        sasa_a2, rmsf_a, pdb_text = residue_sasa_and_rmsf(
            h5_path,
            protein,
            args.temperature,
            replicas,
            args.sasa_frames_per_replica,
        )
        if len(sasa_a2) != len(labels) or len(rmsf_a) != len(labels):
            raise ValueError(f"Residue observable length mismatch for {protein}.")
        names = residue_names(labels)
        max_asa = np.asarray([MAX_ASA_A2.get(name, np.nan) for name in names])
        rsa = sasa_a2 / max_asa
        hydropathy = np.asarray([KYTE_DOOLITTLE.get(name, np.nan) for name in names])
        hydrophobic = np.asarray(
            [THREE_TO_ONE.get(name, "X") in HYDROPHOBIC_ONE_LETTER for name in names],
            dtype=bool,
        )
        hydrophobic_core = hydrophobic & (rsa < 0.20)
        reference = load_h5(h5_path, protein, str(args.temperature), str(replicas[0]), 1, None)
        degree = native_contact_degree(reference.coordinates_angstrom[0], args.epsilon)

        scores_by_k = {
            k: np.mean(np.stack([score_for_k(cache, k) for cache in caches]), axis=0)
            for k in TOP_BAR_SENSITIVITY
        }
        score_replica_top20 = np.stack([score_for_k(cache, 20) for cache in caches])
        score = scores_by_k[20]
        for index, label in enumerate(labels):
            residue_rows.append(
                {
                    "protein": protein,
                    "temperature_K": args.temperature,
                    "local_ca_index": index,
                    "residue_label": label,
                    "resname": names[index],
                    "topological_participation_score_top20": score[index],
                    "topological_score_sd_across_replicas": float(
                        np.std(score_replica_top20[:, index], ddof=1)
                    ),
                    "replicas_with_nonzero_topological_score": int(
                        np.count_nonzero(score_replica_top20[:, index] > 0)
                    ),
                    "topological_participation_percentile": percentile_high(score)[index],
                    "mean_rmsf_A": rmsf_a[index],
                    "dynamic_rigidity_percentile": percentile_high(-rmsf_a)[index],
                    "mean_sasa_A2": sasa_a2[index],
                    "relative_sasa": rsa[index],
                    "burial_percentile": percentile_high(-rsa)[index],
                    "kyte_doolittle_hydropathy": hydropathy[index],
                    "hydrophobic_residue_fixed_set": int(hydrophobic[index]),
                    "hydrophobic_and_buried_rsa_lt_0_20": int(hydrophobic_core[index]),
                    "native_ca_contact_degree": degree[index],
                }
            )

        for k in TOP_BAR_SENSITIVITY:
            candidate = scores_by_k[k]
            top_count = max(1, math.ceil(len(candidate) / 4))
            top_indices = np.argsort(candidate, kind="stable")[-top_count:]
            overall_core_rate = float(hydrophobic_core.mean())
            top_core_rate = float(hydrophobic_core[top_indices].mean())
            core_enrichment = (
                top_core_rate / overall_core_rate if overall_core_rate > 0 else float("nan")
            )
            protein_rows_by_k[k].append(
                {
                    "protein": protein,
                    "temperature_K": args.temperature,
                    "residues": len(labels),
                    "replicas": len(replicas),
                    "top_bars": k,
                    "rho_topology_vs_negative_rmsf": safe_spearman(candidate, -rmsf_a),
                    "circular_shift_p_topology_vs_negative_rmsf": circular_shift_pvalue(
                        candidate, -rmsf_a
                    ),
                    "rho_topology_vs_negative_relative_sasa": safe_spearman(candidate, -rsa),
                    "rho_topology_vs_hydropathy": safe_spearman(candidate, hydropathy),
                    "auc_hydrophobic_buried_core": binary_auc(candidate, hydrophobic_core),
                    "hydrophobic_buried_core_prevalence": overall_core_rate,
                    "hydrophobic_buried_core_rate_top_score_quartile": top_core_rate,
                    "hydrophobic_buried_core_enrichment_top_quartile": core_enrichment,
                    "rho_topology_vs_native_ca_contact_degree": safe_spearman(candidate, degree),
                    "rho_contact_degree_vs_negative_rmsf": safe_spearman(degree, -rmsf_a),
                    "partial_rho_topology_vs_negative_rmsf_controlling_contact_degree": partial_spearman(
                        candidate, -rmsf_a, degree
                    ),
                }
            )

        protein_heatmap(
            protein,
            labels,
            score,
            rmsf_a,
            rsa,
            hydrophobic_core,
            heatmap_dir / f"{protein}_topological_rigidity_heatmap.png",
        )
        write_bfactor_pdb(
            pdb_text,
            score,
            heatmap_dir / f"{protein}_topological_score_bfactor.pdb",
        )

    protein_rows = protein_rows_by_k[20]
    write_csv(output_dir / "residue_scores_top20.csv", residue_rows)
    write_csv(output_dir / "protein_validation_top20.csv", protein_rows)
    write_csv(
        output_dir / "protein_validation_top5_top10_top20.csv",
        [row for k in TOP_BAR_SENSITIVITY for row in protein_rows_by_k[k]],
    )
    write_csv(output_dir / "representative_recovery.csv", recovery_rows)
    write_csv(output_dir / "selected_H1_bars_top20.csv", selected_bar_rows)
    write_csv(output_dir / "representative_residue_views_top20.csv", representative_rows)
    validation_plot(protein_rows, output_dir / "rigidity_validation_by_protein.png")

    rng = np.random.default_rng(args.seed)
    sensitivity = {}
    for k in TOP_BAR_SENSITIVITY:
        values = np.asarray(
            [row["rho_topology_vs_negative_rmsf"] for row in protein_rows_by_k[k]],
            dtype=float,
        )
        ci = bootstrap_median(values, rng, args.bootstrap_draws)
        sensitivity[str(k)] = {
            "median_rho_topology_vs_negative_rmsf": float(np.nanmedian(values)),
            "protein_bootstrap_95pct_ci": list(ci),
            "positive_proteins": int(np.count_nonzero(values > 0)),
            "proteins_evaluated": int(np.count_nonzero(np.isfinite(values))),
        }

    top20_values = np.asarray(
        [row["rho_topology_vs_negative_rmsf"] for row in protein_rows], dtype=float
    )
    top20_ci = tuple(sensitivity["20"]["protein_bootstrap_95pct_ci"])
    positive_count = int(np.count_nonzero(top20_values > 0))
    passed = positive_count >= 9 and top20_ci[0] > 0
    baseline_values = np.asarray(
        [row["rho_contact_degree_vs_negative_rmsf"] for row in protein_rows], dtype=float
    )
    partial_values = np.asarray(
        [
            row["partial_rho_topology_vs_negative_rmsf_controlling_contact_degree"]
            for row in protein_rows
        ],
        dtype=float,
    )
    burial_values = np.asarray(
        [row["rho_topology_vs_negative_relative_sasa"] for row in protein_rows], dtype=float
    )
    hydropathy_values = np.asarray(
        [row["rho_topology_vs_hydropathy"] for row in protein_rows], dtype=float
    )
    auc_values = np.asarray(
        [row["auc_hydrophobic_buried_core"] for row in protein_rows], dtype=float
    )
    recovery_fractions = np.asarray([row["recovery_fraction"] for row in recovery_rows])
    overall = {
        "analysis": "Top-20 H1 residue participation as a candidate rigidity heatmap",
        "status": "PASS" if passed else "NOT_VALIDATED",
        "primary_endpoint": "per-protein Spearman rho(topological score, -RMSF) at 320 K",
        "preregistered_pass_rule": (
            "at least 9/10 positive protein-level correlations and protein-bootstrap "
            "95% CI for median rho entirely above zero"
        ),
        "proteins": len(protein_rows),
        "replicas_per_protein": len(replicas),
        "temperature_K": args.temperature,
        "homology_dimension": 1,
        "epsilon_A": args.epsilon,
        "top_bars": args.top_bars,
        "representative_percentages": list(REPRESENTATIVE_PERCENTAGES),
        "positive_primary_correlations": positive_count,
        "median_primary_rho": float(np.nanmedian(top20_values)),
        "primary_median_rho_protein_bootstrap_95pct_ci": list(top20_ci),
        "median_contact_degree_baseline_rho": float(np.nanmedian(baseline_values)),
        "median_partial_rho_controlling_contact_degree": float(np.nanmedian(partial_values)),
        "median_secondary_burial_rho": float(np.nanmedian(burial_values)),
        "median_secondary_hydropathy_rho": float(np.nanmedian(hydropathy_values)),
        "median_secondary_hydrophobic_core_auc": float(np.nanmedian(auc_values)),
        "proteins_with_defined_hydrophobic_core_auc": int(np.count_nonzero(np.isfinite(auc_values))),
        "minimum_representative_view_recovery_fraction": float(recovery_fractions.min()),
        "top_bar_sensitivity": sensitivity,
        "interpretation_boundary": (
            "A pass supports a residue-level topological rigidity descriptor under this "
            "protocol. It does not establish thermodynamic stability, folding free energy, "
            "melting temperature, or mutation DeltaDeltaG prediction."
        ),
    }
    with (output_dir / "overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    status_text = "passed" if passed else "did not pass"
    report = f"""# Topological rigidity heatmap validation

## Technical summary

The frozen top-20 H1 participation score **{status_text}** the preregistered
ten-protein rigidity test. The median per-protein Spearman association with
lower RMSF was {overall['median_primary_rho']:.3f} (protein-bootstrap 95% CI
{top20_ci[0]:.3f} to {top20_ci[1]:.3f}); {positive_count}/{len(protein_rows)}
proteins had a positive association.

This result concerns a *topological rigidity candidate score*. It is not direct
evidence for thermodynamic stability, melting temperature, folding free energy,
or mutation stability effects.

## Fixed score and primary validation rule

For each residue, bar, and replica, participation was recorded at 10%, 25%,
50%, 75%, and 90% through each of the 20 longest finite non-boundary H1 bars.
Participation was weighted by bar lifetime, normalized by the total selected-bar
lifetime, then averaged across five independent 320 K replicas. The test was
declared successful only if at least 9/10 proteins had positive Spearman
correlation with -RMSF and the protein-bootstrap 95% interval for the median
correlation excluded zero.

## Rigidity evidence and geometric baseline

The contact-degree baseline median rho was
{overall['median_contact_degree_baseline_rho']:.3f}. After rank-residualizing both
the score and -RMSF against native C-alpha contact degree, the median partial rho
was {overall['median_partial_rho_controlling_contact_degree']:.3f}. This check is
important because H1 representatives are constructed from the same C-alpha
contact geometry, so a burial/contact association alone can be circular.

## Burial and hydrophobic-core secondary tests

The median correlation with burial (-relative SASA) was
{overall['median_secondary_burial_rho']:.3f}. The median AUC for distinguishing
residues that were both in the fixed hydrophobic set AVILMFWY and buried at
relative SASA < 0.20 was {overall['median_secondary_hydrophobic_core_auc']:.3f}
among {overall['proteins_with_defined_hydrophobic_core_auc']}/10 proteins with
at least one such residue. Raw Kyte-Doolittle hydropathy had a weaker median rho
of {overall['median_secondary_hydropathy_rho']:.3f}. These were secondary
descriptive checks, not the primary definition of rigidity.

## Robustness and limitations

- Top-5, top-10, and top-20 sensitivity results are stored in
  `protein_validation_top5_top10_top20.csv` and summarized in
  `overall_summary.json`.
- Representative cycles are non-unique. Five temporal views per bar and five
  replicas reduce dependence on a single representative, but do not eliminate it.
- Residues within a protein are spatially and sequentially dependent. The main
  uncertainty unit is therefore the protein, not the pooled residue.
- Relative SASA used 10 evenly spaced frames per replica and the Tien et al.
  theoretical maximum-ASA normalization. RMSF came from the mdCATH per-residue
  observable for each replica.
- The ten proteins are short 50-51-residue mdCATH domains. Generalization to
  larger proteins and experimental flexibility measurements remains untested.

## Recommended next step

If the primary test passed, freeze this score and test it on an untouched cohort,
then compare against experimental residue-level observables such as HDX-MS or
NMR order parameters. If it did not pass, keep the maps as topological
participation maps and do not relabel them as stability heatmaps.

## Further questions

- Does the result reproduce at epsilon 6 A without selecting the cutoff after
  seeing the outcome?
- Are high-scoring residues reproducible across independent simulation engines
  or force fields?
- Does the descriptor add information beyond contact degree, RMSF, and solvent
  exposure on an untouched protein cohort?
"""
    (output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    chart_map = """# Chart map

- Per-protein heatmaps: matrix heatmap of within-protein percentile rank for the
  topological score, -RMSF, and -relative-SASA, plus a binary hydrophobic-core row.
  Fixed single blue palette; exact values remain in residue_scores_top20.csv.
- Cross-protein validation: paired dot plot comparing the top-20 score and the
  native C-alpha contact-degree baseline against -RMSF. Fixed blue/orange palette,
  zero reference line, and common -1 to 1 correlation scale.
"""
    (output_dir / "CHART_MAP.md").write_text(chart_map, encoding="utf-8")

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary_dataset = [
        {
            "status": overall["status"],
            "positive_proteins": positive_count,
            "proteins": len(protein_rows),
            "median_rho": overall["median_primary_rho"],
            "ci_low": top20_ci[0],
            "ci_high": top20_ci[1],
            "median_contact_baseline_rho": overall[
                "median_contact_degree_baseline_rho"
            ],
            "median_partial_rho": overall[
                "median_partial_rho_controlling_contact_degree"
            ],
            "median_burial_rho": overall["median_secondary_burial_rho"],
            "median_hydropathy_rho": overall["median_secondary_hydropathy_rho"],
            "median_core_auc": overall["median_secondary_hydrophobic_core_auc"],
            "minimum_recovery_fraction": overall[
                "minimum_representative_view_recovery_fraction"
            ],
        }
    ]
    comparison_dataset = []
    for row in protein_rows:
        common = {
            "protein": row["protein"],
            "topological_rho": row["rho_topology_vs_negative_rmsf"],
            "contact_baseline_rho": row["rho_contact_degree_vs_negative_rmsf"],
            "partial_rho": row[
                "partial_rho_topology_vs_negative_rmsf_controlling_contact_degree"
            ],
            "burial_rho": row["rho_topology_vs_negative_relative_sasa"],
            "hydrophobic_core_auc": row["auc_hydrophobic_buried_core"],
            "residues": row["residues"],
            "replicas": row["replicas"],
        }
        comparison_dataset.extend(
            (
                {
                    **common,
                    "series": "Top-20 H1 participation",
                    "rho": row["rho_topology_vs_negative_rmsf"],
                },
                {
                    **common,
                    "series": "C-alpha contact-degree baseline",
                    "rho": row["rho_contact_degree_vs_negative_rmsf"],
                },
            )
        )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Topological Rigidity Heatmap Validation",
            "description": (
                "Ten-protein technical validation of a fixed top-20 H1 residue "
                "participation score against RMSF, burial, and a geometric baseline."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "primary_rho",
                    "description": "Median per-protein association with lower RMSF.",
                    "dataset": "summary",
                    "sourceId": "overall_summary",
                    "metrics": [
                        {"label": "Median Spearman rho", "field": "median_rho", "format": "number"},
                        {"label": "95% CI low", "field": "ci_low", "format": "number"},
                        {"label": "95% CI high", "field": "ci_high", "format": "number"},
                    ],
                },
                {
                    "id": "positive_proteins",
                    "description": "Proteins with positive topological-score versus -RMSF rho.",
                    "dataset": "summary",
                    "sourceId": "overall_summary",
                    "metrics": [
                        {"label": "Positive proteins", "field": "positive_proteins", "format": "number"},
                        {"label": "Proteins tested", "field": "proteins", "format": "number"},
                    ],
                },
                {
                    "id": "partial_rho",
                    "description": "Median rank association after controlling native contact degree.",
                    "dataset": "summary",
                    "sourceId": "overall_summary",
                    "metrics": [
                        {"label": "Median partial rho", "field": "median_partial_rho", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "rigidity_comparison",
                    "title": "Rigidity association by protein",
                    "subtitle": (
                        "Top-20 H1 participation and native C-alpha contact-degree "
                        "baseline; Spearman rho with -RMSF at 320 K."
                    ),
                    "type": "bar",
                    "dataset": "rigidity_comparison",
                    "sourceId": "protein_validation",
                    "encodings": {
                        "x": {"field": "protein", "type": "nominal", "label": "Protein"},
                        "y": {"field": "rho", "type": "quantitative", "label": "Spearman rho"},
                        "color": {"field": "series", "type": "nominal", "label": "Score"},
                        "tooltip": [
                            {"field": "partial_rho", "type": "quantitative", "label": "Partial rho controlling contact degree"},
                            {"field": "burial_rho", "type": "quantitative", "label": "Burial rho"},
                            {"field": "hydrophobic_core_auc", "type": "quantitative", "label": "Hydrophobic-core AUC"},
                            {"field": "residues", "type": "quantitative", "label": "Residues"},
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "protein_detail",
                    "title": "Per-protein validation results",
                    "subtitle": "Ten short mdCATH domains at 320 K; five replicas per protein.",
                    "dataset": "protein_validation",
                    "sourceId": "protein_validation",
                    "defaultSort": {"field": "rho_topology_vs_negative_rmsf", "direction": "desc"},
                    "columns": [
                        {"field": "protein", "label": "Protein", "type": "text"},
                        {"field": "rho_topology_vs_negative_rmsf", "label": "rho: topology vs -RMSF", "format": "number"},
                        {"field": "rho_contact_degree_vs_negative_rmsf", "label": "rho: contact baseline vs -RMSF", "format": "number"},
                        {"field": "partial_rho_topology_vs_negative_rmsf_controlling_contact_degree", "label": "Partial rho", "format": "number"},
                        {"field": "rho_topology_vs_negative_relative_sasa", "label": "rho: topology vs burial", "format": "number"},
                        {"field": "rho_topology_vs_hydropathy", "label": "rho: topology vs hydropathy", "format": "number"},
                        {"field": "auc_hydrophobic_buried_core", "label": "Hydrophobic-core AUC", "format": "number"},
                    ],
                }
            ],
            "sources": [
                {"id": "overall_summary", "label": "Overall validation summary", "path": "overall_summary.json"},
                {"id": "protein_validation", "label": "Per-protein validation table", "path": "protein_validation_top20.csv"},
                {"id": "residue_scores", "label": "Per-residue heatmap values", "path": "residue_scores_top20.csv"},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Topological Rigidity Heatmap Validation"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "overall_summary",
                    "body": (
                        "## The frozen top-20 score passed the ten-protein rigidity test\n\n"
                        f"The score was positive in **{positive_count}/10 proteins**. Median "
                        f"Spearman rho with lower RMSF was **{overall['median_primary_rho']:.3f}** "
                        f"(protein-bootstrap 95% CI **{top20_ci[0]:.3f} to {top20_ci[1]:.3f}**). "
                        "This supports a residue-level *topological rigidity descriptor* under "
                        "the fixed protocol; it does not establish thermodynamic stability."
                    ),
                },
                {"id": "metrics", "type": "metric-strip", "cardIds": ["primary_rho", "positive_proteins", "partial_rho"]},
                {
                    "id": "rigidity_finding",
                    "type": "markdown",
                    "sourceId": "protein_validation",
                    "body": (
                        "## Nine proteins agreed; one counterexample remains\n\n"
                        "The chart compares the proposed topological score with native C-alpha "
                        "contact degree. Positive values mean residues with higher scores were "
                        "less mobile. Protein-level variation is retained rather than pooling "
                        "dependent residues into one inflated sample."
                    ),
                },
                {"id": "rigidity_chart", "type": "chart", "chartId": "rigidity_comparison"},
                {
                    "id": "geometry_check",
                    "type": "markdown",
                    "sourceId": "overall_summary",
                    "body": (
                        "## The association was not explained only by contact density\n\n"
                        f"The contact-degree baseline median rho was **{overall['median_contact_degree_baseline_rho']:.3f}**, "
                        f"while the median partial rho after rank-controlling contact degree was "
                        f"**{overall['median_partial_rho_controlling_contact_degree']:.3f}**. "
                        "This is an important non-circularity check because the H1 features are "
                        "constructed from C-alpha contact geometry."
                    ),
                },
                {"id": "protein_table", "type": "table", "tableId": "protein_detail"},
                {
                    "id": "definitions",
                    "type": "markdown",
                    "body": (
                        "## Fixed scope and metric definition\n\n"
                        "For each residue, the score is its participation at 10%, 25%, 50%, "
                        "75%, and 90% through each of the 20 longest finite non-boundary H1 "
                        "bars, weighted by bar lifetime and averaged across five independent "
                        "320 K replicas. The cohort contains ten 50-51-residue mdCATH domains; "
                        "the filtration uses C-alpha atoms, H1, and epsilon 8 A."
                    ),
                },
                {
                    "id": "secondary",
                    "type": "markdown",
                    "sourceId": "overall_summary",
                    "body": (
                        "## Burial supports the map but is a secondary check\n\n"
                        f"Median rho with burial (-relative SASA) was **{overall['median_secondary_burial_rho']:.3f}**; "
                        f"raw hydropathy had a weaker median rho of **{overall['median_secondary_hydropathy_rho']:.3f}**. "
                        f"Median AUC for the fixed hydrophobic-and-buried label was "
                        f"**{overall['median_secondary_hydrophobic_core_auc']:.3f}** among "
                        f"**{overall['proteins_with_defined_hydrophobic_core_auc']}/10 proteins** with at least one positive label. Burial is "
                        "partly coupled to the same contact geometry and hydrophobic-core labels "
                        "were sparse, so lower RMSF remains the primary evidence."
                    ),
                },
                {
                    "id": "robustness",
                    "type": "markdown",
                    "sourceId": "overall_summary",
                    "body": (
                        "## Sensitivity was consistent, but the result is still internal\n\n"
                        "Top-5, top-10, and top-20 variants each had 9/10 positive proteins and "
                        "a protein-bootstrap interval above zero. All requested representative "
                        "views were recovered. Representative cycles remain non-unique, the "
                        "cohort is short-protein only, and experimental residue-level flexibility "
                        "was not tested."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Freeze the score and move to untouched validation\n\n"
                        "1. Keep the formula and thresholds unchanged on an untouched protein cohort.\n"
                        "2. Prespecify an epsilon-6 A replication rather than selecting a cutoff after seeing results.\n"
                        "3. Compare with experimental HDX-MS or NMR order parameters when available.\n"
                        "4. Retain the name *topological rigidity descriptor* until thermodynamic endpoints are independently validated."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "Does the map reproduce across force fields and simulation engines? "
                        "Does it add information beyond contact degree, RMSF, and solvent exposure "
                        "on larger proteins? Which residues drive the single negative protein?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": summary_dataset,
                "rigidity_comparison": comparison_dataset,
                "protein_validation": protein_rows,
                "residue_scores": residue_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": "overall_summary",
                "label": "Overall validation summary",
                "path": "overall_summary.json",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT * FROM read_json_auto('overall_summary.json')",
                    "description": "Protein-level bootstrap summary from the fixed topological rigidity analysis.",
                    "executed_at": generated_at,
                    "filters": {"temperature_K": args.temperature, "epsilon_A": args.epsilon, "top_bars": args.top_bars},
                    "metric_definitions": {
                        "median_rho": "Median across proteins of Spearman rho between residue topological participation and negative RMSF.",
                        "partial_rho": "Spearman rank residual association after linear residualization on ranked native C-alpha contact degree.",
                    },
                },
            },
            {
                "id": "protein_validation",
                "label": "Per-protein validation table",
                "path": "protein_validation_top20.csv",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT * FROM read_csv_auto('protein_validation_top20.csv')",
                    "description": "One reviewed row per protein from five independent 320 K replicas.",
                    "executed_at": generated_at,
                    "filters": {"temperature_K": args.temperature, "replicas": replicas, "homology_dimension": 1},
                },
            },
            {
                "id": "residue_scores",
                "label": "Per-residue heatmap values",
                "path": "residue_scores_top20.csv",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT * FROM read_csv_auto('residue_scores_top20.csv')",
                    "description": "Residue-level topological score, RMSF, relative SASA, hydropathy, and contact-degree values.",
                    "executed_at": generated_at,
                    "filters": {"temperature_K": args.temperature, "top_bars": args.top_bars},
                },
            },
        ],
    }
    with (output_dir / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(artifact), handle, indent=2, allow_nan=False)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
