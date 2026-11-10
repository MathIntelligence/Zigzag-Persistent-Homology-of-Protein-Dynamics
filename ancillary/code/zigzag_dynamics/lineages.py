"""Feature-level kinetics from residue-supported temporal H1 episodes.

The functions in this module do not construct a filtration or choose a zigzag
basis.  They operate on already recovered live-basis representatives, making
the residue-overlap proxy and the resulting observables independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Episode:
    """One selected H1 bar and its residue support through time."""

    episode_id: int
    birth_frame: float
    death_frame: float
    lifetime_ns: float
    frame_supports: Mapping[int, frozenset[int]]
    core_support: frozenset[int]
    interval_type: str = "cc"
    touches_start: bool = False
    touches_end: bool = False

    @property
    def alive_frames(self) -> tuple[int, ...]:
        return tuple(sorted(self.frame_supports))


@dataclass(frozen=True)
class LineageLink:
    predecessor_episode_id: int
    successor_episode_id: int
    gap_frames: float
    support_overlap: float


def consensus_support(
    frame_supports: Mapping[int, frozenset[int]], minimum_fraction: float
) -> frozenset[int]:
    """Residues present in at least ``minimum_fraction`` of recovered frames."""
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must lie in (0, 1]")
    if not frame_supports:
        return frozenset()
    counts: dict[int, int] = {}
    for support in frame_supports.values():
        for residue in support:
            counts[residue] = counts.get(residue, 0) + 1
    denominator = len(frame_supports)
    return frozenset(
        residue
        for residue, count in counts.items()
        if count / denominator >= minimum_fraction
    )


def support_overlap(left: frozenset[int], right: frozenset[int]) -> float:
    """Overlap coefficient, with empty supports treated as nonmatching."""
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def link_episodes(
    episodes: list[Episode],
    *,
    minimum_overlap: float,
    maximum_gap_frames: float,
) -> tuple[dict[int, int], list[LineageLink]]:
    """Link non-overlapping episodes into deterministic, non-branching paths.

    Returns a mapping from episode id to one-based lineage id and the selected
    proxy links.  Each episode has at most one predecessor and one successor.
    """
    if not 0 <= minimum_overlap <= 1:
        raise ValueError("minimum_overlap must lie in [0, 1]")
    if maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be nonnegative")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode ids must be unique")

    by_id = {episode.episode_id: episode for episode in episodes}
    predecessor_of: dict[int, int] = {}
    successor_of: dict[int, int] = {}
    links: list[LineageLink] = []

    successors = sorted(
        episodes, key=lambda item: (item.birth_frame, item.death_frame, item.episode_id)
    )
    for successor in successors:
        candidates = []
        for predecessor in episodes:
            if predecessor.episode_id == successor.episode_id:
                continue
            if predecessor.episode_id in successor_of:
                continue
            gap = successor.birth_frame - predecessor.death_frame
            if gap < 0 or gap > maximum_gap_frames:
                continue
            overlap = support_overlap(predecessor.core_support, successor.core_support)
            if overlap < minimum_overlap:
                continue
            candidates.append(
                (
                    -overlap,
                    gap,
                    -predecessor.lifetime_ns,
                    predecessor.episode_id,
                    predecessor,
                )
            )
        if not candidates:
            continue
        *_key, predecessor = min(candidates)
        predecessor_of[successor.episode_id] = predecessor.episode_id
        successor_of[predecessor.episode_id] = successor.episode_id
        links.append(
            LineageLink(
                predecessor_episode_id=predecessor.episode_id,
                successor_episode_id=successor.episode_id,
                gap_frames=successor.birth_frame - predecessor.death_frame,
                support_overlap=support_overlap(
                    predecessor.core_support, successor.core_support
                ),
            )
        )

    assignment: dict[int, int] = {}
    roots = sorted(
        (episode for episode in episodes if episode.episode_id not in predecessor_of),
        key=lambda item: (item.birth_frame, item.episode_id),
    )
    for lineage_id, root in enumerate(roots, start=1):
        current = root.episode_id
        while True:
            if current in assignment:
                raise ValueError("lineage proxy produced a cycle")
            assignment[current] = lineage_id
            if current not in successor_of:
                break
            current = successor_of[current]
    if set(assignment) != set(by_id):
        raise ValueError("not every episode was assigned to a lineage")
    return assignment, sorted(
        links,
        key=lambda item: (
            item.predecessor_episode_id,
            item.successor_episode_id,
        ),
    )


def residue_loop_observables(
    episodes: list[Episode],
    lineage_by_episode: Mapping[int, int],
    *,
    n_residues: int,
    n_frames: int,
    dt_ns: float,
) -> list[dict]:
    """Return exact dynamic occupancy and episode-weighted lifetime per residue."""
    if n_residues < 1 or n_frames < 1 or dt_ns <= 0:
        raise ValueError("n_residues, n_frames, and dt_ns must be positive")
    occupancy = np.zeros((n_residues, n_frames), dtype=bool)
    lifetimes: list[list[float]] = [[] for _ in range(n_residues)]
    episode_ids: list[set[int]] = [set() for _ in range(n_residues)]
    lineage_ids: list[set[int]] = [set() for _ in range(n_residues)]

    for episode in episodes:
        lineage_id = lineage_by_episode[episode.episode_id]
        for frame, support in episode.frame_supports.items():
            if not 0 <= frame < n_frames:
                raise ValueError("episode support contains an out-of-range frame")
            for residue in support:
                if not 0 <= residue < n_residues:
                    raise ValueError("episode support contains an out-of-range residue")
                occupancy[residue, frame] = True
        for residue in episode.core_support:
            if not 0 <= residue < n_residues:
                raise ValueError("episode core support contains an out-of-range residue")
            lifetimes[residue].append(episode.lifetime_ns)
            episode_ids[residue].add(episode.episode_id)
            lineage_ids[residue].add(lineage_id)

    rows = []
    for residue in range(n_residues):
        values = np.asarray(lifetimes[residue], dtype=float)
        tau = float(np.square(values).sum() / values.sum()) if len(values) else 0.0
        rows.append(
            {
                "local_ca_index": residue,
                "loop_occupancy_fraction": float(occupancy[residue].mean()),
                "loop_lifetime_ns": tau,
                "supporting_episode_count": len(episode_ids[residue]),
                "supporting_lineage_count": len(lineage_ids[residue]),
            }
        )
    return rows


def summarize_lineages(
    episodes: list[Episode],
    lineage_by_episode: Mapping[int, int],
    *,
    dt_ns: float,
) -> list[dict]:
    """Summarize observed episode paths without interpreting them as unique loops."""
    grouped: dict[int, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(lineage_by_episode[episode.episode_id], []).append(episode)

    rows = []
    for lineage_id in sorted(grouped):
        members = sorted(
            grouped[lineage_id], key=lambda item: (item.birth_frame, item.episode_id)
        )
        occupied_frames = set().union(*(set(item.frame_supports) for item in members))
        union_support = set().union(*(set(item.core_support) for item in members))
        lengths = np.asarray([item.lifetime_ns for item in members], dtype=float)
        rows.append(
            {
                "lineage_id": lineage_id,
                "episode_count": len(members),
                "episode_ids": ";".join(str(item.episode_id) for item in members),
                "total_episode_duration_ns": float(lengths.sum()),
                "observed_union_occupancy_ns": len(occupied_frames) * dt_ns,
                "mean_episode_length_ns": float(lengths.mean()),
                "lineage_tau_top_ns": float(np.square(lengths).sum() / lengths.sum()),
                "first_birth_ns": float(members[0].birth_frame * dt_ns),
                "last_observed_loss_ns": float(
                    max(item.death_frame for item in members) * dt_ns
                ),
                "support_size": len(union_support),
                "support_indices": ";".join(str(value) for value in sorted(union_support)),
            }
        )
    return rows
