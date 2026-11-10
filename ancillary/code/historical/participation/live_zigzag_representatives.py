"""Recover actual live zigzag homology cycles for selected barcode intervals.

Unlike the high-level apex helper, ``zigzag_homology_persistence`` maintains a
cycle basis in the original complex after every addition/removal.  Its Python
iterator hides the basis-column identifiers, but those identifiers can be
tracked exactly from the documented insertion/removal order and the birth
operation stored in each diagram point.
"""

from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
import sys
from typing import Any


def _event_table(filtration: Any, times: list[list[float]]) -> list[dict[str, Any]]:
    events = []
    for simplex_index, simplex_times in enumerate(times):
        direction = True
        dimension = filtration[simplex_index].dimension()
        for time in simplex_times:
            events.append(
                {
                    "time": float(time),
                    "simplex_index": simplex_index,
                    "dimension": dimension,
                    "direction": direction,
                }
            )
            direction = not direction
    events.sort(
        key=lambda event: (
            event["time"],
            0 if event["direction"] else 1,
            event["dimension"] if event["direction"] else -event["dimension"],
            event["simplex_index"],
        )
    )
    for operation, event in enumerate(events):
        event["operation"] = operation
    return events


def _birth_dimension(event: dict[str, Any]) -> int:
    return int(event["dimension"] if event["direction"] else event["dimension"] - 1)


def _death_dimension(event: dict[str, Any]) -> int:
    return int(event["dimension"] - 1 if event["direction"] else event["dimension"])


def load_dionysus_with_live_births():
    """Load the minimally patched local Dionysus build."""
    repository_root = Path(__file__).resolve().parents[1]
    dependency_dir = Path(
        os.environ.get(
            "TOPOMOL_DIONYSUS_PATH",
            repository_root / ".dionysus_live_deps_223",
        )
    )
    if not dependency_dir.is_dir():
        dependency_dir = repository_root / ".dionysus_live_deps"
    if dependency_dir.is_dir() and str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))
    import dionysus as d

    if not hasattr(d.ZigzagPersistence, "alive_births"):
        raise RuntimeError(
            "Live representative recovery requires the read-only Dionysus "
            "alive_births binding. Run build_patched_dionysus.sh against the "
            "official Dionysus 2.2.3 source before analysis."
        )
    return d


def _same_time(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)


def _match_bars_to_birth_operations(
    bars: list[Any],
    diagrams: list[Any],
    events: list[dict[str, Any]],
) -> dict[int, int]:
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for dimension, diagram in enumerate(diagrams):
        for point in diagram:
            birth = float(point.birth)
            death = float(point.death)
            if not (math.isfinite(birth) and math.isfinite(death) and death > birth):
                continue
            birth_operation = int(point.data)
            birth_event = events[birth_operation]
            candidates[dimension].append(
                {
                    "birth": birth,
                    "death": death,
                    "birth_operation": birth_operation,
                    "birth_endpoint": "c" if birth_event["direction"] else "o",
                    "used": False,
                }
            )

    matched: dict[int, int] = {}
    for bar in bars:
        compatible = [
            candidate
            for candidate in candidates[int(bar.dimension)]
            if not candidate["used"]
            and candidate["birth_endpoint"] == bar.interval_type[0]
            and _same_time(candidate["birth"], bar.birth_sampled_frame)
            and _same_time(candidate["death"], bar.death_sampled_frame)
        ]
        if not compatible:
            raise ValueError(
                "Could not match a fast-zigzag bar to the direct zigzag basis: "
                f"rank {bar.rank}, {bar.interval_type}, "
                f"({bar.birth_sampled_frame}, {bar.death_sampled_frame})."
            )
        chosen = min(compatible, key=lambda item: item["birth_operation"])
        chosen["used"] = True
        matched[int(bar.rank)] = int(chosen["birth_operation"])
    return matched


def _chain_terms(chain: Any, cells: Any, filtration: Any, dimension: int) -> list[list[int]]:
    support: set[tuple[int, ...]] = set()
    for entry in chain:
        if int(entry.element) % 2 == 0:
            continue
        filtration_index = int(cells[entry.index])
        simplex = filtration[filtration_index]
        if simplex.dimension() != dimension:
            raise ValueError("A live zigzag cycle mixed homological dimensions.")
        key = tuple(sorted(int(vertex) for vertex in simplex))
        if key in support:
            support.remove(key)
        else:
            support.add(key)
    return [list(simplex) for simplex in sorted(support)]


def recover_live_cycles(
    simplices: list[Any],
    times: list[list[float]],
    bars: list[Any],
    frame_count: int,
) -> dict[tuple[int, int], list[list[int]]]:
    """Return one direct live cycle for every selected bar and alive frame."""
    d = load_dionysus_with_live_births()

    filtration = d.Filtration(simplices)
    events = _event_table(filtration, times)
    _persistence, diagrams, _cells = d.zigzag_homology_persistence(
        filtration, times, prime=2
    )
    birth_operation_by_rank = _match_bars_to_birth_operations(
        bars, diagrams, events
    )
    target_birth_operations = set(birth_operation_by_rank.values())
    rank_by_birth_operation = {
        birth_operation: rank
        for rank, birth_operation in birth_operation_by_rank.items()
    }
    dimension_by_rank = {int(bar.rank): int(bar.dimension) for bar in bars}

    frames_by_rank = {
        int(bar.rank): [
            frame
            for frame in range(frame_count)
            if bar.birth_sampled_frame < frame + 0.5 < bar.death_sampled_frame
        ]
        for bar in bars
    }
    pending_frames = sorted(
        {
            frame
            for frames in frames_by_rank.values()
            for frame in frames
        }
    )
    recovered: dict[tuple[int, int], list[list[int]]] = {}
    operation = 0
    current_time: float | None = None
    current_target_cycles: dict[int, list[list[int]]] = {}

    def commit_until(next_time: float) -> None:
        nonlocal pending_frames
        ready = [frame for frame in pending_frames if frame < next_time]
        for frame in ready:
            for bar in bars:
                rank = int(bar.rank)
                if frame not in frames_by_rank[rank]:
                    continue
                birth_operation = birth_operation_by_rank[rank]
                if birth_operation not in current_target_cycles:
                    raise ValueError(
                        f"The direct zigzag cycle for rank {rank} was not alive at frame {frame}."
                    )
                recovered[(rank, frame)] = current_target_cycles[birth_operation]
        pending_frames = [frame for frame in pending_frames if frame >= next_time]

    def callback(simplex_index: int, time: float, direction: bool, zigzag: Any, cells: Any) -> None:
        nonlocal operation, current_time, current_target_cycles
        time = float(time)
        if current_time is not None and not _same_time(time, current_time):
            commit_until(time)
        current_time = time

        event = events[operation]
        if (
            int(simplex_index) != event["simplex_index"]
            or bool(direction) != event["direction"]
            or not _same_time(time, event["time"])
        ):
            raise ValueError("Python and Dionysus zigzag event order diverged.")

        alive_birth_operations = [int(value) for value in zigzag.alive_births()]
        cycles = list(zigzag)
        if len(alive_birth_operations) != len(cycles):
            raise ValueError("Live birth labels and cycles disagree.")
        current_target_cycles = {}
        for birth_operation, chain in zip(
            alive_birth_operations, cycles, strict=True
        ):
            if birth_operation not in target_birth_operations:
                continue
            rank = rank_by_birth_operation[birth_operation]
            dimension = dimension_by_rank[rank]
            current_target_cycles[birth_operation] = _chain_terms(
                chain, cells, filtration, dimension
            )
        operation += 1

    d.zigzag_homology_persistence(
        filtration, times, prime=2, callback=callback
    )
    commit_until(float("inf"))
    expected = sum(len(frames) for frames in frames_by_rank.values())
    if len(recovered) != expected:
        raise ValueError(
            f"Recovered {len(recovered)} live cycles but expected {expected}."
        )
    return recovered
