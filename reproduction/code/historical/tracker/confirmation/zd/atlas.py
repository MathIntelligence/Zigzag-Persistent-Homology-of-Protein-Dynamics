"""Loop atlas: feature-level zigzag across trajectories.

Episodes (H1 bars with residue support) from many trajectories are matched
into *loop families* — the same topological feature recurring within and
across runs — by average-linkage clustering of Jaccard support distance.
Each family is then a named object with

* a residue support and secondary-structure annotation,
* presence or absence in the static starting structure,
* per-trajectory occupancy, number of episodes and censoring,
* a right-censored Kaplan–Meier survival curve of its episode lengths,
* first-death time per trajectory (unfolding order).

This is the information a per-frame descriptor cannot produce: the identity
of a feature followed through time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class Family:
    fid: int
    support: np.ndarray            # canonical support (residues in >= 50% of member episodes)
    members: list[tuple[str, int]] = field(default_factory=list)   # (trajectory tag, episode index)
    label: str = ""


def jaccard_matrix(supports: list[set[int]]) -> np.ndarray:
    n = len(supports)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = supports[i], supports[j]
            D[i, j] = D[j, i] = 1.0 - len(a & b) / len(a | b)
    return D


def build_families(episodes_by_traj: dict[str, list[dict]], max_distance: float = 0.5,
                   weight_by_lifetime: bool = True) -> list[Family]:
    keys, supports, weights = [], [], []
    for tag, eps in episodes_by_traj.items():
        for k, e in enumerate(eps):
            keys.append((tag, k))
            supports.append(set(e["support"]))
            weights.append(e["lifetime_ns"] if weight_by_lifetime else 1.0)
    if not supports:
        return []
    D = jaccard_matrix(supports)
    Z = linkage(squareform(D, checks=False), method="average")
    cl = fcluster(Z, t=max_distance, criterion="distance")
    fams: dict[int, Family] = {}
    for idx, c in enumerate(cl):
        fams.setdefault(int(c), Family(int(c), np.array([]))).members.append(keys[idx])
    out = []
    for c, f in fams.items():
        # canonical support: residues present in >= 50% of (lifetime-weighted) members
        counts: dict[int, float] = {}
        total = 0.0
        for (tag, k) in f.members:
            w = episodes_by_traj[tag][k]["lifetime_ns"] if weight_by_lifetime else 1.0
            total += w
            for r in episodes_by_traj[tag][k]["support"]:
                counts[r] = counts.get(r, 0.0) + w
        f.support = np.array(sorted(r for r, v in counts.items() if v >= 0.5 * total), dtype=int)
        out.append(f)
    out.sort(key=lambda f: -sum(episodes_by_traj[t][k]["lifetime_ns"] for t, k in f.members))
    for i, f in enumerate(out):
        f.fid = i
    return out


# ----------------------------------------------------------------------- #
def parse_ss(pdb_path: str) -> list[tuple[str, int, int]]:
    """(name, start_resSeq, end_resSeq) from HELIX/SHEET records."""
    out = []
    for line in open(pdb_path):
        if line.startswith("HELIX"):
            out.append((f"H{int(line[7:10])}", int(line[21:25]), int(line[33:37])))
        elif line.startswith("SHEET"):
            out.append((f"S{int(line[7:10])}", int(line[22:26]), int(line[33:37])))
    return out


def annotate(support_idx: np.ndarray, ss: list[tuple[str, int, int]], resseq: list[int]) -> str:
    """Secondary-structure elements touched by the support (>= 2 residues each)."""
    res = [resseq[i] for i in support_idx]
    hits = []
    for name, a, b in ss:
        n = sum(1 for r in res if a <= r <= b)
        if n >= 2:
            hits.append(f"{name}({n})")
    loops = len(res) - sum(1 for r in res if any(a <= r <= b for _, a, b in ss))
    if loops >= 2:
        hits.append(f"loop({loops})")
    return "+".join(hits) if hits else "coil"


def family_table(fams: list[Family], episodes_by_traj: dict[str, list[dict]], n_frames: dict[str, int],
                 dt_ns: float) -> list[dict]:
    rows = []
    tags = list(episodes_by_traj)
    for f in fams:
        per = {}
        for tag in tags:
            mem = [episodes_by_traj[tag][k] for (t, k) in f.members if t == tag]
            T = n_frames[tag]
            alive = np.zeros(T, dtype=bool)
            for e in mem:
                alive[e["lo"]:e["hi"]] = True
            censored = any(e["lo"] == 0 or e["hi"] >= T for e in mem)
            first_death = min((e["hi"] for e in mem if e["hi"] < T), default=None)
            per[tag] = dict(occupancy=float(alive.mean()), n_episodes=len(mem),
                            total_ns=float(alive.sum() * dt_ns), censored=censored,
                            first_death_ns=None if first_death is None else first_death * dt_ns,
                            last_death_ns=None if not mem else (max(e["hi"] for e in mem) * dt_ns
                                                                 if all(e["hi"] < T for e in mem) else None))
        rows.append(dict(fid=f.fid, support=f.support.tolist(), n_members=len(f.members), per_traj=per))
    return rows


# ----------------------------------------------------------------------- #
def kaplan_meier(durations: np.ndarray, observed: np.ndarray):
    """Right-censored KM estimator -> (times, survival)."""
    order = np.argsort(durations)
    d, o = durations[order], observed[order].astype(bool)
    times, surv = [0.0], [1.0]
    at_risk = len(d)
    s = 1.0
    i = 0
    while i < len(d):
        t = d[i]
        j = i
        deaths = 0
        while j < len(d) and d[j] == t:
            deaths += int(o[j])
            j += 1
        if deaths:
            s *= 1.0 - deaths / at_risk
            times.append(t)
            surv.append(s)
        at_risk -= (j - i)
        i = j
    return np.array(times), np.array(surv)


def km_median(times, surv):
    below = np.nonzero(surv <= 0.5)[0]
    return float(times[below[0]]) if below.size else float("inf")


def restricted_mean(times, surv, horizon):
    """Restricted mean survival time up to `horizon` (area under KM)."""
    t = np.append(times, horizon)
    s = np.append(surv, surv[-1])
    t = np.clip(t, 0, horizon)
    return float(np.sum(np.diff(t) * s[:-1]))
