"""Reproduce the diagram-tracker continuity tables from saved window features.

Run from anc/: python code/reproduce_tracker.py
No coordinates, third-party Python packages, or original-machine paths are needed.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tracker"
KEY = ("protein", "T", "rep", "start")
VARIANTS = ["full", "eps"] + [f"{v}_g{g}" for g in (0.25, 0.5, 1.0) for v in ("full", "eps")]


def read_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keys = [tuple(r[k] for k in KEY) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate window keys: {path}")
    return rows


def load():
    ex = read_csv(DATA / "exploratory" / "window_tracker_merged.csv")
    gated = read_csv(DATA / "exploratory" / "window_tracker_gated.csv")
    gm = {tuple(r[k] for k in KEY): r for r in gated}
    if set(gm) != {tuple(r[k] for k in KEY) for r in ex}:
        raise ValueError("Exploratory tracker tables do not have identical window keys")
    ex = [{**r, **gm[tuple(r[k] for k in KEY)]} for r in ex]
    confirm = read_csv(DATA / "confirmation" / "window_features_confirm.csv")
    if {r["protein"] for r in ex} & {r["protein"] for r in confirm}:
        raise ValueError("Additional cohort overlaps the exploratory cohort")
    protocol = DATA / "confirmation" / "PROTOCOL.md"
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    if digest != (protocol.parent / "PROTOCOL.sha256").read_text().split()[0]:
        raise ValueError("Confirmation protocol differs from archived hash")
    return ex, confirm, digest


def continuity(rows, occupancy="occupied", spanning="zz_spanning"):
    occupied = [r for r in rows if r[occupancy].lower() == "true"]
    yes = [r for r in occupied if float(r[spanning]) > 0]
    no = [r for r in occupied if float(r[spanning]) == 0]
    result = dict(domains=len({r["protein"] for r in rows}),
                  runs=len({tuple(r[k] for k in KEY[:3]) for r in rows}),
                  windows=len(rows), occupied=len(occupied), continuous=len(yes), replacement=len(no), variants={})
    for v in VARIANTS:
        k = "trk_spanning_" + v
        tp = sum(float(r[k]) > 0 for r in yes)
        fp = sum(float(r[k]) > 0 for r in no)
        result["variants"][v] = dict(true_continuous_called=tp, replacement_called_continuous=fp,
            continuous_recovered_percent=100 * tp / len(yes),
            replacement_called_continuous_percent=100 * fp / len(no),
            balanced_accuracy=(tp / len(yes) + 1 - fp / len(no)) / 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ex, confirm, digest = load()
    result = {"protocol_sha256": digest, "exploratory": continuity(ex),
              "confirmation": continuity(confirm),
              "exploratory_raw": continuity(ex, "raw_occupied", "raw_spanning")}
    for cohort, counts in (("exploratory", (2325, 1932, 1034, 898)),
                           ("confirmation", (4772, 2091, 1246, 845))):
        if tuple(result[cohort][k] for k in ("windows", "occupied", "continuous", "replacement")) != counts:
            raise ValueError(f"Cohort count mismatch: {cohort}")
        print(cohort, "counts match", counts, flush=True)
    domains = {"exploratory": sorted({r["protein"] for r in ex}),
               "confirmation": sorted({r["protein"] for r in confirm})}
    result["cohort_domains"] = domains
    out = ROOT / "reproduced" / "tracker_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
