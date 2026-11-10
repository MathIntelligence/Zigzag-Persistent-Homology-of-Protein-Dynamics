# Reproducibility companion

This archive accompanies *Zigzag Persistent Homology of Protein Dynamics*, Milap Rajgor and Duc Nguyen.
The main manuscript and eleven supplementary sections are compiled together from `main.tex`.

## Run from this directory

Use Python 3.11 or later and install `requirements.txt` in an isolated environment.

```sh
python code/reproduce_counts.py
python code/audit_features.py
python code/check_models.py
python code/reproduce_tracker.py
python code/check_worked_example.py
python code/check_case_attribution.py
python code/make_figures.py
MPLCONFIGDIR=/tmp/zigzag-mpl python code/make_case_5sic.py
```

Outputs go to `reproduced/`; figure generators update `../figures/` in the complete source tree.
The four first commands were rerun for this version. They reproduce counts, six features across 2,325 windows,
combined six-feature models, and independent union/intersection generalized ranks for the worked example.
No new molecular dynamics were run. The case figure was regenerated from included coordinates.
The archived 227,000 independent frame-rank checks and 157 window-rank checks are recorded in
`data/verification.json`; they were not all recomputed for this revision.

## Input data

- `data/global/`: saved intervals, framewise Betti counts, Q, windows, parameter and source hashes.
- `data/baseline_test/`: saved window features, baseline outputs and per-domain AUC tables.
- `data/review2_tests/`: contact-timing null draws and window-length sensitivity.
- `data/worked_example_ca.npy`: complete 500-frame 2fd5A02 450 K replica 1 C-alpha coordinates in angstroms.
- `data/5sic_450_rep0_ca.npy`: 5sicI00 case coordinates in angstroms; native PDB numbering is 7 through 113.
- `data/screen_5sic/`: coarse screen at 348 and 450 K, replica 0 for both panels.
- `data/t4l/`: input structure, mutation metadata and coordinates subsampled at 250 ps. For 1 ns use every fourth included frame.
- `data/participation/`: sampled representative residue memberships and protein-level comparisons.
- `protocols/`: frozen records, including subsequent amendments and originally stated hypotheses.

Coordinate provenance is mdCATH revision `5e3ed8aec62b689e01751db16275fdcdbc39e47f`,
https://huggingface.co/datasets/compsciencelab/mdCATH . The manuscript cites the original data paper.
The T4 lysozyme arrays come from the existing author-generated ISAAC simulation campaign.
Other trajectory-level recomputations require the pinned HDF5/coordinate files and topology metadata;
the full mdCATH collection is not included.

## Historical scripts

`code/historical/` preserves analysis drivers and tests as archived. Several import source-specific loader
modules or contain paths from the original machines. They are provenance, not portable entrypoints.
Adapt those paths, provide the relevant source coordinates and external loaders, and retain the frozen
protocol before using them to recompute the original experiments. Use the portable scripts above for
table/feature reproduction. `code/zigzag_dynamics/` is the accompanying topology implementation.

## Interpretation and conventions

Topology uses all C-alpha pairs, 6 A, debounce 3 at primary settings; native contacts use the distinct
8 A sequence-separated definition. Full-record lifetime is retrospective and is not clipped to each window.
The held-out-domain model scaling uses that domain's unlabelled evaluation-cohort distribution.
The worked example has beta1 between 1 and 3; no observed primary hidden-replacement window has beta1 identically one.
Cycles displayed at selected frames are independently chosen non-boundaries, not matched physical rings.
Critical-simplex residue attribution depends on the within-step update order.
`reproduced/figure3_contact_partners.json` records independently recalculated exact partners for Figure 3A;
plotted residue ranges bracket nearby contacts and may include gaps.

`MANIFEST_SHA256.json` records the exact package bytes excluding itself and transient cache files.
Recomputed outputs can differ in runtime timings or last floating-point digits across dependency versions.
Versions used for fresh audits: Python 3.13, NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.8.0,
Matplotlib 3.10.8 and Dionysus 2.1.8. Pandas and NetworkX are required by the saved general figure generator.

The Figure 3 attribution excerpt in `data/5sic_attribution/transition245_246.csv` retains the twelve
transition endpoints from the archived `events_full_0.csv` (450 K, replica 0, raw 8 A complexes).
`check_case_attribution.py` reproduces these assignments from the included coordinates. The two cited
critical edges are distinct from the contact-based snapshot colouring.

## Diagram tracker continuity comparison (22 October 2026)

Section 6 adds a compact continuity comparison; Supplementary Section S11 contains only the
cohort/matching methods, eight-variant table and raw-reference control. No new classification
results, AUC table or classifier significance discussion are added to the manuscript.

Run `python code/reproduce_tracker.py` to reconstruct both cohorts, all eight tracker variants,
the exploratory raw-reference control, the cohort rosters and the frozen-protocol hash check.
This uses the Python standard library and the saved CSVs in `data/tracker/`; no trajectory
download or model fitting is needed. `reproduced/tracker_comparison.json` records the results.
The original protocol, scripts and logs are archived unchanged as provenance, including their
original classification analyses; these are not new manuscript results. Coordinate-level
recomputation requires the original mdCATH arrays and the path adaptations in the archived code.
