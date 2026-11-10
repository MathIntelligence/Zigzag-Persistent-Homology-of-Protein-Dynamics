# Zigzag Persistent Homology of Protein Dynamics

Code and supporting data for the paper *Zigzag Persistent Homology of Protein Dynamics* by Milap Rajgor and Duc Nguyen. The workflow computes and analyzes residue-labelled zigzag persistence on Cα contact complexes from molecular dynamics trajectories.

## Reproduce the paper’s results

From the repository root, with Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ancillary/requirements.txt
python ancillary/code/reproduce_counts.py
python ancillary/code/audit_features.py
python ancillary/code/check_models.py
python ancillary/code/reproduce_tracker.py
python ancillary/code/check_worked_example.py
python ancillary/code/check_case_attribution.py
```

These commands use included inputs to reproduce the headline counts, window features, model comparisons, diagram-tracker results, worked-example ranks, and case-study attribution. Outputs are written to `ancillary/reproduced/`. See the [companion README](ancillary/README.md) for figure generation, tested dependency versions, and coordinate-level reproduction.

| Path | Contents |
| --- | --- |
| [`ancillary/code/`](ancillary/code/) | Reproduction scripts, topology implementation, and archived analysis drivers |
| [`ancillary/data/`](ancillary/data/) | Saved barcodes, window features, tracker outputs, selected coordinates, and validation records |
| [`ancillary/protocols/`](ancillary/protocols/) | Frozen analysis protocols and amendments |
| [`figures/`](figures/) | Figures from the paper |

The primary trajectories come from [mdCATH](https://huggingface.co/datasets/compsciencelab/mdCATH) at revision `5e3ed8aec62b689e01751db16275fdcdbc39e47f`. The included data support the reproduction commands above. Full-cohort coordinate-level recomputation requires the pinned trajectories and the path adaptations described in the companion README. Source datasets retain their original terms and attribution requirements.
