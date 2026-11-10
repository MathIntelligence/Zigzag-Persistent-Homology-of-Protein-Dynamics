# WT vs mutant: does zigzag replacement track known stability at matched Q?
Frozen 2026-09-23 before computing replacement on these trajectories.
Data: existing folder benchmarks with known stability order (to be listed exactly in the report as found:
T4 lysozyme WT/S117V/G30F; RNase HI WT/mutants if present). Pipeline as in PROTOCOL_CONFIRM_EPS5.md, but windows
from every run (no unfolding qualification needed), at eps 5 and 6 A (both reported; eps 5 primary for consistency).
Observable per variant: fraction of beta1-occupied windows with no spanning interval (replacement rate),
and the mean spanning fraction (spanning / min_beta). Covariate: window Q relative to that variant's own native.
Hypothesis: less stable variant shows higher replacement at matched Q. Decision: a descriptive consistent ordering
across replicas (replica = unit) with a replica-level permutation test; with few replicas this is exploratory.

## Exact design (fixed 2026-09-23 before computing)
T4 lysozyme (fireprotdb_md_validation/ca_coords): WT, S117V (ddG -2.0, stabilizing), G30F (+1.5, destabilizing);
3 x 100 ns at 300 K each; 2001 frames at 50 ps.
Primary: sample every 20th frame (1 ns, as mdCATH), windows of 20 frames (5 per run), eps 5 A, debounce 3.
Run-level observable: continuity C = mean over beta1-occupied windows of spanning / min_beta
(fraction of the loops present throughout the window that are one continuous class); also the replacement rate.
Prediction: C(S117V) > C(WT) > C(G30F). Test: Spearman between run-level C and -ddG over the 9 runs, with the exact
permutation p over all 1,680 assignments of the 3/3/3 labels (one-sided). Also compare framewise baseline: mean beta1 per run.
Secondary: eps 6; and 250 ps sampling (every 5th frame, 80-frame windows = 20 ns).
RNase HI (DynamicPDB, one 100 ns / 1 ns run each: WT 2RN2, K95G stabilizing, G77A destabilizing): descriptive only,
prediction C(K95G) > C(WT) > C(G77A). Q per run is reported to check that variants are at comparable nativeness.
