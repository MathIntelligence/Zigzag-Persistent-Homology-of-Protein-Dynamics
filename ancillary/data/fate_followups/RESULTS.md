# Two follow-up tests (2026-09-23). Protocols frozen before outcomes: PROTOCOL_CONFIRM_EPS5.md, PROTOCOL_WT_MUTANT.md

## Pipeline check
The standalone driver (cont.py, which imports zigzag_dynamics build_filtration/compute_bars) reproduces window_summary.csv
EXACTLY (min/max beta1, spanning count, Q, qualifying flag) for 2np9A01 450/0, 450/1 and 413/2 at eps 5 and 6.

## Test 1: eps-5 confirmation of "replaced windows lose more native contacts next" -> INCONCLUSIVE (not confirmed)
HuggingFace is blocked from both machines, so ranks 16-20 could not be downloaded. The amended rule (written before
computing) used the six on-disk ranked domains not in the exploratory 14.
Qualifying unfolding runs (413/450 K): 2k0sA00 2/2, 4d70A00 0/2, 3or1B01 0/1, 3mwnA00 5/5, 3k6sD04 0/0, 2kyyA00 0/0
(identical to the earlier screen).
eps 5: 236 eligible windows (124 replaced), but only 10 matched pairs, all from 3mwnA00. Matched dQ1 difference +0.0035
(wrong sign). Testable domains 1 < 3 -> INCONCLUSIVE.
eps 6: 12 pairs, 2 domains, pooled +0.0027 -> INCONCLUSIVE.
Conclusion: the exploratory eps-5 signal is not confirmed. With matching this strict, most domains yield no pairs.

## Test 2: WT vs mutant continuity vs experimental stability -> NO RELATION
T4 lysozyme (3 x 100 ns, 300 K; S117V ddG -2.0, G30F +1.5), continuity C = mean spanning/min_beta per run:
- eps 5, 1 ns sampling (primary): S117V 0.300, WT 0.317, G30F 0.376. rho vs stability -0.53 (wrong direction),
  one-sided exact p = 0.94.
- eps 6: WT 0.885, S117V 0.832, G30F 0.801. rho +0.37, p = 0.18 (not ordered; WT highest).
- eps 5, 250 ps sampling: rho -0.37, p = 0.86.
- Framewise mean beta1 behaves the same way (no relation).
Replacement windows: 0 in every run. Proteins at 300 K near native (Q ~ 0.90) never show replacement,
consistent with the 0/625 mdCATH 320 K windows.
RNase HI (one run each, descriptive): C at eps 5 = WT 0.635, K95G 0.727, G77A 0.726; at eps 6 = 0.462, 0.453, 0.626.
No consistent order. Replacement 0.
Interpretation: 1.5-2 kcal/mol stability differences act on unfolding timescales (ms-s). 100 ns near-native MD does
not sample them, and neither zigzag continuity nor framewise topology resolves them here.
