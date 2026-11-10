# Added-value audit of the loop-family ordering study

2026-09-10. This is a retrospective diagnostic and method-development study.
All existing results remain unchanged. Previously observed positive proteins are
not an independent validation cohort. No promise of a positive outcome is made.

## Fixed comparison cohort

Include all nine currently testable 450 K domains from the three completed
cohorts, irrespective of their ordering result: 2d1pA00, 2d4uB00, 2dawA01,
2ervA00, 2fd5A02, 4a4aA02, 2ivxB01, 3hwcA01, 3hbvP01.
Across-temperature comparisons include all three currently testable domains:
2ervA00, 3hwcA01, 3hbvP01. Use the same qualifying runs, 320 K family definitions,
500-frame windows, and residue supports as the completed studies.

## Questions and measurements

1. Does the ordering survive removing absent and right-censored observations,
   and discarding ambiguous episode assignments? Recompute tau-b on common
   observed families (at least four) for each run pair. Report missing comparisons.
   Bootstrap runs independently within each temperature, not individual run-pair
   correlations. These intervals describe uncertainty conditional on the fixed
   families and selected domains, not population or selection-adjusted inference.
2. Is agreement specific to zigzag? Compare last appearances of:
   - Original zigzag family occupancy.
   - Local native-contact fraction Q >= 0.5 (existing support-union definition).
   - Nonzero framewise H1 in the induced complex on the SAME fixed family support,
     epsilon 6 A, with both raw and debounce-3 edges. Suppress positive pulses
     shorter than three frames as a matched episode-duration sensitivity.
   This is a localized Betti-number baseline, not a complete reproduction of
   published volume-optimal-cycle PH or an optimized static-PH competitor.
   Use exact F2 triangle-boundary rank, not graph cycle rank alone. Independently
   compare synthetic and selected real-frame Betti numbers with Dionysus.
   Summarize within/cross-temperature ordering and within-run agreement between
   methods. Differences in raw repeatability do not establish added value.
3. Can native-contact ordering account for the zigzag order? Report within-run
   associations, including observed-only sensitivity and family support sizes.
   Preserve opposing or null associations. A different descriptor is not itself
   a better descriptor.
4. Inspect the physical content of temporal features. For the three cross-
   temperature domains, recover representatives of all >=20-frame episodes
   extending >=20 frames beyond independently confirmed native-contact loss.
   Inspect one representative every 10 frames including the first and last
   post-confirmation frames. Retain all episodes and describe matched families,
   edge turnover, sequence separation, and non-native long-range edges. Require
   cycle boundary zero and nontrivial homology in the corresponding full complex
   for checked examples. Representative turnover is basis-dependent and must not
   be equated with bond rupture or unique physical-loop identity.

## Claims and decision rule

No new statistical pass will be advertised from this outcome-aware audit.
Do not claim zigzag is necessary unless a fair temporal-linkage ablation fails
while independent biological evidence supports the zigzag interpretation.
If the current headline reduces to a simpler observable or fails robustness,
record that result and redirect the paper claim. If a reproducible mechanistic
lead survives, specify a frozen independent validation experiment. A good paper
requires a distinguishable question and reliable evidence, not more positive
p-values from repeated searches.

## Deliverables

Reproducible code, complete cohort tables, source hashes, meaningful mathematical
tests, independent audit, scientific figures, and an evidence-based paper decision.
