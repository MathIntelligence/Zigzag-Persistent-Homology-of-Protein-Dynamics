# Temporal and bridge sensitivity of the validation diagnostic

Post-validation sensitivity, fixed before computing these alternative ranks.
No new validation pass is introduced and no primary definition is changed.

Use all 25 qualifying 450 K validation runs, epsilon 6 A and original debounce-3
edge traces. Keep the original nonoverlapping 20-frame windows, the 19 ns endpoint
span, and the denominator of windows occupied at all original 20 frames.

For every such window compute exact generalized ranks for:
- union zigzag after keeping every second frame and the final frame;
- union zigzag after keeping every fifth frame and the final frame;
- intersection zigzag at the original 1 ns sampling, reversing the arrows to
  `K_t <- K_t intersection K_(t+1) -> K_(t+1)`.

Smoothing is performed at original resolution before subsampling. The irregular
final subsampling interval preserves both endpoints; these are controlled
temporal ablations, not newly simulated coarse trajectories. Report all changes,
including cases that switch interpretation. The alternative bridge conventions
are different mathematical measurements, not expected equivalents.

This does not replace higher-frequency MD or establish the physical state of
unobserved intermediate configurations.
