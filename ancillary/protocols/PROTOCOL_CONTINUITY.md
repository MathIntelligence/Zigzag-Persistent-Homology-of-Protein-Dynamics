# Exact regional continuity versus continuous loop occupancy

Frozen before computing regional zigzag results, 2026-09-10. This is a new
exploratory analysis motivated by the completed ordering-baseline audit, which
did not establish general superiority over localized framewise H1.

Question: how often does continuous regional H1 occupancy fail to correspond to
even one homology interval spanning the observation window? This tests an exact
temporal distinction rather than strength of rank agreement.

Use all nine domains in PROTOCOL.md, all five 320 K reference runs, all previously
qualifying 450 K runs, and all qualifying 413 K runs for the three cross-temperature
domains. Regions are the previously fixed core-family consensus residue supports;
they are NOT individual loops, and induced subcomplexes may contain holes that
fill in the full protein. They are fixed using only 320 K training data.

Construct the regional induced Rips complex at 6 A with debounce 3, through
dimension 2. All H1 bars count, including <3-frame bars, so the instantaneous
Betti number reconstructed from the barcode must match independently calculated
framewise H1 exactly. Do not apply a duration filter to this equality.

Partition each 500-frame trajectory into 25 nonoverlapping windows of 20 sampled
frames (19 ns between endpoints). For each region-window record minimum/maximum
beta1, number of bars covering every frame, longest overlap fraction, Q, and
location relative to the independent global contact-loss event. A hidden-
replacement window has beta1>0 at every sampled frame but zero spanning bars.
A stricter diagnostic requires beta1=1 at every frame and zero spanning bars.
The descriptor is the number of interval summands covering the window (generalized
rank for this zigzag), not a uniquely identifiable physical ring. Intermediate
union complexes are part of the temporal construction.

Record all maximal constant-beta1=1 bouts lasting >=20 frames, and their spanning
bar count. Inspect a concrete example chosen deterministically: longest such
bout with zero spanning bars among qualifying 413/450 K data, tie by domain,
temperature, replica, family, start. Compare a continuous-class bout as control.
Validate barcode/Betti equality, an independent F2 calculation and direct
Dionysus calculations. Document whether interruptions occur only in the union
bridges between snapshots or also at sampled frames.

Sensitivity: recompute the three cross-temperature domains on their 413/450 K
runs at 5 and 7 A (debounce 3), and at 6 A without debounce. Keep all outcomes.
These are geometric sensitivities, not parameter equivalence or physiological
validation. Do not optimize the cutoff after results.

This can establish information lost by regional Betti counts/occupancy. It cannot
establish superiority over all static PH feature pipelines, contact-map sequence
models, or independently prove a physical unfolding mechanism. Because the
question is developed on previously inspected data, any promising biological
association requires a separately frozen untouched cohort.
