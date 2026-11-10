# Early-warning test: results (2026-10-01)

Protocol frozen before outcomes: PROTOCOL.md. 66 qualifying 450 K runs, 14 domains, eps 6 A, debounce 3.

## Primary (frozen): step survival S vs Q, half-transition onset
Run median lead 0 frames; S earlier in 50% of runs; 7/14 domains; domain median -1; one-sided sign p = 0.60.
**Primary test failed.** Loss of step-to-step loop survival coincides with loss of native contacts.

## Secondary (frozen, descriptive)
C10 (classes spanning trailing 10 frames) vs Q: +14.5 frames, 14/14 domains, p = 0.0001.
S vs RMSD +13, vs Rg +46.5, vs contacts +3, vs Jaccard -11, vs beta_1 +2 (none significant).
Baselines vs Q: Jaccard leads Q (+7.5, 12/14); RMSD, Rg, contacts, beta_1 lag Q.

## Post hoc
Matched look-back (trailing 10 frames): C10 vs retained contacts +10.5 (11/14, p = 0.03); vs windowed min beta_1 +10.5 (12/13, p = 0.002).
Shape-independent onset (first exit from early noise band, mean +/- 3 SD, sustained 10 frames):
the result reverses. C10 vs Q -41 (2/13 domains); S vs Q -56.5 (0/13); C10 vs retained contacts -29; C10 onset undefined
in 25/66 runs (noisy signal).

## Conclusion
No robust early-warning signal. The C10 lead under the half-transition rule depends on how the onset is defined
(signal shape), and disappears under a noise-band rule. Zigzag continuity changes together with, not before,
native-contact loss. Report as a negative result.
Files: signals.py, signals.csv, onset.py, onsets.csv, leads.csv, windowed_baselines.py, onsets_with_windowed.csv,
band_onset.py, onsets_band.csv.
