# Early-warning test (frozen 2026-10-01, before any outcome was computed)

Question: in unfolding trajectories, does loss of zigzag loop continuity occur earlier than changes in standard
descriptors?

Runs: all 66 qualifying 450 K runs (14 domains), eps = 6 A, debounce 3 (same complexes as the paper).
Per-frame signals (t = 0..499):
  Zigzag (primary): step survival S(t) = fraction of H1 classes alive at t still alive at t+1 (1 if none alive).
  Zigzag (secondary): C10(t) = number of classes spanning the trailing window [t-9, t].
  Baselines: Q(t) (8 A native contacts), RMSD to native (C-alpha, Kabsch), radius of gyration, number of contacts
    (6 A debounced), consecutive contact Jaccard J(t) = |E_t n E_{t-1}| / |E_t u E_{t-1}| (6 A debounced),
    framewise beta_1(t) (debounced complex).
Smoothing: identical trailing 10-frame mean for every signal (no look-ahead).
Onset (half-transition time) for each signal: first frame t at which the smoothed signal crosses the midpoint between
  its early level (mean of raw signal over frames 0-19) and its late level (mean over frames 450-499), moving from the
  early side to the late side, and stays on the late side for at least 10 consecutive frames.
Lead of a signal = onset(Q) - onset(signal), in frames (ns); positive = signal changes before Q.
Primary test: lead of S relative to Q. Per domain take the median lead over its runs; one-sided sign test over the
  14 domains (H1: median lead > 0); success if p <= 0.05 and median domain lead >= 2 frames.
Secondary (descriptive, same statistic): lead of S relative to each baseline (RMSD, Rg, contacts, J, beta_1); lead of
  C10 relative to Q.
Runs in which a signal has no defined onset are reported and excluded for that signal only.
