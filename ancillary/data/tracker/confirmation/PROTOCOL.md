# Confirmatory test: does zigzag add to a diagram-only tracker? (frozen 2026-10-09, before any computation)

Motivation (exploratory, 14 global domains, framewise_tracker/RESULTS.md): for native-like vs unfolded occupied
windows, LODO tracker + zigzag 0.998 vs tracker 0.960 (median within-domain AUC); paired median +0.020, 9/13 wins,
bootstrap 95% CI [0.000, 0.061], P(<=0) 0.043. The diagram-only tracker labelled 83-94% of hidden-replacement windows
continuous. This run tests both on new domains.

Cohort: the 59 mdCATH domains in ../mdcath_order_added_value/compact_confirm/domains.txt (none of the 14 global
domains), C-alpha coordinates as already extracted in compact_confirm/ca/*.npz (ref = 320 K replica 0 frame 0;
413 K and 450 K, replicas 0-4). Disclosure: compact_confirm (2026-10-03) already analysed these runs with zigzag
mean overlap, step survival, windowed tau_top and framewise beta_1 / Rips summaries (its secondary analysis used the
same Q>=0.6 vs Q<0.3 task). The diagram-only tracker and the tracker + zigzag combination have never been computed on
this cohort. All domains are kept regardless of outcome.

Pipeline (unchanged from the paper and the exploratory run):
- Q: native pairs d<8 A, |i-j|>=4 in ref; contact if d<8 A. Qualifying run: Q[0]>=0.6 and >=20 consecutive frames
  with Q<0.2. Only qualifying runs are used. Runs shorter than 500 frames use their available frames.
- Zigzag: C-alpha, eps 6 A, max dim 2, debounce 3, union zigzag (zzlib / zigzag_dynamics, as in compact_confirm).
- Windows: non-overlapping 20 frames; occupied = min beta_1 > 0 over the window (debounced zigzag beta_1).
- Zigzag window features (8): spanning, retained fraction, max overlap, mean overlap, mean lifetime, step survival,
  turnover, number of classes (definitions as baseline_test/zigzag_features.py).
- beta_1 trace (6): min, mean, max, SD, lag-1 autocorrelation, mean |delta beta_1| (as baseline_test/features.py).
- Tracker (40): per-frame Rips H1 diagram (gudhi, max edge 10 A, max dim 2); consecutive diagrams matched by optimal
  W1 matching (L-inf ground, diagonal allowed); variants full/eps x {ungated, gate 0.25, 0.5, 1.0 A}; 5 features each
  (tracked spanning, step survival, mean overlap, mean life, max overlap) (as framewise_tracker/tracker.py, gated.py).
- Models: logistic regression (sklearn defaults, max_iter 5000), features z-scored within domain, leave-one-domain-out.
  Score = within-domain AUC of LODO predictions.

PRIMARY. Task: native-like (window mean Q>=0.6) vs unfolded (Q<0.3), occupied windows. Eligible domain: >=5 windows
in each class. Statistic: median over eligible domains of AUC(tracker + zigzag) - AUC(tracker).
SUCCESS iff the hierarchical bootstrap (domains, then runs within domain; 2,000 resamples; seed 20261009; fixed LODO
predictions) 95% CI lower bound is > 0 AND tracker + zigzag beats tracker (strictly) in more than half of eligible
domains. If fewer than 8 domains are eligible the test is inconclusive.

SECONDARY (reported whatever the outcome; not used for success):
 S1 zigzag vs tracker; beta + tracker + zigzag vs beta + tracker; tracker vs beta trace (same task and statistics).
 S2 Hidden replacement: among occupied windows, fraction of zigzag hidden-replacement windows (spanning = 0) that each
    tracker variant labels continuous (tracked spanning > 0), and balanced accuracy of each variant.
 S3 Predicting zigzag continuity (spanning > 0) among occupied windows: LODO beta trace vs beta + tracker vs tracker.
 S4 Transfer: models trained on all 14 exploratory domains, tested on each new domain (primary comparison repeated).
Failure is reported as failure; no threshold, feature or cohort is changed after seeing outcomes.
