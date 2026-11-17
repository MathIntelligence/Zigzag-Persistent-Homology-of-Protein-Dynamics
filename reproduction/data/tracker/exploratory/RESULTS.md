# Diagram-only tracker vs zigzag continuity (run 2026-10-09)

Question: can a tracker built only from framewise persistence diagrams (no residue labels) reproduce zigzag continuity,
and does zigzag add anything when the tracker is already in the model?

Data: the paper's 14 global domains, 413-450 K qualifying runs (93 runs, 2,325 windows of 20 frames; 1,932 occupied
under eps 6 A / debounce 3). Same windows as baseline_test/ and the paper tables. Exploratory, not pre-registered.

## Tracker (tracker.py, gated.py)
Per frame: Rips H1 diagram of the C-alpha cloud (gudhi, max edge 10 A, max dim 2). Consecutive diagrams matched by the
optimal W1 matching (L-inf ground metric, diagonal allowed); matched points are chained. A point is "present at eps" if
birth <= 6 < death; these are exactly the beta_1 classes of the eps-complex (check: framewise min beta at 6 A equals the
raw (no-debounce) zigzag min beta in 100% of windows). Variants: match whole diagrams (full) or only eps-present points
(eps); ungated, or keep a match only if the matched points are within delta = 0.25, 0.5, 1.0 A (8 variants, 40 features).
Window features per variant: tracked spanning chains, step survival, mean run overlap, mean run length, longest run.

## 1. The tracker does not recover continuity (gate_table.txt, analysis3_output.txt)
Among occupied windows (zigzag eps 6, debounce 3; 1,034 continuous, 898 hidden replacement):
- Ungated tracker calls 83% (full) / 94% (eps) of hidden-replacement windows continuous; same with the raw
  (no-debounce) zigzag as reference: 75% / 89%. W1 matching links a loop to whichever loop has similar (b, d).
- Tight gates (0.25-0.5 A) call almost nothing continuous (0.2-7.8% of truly continuous windows).
- No gate separates the two: best balanced accuracy 0.64 (debounced reference) / 0.68 (raw reference), at delta = 1 A,
  where 22-33% of hidden-replacement windows are still called continuous and 37-45% of continuous windows are missed.
- Predicting continuity (LODO logistic, median within-domain AUC): tracker (all 40 features) 0.79 (worst 0.60);
  framewise beta_1 trace 0.91 (worst 0.78); beta trace + tracker 0.90. Tracker adds nothing to the beta trace
  (diff -0.002, 95% CI [-0.015, +0.004]); beta trace beats tracker in 14/14 domains. Raw reference: 0.81 / 0.92 / 0.91.
  Note the beta_1 trace already predicts continuity at 0.91 (as in the paper); the tracker does not add to that.

## 2. Native-like (Q>=0.6) vs unfolded (Q<0.3), occupied windows (paper table task; 1,554 windows, 13 domains)
LODO logistic, median within-domain AUC (worst domain):
  beta trace 0.939 (0.52) | tracker, all variants 0.960 (0.77) | zigzag 0.980 (0.76)
  tracker + zigzag 0.998 (0.89) | beta + tracker 0.956 (0.76) | beta + tracker + zigzag 0.998 (0.89)
Paired, hierarchical bootstrap (domains then runs, 2,000 resamples, fixed LODO predictions):
  zigzag - tracker            +0.000, wins 4/13 (3 ties), 95% CI [-0.034, +0.048], P(<=0) 0.76  -> no difference
  tracker + zigzag - tracker  +0.020, wins 9/13 (3 ties), 95% CI [+0.000, +0.061], P(<=0) 0.043
  tracker + zigzag - zigzag   +0.010, wins 8/13 (3 ties), 95% CI [+0.000, +0.060], P(<=0) 0.21
  beta+tracker+zigzag - beta+tracker  +0.014, wins 8/13, 95% CI [+0.000, +0.030], P(<=0) 0.22
  tracker - beta trace        +0.018, wins 7/13, 95% CI [-0.035, +0.099], P(<=0) 0.25
Single features: ungated tracker features are near chance (0.43-0.60); the best tracker features are the tightly gated
ones (eps, delta 0.25: mean overlap 0.86, worst 0.52), i.e. how little the diagram jitters between frames, not loop identity.
zz mean class overlap alone: 0.985 (0.74).

## Reading
- The diagram-only tracker is the strongest framewise imitation of continuity we tried, and it fails at the thing the
  paper claims: it cannot tell hidden replacement from continuity (best balanced accuracy 0.68). This is the empirical
  counterpart of the proposition that framewise diagrams do not determine the zigzag barcode.
- For native vs unfolded, gated tracker features (diagram stability) are competitive with zigzag; zigzag is not
  significantly better than the tracker. Adding zigzag to the tracker gives the best model (median 0.998, worst domain
  0.89 vs 0.77), a gain over tracker alone of borderline significance (P 0.043; CI touches 0). Ceiling effects: many
  domains are at AUC 1.0 (3-4 ties).
- Claims supported: "a diagram-matching tracker labels most hidden-replacement windows as continuous" and "zigzag
  adds to the best framewise model, mainly in the worst domains". Not supported: "zigzag beats framewise tracking at
  separating native from unfolded windows".

Files: tracker.py, gated.py, analyze.py (first pass, analysis_output.txt), analyze2.py (gate table, gate_table.txt),
analyze2_lib.py + analyze3.py (LODO + bootstrap, analysis3_output.txt), window_tracker_merged.csv, window_tracker_gated.csv.
