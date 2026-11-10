# Confirmatory test result (2026-10-09): PRIMARY = FAIL (narrowly)

Protocol: PROTOCOL.md (sha256 f24a779a...), frozen and committed before any computation. No deviations.
Cohort: 44 domains / 204 qualifying runs / 4,772 windows (2,091 occupied); 17 domains eligible for the primary task.
Compute check: code reproduced all 60 window features of the exploratory run exactly (2np9A01 450 K r0), and the
window and eligibility counts match compact_confirm (4,772 windows; 17 eligible domains).

## Primary: native-like (Q>=0.6) vs unfolded (Q<0.3), LODO, median within-domain AUC (worst domain)
  beta trace 0.967 (0.40) | tracker 0.950 (0.42) | zigzag 0.984 (0.69) | tracker + zigzag 0.992 (0.81)
  beta + tracker 0.981 (0.53) | beta + tracker + zigzag 0.981 (0.80)
  tracker + zigzag - tracker: +0.021, wins 13/17 (2 ties), 95% CI [+0.000, +0.041], P(<=0) 0.059
  -> FAIL: wins criterion met (13 > 8.5) but CI lower bound is not > 0.

## Secondary (descriptive)
S1  zigzag - tracker +0.011, 10/17, CI [-0.001, +0.043], P 0.34
    beta+tracker+zigzag - beta+tracker +0.002, 10/17, CI [0.000, +0.018], P 0.39
    tracker - beta trace +0.000, 7/17, P 0.86; zigzag - beta trace +0.008, 10/17, CI [0.000, +0.025], P 0.43
S2  Hidden replacement in 40.4% of occupied windows (845/2,091). Ungated tracker labels 55% (full) / 73% (eps) of them
    continuous; tight gates label almost nothing continuous; best balanced accuracy 0.63 (delta 1 A).
    Replicates the exploratory finding (there 83/94%, best 0.68): no tracker variant separates hidden replacement
    from continuity.
S3  Predicting zigzag continuity (36 domains): beta trace 0.911, tracker 0.859, beta + tracker 0.911;
    beta trace beats tracker (+0.032, 26/36, CI [+0.004, +0.077]); tracker adds nothing to beta (9/36 wins).
S4  Transfer (trained on the 14 exploratory domains): tracker 0.970 (0.54), zigzag 0.991 (0.64),
    tracker + zigzag 0.989 (0.81); tracker + zigzag - tracker +0.014, 11/17, CI [0.000, +0.039], P 0.27.

## Reading
- Not confirmed: "adding zigzag significantly improves AUC over framewise tracking". Same direction and size as the
  exploratory run (+0.020 there, +0.021 here; 13/17 wins), but the interval still touches zero. Ceiling: most
  domains are near AUC 1 for every model.
- Consistent across both cohorts and in transfer, but not a pre-registered endpoint: adding zigzag lifts the worst
  domain (tracker 0.77 -> 0.89 exploratory; 0.42 -> 0.81 here; 0.54 -> 0.81 in transfer).
- Confirmed qualitatively: a diagram-only tracker cannot recover loop continuity (best balanced accuracy 0.63),
  and continuity is not a by-product of the tracker (beta trace predicts it better; tracker adds nothing).
