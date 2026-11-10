# Confirmatory test: zigzag replacement vs future native-contact loss at eps 5 A
Frozen 2026-09-23, before any topology of the confirmation domains is computed.
Motivation: exploratory fate test (fate_matched_test/) failed at eps 6 A and was positive at eps 5 A (12/13 domains).
This test checks whether the eps-5 signal transfers. The eps 6 result stays reported as the primary exploratory failure.

Domains: the next five of the frozen 413 K expansion ranking (mdcath_order_targeted_10/ranked_candidates.json,
sha256 fde66a28...), i.e. ranks 16-20: 2xtqA01, 3bt7A02, 3zmkB02, 2kptA00, 2mcfA00. All retained regardless of outcome.
Pipeline identical to global_continuity.py: C-alpha, Rips eps 5 A, max dim 2, debounce 3, zigzag_dynamics
build_filtration/compute_bars, 20-frame non-overlapping windows, Q from 320 K replica-0 frame-0 native contacts
(8 A, |i-j| >= 4). Qualifying run: Q(0) >= 0.6 and 20 consecutive frames with Q < 0.2. Temperatures 413 and 450 K, all 5 replicas.
Implementation validated first against existing window_summary.csv rows (exact equality of min/max beta, spanning, Q).
Test (unchanged from the exploratory run): windows with beta1 > 0 at every frame in qualifying runs; replaced = no spanning
interval; match each replaced window to the persistent window from the same domain/temperature with identical
(min_beta, max_beta) and nearest Q_mean within 0.05. Outcome dQ1 = Q_mean(next window) - Q_mean(window).
Pass criterion (single, fixed): the pooled matched mean dQ1 difference (replaced - persistent) < 0 AND
at least 4 of 5 domains negative. Also report a within-stratum permutation p and dQ2. If a domain has no
matched pairs it counts as not negative.

## Amendment (2026-09-23, before any confirmation topology was computed)
HuggingFace downloads are blocked from both the cloud workspace and the local VM, so ranks 16-20 cannot be fetched now.
Substitute rule, fixed now: use ALL domains of the same frozen ranking that are already on disk as verified C-alpha arrays
(mdcath_order_targeted_10/input) and are NOT among the 14 exploratory domains: 2k0sA00, 4d70A00, 3k6sD04, 3or1B01,
3mwnA00, 2kyyA00. Their window-level topology and fate outcome have never been computed; their loop-ORDER analysis
was (targeted_10), so they are untouched for this question but not for the dataset. Known in advance from that
screen: qualifying 450 K runs = 2, 2, 0, 1, 5, 0.
Revised pass rule: testable domain = at least one matched pair. Require >= 3 testable domains (else INCONCLUSIVE);
PASS if the pooled matched mean dQ1 < 0 AND >= 80% of testable domains have a negative domain mean.
Ranks 16-20 remain the preferred cohort and can be run with the same script once downloaded.
