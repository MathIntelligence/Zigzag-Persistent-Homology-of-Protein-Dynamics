# Contact-history comparator for full-complex continuity

2026-09-10. Additional development analysis, frozen while the five new public
validation files are being prepared and before their topology is computed.
Use the same domains, 20-frame windows and geometric parameters as the global
continuity analysis. Do not select domains or windows by a desired result.

For each window intersect its edge sets, then construct the clique complex of
the contacts present at EVERY sampled frame. Compute its exact H1 dimension.
This is a fixed-contact-cycle baseline, not a full contact-history model.

Report two mathematically distinct failures of scalar interpretations:
1. H1 present at every frame but no spanning zigzag interval: uninterrupted
   loop occupancy does not imply temporal continuity.
2. A spanning zigzag interval but H1 of the contact-intersection complex is zero:
   temporal homology can survive without a nontrivial cycle using exclusively
   contacts retained throughout the window. Representatives must change; this
   alone does not identify a unique physical loop or biochemical mechanism.

All sensitivities and validation domains are retained. Native-contact loss is
reported separately; the intersection test itself uses no native reference.
Show conditional denominators and run/domain summaries, not window-level
significance. Check intersection Betti numbers independently with Dionysus.
The criterion is a specific descriptor comparison, not superiority over all
contact-based or static-PH time-series methods.
