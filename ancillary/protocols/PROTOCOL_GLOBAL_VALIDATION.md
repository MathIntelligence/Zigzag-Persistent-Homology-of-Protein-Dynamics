# Full-protein continuity and untouched validation

Frozen 2026-09-10 before global continuity outcomes and before downloading the
validation domains. Regional continuity results motivated checking the full
complex: induced-subcomplex holes can be filled by omitted residues.

## Development and validation

Development: all nine domains in PROTOCOL.md; all five 320 K runs, previously
qualifying 450 K runs, and qualifying 413 K runs for the three cross-temperature
domains. Compute the WHOLE protein complex, without atlas matching or regions.

Untouched validation: exactly the next five domains (ranks 11--15) of the already
saved expansion-metadata ranking, after the previous targeted ten. Freeze IDs,
ranking hash, protocol hash, and raw-file absence before downloading. All five
are retained regardless of outcome. Extract five runs each at 320, 413, 450 K
from pinned mdCATH revision 5e3ed8aec62b689e01751db16275fdcdbc39e47f.
Compute all 75 trajectories, report each temperature and each domain. For
unfolding-specific descriptions use the unchanged independent Q eligibility
rule; no minimum family count is needed for a whole-complex diagnostic.

## Observable

Same exact observable as PROTOCOL_CONTINUITY.md, now for the full protein:
Rips epsilon 6 A, max simplex dimension 2, debounce 3. Nonoverlapping 20-frame
windows (19 ns endpoint span); minimum and maximum instantaneous beta1; number
of bars spanning all 20 frames. Do not filter short bars. A window has hidden
replacement when min beta1>0 but no interval spans it. A stricter example has
beta1=1 at every sampled frame with no spanning interval. Both are mathematical
statements conditional on the chosen complex and intermediate union maps.

Describe rates with denominators, aggregate equally by run and then domain;
do not treat windows or homology bars as independent experimental samples.
Separate 320 K, 413 K and 450 K and independently Q-confirmed runs. Display
selected examples using deterministic rules; no unreported search for a preferred
cutoff. Recompute 5 and 7 A with debounce 3, and 6 A without debounce for all
development and validation high-temperature runs that meet Q eligibility.

Operational replication criterion: at 6 A/debounce 3, at least three of the five
untouched domains have hidden-replacement windows in at least two qualifying
high-temperature runs each. Also report the stricter constant-one cases whether
or not they replicate. This criterion documents transfer of the descriptive
phenomenon, not statistical superiority or a p-value. No replacement batch if
it fails. Parameter sensitivities are separate outcomes.

## Value and limits

This is a direct test of information lost by a common scalar baseline: continuous
Betti occupancy need not imply surviving temporal homology. It is not a new
mathematical theorem; it does not show that all framewise PH methods, sequence
models, or contact graph histories are unable to detect change. Native-contact
loss and loop disappearance are distinct. No biological memory, folding nucleus,
bond rupture, or identical unfolding-pathway claim follows automatically.

Deliver a transparent paper candidate only if exact examples, untouched transfer,
robustness and source validation support it. Otherwise document the failed
validation rather than advertise a publishable discovery.
