# Source and result boundaries for the B/C release candidates

## Authoritative sources

The science and pixel archives use the original frozen cache, not a later raw
archive repair. Full aggregation parity is a release gate. Detection uses
unchanged defaults and explicit original per-band seeds after ID substitution.
The package preserves input order and missing months. No source data are
modified or reconstructed from figures.

The new aggregate directory results/nrt_ccdc/ contains the actual September 5
six-row comparison plus direction-controlled diagnostics and paired intervals.
The original results/formal_validation/ files remain the earlier benchmark
version. A figure referencing the earlier version must not be silently assigned
the new diagnostic labels or intervals.

## What can and cannot be claimed

| Claim | Required evidence |
|---|---|
| Source parses and release audit passes | `scripts/audit_source_syntax.py` and `provenance/audit_public_package.py` receipts |
| B and C represent the same signal | All 4,001 medians and NaN masks equal |
| Frozen statistical results reproduce | Six configurations, event matching and 5,000 paired replicates equal |
| National VV detection reproduces | Fresh 4,001-track F4 result matches every stored track-event month |
| F0–F4 reproduce from signals | Fresh component candidates feed common reference evaluation, not frozen fallback |
| Pixel-first extension runs | Real per-pixel N1 detection, local-adjacency support, then track fusion |
| Official CCDC cached results reproduce | Decode API segments, restore batch offsets, select frozen folds and match all predictions |
| New official cloud computation reproduces | Separate authorized cloud run; NOT established by cached replay |
| Exact national figures reproduce from B/C | NOT claimed; geometric/basemap D inputs are excluded |

Pixel-first N1 results must not be called a bit-exact replay of the older L0
Pixel-first experiment. Fixed 48/12 sensitivity results are not the
held-out-selected 36/12 Windowed benchmark. A within-window permutation p value
does not imply a national false-positive rate.

## Publication gates

No external release occurred. Select the code and data licenses, finalize
author/citation metadata and a public deposit, and record any required
third-party figure permissions before public release. Exact geometry and original
imagery stay excluded even if the two B/C archives are subsequently deposited.

The privacy scanner uses actual Git inclusion rules and a separate archive
schema/hash audit. A green technical check is deliberately not the same as
publication_ready. No Chinese-language file is classified as a privacy violation
merely because of its language.
