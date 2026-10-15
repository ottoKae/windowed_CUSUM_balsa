# Final reference scoring: 2026-10-15

These aggregate tables supersede the probability precision numbers in
`../w36_standard/precision.csv`. The latter remains the historical reproducible
release snapshot. The detector and B20 reports did not change.

| File | Meaning |
|---|---|
| `main_results.csv` | Final probability precision endpoints and separately scoped frozen catalogue recall |
| `weighted_precision.csv` | Original two-stage weighted estimator, U scenarios, and approximate outer intervals |
| `error_composition.csv` | Aggregate count and weighted contribution of each error category |
| `evaluation.json` | Current scoring overlay; +/-3 months primary, +/-1 and +/-2 sensitivity |
| `SOURCES.json` | Source and export hashes; no report IDs, geometry, or private notes |

The 307 sampled reports retain their original weights and represent 1,266
reports. The 1,049 catalogue events are a separate reference population. Do not
combine these precision and recall estimates into an overall F1. A nonzero U
count is retained; endpoint ranges are not confidence intervals. The primary
three-month tolerance was chosen after inspecting previous results. A +/-3-month
tolerance accepts seven discrete month labels, not a three-month aggregation
window. B20 aggregation still uses three consecutive calendar months.

`configs/standard_w36.json` and `scripts/evaluate_standard.py` reproduce the
historical detector/capsule release, whose displayed primary tolerance is two
months. This directory supplies the final scoring overlay and aggregate results;
the older deidentified evidence capsule has not been silently relabeled as the
updated reference evidence. Raw review records remain private. Reproducing these
updated aggregates from private evidence requires the versioned source data;
no public download or additional redistribution right is implied.

Create these aggregate-only copies with
`python tooling/release/prepare_public_aggregates.py` in the author workspace.
This copies recorded results, translates error-category labels, checks the 307
sample totals, and does not rerun detection, matching, or estimation.
