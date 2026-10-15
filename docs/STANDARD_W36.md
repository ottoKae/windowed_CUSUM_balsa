# Frozen W36+B20 method and evaluation versions

The detector configuration is `configs/standard_w36.json`. W36 means 36 months,
not weeks: 12 months before the central 12-month decision interval and 12 months
after it. The step is 12 months. VV is the main input, using the original joint
validity mask. Missing observations remain missing. Observation context spans
2016-01 through 2026-06; the reporting period is 2017-01 through 2025-12.

## Detection and B20 aggregation

The original CUSUM statistic and negative-shift screening are unchanged. The IID
permutation test uses 3,999 initial permutations. Candidates with initial p in
[0.015, 0.035] receive an independent refinement to 19,999 permutations; the
admission threshold is p <= 0.025. Identity, global window offsets and random
seeds are frozen. This is a window-specific null test, not a posterior harvest
probability or a calibrated regional false-discovery rate.

B20 uses all fixed master pixels of each stand. Pixel centres must satisfy the
frozen stand-membership rule. All consecutive three-calendar-month windows are
considered, including those spanning year boundaries. Windows are ranked by
remaining distinct master pixels, then remaining pixel-month response nodes,
then earlier start. All remaining nodes in the selected window are assigned at
once and cannot be assigned again. Support must reach at least 20% of the fixed
stand denominator. The representative month is the response-node median, mapped
to an integer month using floor(median + 0.5). Reference dates never select a
window or a representative month.

`scripts/reproduce.py` uses the installed `lr_cusum` numerical and aggregation
implementation. Legacy script names retain compatibility imports only.

## Final evaluation versus historical replay

The original configuration and evaluation capsule use +/-2 months as primary.
They remain unchanged for historical reproduction. `scripts/evaluate_standard.py`
replays that capsule; it does not replace its evidence with later revisions.

The final overlay is `results/final_reference_score_20261015/evaluation.json`.
It uses +/-3 months as the primary display, chosen after earlier scores had been
examined. +/-1 and +/-2 months are also reported. This changes the displayed
scoring tolerance, not the detector, B20, representative month or polarization
pairing. +/-3 months accepts seven discrete month labels; it is not the same as
B20's three-month aggregation window. Monthly output is not proof of monthly
accuracy.

- Probability precision: 307 two-stage probability samples from 1,266 frozen
  reports. Retain original selection, weights and T/F/U states. Identity,
  location, temporal compatibility and one-to-one credit must all qualify.
- Estimate with the original two-stage Hajek ratio and design variance. Unknowns
  contribute 0/1 to scenario endpoints; an approximate 95% outer interval uses
  the original degrees-of-freedom choice of 26. Endpoints are not confidence
  limits, and the optimistic endpoint need not be jointly achievable under
  one-to-one credit.
- Catalogue recall: 1,049 eligible monthly events from frozen R0. Ten eligible
  interval references are evaluated separately. New report-discovered events do
  not automatically enter the primary recall denominator.
- Recall uncertainty is a spatial-block bootstrap conditional on the existing
  catalogue, not a national probability estimate.
- Do not combine probability precision and selective catalogue recall into
  population F1. No qualified complete-history common domain is available.
- Multi-event history recovery is the fraction of stands with at least two
  eligible catalogue events for which all those registered events were found.
  It does not demonstrate recovery of all real harvests.

## Existing sensitivity experiments

Polarization pairing uses its own +/-2-month tolerance. The intersection retains
VH-supported VV nodes at VV dates. The union retains VV nodes and adds unpaired
VH nodes at their original dates. Scoring sensitivity never regenerates fusion.

The three-month calendar-block null examines the complete pre-permutation testable
candidate set, including candidates that failed IID. Each null uses its own initial
p-values to choose refinement and recomputes the original full CUSUM extremum on
each resample. The 512-pixel seed diagnostic compares pixel responses only; it is
not a valid stand-B20 experiment on incomplete stand membership.

Truncation and the additional Polygon-first pathway were conditional experiments,
not automatically completed W36 results. If later executed, truncation must first
clip observations, admit only fully ended B20 windows, preserve global month/seed
identities and retain all pixels and candidate history of each evaluated stand.
Cache by configuration, stand and cutoff month. A fixed eligible reference cohort
must distinguish insufficient computation support from supported non-detections.

## Data and manuscript boundaries

Real review rows, imagery, exact locations and workbooks remain private. Separate
capsules do not replace visual evidence and are not assumed publicly deposited.
The current author manuscript and figure assets are not changed by this release.
Historical W48/Polygon-first graphics and F1 values do not become W36 results by
being copied into a public source tree. Consult the result and figure indexes.
