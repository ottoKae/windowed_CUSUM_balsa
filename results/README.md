# Frozen public result tables

These are compact, non-identifying exports from the frozen analysis. They are
included so readers can verify manuscript values without downloading the
private extraction caches.

## Formal validation

- `formal_validation/nested_spatial_oof_metrics.csv`: primary nested spatial
  out-of-fold comparison at the manuscript tolerance.
- `formal_validation/method_comparison_metrics.csv`: Windowed CUSUM and external
  detector baselines under the common evaluation protocol.
- `formal_validation/stand_representation_metrics.csv`: final Figure 5
  polygon-first and pixel-first representation routes under the fixed N1
  sensitivity protocol.
- `formal_validation/window_configuration_metrics.csv`: full window/step and
  tolerance table used during sensitivity analysis.

The nested spatial benchmark endpoint is polygon-first VV, spatial OOF F1 with
one-to-one event matching within +/-2 calendar months: TP=880, FP=238, FN=213,
precision=0.7871, recall=0.8051, F1=0.7960, and temporal MAE=0.7784 months.
The paper's operational N1 detector is a distinct frozen run: 3,999 initial
permutations, refinement to 19,999 permutations for candidates with preliminary
`p_perm` in [0.015, 0.035], and admission at `p_perm <= 0.025`. The legacy
800-replicate / confidence-0.975 metadata belongs to the archived L0 formal
release and must not be reported as the N1 operational configuration.

## Component and configuration sensitivity

`component_ablation/` contains the F0--F4 stage estimates and paired spatial
block-bootstrap contrasts. `configuration/` contains the final Figure 5
window geometry table (including 60- and 72-month windows) and temporal-
truncation summaries. All tables are aggregate.

## National chronology

The manuscript's national chronology is **only** the 6,658 qualified
polygon-first VV events. The additional 6,241 pixel-first VV rows represent a
separate route. Their arithmetic sum, 12,899, is a route-concatenated row count,
not a de-duplicated event estimate and not the reported national chronology.

- `national/annual_polygon_first_vv_events_2017_2025.csv` sums to 6,658.
- `national/polygon_event_count_distribution.csv` covers all 3,498 validated
  polygons, including 349 with zero detected events.
- `national/adjacent_interval_distribution_6month_bins.csv` summarizes 3,509
  adjacent intervals from 2,194 polygons with at least two events; the overall
  median is 36 months.

## Polarization sensitivity

The revised +/-2-month, one-to-one VH-support analysis retained 4,061 of the
6,658 polygon-first VV events. It is a sensitivity subset and does not replace
the primary VV chronology. `strategy_metrics.csv` and
`strategy_spatial_bootstrap.csv` are the fixed-configuration polarization
inputs used in Figure 5; the other files describe recurrence-interval
sensitivity to one-to-one VH support.

All authoritative definitions and source hashes are recorded in
`count_ledger.json` and `../provenance/SOURCE_FILES.json`.
