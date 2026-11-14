# Paper experiment index

The current paper is W36 first-post windowed CuSum -> TDA -> conditional STP.
Primary matching is one-to-one TDA detection-range overlap with reference +/-1 month.
All main comparisons use 459 reference stands, 1,049 records and 590 adjacent pairs.

| Paper experiment | Author implementation (relative to `tooling/diagnostics/`) | Public results |
|---|---|---|
| Section 4.1 / Figure 5: VV/VH signatures | `balsa-results-optimization-20261105/signatures.py`; `results-refresh-20261111/signatures.py` | `backscatter_*.csv` (978 qualifying events) |
| Table 3: progressive components | `cumulative-components-20261111/compute.py`, `range_overlap_pm1.py` | `table03_components.csv`, `table03_adjacent_effects.csv` |
| Table 4: geometry, polarization, threshold | `range-primary-20261111/sensitivity.py`; `configuration-extension-20261111/geometry.py`, `score.py` | `table04_configuration.csv`, `table04_paired_effects.csv` |
| Temporal null sensitivity | `block-length-sensitivity-20261111/run_blocks.py`, `score.py` | `temporal_blocks*.csv` |
| Table 5: 14 benchmark configurations | `detector-tda-benchmark-20261109/{local,recusum,cloud,aggregate}.py`; `results-refresh-20261111/benchmark.py` | `table05_*.csv` |
| Section 4.4 / Figures 6-7: national histories | `harvest-inventory-20261104/{tda,stp,products}.py`; `results-refresh-20261111/histories.py` | `national_*.csv`; final figures |

Results above are in [`results/summary/`](../../results/summary/).
[Machine-readable catalogue](../../configs/experiments/catalog.json).
Paths are retained because frozen functions import each other. Several author
scripts also perform receipt checks, use old caches or prepare manuscript tables;
they require the private workspace, not just the aggregate CSVs. Follow the
supplied-table commands below for a new application. Cloud comparators additionally
require the relevant Earth Engine access and monthly inputs.

## Public commands

- Current detection and aggregation: `python scripts/run_current_pipeline.py --output .test-tmp/demo`
- Check all published table metrics: `python scripts/experiments/check_published_results.py`
- Score your own configurations: `python scripts/experiments/evaluate_reports.py --help`

[Input formats, bootstrap settings and replay limits](../../docs/REPRODUCIBILITY.md).
Original detector and aggregation outputs were not rerun to assemble this release.
Table 3 is **cumulative**, not leave-one-component-out: global context -> local
contexts -> decision admission -> negative direction -> permutation screening.
The released sensitivity table also includes supplementary configurations; it is
not a claim that every combination of W, D, S and polarization was tested.

## Figures

Use the [seven-figure index](../../docs/FIGURES.md).
Statistical recomputation, spatial product construction and figure rendering are
separate operations. STP children never increase national parent-event counts.
