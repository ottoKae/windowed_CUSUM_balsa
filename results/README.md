# Results behind the paper

[`summary/`](summary/) contains **22 aggregate CSVs**. These are the small tables needed to inspect the paper's results without downloading the full time series. [`SOURCES.json`](summary/SOURCES.json) links each table to its original computation and records its row count and SHA-256.

| Paper content | Files in `summary/` |
|---|---|
| VV/VH signatures, Figure 5 | `backscatter_signatures.csv`, `backscatter_absolute.csv`, `backscatter_aligned_curve.csv` |
| Progressive components, Table 3 | `table03_components.csv`, `table03_adjacent_effects.csv` |
| Configuration sensitivity, Table 4 | `table04_configuration.csv`, `table04_paired_effects.csv` |
| Detector benchmarks, Table 5 | `table05_benchmarks.csv`, `table05_primary_contrasts.csv`, `table05_paired_effects.csv` |
| Temporal block permutations | `temporal_blocks.csv`, `temporal_blocks_effects.csv` |
| Nationwide histories | `national_metrics.csv`, `national_annual_reports.csv`, `national_stand_count_distribution.csv`, `national_stp_counts.csv` |
| Canton and climate summaries, Figure 7 | `national_canton_annual_reports.csv`, `national_canton_quarterly_reports.csv`, `national_canton_ranking.csv`, `national_canton_classes.csv`, `national_climate_interval_distributions.csv`, `national_climate_interval_summary.csv` |

The canton counts sum to 6,303 TDA reports. Climate interval summaries use the first detection month of one TDA parent to the first detection month of the next parent in the same stand. They contain 3,372 intervals nationally; STP children do not add events or intervals.

```sh
python scripts/experiments/check_published_results.py
```

This verifies table hashes, metric arithmetic, cumulative increments, baseline agreement and the consistency of annual, quarterly and climate summaries. It does not rerun detection or estimate confidence intervals from the aggregate tables. [Experiment scripts](../scripts/experiments/README.md) provide the calculation sources.

For individual stands, pixels and reports, use the [Zenodo V1.0.0 dataset](https://doi.org/10.5281/zenodo.24223182). Maps and case figures are in [the manuscript figures](../manuscript_JAG/figures/README.md), with their [code and input index](../docs/FIGURES.md).

Some frozen columns retain earlier names such as `octa`, `cstr` and `catalogue_precision`; these correspond to TDA, STP and reference precision. Other folders under the author's local `results/` contain earlier configurations or detailed evidence. They are ignored, kept at their original paths for historical replay, and are not the current published result tables.
