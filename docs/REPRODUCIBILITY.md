# Reproducibility

## Three supported levels

1. **Run the method:** `scripts/run_current_pipeline.py` runs the current detector,
   TDA and standalone STP using invented or supplied pixels.
2. **Inspect paper results:** `results/summary/` contains frozen aggregate CSVs;
   `scripts/experiments/check_published_results.py` checks counts, metrics and
   common-baseline consistency. No real individual records are needed.
3. **Replay the frozen baseline:** download dataset V1.0.0 from
   [Zenodo](https://doi.org/10.5281/zenodo.24223182) and follow
   [the dataset guide](../dataset/README.md). It includes numerical trajectories,
   references, frozen products, paired bootstrap draws and STP replay inputs.
   Historical experiments and figure builders can require additional author-side
   inputs; the data release does not include commercial optical imagery.

## Supplied-pixel inputs

```sh
python scripts/run_current_pipeline.py --input-dir data/local/my_inputs --output runs/my_application
```

The input directory contains:

| File | Contract |
|---|---|
| `units.csv` | Unique `unit_id`, `track_key`, integer `native_row`, `native_col`; one row per fixed native pixel. Preserve `pixel_track_id` when supplied: it fixes the original detector permutation identity. Otherwise unit_id is used. |
| `membership.csv` | `unit_id`, `sample_id`, `source_row_original`; exactly one stand per pixel |
| `monthly_vv.npy` | Float matrix in units.csv row order, shape `(pixels,126)`, dB values from 2016-01 to 2026-06; NaN for absent or joint VV/VH-ineligible observations |

Values must already follow the paper's track selection and joint-observation mask.
This entry does not download data, reproject polygons or form monthly composites.
The fixed calendar reflects the existing frozen detector interface.

Outputs include accepted pixel dates, execution eligibility, TDA monthly states,
phase links, source ownership, parent reports and conditional STP groups. STP
uses a fresh stream by default. With the V1.0.0 dataset, pass
`--stp-replay dataset/reproduction/stp_replay` to reproduce the historical cached
calibration prefix. `--workers 4` parallelizes independent pixel detection without
changing output order or pixel seeds. The [dataset guide](../dataset/README.md)
explains the reference and national replay commands and the scope of randomization.

`pixel_detections.csv`, `tda_events.csv` and `stp_groups.csv` always retain their
column headers, including empty results. `stand_monthly_observation.csv` records
the same active / observed non-detection / unknown states as TDA.
`stand_summary.csv` distinguishes fully observed, partially observed and
unassessable supplied stands over 2017–2025. A stand with no observed or active
reporting month has a missing `report_count`, not an assessed zero. Partial
coverage remains explicit; a count of zero is not proof of no harvest.
Only stands represented in the supplied membership table are included.

## Supplied-report evaluation

```sh
python scripts/experiments/evaluate_reports.py --references data/local/references.csv --reports baseline data/local/baseline.csv --reports comparison data/local/comparison.csv --output runs/my_evaluation
```

References require `event_id`, `sample_id`, `reference_month` (YYYY-MM).
Reports require `prediction_id`, `sample_id`, `first_response_month`,
`last_response_month`, `representative_month` (YYYY-MM).
Dates must be nonmissing; the first month cannot follow the last month, and
the representative month must lie within that range. Header-only report tables
are valid zero-report inputs.
Only stands in the reference cohort are evaluated. The existing one-to-one solver
uses overlap with reference +/-1 month and deterministic tie breaking. A long report
cannot match multiple references. Stored report spans are not enlarged on disk.

All configurations share 5,000 stand bootstrap samples. Adjacent contrasts follow
the supplied configuration order (current minus previous). Use `--ci 0.975` for a
prespecified comparison requiring that interval, otherwise the default is 0.95.
`--draws` accepts the original integer sample-index array; its columns/index values
must follow sorted stand IDs. Without it, the script generates new draws using
`--seed` (default 20261114), so confidence limits need not equal the paper's frozen
limits. This command does not tune detectors or invent missing reference events.

## Figures and manuscript

The seven final PDFs are readable without raw inputs. Their scripts, dependencies
and commands are listed in `manuscript_JAG/figure_sources/FIGURE_BUILDERS.csv`.
Some figures require private imagery/geometry; Figures 1-4 also retain frozen
artwork or layout dependencies. These are not seven independent raw-data builders.
The public statistical summaries are useful for checking results and plotting new
summaries, but do not replace all private figure inputs.

See `manuscript_JAG/BUILDING.md` for the official CAS template and TeX build.
Building TeX reuses approved figures; it does not run experiments or redraw figures.
