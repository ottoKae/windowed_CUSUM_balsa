# Paper results

[`summary/`](summary/) contains 16 frozen aggregate CSVs for the current W36/TDA
paper: backscatter signatures, cumulative components, configuration sensitivity,
block permutations, benchmarks and national histories. `summary/SOURCES.json`
records each source path, row count and SHA-256. Individual stand/reference rows,
coordinates, optical judgements and original imagery remain private.

```sh
python scripts/experiments/check_published_results.py
```

This checks metric arithmetic, cumulative increments and baseline agreement across
Tables 3-5. It does not recreate detector outputs or confidence intervals from
aggregate tables. To score supplied individual reports with the real matcher and
paired bootstrap, use `scripts/experiments/evaluate_reports.py`.
See the [experiment index](../scripts/experiments/README.md).

Frozen IDs `octa`, `R2_TIME`, `cstr` and `catalogue_precision` are retained in some
columns for traceability; the paper calls these TDA, STP and reference precision.
Older B20 and polygon-first result directories are private historical evidence,
not current paper results.
