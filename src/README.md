# Source code

`windowed_cusum` is the installed Python package. The older `lr_cusum` import
name and its historical command-line interface are retired.

```text
src/
  windowed_cusum/
    detector/      CuSum, window geometry, screening and temporal nulls
    stand/         Pixel support, aggregation primitives and STP
    evaluation/    Matching, scoring and reference-evaluation helpers
    benchmarks/    Comparator detector implementations
  archive/         Local historical sources; not installed or published
```

Start the current detector -> TDA -> STP workflow from the repository root:

```sh
python scripts/run_current_pipeline.py --output .test-tmp/current-demo
```

The production first-post and TDA adapters retain their dated paths under
`tooling/diagnostics/`; [ALGORITHM.md](../docs/ALGORITHM.md) identifies their
callable interfaces. The four package folders contain their numerical
dependencies, not four separate alternative pipelines.

Some filenames describe earlier methods because current experiments reuse
their functions. In particular, `paper.py` supplies configuration primitives,
and `harvest_bounded_event_aggregation.py` supplies shared support utilities.
Their presence does not change the current W36 or TDA defaults.

Paper experiments are indexed in [scripts/experiments](../scripts/experiments/README.md),
and the seven figure builders in [docs/FIGURES.md](../docs/FIGURES.md).
Neither manuscript files nor released data belong in the Python package.
