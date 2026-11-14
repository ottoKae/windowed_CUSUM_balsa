# Public tests

The public export contains selected tests whose implementation dependencies are
included. These cover the real temporal relation and assembly functions, TDA
over synthetic tables, STP source conservation and cached-prefix RNG behavior,
and release selection/verification. Additional author-workflow tests in the local repository
may need private data or GUI dependencies and are not all exported.

From a clean public checkout with Python 3.12:

```sh
python tooling/release/export_public.py --verify .
python -m pip install ".[io,validation,geo,figures]" -c configs/requirements-reproduce.txt
python -m pip install pytest -c configs/requirements-reproduce.txt
python -I -c "import windowed_cusum; from pathlib import Path; assert 'site-packages' in Path(windowed_cusum.__file__).parts"
python -c "from pathlib import Path; Path('.test-tmp').mkdir(exist_ok=True)"
python -I -m pytest tests --import-mode=importlib -q -p no:cacheprovider --basetemp .test-tmp/public-check
```

The STP integration test
`test_frozen_cohort_actual_artifact_preserves_original_full_chain` skips when its
private frozen output directory is absent. Where those inputs are available, it
still performs the original parity checks. Synthetic tests do not need those
artifacts and do not reproduce the paper's real-data results.

`test_public_entry_imports.py` loads experiment and figure entries in separate
processes. It checks transitive helpers and configuration paths without private
inputs or the retired `lr_cusum` package. Local comparator tests also exercise
recursive dates, missing-month support and cloud-output parsing offline.

CI also runs `scripts/run_current_pipeline.py`, exercising the actual first-post
W36 → TDA → STP chain on invented pixels, and checks the published aggregate
tables with `scripts/experiments/check_published_results.py`. Historical source
code is excluded from the installed package and public snapshot.

Use a unique `.test-tmp/<run-id>/` for concurrent runs. Pytest may clear that exact
basetemp directory; never point it at production data, the repository root or a
shared parent. Keep scientific/release acceptance receipts outside disposable
test directories. Do not delete evidence merely because its filename contains
`test`, `tmp` or `review`.
