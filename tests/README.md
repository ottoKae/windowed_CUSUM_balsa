# Tests and disposable artifacts

`tests/` contains reusable regression tests. The public export selects the core
aggregation, matching, probability-estimation and release checks that have their
dependencies in that export. Additional author-workflow tests remain in the local
checkout; their presence does not imply they all run without private inputs or
GUI dependencies.

To check a clean public export after a normal installation:

```sh
python -m pip install ".[io,validation,replay]" -c configs/requirements-reproduce.txt
python -m pip install pytest geopandas -c configs/requirements-reproduce.txt
python -m pytest tests --import-mode=importlib -q -p no:cacheprovider --basetemp .test-tmp/public-check
```

Use a unique basetemp for simultaneous runs. Pytest may clear that exact directory;
never point it at production data, the project root or a shared test parent.
Verify that `lr_cusum.__file__` resolves inside the environment's `site-packages`
when checking a normal installation.

- Reusable source: `tests/`.
- Disposable artifacts: `.test-tmp/<run-id>/`, ignored.
- One-off author diagnostics: `tooling/diagnostics/<run-id>/`, ignored.
- Scientific/release acceptance receipts: `runs/<run-id>/`, ignored but retained.

Do not delete a file merely because its name includes `test`, `tmp` or `review`.
Historical reference and audit records remain evidence. The public release
inventory, rather than recursive copying, determines which tests are shipped.
