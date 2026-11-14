# What belongs where

The public repository contains the method, its tests, the paper and compact result tables. Large numerical data are distributed through [Zenodo](https://doi.org/10.5281/zenodo.24223182). The author's workspace also holds earlier experiments and source imagery; a public checkout does not include those folders.

## Keep and publish

| Location | Purpose |
|---|---|
| `src/windowed_cusum/` | Current numerical package: detector, stand aggregation, evaluation, benchmarks |
| `scripts/`, selected `tooling/diagnostics/` and `analysis/workflows/` files | Pipeline, experiment and figure implementations |
| `configs/` | Settings, experiment catalogue and dependency versions |
| `tests/`, `data/demo/`, `data/schemas/` | Reusable tests, synthetic examples and format descriptions |
| `results/summary/` | Current paper's 22 aggregate tables and their sources |
| `dataset/` | Documentation and citation for the Zenodo deposit |
| `manuscript_JAG/` | Current TeX, bibliography, readable PDFs and seven final figures |
| `provenance/`, selected `docs/` | Figure bindings, source credits and use instructions |

Dated filenames are not a reason to archive code. Some helpers still supply TDA assembly, STP and figure rendering. Their original paths remain in use and are recorded in the [algorithm](ALGORITHM.md), [experiment](../scripts/experiments/README.md) and [figure](FIGURES.md) indexes.

## Keep locally and ignore

| Location in the author workspace | Why it is retained |
|---|---|
| `src/archive/`, `archive/`, `manuscript_JAG/archive/` | Retired implementations, old builds and earlier manuscripts |
| Unlisted `data/` folders | Downloaded arrays, source imagery, extraction caches and private records |
| `runs/`, unlisted `results/` folders | Detailed computation outputs, frozen ledgers and earlier experiment results |
| Unlisted `docs/`, `tooling/`, `tests/` files | Historical plans, diagnostics and tests for retired workflows |
| `deliverables/`, `dist/` | Earlier delivery packages, release snapshots and the isolated publication checkout |
| `.qa/`, `.hallmark/`, local editor/service settings | Local review material and environment configuration |

Keep frozen inputs and evidence where recorded scripts expect them. Do not move a historical result merely to make the root directory look smaller. New acceptance reports belong in an ignored `data/private/<review-name>/` directory; new test scratch data belong in `.test-tmp/<run-id>/`.

## Delete only disposable products

Completed pytest working directories containing generated fixtures, Python bytecode caches and superseded synthetic demo outputs can be removed once their checks have finished. Keep test source files, intentional demo inputs and test expectations. A file containing `test`, `review` or `tmp` in its name may still be scientific evidence.

Real-data parity reports and manual-review evidence found inside a temporary directory should be preserved before cleanup. Active environments, uncertain folders and paths that cannot be read are left alone. Maintenance receipts record exactly what was removed or relocated.

## How publication is controlled

[`configs/public_export.json`](../configs/public_export.json) selects files individually. The generated `.gitignore` ignores everything else, including newly downloaded data and future temporary outputs. Update the selection first, then run:

```sh
python tooling/release/update_public_ignore.py
python tooling/release/export_public.py --output dist/public-review
```

Use a new output directory. Inspect and test that clean snapshot before publishing it. Ignore rules do not remove already tracked files or old Git history; the author workspace's full Git history is not used as the public release. See [publication instructions](PUBLIC_RELEASE.md) and [data/privacy boundaries](PRIVACY.md).
