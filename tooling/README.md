# Repository tooling

The public snapshot retains selected tools and algorithm dependencies at their
original paths. The [algorithm guide](../docs/ALGORITHM.md) maps the current
first-post W36 → TDA → STP implementation and its required input tables.

| Path | Responsibility |
|---|---|
| `release/export_public.py` | Build a fresh allowlisted candidate or verify an exported `PUBLIC_FILES.json` |
| `diagnostics/balsa-national-firstpost-20261030/detector_engine.py` | Current first-post pixel detector adapter |
| `diagnostics/harvest-inventory-20261104/tda.py` | Current TDA supplied-table interface and frozen inventory adapter |
| `diagnostics/harvest-inventory-20261104/stp.py` | Current STP adapter with frozen calibration and RNG continuation |
| Selected episode, baseline and release helpers | Existing imports required by the unchanged TDA implementation |
| Selected figure diagnostic scripts | Current figure renderers and their preserved helpers |

Only explicitly selected Python files are public. Other diagnostics, cataloguing
work, review helpers and execution receipts remain local. Importing a retained
helper does not publish its private inputs. Some historical release helpers are
retained because the current algorithm imports them, not because their old data
packages are included or should be rebuilt.

Run release commands from the repository root:

```sh
python tooling/release/export_public.py --output dist/public_repository_20261113
```

Choose a fresh destination for each candidate. The allowlist is
`configs/public_export.json`; approved PDF paths have exact hashes. The exporter
writes the public file inventory without copying `.git` or old repository
history. It does not upload anything. In the exported directory:

```sh
python tooling/release/export_public.py --verify .
```

An exported checkout does not contain the private inputs used to assemble or
replay author-side research outputs. Readers can inspect the implementation and
run selected synthetic tests without those inputs. Full workflow and figure
replay remain separate operations with the boundaries documented in the
[release guide](../docs/PUBLIC_RELEASE.md).

Reusable tests belong in `tests/`; disposable output belongs in
`.test-tmp/<run-id>/`. Preserve scientific acceptance receipts and historical
records locally. Old organization/packaging scripts describe their original
migrations and are not current cleanup commands.
