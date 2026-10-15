# Repository tooling

| Path | Purpose |
|---|---|
| `catalogs/build_sample_catalog.py` | Build a private, searchable collection of existing interpretation records and public schema/cohort summaries |
| `catalogs/build_figure_provenance.py` | Trace actual TeX figure inclusions to assets, receipts, scripts and input versions |
| `release/export_public.py` | Create a fresh public source export from an explicit allowlist |
| `release/prepare_public_aggregates.py` | Copy non-identifying final-score summaries with source hashes |
| `diagnostics/<run-id>/` | Ignored author-side one-off checks, review helpers and execution receipts |

Run helpers from the repository root. Catalogues preserve original files and
scientific identities; they do not re-detect events or re-score samples.

```sh
python tooling/catalogs/build_sample_catalog.py --help
python tooling/release/export_public.py --help
```

For example, choose a fresh public candidate directory:

```sh
python tooling/release/export_public.py --output dist/public_repository_NEW_VERSION
```

This export command and its source mappings are **author-workspace tools**. The
source-only public clone intentionally lacks the archived source locations and
private inputs used to assemble a release; it is not a re-export workspace.
Readers can install the package, run the synthetic demo and use the included code
without running these assembly helpers.

On the author workstation, `python tooling/catalogs/build_figure_provenance.py`
refreshes the figure index without rendering. It requires the original manuscript
assets and receipts; a source-only public export intentionally omits those inputs.

Use the exporter's output option to choose a new destination. Never copy the
entire author workspace to GitHub: it contains private inputs, historical review
records and licensed assets. The exporter writes `PUBLIC_FILES.json` and checks
the selected text files for accidental private filesystem paths.

The older `organize_w36.py`, `package_artifacts.py`, `write_readme.py` and related
October 4 helpers describe a completed migration/release. They are not an
idempotent current cleanup command. In particular, rerunning old packaging can
select historical exports or real capsules. Keep their historical receipts;
use the explicit exporter for the current public source candidate.

Reusable tests belong in `tests/`, not in diagnostics. Disposable test output
belongs in `.test-tmp/<run-id>/`; final verification receipts belong in `runs/`.
