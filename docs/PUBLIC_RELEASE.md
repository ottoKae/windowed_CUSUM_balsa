# Public snapshot

The current selection contains the numerical library, the original first-post
W36/TDA/STP dependency chain, current figure renderers/helpers, synthetic examples,
schemas, selected tests, and the author TeX, bibliography, main/supplementary PDFs
and seven approved figure PDFs, plus current experiment sources and 22 checked
aggregate result tables. The numerical study dataset is published separately on
Zenodo (see the data guide). GitHub does not distribute raw data, real row-level
tables, exact geometry, private experiment records, PPTX sources or
archives. The manuscript content and scientific results are unchanged by this
organization task.

## Selection and history

`configs/public_export.json` is the explicit allowlist. Its `approved_pdfs`
section authorizes particular PDF paths and SHA-256 hashes; it is not a blanket
permission to export PDF files. The generated `PUBLIC_FILES.json` records every
exported file. The default-deny `.gitignore` follows the public file selection;
files already tracked by Git need separate index handling.

Prepare a fresh candidate from the author workspace:

```sh
python tooling/release/export_public.py --output dist/public_repository_20261114
```

The destination must not already exist. Use another fresh name for a subsequent
candidate. The export contains no `.git` directory or old repository history.
This boundary matters because historical objects, including earlier PPTX files,
can retain personal metadata even when those files are absent from the current
working tree. Existing author history is retained locally. This preparation
does not push, change repository visibility or rewrite history.

## Verify the public checkout

From the exported directory, using Python 3.12:

```sh
python tooling/release/export_public.py --verify .
python -m pip install ".[io,validation,geo,figures]" -c configs/requirements-reproduce.txt
python -m pip install pytest -c configs/requirements-reproduce.txt
python -I -c "import windowed_cusum; from pathlib import Path; assert 'site-packages' in Path(windowed_cusum.__file__).parts"
python -c "from pathlib import Path; Path('.test-tmp').mkdir(exist_ok=True)"
python -I -m pytest tests --import-mode=importlib -q -p no:cacheprovider --basetemp .test-tmp/public-check
python scripts/run_current_pipeline.py --output .test-tmp/current-demo
python scripts/experiments/check_published_results.py
```

The public checkout contains only selected tests. The STP integration test that
needs private frozen outputs skips when those inputs are absent. Use a unique
basetemp for concurrent runs. The example runs the current first-post/TDA/STP
chain on invented data; it does not reproduce real-data study results.
CI uses the selected tests and this demonstration; local checks
do not imply that hosted CI has run.

## Manuscript and figures

Read `manuscript_JAG/main.pdf` and `manuscript_JAG/supplementary.pdf` directly.
The seven approved figure PDFs accompany the unchanged TeX. Figure rendering
from source still requires the private frozen inputs described in
`provenance/figures/source_spec.json`; supplying finished PDFs does not release
those inputs. The historical `scripts/figures/build_all.py` is not the current
seven-figure renderer.

The partial local CAS class/style bundle remains private. Obtain Elsevier's
complete CAS template from the [official CTAN package](https://ctan.org/pkg/els-cas-templates)
and retain its license/distribution metadata before compiling the TeX. The
project's MIT license does not replace the template's license. A separately
installed template may produce layout differences from the supplied author PDFs.

For licensing and data boundaries, see the
[release policy](../provenance/PUBLIC_RELEASE_POLICY.md). The separate V1.0.0 data
release is available at [Zenodo](https://doi.org/10.5281/zenodo.24223182) under
CC BY-NC-SA 4.0; software remains MIT-licensed. Data archives stay out of Git.
