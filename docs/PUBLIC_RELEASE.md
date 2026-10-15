# Public source release and GitHub preparation

Prepared: 2026-10-16. Target remote from the author repository:
`https://github.com/ottoKae/windowed_CUSUM_balsa.git`.

## Contents and reproducibility

The public source provides the W36+B20 implementation, frozen configuration,
synthetic example, input schemas, selected tests, current aggregate score tables,
benchmark summaries, figure-source provenance and author manuscript TeX/bibliography.
Read `README.md` first and use the final reference-score directory for current
precision. Historical result versions are explicitly retained as such.

Raw imagery, exact geometry, real review rows, private sample databases, local
capsules and run caches are not part of this source snapshot. Real numerical
replay requires the matching separately available inputs. A prepared local capsule
is not a deposited public dataset; no download URL or DOI is invented.

## Manuscript and figure scope

The manuscript text is copied without revision. Some passages and figures still
describe older configurations; including them here does not synchronize them to
the current results or establish that the paper is ready for submission.

Figure code and input provenance are available in `provenance/figures/`. No figure
was regenerated for this release. Figure PDF/PPTX binaries and third-party image
panels are omitted. The source-only manuscript therefore cannot produce the
complete paper without authorized assets.

The partial local CAS template bundle is also omitted. Obtain Elsevier's CAS
templates from the [official CTAN package](https://ctan.org/pkg/els-cas-templates)
and retain its license/distribution metadata. The local typesetting environment
is not an exact-layout reproduction claim, and the project MIT license does not
replace the CAS license.

## Verify a public checkout

```sh
python tooling/release/export_public.py --verify .
python -m pip install ".[io,validation,replay]" -c configs/requirements-reproduce.txt
python -m pip install pytest geopandas -c configs/requirements-reproduce.txt
python -c "from pathlib import Path; Path('.test-tmp').mkdir(exist_ok=True)"
python -I -m pytest tests --import-mode=importlib -q -p no:cacheprovider --basetemp .test-tmp/public-check
python -I -m lr_cusum paper data/demo/monthly_s1_synthetic.csv .test-tmp/demo_events.csv --config configs/standard_w36.json
```

`.github/workflows/ci.yml` runs these checks with read-only repository permissions
on Python 3.12. It uses the official [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) actions. Local checks
do not imply that GitHub-hosted CI has already run; that occurs after pushing.

## Preparing a push

The author working tree contains private data and ongoing edits. Assemble a fresh
allowlisted source export, then initialize a separate Git repository with a new
publication branch and a clean root commit. This avoids copying the author
repository's old objects into the prepared release. Verify the exported manifest
again after Git staging and a
fresh checkout. Inspect the commit's file list before pushing. The author tree's
index, active branch and uncommitted manuscript edits must remain unchanged.

Preparation does not push, merge, tag, create a release or change repository
visibility. A new root publication branch is independent of the old branches;
it does not erase or sanitize branches already present on the remote. If an
existing private repository is made public, those other branches and their
history are a separate consideration. Do not force-push or automatically merge
unrelated histories as part of this preparation.

The local preparation receipt records the actual commit, branch, remote lookup
status, file inventory and checks. Authenticate GitHub and refresh the remote refs
before the eventual ordinary push of the prepared branch.
