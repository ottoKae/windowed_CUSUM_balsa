# Windowed CUSUM: W36 + B20

Monthly Sentinel-1 disturbance detection and stand-level harvest-report evaluation
for Ecuadorian balsa plantations. The frozen detector is **W36 / D12 / S12, VV,
B20**. The final reference-score release is **2026-10-15**, with **+/-3 months**
as the primary display and +/-1 and +/-2 months reported alongside it.

The primary display tolerance was selected after inspection of earlier results.
Monthly output resolution does not imply one-month dating accuracy. The original
detector configuration and older +/-2-month evaluation capsule remain intact.

## Find the right material

| Need | Entry point |
|---|---|
| Layout and version boundaries | [Repository guide](docs/REPOSITORY_GUIDE.md) |
| Data sources and retrieval | [Data catalogue](provenance/DATA_CATALOG.csv), [source guide](provenance/DATA_SOURCES.md) |
| Unified visual interpretation records | [Sample schema](data/schemas/sample_collection.md), [data guide](data/README.md) |
| Current precision and evaluation definition | [Final aggregates](results/final_reference_score_20261015/README.md) |
| Benchmarks, windows, components and polarization | [W36 result index](results/w36_standard/result_index.csv) |
| TeX figures and their actual source scripts | [Figure provenance](provenance/figures/README.md) |
| Author manuscript | [Main TeX](manuscript_JAG_v2/main.tex) |
| Public export and maintenance | [Tooling guide](tooling/README.md) |
| GitHub preparation and public checks | [Public release guide](docs/PUBLIC_RELEASE.md) |
| Software citation | [CITATION.cff](CITATION.cff) |

## Run the synthetic example

Use Python 3.12 in a fresh environment from this directory:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install ".[io,validation,replay]" -c configs/requirements-reproduce.txt
windowed-cusum paper data/demo/monthly_s1_synthetic.csv demo_events.csv --config configs/standard_w36.json
```

This example needs no imagery or account. It exercises the detector; it does not
reproduce Ecuadorian sample scores. See [tests](tests/README.md) for the checks
shipped with the public source export.

In a public checkout, first verify the distributed bytes without private inputs:

```sh
python tooling/release/export_public.py --verify .
```

For a separately supplied real pixel-input capsule:

```sh
python scripts/reproduce.py detect --data data/capsules/w36_pixels --output runs/reproduced_w36 --workers 6
```

The capsule must contain `units.parquet`, `membership.parquet`, `series.npz` and
`manifest.json`. Complete stand membership is required for B20. No public download
URL or deposited dataset is implied. The author-local input,
`data/local/standard_w36_v1`, is restricted and excluded from Git.

`scripts/evaluate_standard.py` reproduces the older frozen evaluation capsule
when available. It does not silently substitute the October 15 reference
revisions. Use the final aggregate release for current numbers.

## Frozen method and current evidence

The context window contains 12 months before a central 12-month decision interval
and 12 months after it, advancing by 12 months. B20 selects three consecutive
calendar months, including across year boundaries. At least 20% of each stand's
fixed master pixels must respond. Each source response is assigned once; the
response-node median gives the representative month. Reference dates do not select
windows or representative months.

The frozen detector produced **44,386 pixel responses and 1,266 stand reports**.
The final evidence retains the original 307 probability samples and weights,
including unresolved observations.

| Month tolerance | T / F / U | Weighted precision scenario endpoints | Catalogue recall |
|---|---|---|---|
| +/-1 | 126 / 167 / 14 | 43.47%--46.19% | 833 / 1,049 (79.41%) |
| +/-2 | 176 / 117 / 14 | 60.74%--63.47% | 934 / 1,049 (89.04%) |
| **+/-3 (primary display)** | **202 / 91 / 14** | **69.62%--72.34%** | **955 / 1,049 (91.04%)** |

Scenario endpoints are not confidence limits. The approximate 95% outer interval
at +/-3 months is 65.17%--76.88%; the final tables contain all tolerances. The
optimistic endpoint does not establish that all unknown reports could
simultaneously receive unique event credit.

Probability precision concerns the 1,266-report population. Catalogue recall
concerns the selectively assembled reference events. **Do not combine them into
population F1.** The seven reviewed history domains have not yielded a qualified
complete reference domain. Benchmark catalogue comparisons do not establish their
report-population precision or overall superiority.

## Directory responsibilities

| Directory | Responsibility |
|---|---|
| `src/lr_cusum/` | Installed numerical and evaluation implementation |
| `scripts/` | Replay, evaluation, review and figure entry points |
| `configs/` | Frozen detector settings and dependency constraints |
| `data/demo/`, `data/schemas/` | Public synthetic data and data contracts |
| `data/private/`, `data/local/`, `data/capsules/` | Ignored local samples, inputs and separate capsules |
| `results/final_reference_score_20261015/` | Current non-identifying precision aggregates |
| `results/w36_standard/` | Frozen detector comparisons and historical scoring tables |
| `provenance/` | Source catalogue, rights and figure lineage |
| `manuscript_JAG_v2/` | Current author-controlled TeX and figure assets |
| `tests/` | Reusable tests; disposable output goes under `.test-tmp/` |
| `tooling/` | Cataloguing and release helpers; local diagnostics are ignored |
| `runs/`, `archive/` | Ignored run evidence and historical implementations |

The manuscript and some figures still describe historical configurations. They
are not automatically updated by this cleanup or a reference-score release.
The figure catalogue records that distinction for every included asset.

## Public release

Publish a fresh allowlisted export, not the entire author working tree.
`configs/public_export.json` defines the selection; `PUBLIC_FILES.json` in the
export records hashes. Private review text, exact geometry, source imagery,
temporary artifacts and archived experiments are excluded. Selected historical
figure code is retained to preserve its actual implementation and imports.
Figure assets requiring separate third-party clearance are not automatically
exported; the source-only manuscript is not a complete typesetting bundle.

This work prepares a local release candidate; it does not push to GitHub or
deposit data. Original software uses [MIT](LICENSE); authorized processed research
data and aggregates use [CC BY 4.0](DATA_LICENSE). Neither license grants rights
over source imagery, basemaps, inventory geometry or private review workbooks.
