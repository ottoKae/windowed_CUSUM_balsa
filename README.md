# Windowed CUSUM

Windowed CUSUM detects recurrent negative shifts in stand-level Sentinel-1
backscatter using overlapping local temporal windows. This repository provides
the detector, controlled comparisons, and real-data reproducibility packages
for Ecuadorian balsa plantations. Detections indicate **stand disturbances**, not
independently verified harvesting causes or biological rotation ages.

**Release status:** preparation for version 1.0.0. Public repository and archival
DOI links will be added after deposition; local packages are not yet published.

## Algorithm

1. Aggregate polygon-interior observations into monthly, track-specific medians.
2. Compute CUSUM candidates within overlapping local windows.
3. Admit candidates inside the central decision interval and retain negative shifts.
4. Apply the finite-sample-corrected permutation screen (N1).
5. Consolidate overlapping-window and same-band track detections, then annotate
   the VV chronology with one-to-one VH support.

The frozen national configuration uses a **48-month window**, **12-month step**,
central **24-month decision interval**, and `p_perm <= 0.025`. Reference matching
and VV--VH support use a tolerance of **two months**. Missing observations remain
missing; they are not replaced with zero. See [configuration](configs/paper_v1.json)
and [exact index/calendar semantics](docs/METHOD_SEMANTICS.md).

Fixed-configuration ablations and spatially selected benchmark configurations
are different experiments. Do not substitute one for the other; see
[result boundaries](docs/RELEASE_SOURCE_AND_RESULT_BOUNDARIES.md).

## Data sources and reproducibility packages

The study combines Sentinel-1 VV/VH observations organized through S1-GRiTS,
the official Ecuadorian BALSA plantation inventory, and visually dated historical
disturbance references. The full official inventory and the validated analysis
cohort are different populations. [Source inventory](provenance/DATA_SOURCES.md)
records their provenance.

| Package | Contents | Purpose |
|---|---|---|
| `science.zip` (B) | 3,498 opaque polygon identities; 4,001 polygon-track stand series; 1,093 reference events in 462 polygons; 57 evaluation blocks and five folds; event ledgers and bootstrap evidence | Stand-level detection and evaluation |
| `pixels.zip` (C) | 176,907 unique pixel-track series; 176,935 polygon-pixel memberships; local pixel adjacency | Pixel-first detection and comparison with stand medians |

Inputs cover **January 2016--June 2026**; national events are reported for
**2017--2025**. Pixel arrays have shape `[pixel, 2, 126]`, ordered VV then VH in dB.
NaNs, validity masks, observation counts and source-month flags are preserved.

Both packages omit exact geometry, absolute coordinates, original catalog/Zarr,
static rasters, source imagery, administrative vectors and basemap caches (D).
Opaque IDs and local adjacency reduce direct geolocation disclosure but do not
guarantee anonymity. Public packages reproduce signal-to-event calculations and
statistics, **not original satellite extraction or all georeferenced maps**.

See [extraction and replay instructions](docs/REAL_CAPSULES.md).
Data archives are distributed separately from Git history.

## Installation and quick start

Use Python **3.12** for the tested workflow (package minimum: 3.10).
From a source checkout, create a fresh environment and activate it:

```text
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`.
Linux/macOS: `source .venv/bin/activate`.

```text
python -m pip install -e ".[io,validation,replay]" -c configs/requirements-replay-20260905.txt
windowed-cusum paper data/demo/monthly_s1_synthetic.csv demo_events.csv --config configs/paper_v1.json
python provenance/audit_public_package.py
```

The example needs no private data or Earth Engine account. Its output contains
accepted **window candidates**, not a fully consolidated national event table.
Input columns are `ID`, `time`, and `VV_dB` and/or `VH_dB`. Keep a row for every
calendar month and use NaN for unavailable observations.

## Reproduce the real-data experiments

Extract the archives so `data/capsules/science/` and `data/capsules/pixels/`
contain their manifests. Recipients do **not** need to run the author's exporter.

```text
python scripts/replay_capsule.py verify --capsules data/capsules
python scripts/replay_capsule.py parity --capsules data/capsules
python scripts/replay_capsule.py detect --capsules data/capsules --output runs/smoke --limit 4
python scripts/replay_capsule.py evaluate --capsules data/capsules --output runs/evaluation
```

`--limit 4` is a small integration test. Use `--limit 0` for all polygon tracks.
`evaluate` replays frozen predictions; fresh F0--F4 signal-to-event recomputation
requires `detect --components` followed by `evaluate --candidate-source`.
[Full commands](docs/REAL_CAPSULES.md) also cover Pixel-first, seven stand
representations, CuSum NRT and cached official CCDC responses.

Cached CCDC replay is offline; it does **not** execute Earth Engine again.
A new official cloud run requires your own account and explicit opt-in.
See [NRT/CCDC reproduction](docs/NRT_CCDC_REPRODUCTION.md).

## Outputs and verification

The frozen national VV chronology contains **6,658 events**, **3,509 adjacent
intervals**, and an overall interval median of **36 months**. These describe
detected disturbance recurrence, not independently verified harvest rotations.
Legacy L0 and current N1 sensitivity results must remain separately labelled.

[The B/C validation report](docs/20260905_BC_REAL_REPRODUCIBILITY_RELEASE_REPORT.md)
documents the full replay checks. Archives include per-file hashes and SHA-256
sidecars. [Release checklist](docs/RELEASE_CAPSULE_CHECKLIST_20260905.md) distinguishes
technical verification from public deposition.

## Repository structure

| Path | Purpose |
|---|---|
| `src/lr_cusum/` | Detection, aggregation, fusion, evaluation and adapters |
| `configs/` | Frozen protocols and environment constraints |
| `data/demo/`, `data/schemas/` | Synthetic example and input contracts |
| `scripts/replay_capsule.py` | Public real-data replay entry point |
| `scripts/figures/` | Figure builders; some need excluded spatial/imagery inputs |
| `results/` | Small aggregate tables, not raw data |
| `provenance/audit_public_package.py` | Release-integrity, privacy, and manuscript consistency audit |
| `docs/`, `provenance/` | Instructions, source/rights boundaries and verification |

The author's full raw-to-map workflow is documented separately in
[Full reproduction](docs/FULL_REPRODUCTION.md). Manuscripts and third-party
figure assets in a local working copy are not automatically part of the public
software release. Installing geospatial dependencies alone does not supply D.

## Citation and licenses

Use [CITATION.cff](CITATION.cff) for software attribution; archival identifiers
will be supplied only after they exist. The accompanying data have their own
[citation metadata](data/CITATION.cff).

Original software is licensed under [MIT](LICENSE). Owner-authorized processed
B/C data and study-generated aggregate tables are licensed under
[CC BY 4.0](DATA_LICENSE). Third-party code retains its original notices, including
[CuSum NRT](docs/third_party/CuSum_NRT_LICENSE.txt). Neither license grants rights
to excluded raw data, basemaps, commercial imagery or third-party manuscripts.
