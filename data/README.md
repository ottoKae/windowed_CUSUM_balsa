# Data package

## Current entry points (2026-10-15)

- [Data catalogue](../provenance/DATA_CATALOG.csv): source families, roles, access
  boundaries and relative retrieval entries.
- [Unified visual sample collection](schemas/sample_collection.md): how existing
  observations, event identities, report links and sampling roles fit together.
  The actual searchable collection is local and ignored at
  `private/sample_collection_20261015_v1/`; it preserves original source hashes
  and does not replace the original review workbooks or GUI answers.
- Current W36 input: `local/standard_w36_v1/` (`units.parquet`,
  `membership.parquet`, `series.npz`, `manifest.json`).
- Current public score aggregates:
  [final_reference_score_20261015](../results/final_reference_score_20261015/README.md).

The older contracts below remain available for historical reproduction. Their
reference versions and workflow counts must not be substituted for the current
W36 pixel+B20 sample frame. The unified collection keeps the original catalogue,
307 formal reports, repeat/relationship checks and incomplete history domains as
different cohorts, not as one interchangeable set of independent samples.

## FC001 2025 BALSA sample management

The historical author-local `data/SAMPLE_MANAGEMENT_PLAN.md` describes
the existing official polygon IDs, historical reference versions, 2026 reviews,
29 proposed additions, and pixelwise review cases. The executed shared local root
is `data/balsa_FC001_dated_event_subset/`, used by both the balsa and pixelwise
projects. The current read-only snapshot is selected by its `CURRENT.json` file;
editable-ledger migration has not yet been switched on.

Update, 2026-09-05: the owner approved B (minimal real science) and C (pixel
extension) capsules. See [the capsule guide](../docs/REAL_CAPSULES.md).
These are separate local ZIP release candidates, not new Git-tracked raw data.
The exact-location/raw-data layer remains outside the approved release scope.

This repository intentionally does **not** track the source geospatial data
used in the Ecuador analysis. The public GitHub package contains only:

- a fully synthetic, non-geographic monthly Sentinel-1 example in `demo/`;
- machine-readable schemas in `schemas/`;
- aggregate, non-identifying results under `../results/`.

## Why the real records are not here

The private analysis uses official plantation polygons, exact reference-event
locations, S1-GRiTS Zarr stores, polygon-interior extraction caches, imagery
used during interpretation, and map-tile caches. Some of these are very large;
others contain exact locations, source-system identifiers, reviewer evidence,
or material whose redistribution terms have not been confirmed. They are not
required to inspect the algorithm or reproduce the included synthetic example.

In particular, this repository does not include MAG plantation geometries,
GADM vectors, Planet imagery, Esri basemap tiles, raw Sentinel-1 products,
S1-GRiTS Zarr stores, or private event-review workbooks.

See `../provenance/DATA_SOURCES.md` and
`../provenance/PUBLIC_RELEASE_POLICY.md` for access and release boundaries.

## Synthetic demonstration

`demo/monthly_s1_synthetic.csv` follows the core long-table API: one row per
stand-month, with `ID`, `time`, `VV_dB`, and `VH_dB`. Blank backscatter values
are genuine missing values; they must remain missing and must never be replaced
with physical zero. `demo/reference_events_synthetic.csv` supplies invented
event months for demonstration only. These rows are not derived from Ecuadorian
plantations and must not be used to reproduce paper statistics.

Regenerate the two files with:

```bash
python data/demo/generate_synthetic_demo.py
```

## Real-data reproduction contract

This section describes the historical `paper_v1` contract. For the current W36
pixel input use the capsule contract in `schemas/w36_capsules.md` and the entry
point above. Neither contract includes a new download or permission to publish
the original spatial records.

An authorized user must construct a long table satisfying
`schemas/monthly_s1.schema.json`, preserve missing months as null/NaN, and keep
one stable object identifier across the monthly sequence. The frozen study used
2016-01 through 2026-06 as temporal context and admitted reported events only
within 2017-01 through 2025-12. These dates describe the published experiment;
they are not hard-coded requirements of the core detector.

On an authorized workstation, `scripts/reproduce.py prepare` assembles the
ignored `data/local/paper_v1` package. Its canonical files are:

| Path | Unit |
|---|---|
| `inventory/balsa_validated.parquet` | one row and geometry per validated polygon |
| `splits/polygon_roles.parquet` | one immutable spatial role per polygon |
| `splits/spatial_blocks.parquet` | spatial-block geometry |
| `polygon_pixels/pixel_associations.parquet` | polygon-to-native-pixel membership |
| `polygon_pixels/cache/*.npz` | VV, VH, validity, n_obs, and calendar by native pixel-track |
| `references/reference_ground_events.parquet` | one row per dated ground event |
| `references/reference_track_events.parquet` | reference events expanded to observed tracks |
| `s1grits/catalog.parquet` | orbit-aware source catalog |
| `frozen_results/` | candidates, events, F0--F4, spatial bootstrap, and national summaries |
| `admin/`, `basemap/` | local map context; Esri cache must not be redistributed |

`MANIFEST.json` hashes every local file. `AUDIT.json` verifies the
study-defining counts. The Zarr stores remain in their original archive and are
resolved using `s1grits/SOURCE.local.json`; copying them would add terabytes
without improving reproducibility.
