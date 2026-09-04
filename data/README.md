# Data package

This repository intentionally does **not** redistribute the source geospatial
data used in the Ecuador analysis. The public GitHub package contains only:

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

An authorized user must construct a long table satisfying
`schemas/monthly_s1.schema.json`, preserve missing months as null/NaN, and keep
one stable object identifier across the monthly sequence. The frozen study used
2016-01 through 2026-06 as temporal context and admitted reported events only
within 2017-01 through 2025-12. These dates describe the published experiment;
they are not hard-coded requirements of the core detector.

