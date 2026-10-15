# Data sources and redistribution status

## Find data and interpretation records (2026-10-15)

Start with [DATA_CATALOG.csv](DATA_CATALOG.csv). Each row identifies a source
family, its scientific role, version, local retrieval entry and public metadata.
Paths are relative to the repository. An ignored path identifies a local input;
it does not promise that the input is included in a downloaded public export.

The current radar input is `data/local/standard_w36_v1/manifest.json`, with
`units.parquet`, `membership.parquet` and `series.npz`. The historical
`data/local/paper_v1/` contract remains separate. Original image/Zarr archives are
resolved by existing manifests and evidence records, not copied into this repo.

All identified visual-interpretation workflows are indexed in
`data/private/sample_collection_20261015_v1/`. See the
[sample schema](../data/schemas/sample_collection.md) and
[public cohort summary](../data/public/sample_collection_summary.json).
The collection retains the original sources, revisions, review notes, event
identities, report links, sampling roles and probability weights. It combines
retrieval, not scientific denominators: a repeated reading or relationship check
does not create another independent event or another probability sample.

Current probability-score aggregates are in
`results/final_reference_score_20261015/`. Earlier `results/w36_standard/precision.csv`
and the old manuscript results remain versioned historical records. TeX assets,
their actual builders and their data requirements are indexed in
[the figure catalogue](figures/README.md).

The table below describes provider roles and the existing project release
boundaries. It is not a new legal review of provider terms. Reuse existing local
data first; no imagery download or new reference annotation was performed during
this cataloguing task.

Owner authorization update (2026-09-05): B/C real time-series capsules, dated
events and opaque evaluation groups are now approved for preparation. Exact
geometry and the full spatial/raw layer remain withheld by the owner's explicit
scope decision, not because the owner has failed to respond. Original imagery
and third-party layers remain separate. See docs/REAL_CAPSULES.md.

This register separates scientific provenance from redistribution permission.
Public availability of a source does not automatically grant permission to
republish a local copy.

| Source | Use in study | Included here | Release status / action |
|---|---|---:|---|
| Ecuador MAG plantation inventory | Plantation and BALSA polygon inventory | No | Exact geometries and source identifiers are withheld. The private source registry retains the original acquisition record; no public download endpoint or redistribution permission is asserted here. |
| S1-GRiTS monthly Sentinel-1 stores | 2016-01--2026-06 monthly VV/VH context | No | Multi-terabyte derived archive; regenerate/acquire through S1-GRiTS under its own terms. |
| Sentinel-1 mission data | Radar observations underlying S1-GRiTS | No | Cite Copernicus/ESA and the processing source; do not mirror the source archive in GitHub. |
| Historical disturbance reference material | Event-month interpretation and validation | No | Exact locations, source rows, historical IDs, personnel/evidence filenames, and review workbooks withheld. |
| Planet imagery | Visual interpretation and illustrative evidence | No raw imagery | Do not redistribute raw/original imagery. Any manuscript screenshot must be checked against the account's current Education & Research sharing terms and carry required attribution. |
| GADM administrative boundaries | Private analysis/cartographic context | No | Local license restricts redistribution/use; acquire independently from the provider or replace with a compatible open boundary. |
| Esri World Hillshade tile cache | Private cartographic background | No | Cached tiles are not redistributed. Fetch at runtime under applicable terms or replace with an openly redistributable relief layer. |
| Natural Earth | Small-scale cartographic context where applicable | No source copy needed | Public-domain source; obtain from the official Natural Earth distribution and retain attribution/provenance metadata. |

## Included derived material

The `results/` directory contains only aggregate metrics and distributions.
No included table allows a reader to recover an official source row, exact
plantation geometry, exact coordinate, administrative assignment, spatial-fold
assignment, reviewer identity, or evidence filename.

The real row-level input and event ledgers are intentionally withheld pending a
separate data-release decision. If released later, they should receive their own
repository record, data license, version, DOI, checksums, and disclosure review.
