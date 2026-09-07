# Data sources and redistribution status

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
| Ecuador MAG plantation inventory | Plantation and BALSA polygon inventory | No | Exact geometries and source identifiers withheld because redistribution terms were not established in the project record. Link to the authoritative portal and cite the provider. |
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
