# Public-release policy

## Source-only manuscript support update, 2026-10-15

The current explicit public candidate includes the author's unchanged TeX and
bibliography, together with selected figure code
and non-identifying aggregate inputs. This expands the earlier software-only
selection; it does not export licensed image panels, manuscript PDF/figure
binaries, exact geometry or private interpretation records. The author workspace
retains all assets. The source-only export cannot typeset a complete manuscript
until authorized assets are supplied, and the prose has not been synchronized to
the final W36 scoring version. Actual selection is recorded in
`configs/public_export.json` and the exported `PUBLIC_FILES.json`.

## Owner-approved scope, 2026-09-05

The owner has approved production of the minimal real scientific and pixel-level
extension packages. On 2026-09-05 the owner also approved the proposed MIT code
and CC BY 4.0 processed-data licenses and requested publication after validation.
Packages remain in ignored dist/ until actual deposition. This supersedes earlier pending-owner-approval statements
below for B/C only, not for exact geometry, original catalog/Zarr or imagery.
Opaque group identities and local pixel adjacency are retained for scientific
reproduction; the private lookup and absolute locations are not exported.
See docs/REAL_CAPSULES.md. Authorization to build is not an external upload.

## Included in the GitHub release

- source code and tests selected by the repository maintainers;
- synthetic demonstration data;
- input/output schemas;
- aggregate validation, ablation, configuration, polarization, and national
  distribution tables;
- a count ledger and cryptographic provenance manifests;
- publication figures whose third-party imagery has been separately cleared.

## Excluded from the GitHub release

- raw Sentinel-1 scenes and S1-GRiTS Zarr stores;
- polygon-interior pixel caches and large intermediate tables;
- MAG/GADM shapefiles or GeoPackages;
- exact coordinates, geometries, administrative labels, source rows, original
  polygon identifiers, historical identifiers, spatial-block identifiers,
  reviewer names, evidence filenames, and local filesystem paths;
- Planet source imagery and Esri tile caches;
- private review workbooks and manually annotated evidence cards.

## Public identifier rule

If a real row-level research table is released later, identifiers must be
generated once from a private mapping and reused consistently across all
released tables (for example `PUB_P000001`). The private-to-public mapping must
remain outside the public repository. Hashing source IDs without a secret key is
not sufficient anonymization because source IDs may be enumerable.

No real row-level table is included in this GitHub package. The `DEMO_STAND_*`
identifiers identify invented sequences only.

## Separate archival release

A larger reproducibility capsule may be deposited separately only after the
authors confirm that its underlying inventory and reference records may be
redistributed. A suitable capsule could contain de-identified polygon-level
monthly series, a de-identified event ledger, and spatial-bootstrap grouping at
a deliberately coarsened scale. It must be reviewed for linkage risk before
deposit and should not be silently added to Git history.

## Licensing

Original software uses MIT (LICENSE); approved processed B/C data and study
aggregates use CC BY 4.0 (DATA_LICENSE). Retain third-party code notices.
Neither license implies rights over source imagery, basemaps or papers.
The first software-only distribution excluded both manuscript trees and their
rendered assets. The October 15 source-only selection above supersedes that file
selection, while retaining the restrictions on rendered third-party assets.

On 2026-10-16 the partial local CAS class/style bundle was removed from the
public selection. Obtain the official complete bundle separately as described
in `docs/PUBLIC_RELEASE.md`; the local author files remain unchanged.
