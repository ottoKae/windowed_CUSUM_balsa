# Public-release policy

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

The source repository had no approved project-wide software license at the time
of this packaging pass. The authors must select a code license and must not use
that license to imply rights over third-party data, imagery, or basemaps. Data
and figure rights should be declared separately.

