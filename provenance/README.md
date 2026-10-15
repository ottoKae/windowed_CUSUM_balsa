# Provenance

## Current navigation

- `DATA_CATALOG.csv`: searchable source-family registry and retrieval entries.
- `DATA_SOURCES.md`: current data, sample, provider and access explanation.
- `figures/README.md`: current TeX inclusion order, asset hashes, producer scripts
  and verification limits. It supersedes old printed-order builder lists.
- `../data/schemas/sample_collection.md`: unified local interpretation collection.
- `../results/final_reference_score_20261015/SOURCES.json`: final aggregate sources.

Legacy release manifests below describe the files of their original release.
For a newly generated public export, use its own `PUBLIC_FILES.json` inventory.

- `DATA_SOURCES.md` records source roles and redistribution boundaries.
- `PUBLIC_RELEASE_POLICY.md` defines what belongs in GitHub versus a controlled
  or separately archived data release.
- `ID_AND_PRIVACY_POLICY.md` defines fields and linkage routes excluded from
  public row-level data.
- `SOURCE_FILES.json` records the hashes of authoritative private inputs without
  exposing workstation paths.
- `RELEASE_MANIFEST.json` and `CHECKSUMS.sha256` are generated after packaging
  and cover every file under `data/`, `results/`, and `provenance/`, excluding
  the self-referential checksum and manifest files themselves.
- `PRIVACY_AUDIT.json` records the automated release checks.

The manifests establish byte-level provenance; they do not confer a license for
third-party datasets.

