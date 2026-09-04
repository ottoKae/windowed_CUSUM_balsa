# Provenance

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

