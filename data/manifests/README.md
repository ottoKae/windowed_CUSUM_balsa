# Dataset manifests

`configs/data.example.json` defines the logical inputs. An authorized local
checkout copies it to `configs/data.local.json` and supplies machine paths.
Run:

```bash
python scripts/reproduce.py prepare --config configs/data.local.json
python scripts/reproduce.py audit --config configs/data.local.json
```

The first command creates `data/local/paper_v1/`; that directory is ignored by
Git because it contains exact locations, licensed inputs, and large arrays. It
also writes `MANIFEST.json`, whose relative paths, sizes, row counts, and SHA-256
digests make the local dataset auditable without exposing workstation paths.

The Git-tracked repository contains only the schema, example configuration,
aggregate results, and synthetic data. Publishing the local package is a
separate data-release decision; see `provenance/PUBLIC_RELEASE_POLICY.md`.
