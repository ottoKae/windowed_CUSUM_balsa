# Reproducibility guide

The release separates three reproducibility levels so that public claims do not
depend on unavailable private data.

## Level 1: algorithm and synthetic example

This level is fully public.

```bash
python -m pip install -e ".[io,validation,replay]"
windowed-cusum paper data/demo/monthly_s1_synthetic.csv demo_events.csv \
  --config configs/paper_v1.json
python provenance/audit_public_package.py
```

The synthetic files contain no Ecuadorian coordinates and do not reproduce the
paper's numerical results. They verify calendar handling, change-boundary
semantics, and N1 candidate admission.

## Level 2: manuscript tables and frozen figure assets

Compact non-identifying source tables are stored under `results/`. Verify their
counts and interpretation against `results/count_ledger.json` and the hashes in
`provenance/`. These tables support numerical auditing, but they are not a
complete public source package for every manuscript figure.

The production source registry for the final manuscript is
`manuscript_JAG_v2/figure_sources/FIGURE_BUILDERS.csv`. All known Python
builders and inherited Python modules are stored in this repository. Public
aggregate figure inputs are under `data/public/figure_inputs/`; inputs that
contain plantation geometry, event-level time series, or licensed imagery stay
under the ignored `data/local/paper_v1/` author bundle.

Consequently, the public checkout can audit every frozen figure and rebuild the
figures driven only by public aggregates. An authorized author checkout can
also run the restricted-input builders. The frozen manuscript PDFs remain the
publication assets until a rebuild passes the exact-layout comparison described
in `docs/20260906_MANUSCRIPT_FIGURE_REPRODUCIBILITY_AUDIT.md`.

Build the LaTeX manuscript on Windows with:

```powershell
cd manuscript_JAG_v2
.\build_latex.ps1 -Clean
```

The active paper uses only the files listed in the final-manuscript figure registry.
The manuscript build consumes the eight released PDFs and does not regenerate
Figures 1, 2, or 4--8. Generated LaTeX auxiliaries and historical figure
versions are not release sources.

## Level 3: complete raw-to-paper replay

This level requires authorized external inputs that are not included:

- the official Ecuador plantation inventory and redistribution permission;
- exact dated reference-event records and their frozen spatial roles;
- orbit-specific S1-GRiTS monthly Zarr stores;
- GRiTS-Resolver for raw Zarr/catalog access;
- any licensed optical imagery used for interpretation; and
- suitable administrative boundaries and an attributed basemap.

An authorized replay must first produce one complete monthly row per trajectory,
retain missing values as NaN, and freeze input hashes before detection. It then
applies `configs/paper_v1.json`, performs same-band consolidation, creates the VV
chronology and VH-supported subset, evaluates with locked spatial blocks, and
only then regenerates aggregate tables and figures.

## Determinism and audit records

- Candidate seeds are derived from the stable trajectory-and-band identifier.
- Near-threshold N1 refinement reuses that deterministic seed.
- Spatial bootstrap seeds and replicate counts are frozen in the protocol.
- Input tables, configuration, code revision, and output tables require SHA-256
  entries in a run manifest.
- A changed source hash defines a new run; do not overwrite a frozen release.
