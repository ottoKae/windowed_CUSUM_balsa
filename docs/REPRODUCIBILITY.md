# Reproducibility guide

The release separates three reproducibility levels so that public claims do not
depend on unavailable private data.

## Level 1: algorithm and synthetic example

This level is fully public.

```bash
python -m pip install -e ".[test]"
python -m pytest
windowed-cusum paper data/demo/monthly_s1_synthetic.csv demo_events.csv \
  --config configs/paper_v1.json
```

The synthetic files contain no Ecuadorian coordinates and do not reproduce the
paper's numerical results. They verify calendar handling, change-boundary
semantics, and N1 candidate admission.

## Level 2: manuscript tables and frozen figure assets

Compact non-identifying source tables are stored under `results/`. Verify their
counts and interpretation against `results/count_ledger.json` and the hashes in
`provenance/`. These tables support numerical auditing, but they are not a
complete public source package for every manuscript figure.

Figure 3 is the only manuscript figure that can be rebuilt entirely from this
public checkout. Build the standalone method workflow with:

```bash
python scripts/figures/build_workflow.py manuscript/figures/workflow_rebuilt.pdf
```

Figures 1, 2, and 4--8 are distributed as frozen, checksum-verified publication
PDFs. Rebuilding those seven figures requires authorized plantation geometry,
reference events, S1-GRiTS products, administrative or basemap layers, or
licensed interpretation imagery that cannot be redistributed here. The private
authoring-script paths recorded in `manuscript/figures/FIGURE_MANIFEST.md` are
provenance notes only: those paths are not expected to exist in this public
repository. Verify the released assets with
`manuscript/figures/SHA256SUMS.txt` instead of treating the private paths as
public build commands.

Build the LaTeX manuscript on Windows with:

```powershell
cd manuscript
.\build_latex.ps1 -Clean
```

The active paper uses only the files listed in the manuscript figure manifest.
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
