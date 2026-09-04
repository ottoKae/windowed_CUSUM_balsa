# Public release scope

The public repository was assembled from an internal research workspace through
an explicit allow-list. This avoids publishing machine paths, private review
records, licensed imagery, and many historical script versions.

## Included code

- core CUSUM statistic and calendar-window generator;
- archived legacy detector, visibly separated from the paper protocol;
- frozen N1 detector and finite-sample permutation null;
- polygon-first and pixel-first representation utilities;
- event consolidation, one-to-one matching, spatial-fold, and block-bootstrap
  utilities;
- component, representation, polarization, and interval-analysis utilities;
- optional S1-GRiTS and reference-table adapters; and
- focused unit tests for these modules.

## Excluded code and artifacts

- notebook-derived `src/workflows` code with workstation-specific paths;
- first-establishment analysis, which is outside this manuscript;
- superseded event-calibration implementations;
- blinded/manual review cards and private reviewer workbooks;
- versioned plotting wrappers whose only purpose was iterative layout;
- raw or intermediate candidate ledgers and bootstrap replicate tables;
- cache directories, basemap tiles, raw Zarr stores, shapefiles, and commercial
  imagery; and
- obsolete manuscript snapshots and compiled auxiliary files.

## Workflow figure provenance

The development workflow figure originated in the evidence-building script's
`save_workflow_figure` routine. The public release extracts that routine into
`scripts/figures/build_workflow.py` so it no longer imports unrelated tables,
private paths, or national-product code. Its labels were updated to state the
final N1 Windowed CUSUM semantics.

## Release gates

Before the repository becomes public, confirm that:

1. an author-approved `LICENSE` replaces `LICENSE_PENDING.md`;
2. a permanent repository URL, archival DOI, and `CITATION.cff` are supplied;
3. every public data and figure asset has a documented redistribution basis;
4. no absolute paths, secrets, or non-English working comments remain;
5. all tests pass in a clean environment; and
6. aggregate counts, manuscript text, and the frozen configuration agree.
