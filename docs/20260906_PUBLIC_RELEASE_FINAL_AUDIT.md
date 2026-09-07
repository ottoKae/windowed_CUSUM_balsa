# Public release audit (2026-09-06)

## Decision

The repository is technically suitable as a public companion containing the
algorithm, public aggregate results, the final JAG manuscript, frozen figure
assets, and figure-building code. Author metadata, the GitHub location, figure
permissions, and the editable workflow source are confirmed. An archival DOI
remains unassigned until a versioned release is deposited.

## Completed cleanup

- Removed the pytest suite, pytest workflow, and pytest configuration from the
  final companion repository.
- Added a dependency-free Python source parser and retained the public-package
  audit as the release smoke checks.
- Localized the known production plotting modules and registered one
  authoritative source route for each of the eight final manuscript figures.
- Localized public aggregate figure inputs under `data/public/figure_inputs/`.
- Kept restricted geometry, event-level Sentinel-1 series, licensed imagery,
  basemap caches, and detailed acquisition receipts outside the public file
  set through explicit ignore rules.
- Removed machine-specific absolute paths from public scripts and documents.

## Verification result

- Python syntax audit: 108 files parsed successfully.
- Public-package technical audit: passed.
- Numerical ledgers: 3,498 polygons, 6,658 primary VV events, 3,509 adjacent
  intervals, 4,061 VH-supported events, and formal confusion totals all agree
  with their frozen aggregate sources.
- Manuscript structure: eight referenced figures and eight registered canonical
  builders are present; the current manuscript compiles successfully.
- Secrets, forbidden binary data, private absolute paths, CSV parse errors, and
  sensitive tabular headers: none in the public file set.

## Figure reproducibility boundary

All known Python source code is local. Exact public reconstruction of every
figure is nevertheless neither possible nor appropriate because some figures
depend on restricted plantation geometry, event-level Sentinel-1 data,
PlanetScope imagery, or Esri basemap tiles. These inputs belong in the ignored
author-side bundle. The manuscript therefore ships frozen figure PDFs plus a
source registry. Figure 8 has a pixel-verified local rebuild; Figures 4 and 6
have functional production wrappers but retain an approximately 1-pt crop
difference from their frozen PDFs. See
`20260906_MANUSCRIPT_FIGURE_REPRODUCIBILITY_AUDIT.md`.

## PowerPoint source policy

The editable source for the manually authored workflow figure belongs at:

`manuscript_JAG_v2/figure_sources/powerpoint/windowed-CUSUM-workflow-figures-01.pptx`

Commit it only when every embedded element is author-owned or licensed for
redistribution and document metadata has been reviewed. A small author-owned
PPTX can be stored directly in Git. A large file should use Git LFS or an
archival data repository. If it contains restricted third-party material, keep
the PPTX private and publish only the frozen PDF and provenance note.

## Remaining publication gates

1. Create a versioned GitHub release and archive it in Zenodo or an equivalent
   repository.
2. Add the assigned software DOI and, if deposited separately, the data DOI to
   the citation metadata and manuscript availability statements.

Until DOI deposition is complete, the codebase is a technically clean public
companion, but not yet a finalized DOI-bearing archival release.
