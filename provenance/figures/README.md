# Current manuscript figure provenance

This is the source index for the **eight PDF assets actually included by the current
`manuscript_JAG_v2/main.tex`**. The supplementary TeX currently includes no graphics.
Nothing here renders a figure, changes scientific results, or edits the author's TeX.

## Find a figure and its code

- [figure_bindings.csv](figure_bindings.csv): actual TeX file and line, source encounter
  order, asset SHA-256, producing source route, historical version and rebuild limitations.
- [figure_dependencies.csv](figure_dependencies.csv): one row per builder, helper,
  input or receipt, with relative path, availability and file hash. Directory entries
  indicate a required input group; they are not recursive content checksums.
- [figure_code_inventory.csv](figure_code_inventory.csv): figure entry points, shared
  renderers, previews and additional builders, including code not used by current TeX.
- [public_code_files.json](public_code_files.json): explicit source-to-destination paths and
  hashes for source-only export; includes the archived helpers at their original import locations.
- [figure_provenance.json](figure_provenance.json): the same relationships as a nested
  machine-readable record, with hashes of the inspected TeX files and PDF assets.
- [source_spec.json](source_spec.json): the small maintained source-route specification.
  The catalog builder joins it to actual TeX references; it does not guess that similarly
  named PDFs came from similarly named scripts.

All catalog paths are relative to the repository root. A path beginning `../LR-CuSum_v100/`
is a private sibling-archive dependency, **not a file included in this repository**.
Paths, hashes and availability are recorded; coordinates, geometry, sample observations,
licensed raster content and absolute machine/user paths are not copied into these catalogs.

## Actual TeX order and scientific version

The order below is source encounter order after expanding `\\input`; it is not a claim
about float placement in a compiled PDF. The old `FIGURE_BUILDERS.csv` and `build_all.py`
use a historical numbering order and must not be treated as the current figure numbers.

| TeX order | Asset | Authoritative route or best supported source | Status |
|---:|---|---|---|
| 1 | `fig02_study_area.pdf` | Archived `build_study_area_bursts_v5.py` plus `build_study_area_figure.py` | Current PDF matches both accepted replacement receipt and saved output hashes. Static burst footprint coverage; not historical acquisition availability. |
| 2 | `fig06_representative_timeseries.pdf` | Archived `build_timeseries_signature_polygon_iqr.py` plus `build_timeseries_signature_polygon_median.py` | Historical median/IQR example with PlanetScope panels. Formal source route known; complete frozen input bundle and exact export binding unresolved. |
| 3 | `fig03_workflow.pdf` | Editable `figure_sources/powerpoint/windowed-CUSUM-workflow-figures-01.pptx` | Authoring source exists; exact export receipt unresolved. Python workflow script is an alternative W48 polygon-first diagram. |
| 4 | `fig04_windowed_cusum_concept.pdf` | `preview_concept_label_positions_v3.py` plus `build_windowed_cusum_concept_n1_current.py` | Current PDF matches output manifest and saved output. Historical Polygon-first W48 example, not W36+B20. |
| 5 | `fig05_population_s1_signatures.pdf` | `build_population_s1_signatures.py` to population-signature v11 renderer | Historical signal characterization. Previous local rebuild worked but page crop differed; not exact frozen-PDF reproduction. |
| 6 | `fig07_representation_configuration.pdf` | `build_representation_configuration.py` to metric-encoding v19 and inherited renderers | Historical representation/parameter P/R/F1. Do not use these as current weighted W36 precision. Previous crop mismatch remains documented. |
| 7 | `fig01_national_recurrence.pdf` | `build_national_recurrence.py` to `build_figure6_recurrence_10km.py` | Historical Polygon-first national chronology (3,498 stands, 6,658 events). Historical audit reported identical rendered pixels; current binary hash is a separate check. |
| 8 | `fig08_polarization_intervals.pdf` | Archived `summarize_n1_polarization_interval_sensitivity.py` | Historical VV-primary stand intervals with VH support. Different from current pixel VV/VH+B20 experiment; exact output binding unresolved. |

Two important corrections to the older index are recorded here:

1. The current study-area asset is the **September 11 burst-v5 replacement**, not the
   earlier green/yellow study-area renderer. The replacement receipt hashes establish this.
2. The current CUSUM concept asset is the **v3 label-placement output**. The base workflow
   alone does not identify the accepted rendering revision.

Source hashes bind file identity, not scientific correctness or third-party rights.
`source_route_only` means source identification is supported by the existing source index,
code and historical audit; it does not mean the current PDF has been reproduced.
`output_and_receipt_hash_verified` establishes byte identity with an accepted saved output
and its receipt. No new exact-layout or numerical reproduction is claimed in this task.

## Code organization and public export

- `scripts/figures/`: local entry points, one `build_all.py` orchestrator, and author-side
  `preview_*` tools. Several entry points are explicitly simplified diagnostic renderers.
- `analysis/workflows/`: shared formal renderers and their versioned helper chain. Keep
  this chain intact; identical directory names do not imply identical scientific versions.
- `archive/legacy_20261004/`: immutable historical source for the current study-area,
  representative-series and interval figures, and the v2--v8 sensitivity helpers.
- `data/public/figure_inputs/`: already designated aggregate source tables, retaining
  their original method and evaluation labels.
- `data/local/`, `runs/` and the sibling archive: restricted inputs and historical receipts,
  retained locally and excluded from ordinary public export.

The dependency catalog's `public_export_path` maps the selected archived Python sources
to their pre-archive relative locations for an explicit source-only export. The local archive
is not moved or rewritten. These include `scripts/build_study_area_bursts_v5.py`, its base
renderer, the two representative-series renderers, the interval renderer and sensitivity
helpers v2--v8. Restoring source location does not supply missing/licensed data or establish
an exact rebuild. The repository's public-export manifest is authoritative about which
files actually ship.

The author-side `scripts/figures/preview_reference_v2.py` remains in the inventory
but is excluded from public export. It imports the unshipped `rerun_reference_v2`
review workflow and is not a producer of an asset currently used by TeX. The code
inventory records this exclusion explicitly; the public code mapping contains 46
files rather than all 47 author-side figure scripts.

Actual map renderers also require Cartopy in addition to the plotting and geospatial
dependencies. For an authorized local figure-input bundle, install the required extras
and Cartopy in the same environment:

```bash
python -m pip install ".[io,validation,geo,figures]" cartopy -c configs/requirements-reproduce.txt
```

This installs libraries only. It does not fetch restricted inputs, supply licensed
basemaps, or make every historical figure reproducible from the source-only export.

The restored archived study-area and representative-series code expects older private
input conventions; supply authorized input paths explicitly. Do not run the concept
preview as a generic rebuild: it asserts the previous manuscript asset hash to protect the
author's file, which differs from the currently accepted v3 asset. No preview or production
renderer should overwrite manuscript assets during a source-provenance check.

The representative-series PDF contains PlanetScope-derived panels. Basemap-derived maps,
inventory geometry, editable PowerPoint embedded assets and document metadata also have
their own release conditions. Public figure **code** is separate from permission to publish
the corresponding binary PDF or original imagery. Retaining a local file is not a grant
of redistribution rights.

## Refresh and verify without rendering

From the repository root:

```bash
python tooling/catalogs/build_figure_provenance.py
```

The standard-library-only builder ignores commented-out graphics, follows current TeX
inputs, checks all eight assets, hashes existing dependencies, and asserts that the inspected
TeX/PDF files did not change during indexing. It fails on an unexpected graphic or input
cycle so that a new author figure receives an explicit source definition.

An exported source-only package may omit the private receipts and licensed PDFs. Use its
frozen catalog for inspection; regenerate locally only where the authorized dependencies
are available. Missing private data is recorded as missing, never converted into dummy data.

## Remaining limits

All eight current assets have a source route; two are byte-bound to saved outputs and
receipts. Six have no equivalent current-PDF byte binding in the audited records. A previous
national-map rendered-pixel audit is preserved as a historical claim. This task did not rerun
that audit or any renderer. The representative-series `figure_inputs/fig06` directory is
currently absent. The existing manuscript still contains historical figures; this catalog
does not relabel them as current W36+B20 results or authorize replacement figures.
