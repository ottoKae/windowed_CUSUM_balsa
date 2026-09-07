# Manuscript figure reproducibility audit (2026-09-06)

## Scope

This audit follows the order in which figures are referenced by
`manuscript_JAG_v2/main.tex` and its included section files. A figure is marked
**exact** only when the production Python workflow can recreate the same panel
geometry and rendered pixels, rather than merely drawing the same quantities.

## Current TeX order and production status

| Printed figure | TeX asset | Production Python source | Exact-layout status |
|---:|---|---|---|
| 1 | `fig03_workflow.pdf` | `manuscript_JAG_v2/figure_sources/powerpoint/windowed-CUSUM-workflow-figures-01.pptx` | **Source available.** The exact editable PowerPoint source and frozen manuscript PDF are both included. The alternative Python workflow is not asserted to be pixel-equivalent. |
| 2 | `fig02_study_area.pdf` | `scripts/build_study_area_green_yellow.py` + `scripts/build_study_area_figure.py` | **Partial.** The formal renderer exists, but its configuration and restricted geometry/reference inputs still need a fully local command wrapper. The simplified `scripts/figures/build_study_area.py` is not layout-equivalent. |
| 3 | `fig04_windowed_cusum_concept.pdf` | `analysis/workflows/build_windowed_cusum_concept_n1_current.py` | **Partial.** The formal renderer exists; the simplified wrapper is not layout-equivalent and the frozen input paths are not yet localized in the all-figure runner. |
| 4 | `fig05_population_s1_signatures.pdf` | `scripts/figures/build_population_s1_signatures.py` -> `analysis/workflows/build_population_s1_signature_figure_v11.py` | **Functional, not exact.** The inherited modules and frozen local inputs are localized. A clean rebuild succeeds, but the page crop differs from the frozen PDF by about 1 pt; publishing over the manuscript asset remains disabled. |
| 5 | `fig06_representative_timeseries.pdf` | `analysis/workflows/build_timeseries_signature_polygon_iqr.py` + `build_timeseries_signature_polygon_median.py` | **Partial.** The formal renderer exists and the figure inputs are partly localized; the simplified wrapper is not layout-equivalent. |
| 6 | `fig07_representation_configuration.pdf` | `scripts/figures/build_representation_configuration.py` -> `analysis/workflows/build_figure5_abcde_metric_encoding_v19.py` | **Functional, not exact.** The inherited modules and aggregate source tables are localized. A clean rebuild succeeds, but the page crop differs from the frozen PDF by about 1 pt; publishing over the manuscript asset remains disabled. |
| 7 | `fig08_polarization_intervals.pdf` | `scripts/summarize_n1_polarization_interval_sensitivity.py` | **Partial.** The formal renderer exists, but the current `scripts/figures/build_polarization_intervals.py` is a simplified reimplementation and is not layout-equivalent. |
| 8 | `fig01_national_recurrence.pdf` | `analysis/workflows/build_figure6_recurrence_10km.py`; wrapper `scripts/figures/build_national_recurrence.py` | **Exact.** The local-only rebuild has identical dimensions and is pixel-exact against the frozen production PNG. |

The filenames preserve earlier manuscript numbering and therefore do not match
the current printed figure numbers. The current printed order is authoritative;
renaming the PDF assets is optional and should not be mixed with scientific
updates.

## Diagnostic rebuild result

Running the existing simplified all-figure runner produced no bit-exact PDF.
Figure 6 did not render because the current representation labels no longer
match the old simplified wrapper. The other seven diagnostic outputs used
different page sizes or panel geometry. Consequently, the simplified runner
must not publish over the manuscript figures.

## Figure 8 exact-layout verification

The production Figure 8 builder was restored with its two original helper
modules and frozen layout configuration. The following local inputs are now
used:

- 3,498 validated BALSA polygons;
- the frozen Polygon-first VV and VH event ledgers;
- the polygon-to-track table;
- local Ecuador mainland and Amazon context GeoParquet files;
- the local Esri hillshade cache and Natural Earth administrative context.

The no-VH-support build was compared with the frozen production PNG:

- identical raster dimensions: `4261 x 2568` pixels;
- changed-pixel fraction: `0.0`;
- mean absolute pixel difference: `0.0`.

This establishes exact layout and exact rendered content for the default
VV-primary Figure 8 route. PDF binary hashes can still differ when export
metadata timestamps differ; rendered-pixel identity is the relevant layout
test.

## VV-primary / VH-evidence preview contract

The optional Figure 8 preview preserves the production 180 x 107 mm canvas,
left national map, right three-panel stack, basemap, map extent, grid, fonts,
axes, margins, and panel locations.

- Panel (a) remains the median Polygon-first VV event count in each occupied
  10-km-wide hexagon.
- Panel (b) retains annual primary VV counts and overlays the subset supported
  by one Polygon-first VH event within +/-2 calendar months.
- Panel (c) remains the primary VV polygon event-count distribution.
- Panel (d) retains the primary VV adjacent-interval histogram and overlays
  the interval distribution for which both primary-event endpoints have VH
  support.
- Unmatched VH detections remain in a separate ledger and never enter the
  primary disturbance chronology or a VV/VH average.

Frozen counts are 6,658 primary VV events, 6,144 Polygon-first VH events,
4,061 VH-supported primary events (61.0%), 2,083 unmatched VH events, 3,509
primary adjacent intervals, and 1,385 intervals with both endpoints supported
(39.5%). The primary and supported interval medians are 36 and 40 months,
respectively.

## Remaining actions before claiming eight-figure exact reproduction

1. Preserve the included workflow PPTX and its frozen exported PDF together.
2. Complete thin production entry points for Figures 2, 3, 5, and 7; Figures 4
   and 6 now have localized thin wrappers and dependency chains.
3. Keep restricted geometry, event-level time series, and licensed imagery in
   the ignored author-side bundle; expose only aggregate source tables that
   satisfy the release policy.
4. Rebuild each figure to a versioned audit directory and compare page size,
   rendered pixels, panel boxes, fonts, and source-data hashes.
5. Enable `--publish` only after all eight rows pass; until then, manuscript
   assets remain frozen and unchanged.
