# Fig1: current approved source

Approved on 2026-10-29: **turbo**, based on the selected v32 comparison.

| Item | Current location (relative to repository root) |
|---|---|
| Canonical source entry | `scripts/figures/build_fig1.py` |
| Installed manuscript PDF | `manuscript_JAG/figures/fig01_study_area.pdf` |
| Named final export | `runs/figures/fig1_final_20261029/Fig1.pdf` |
| Final preview | `runs/figures/fig1_final_20261029/Fig1_preview.png` |
| Installation/hash receipt | `runs/figures/fig1_final_20261029/integration_receipt.json` |
| Obsolete PDF backup cleanup | Workspace `tooling/workspace-maintenance/receipts/balsa_figures_20261113/figure_backups_cleanup.json` |

The first figure included by `sections/02_materials.tex` is now named
`fig01_study_area.pdf`. The November 13 filename organization updated its TeX
reference while preserving the approved PDF bytes and caption.

## Rebuild

From the repository root, using the existing plotting/geospatial Python environment:

```powershell
python scripts/figures/build_fig1.py --output runs/figures/fig1_rebuild/Fig1.pdf
```

Use a new output path: the builder does not overwrite prior exports or install
manuscript assets automatically. It does not launch a GUI or create palette trials.

This export reuses frozen vector artwork at
`runs/figures/fig1_harvest_concept_v31_white_ticks_20261029/study_area_fig1_v31.pdf`.
The coverage-bin check uses
`runs/figures/s1_cumulative_coverage_20261029_v25/s1_cumulative_ascending_201601_202606.tif`.
The source imports retained helpers from `build_study_area_fig1_v32.py` and its
existing helper chain. Those source files and the frozen input PDF must remain.
This is an exact-layout export route, not a standalone rebuild from raw imagery.

## Frozen design

- Panel (a): inventory, dated subset and MGRS tiles; inventory legend right-aligned
  with the South America inset; mainland north/south margins tightened.
  Area and MGRS legends are shifted up 19.4153595 pt from the frozen source.
  Their final adjustment is 10.5846405 pt downward, aligning the Area frame top
  with the South America inset while preserving the gap between the two legends.
  The single-line bottom-right credit is `Hillshade: Esri World Hillshade © Esri` at 5.5 pt.
  The South America inset is shifted up 6 pt to clear the attribution row.
- Panel (b): turbo coverage, linear 0-900 scale, opacity 0.78; no terrain raster;
  title `Sentinel-1 observations from 2016 to June 2026`; right latitude labels are rotated
  180 degrees about their original centers to match panel (a).
- Panels (c-f): optical imagery, gray dashed 0.55 pt grid, white inward ticks.
- Panels (g-j): SAR imagery, black solid 0.55 pt grid with original 0.52 opacity,
  black inward ticks; bottom longitude labels retain an additional 1.0 pt gap relative to v29 (0.8 pt farther than the initial approved turbo export).
- Dates: Jul. 2021, Jun. 2022, Apr. 2024, Jun. 2026; 250 m scales.
- Page: approximately 180 x 177.55 mm. No imagery or coverage values changed.

The initially accepted canonical rebuild matched the selected turbo trial at
200 dpi. A subsequent revision moved only the four bottom longitude labels down
by 0.8 pt. A later revision added the hillshade credit and moved the Area/MGRS
legends up 10 pt. The latest revision relocates the credit to one line at bottom-right,
raises those legends another 20 pt and the inset 6 pt, and rotates seven panel (b)
latitude labels. The final adjustment lowers only the Area/MGRS legends by
10.5846405 pt to align the Area and inset frame tops. All pixels outside the edited regions remain identical at 200 dpi. The installed manuscript PDF and named final export match by SHA-256.
Palette-trial HTML, previews and PDFs were removed after acceptance. The selection
QA remains in the final export directory. Earlier source-dependent figure versions
were retained. Cleanup paths, hashes and verification are recorded under workspace
`tooling/workspace-maintenance/receipts/fig1_palette_20261029/`.

The manuscript was not recompiled during this installation; its next compilation
will pick up the replaced figure. Caption text remains the author's responsibility.
