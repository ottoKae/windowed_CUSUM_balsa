# Seven manuscript figures

The final PDFs follow the printed order. Use the entry below for each figure;
numbered helper versions are dependencies, not alternative current figures.

| Figure | Final PDF | Current Python entry | Required inputs |
|---|---|---|---|
| 1. Study area | [PDF](../manuscript_JAG/figures/fig01_study_area.pdf) | [build_fig1.py](../scripts/figures/build_fig1.py) | Frozen vector artwork, SAR coverage raster and inventory context |
| 2. Framework | [PDF](../manuscript_JAG/figures/fig02_workflow.pdf) | [draw_workflow_from_pptx.py](../tooling/diagnostics/workflow-sts-construction-20261105/draw_workflow_from_pptx.py) | Private editable layout and case trajectories |
| 3. STS construction | [PDF](../manuscript_JAG/figures/fig03_sts_construction.pdf) | [draw_workflow_v35_border_padding.py](../tooling/diagnostics/workflow-sts-construction-20261105/draw_workflow_v35_border_padding.py) | Frozen layouts, geometry and case trajectories |
| 4. Pixelwise CuSum | [PDF](../manuscript_JAG/figures/fig04_windowed_cusum_concept.pdf) | [restore_original_layout.py](../tooling/diagnostics/fig4-pixel-firstpost-20261109/restore_original_layout.py) | Pixel evidence and frozen layout PDF |
| 5. Backscatter signatures | [PDF](../manuscript_JAG/figures/fig05_backscatter_signatures.pdf) | [build_population_s1_signatures.py](../scripts/figures/build_population_s1_signatures.py) | Frozen event/trajectory statistics; aggregate curves also in results/summary |
| 6. Stand histories | [PDF](../manuscript_JAG/figures/fig06_stand_harvest_histories.pdf) | [build_harvest_stand_histories_2024_reports.py](../scripts/figures/build_harvest_stand_histories_2024_reports.py) | Private geometry, optical imagery and TDA/STP member records |
| 7. National patterns | [PDF](../manuscript_JAG/figures/fig07_national_harvest_patterns.pdf) | [build_national_harvest_context_cantons.py](../scripts/figures/build_national_harvest_context_cantons.py) | Private stand/context tables plus administrative and climate inputs |

[Full source specification](../provenance/figures/source_spec.json) records commands,
helpers, source paths and approved PDF hashes. [CSV index](../manuscript_JAG/figure_sources/FIGURE_BUILDERS.csv)
provides the same paper-to-code mapping. Listed private input paths document
provenance; they are not downloadable datasets in this public repository.

Compiling the manuscript reuses these PDFs and does not require their raw inputs.
Rendering every figure from scratch requires the documented source assets; some
renderers reuse existing artwork rather than rebuilding it from sensor data.
Figure 1's original rendering commands also require Windows Arial fonts. Its
helpers can be imported on other platforms; this does not guarantee identical
figure rendering without the original fonts and artwork.
