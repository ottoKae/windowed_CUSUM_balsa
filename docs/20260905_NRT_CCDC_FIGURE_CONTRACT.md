# NRT / official CCDC comparison figure contract

Status: new diagnostic figure only; do not replace or renumber manuscript figures.

Question: does the recurrent-event advantage survive credible alternative detectors,
direction controls and a distinction between retrospective accuracy and alert timeliness?

Backend: Python / matplotlib. Fixed 183 x 138 mm, 2 x 2 panels, Arial 8 pt
body/axes, 9 pt panel labels; vector PDF and editable SVG, 600 dpi PNG.
Green identifies Windowed CUSUM, yellow identifies the direction-controlled NRT
adaptation, blue identifies CCDC, and grey identifies native-direction NRT.
Markers and line styles provide redundant, colour-independent encodings. No red.

| Panel | Unique evidence role | Source | Uncertainty |
|---|---|---|---|
| a | Full-horizon, spatial-OOF catalogue-conditional F1; includes direction and dual-band controls | diagnostics/method_intervals.csv | 95% spatial-block bootstrap CI |
| b | Paired F1 effects for the two declared primary comparisons | paired_contrasts.csv | 97.5% individual CIs, Bonferroni coverage for two F1 contrasts |
| c | Correct, still-active alerts as a function of time after the reference month | latency/latency_metrics.csv | 95% spatial-block bootstrap CI; censored references excluded equally |
| d | Whether recall is concentrated in the first rather than later known events | diagnostics/stratified_reference_recall.csv | 95% spatial-block bootstrap CI; exploratory, not multiplicity-adjusted |

All four panels use 57 complete spatial blocks, 5,000 resamples, seed 20260829.
The population is 1,093 references in 462 polygons / 474 polygon-track sequences;
pixel-level cross-Tc is a supplementary control, not mixed with stand-level input.
The first/subsequent distinction concerns known references, not biological generations.

Do not claim nationwide precision, daily timeliness, full-scene reproduction of the
published NRT workflow, or independent prospective validation. The catalogue is
not an exhaustive no-disturbance census. Predictions unmatched to it may include
real but undated events. The timeline is a retrospective causal-input replay with
spatially held-out parameter selection, not a historical deployment trial.

QA: run the skill's source validator and PDF glyph audit, inspect each rendered
panel plus the assembled PNG. Archive the plotted source CSVs and QA notes next
to the figure. A missing input or incomplete replay must stop figure generation.
