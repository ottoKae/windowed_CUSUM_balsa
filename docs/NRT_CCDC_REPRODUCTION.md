# Monthly CuSum NRT and official CCDC comparison

This isolated workflow implements the approved 2026-09-05 experiment. It does not
replace the frozen national outputs, manuscript text, or existing figure layouts.
The dated-reference benchmark is **1,093 events, 462 polygons, 474 polygon-track
sequences, 57 spatial blocks and five frozen outer folds**. The polygon median
series spans January 2016–June 2026; admitted historical events span 2017–2025.
The native monthly axis has 126 slots, with missing observations retained as missing.

## What runs where

- `src/lr_cusum/cusum_nrt_adapter.py`: the source-audited scalar numerical core,
  finite-observation/calendar windows, original empirical confidence and the
  separate corrected-permutation/decrease controls.
- `src/lr_cusum/ccdc_adapter.py`: **official**
  `ee.Algorithms.TemporalSegmentation.Ccdc`, called from Python. No local CCDC
  approximation, JavaScript editor, replacement SAR archive, or interpolation.
- `scripts/run_cusum_nrt_ccdc_comparison.py`: input hash checks, author-core and
  cloud preflight, parameter grids, frozen-block OOF selection and paired metrics.
- `scripts/run_nrt_ccdc_local_controls.py`: existing F0–F4/R1 engine and Windowed
  CUSUM monthly prefix replay. F4 and final replay outputs must match frozen events.
- `scripts/run_nrt_cross_tc_control.py`: polygon-interior pixel adaptation of
  low/high confidence spatial support, using native coordinates and physical area.
- `scripts/run_ccdc_followups.py`: VV/VH/ratio controls and strict cloud prefix replay.
- `scripts/summarize_nrt_ccdc_replay.py`: first-issued/updated/withdrawn alert
  histories, first correctly localized alerts, censoring and time-specific recall.
- `scripts/diagnose_nrt_ccdc_comparison.py`: direction-controlled comparisons,
  subgroup recall, unmatched cases, common reporting period and CCDC single factors.
- `scripts/figures/build_nrt_ccdc_comparison.py`: a new, standalone figure bundle.

Existing matching, multitrack fusion, component detection and spatial bootstrap
implementations are reused rather than replaced.

## Cloud permission and scope

The user authorized processing the sample time series in Earth Engine. The
public configuration contains no author project. Each sample-track series is represented by
one **artificial** pixel in a narrow 1-m computational grid. Its actual VV/VH values,
masks and month timestamps are submitted in Python API requests; **real polygon
geometries, field labels, and credentials are not embedded in these images**.
The virtual pixel does not imply a 1-m SAR resolution. Missing-value backing zeros
are masked; real 0 dB observations remain valid. Requests return segment arrays
to local JSON caches. No Earth Engine asset publication or Drive export is needed.

Authentication is a local user action if credentials expire:

```powershell
conda activate py312_torch
earthengine authenticate
earthengine set_project YOUR_EE_PROJECT
$env:EE_PROJECT = "YOUR_EE_PROJECT"
$env:S1GRITS_ALLOW_CLOUD = "1"
```

## Run order

From the repository root, with the existing local data release mounted:

```powershell
python scripts/run_cusum_nrt_ccdc_comparison.py preflight
python scripts/run_cusum_nrt_ccdc_comparison.py nrt
python scripts/run_cusum_nrt_ccdc_comparison.py ccdc
python scripts/run_cusum_nrt_ccdc_comparison.py nrt-ablation
python scripts/run_cusum_nrt_ccdc_comparison.py evaluate
python scripts/run_nrt_ccdc_local_controls.py components
python scripts/run_nrt_ccdc_local_controls.py replay
python scripts/run_nrt_cross_tc_control.py
python scripts/run_ccdc_followups.py bands
python scripts/run_ccdc_followups.py replay
python scripts/diagnose_nrt_ccdc_comparison.py
python scripts/summarize_nrt_ccdc_replay.py
python scripts/figures/build_nrt_ccdc_comparison.py
python scripts/audit_source_syntax.py
python provenance/audit_public_package.py
```

The default output is `runs/cusum_nrt_ccdc_20260905_v1/`. `--run-dir` and `--config`
can select a new version. The short `all` command on the main runner covers its own
stages only, not the separate cross-Tc, replay, diagnostics or plotting workflows.
Cloud replay defaults to three concurrent requests, 80 sequences/request, with
bounded retries. A failed request is never converted into an empty detection result.

The initial grid contains 27 NRT and 36 CCDC configurations. Three CCDC configurations
(`minObservations=3`, `lambda=0`, all three chi-square values) fail on the same real
trajectory. They are recorded as technically unavailable and excluded for **all**
outer folds; no polygon is removed. The other 33 CCDC configurations cover all 474
sequences. Do not silently delete failure receipts or relabel them as zero events.

## Resume and reproducibility limits

`manifest.json` checks input/config hashes before resume. Raw scans, cloud batches,
segments, selections and predictions are cached separately. Use a **new run directory
after any algorithm change**: this version does not invalidate every cached stage
automatically from a source-code hash. The final receipt records delivery source
hashes; it is not a claim that intermediate development versions were identical.

The ReCuSum comparison originally comes from the sibling `LR-CuSum_v100` frozen
OOF cache; its copied result and checksum are in this run. Regenerating ReCuSum from
raw data is not part of the new NRT/CCDC adapter. Preserve that cache for portability.
The real local time-series archive and caches are not automatically public-release
assets. The author NRT source is MIT-licensed; its notice is retained in
`docs/third_party/CuSum_NRT_LICENSE.txt`. GEE processing permission is not permission
to publish private source data.

## Interpretation rules

Primary F1 contrasts use 5,000 paired resamples of complete spatial blocks and
97.5% individual CIs for the two-comparison Bonferroni family. Other intervals are
95% exploratory CIs, not an unreported family of confirmatory tests. F1 is computed
from pooled event TP/FP/FN, not the arithmetic mean of block F1 scores.

All event matches are one-to-one within ±2 calendar months. Precision and F1 are
**conditional on the dated-reference catalogue**: unmatched predictions are not
independently verified false disturbances. Timing MAE is conditional on matched
events. The NRT monthly adaptation is not the complete published high-frequency,
full-scene workflow; the cross-Tc branch is restricted to polygon interiors, with
no permanent forest exclusion mask. At the native 30-m grid a cell is 900 m², so
a nominal >300 m² minimum does not reproduce a finer-resolution area filter.

CCDC receives SAR dB features, not the Landsat units used for its common defaults;
lambda sensitivity therefore matters. The optional ratio is explicitly
`10**((VV_dB - VH_dB)/10)`, not a quotient of dB numbers or an asserted exact
reproduction of another paper's feature engineering.

Monthly replay is a retrospective simulation with strictly truncated **inputs**;
spatial calibration uses the benchmark's other four folds, not an independently
held-out historical deployment period. First-issue date, first-correct date and
active correctness are distinct. Right-censored horizons are excluded equally for
all methods. It cannot establish daily latency, operational data-delivery latency,
or a nationwide false-alert rate.

## Sources

- [CuSum NRT author repository](https://forge.inrae.fr/bertrand.ygorra/cusum-deforestation_monitoring),
  audited commit `8ed46c036d7947633b4e7eedf495d295f660d603`.
- [Ygorra et al., 2024](https://doi.org/10.3389/frsen.2024.1416550), existing key
  `Ygorra2024CuSumNRT` in `manuscript_JAG_v2/references.bib`.
- [Official Earth Engine CCDC API](https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc).
- [Bullock et al., 2022](https://doi.org/10.1016/j.rse.2022.113043), existing key
  `Bullock2022CCDC_S1`. No duplicate bibliography entries were added.

After review of the report and new figure, manuscript changes require a separate
approval. No old figures, national detection products, Git commits or pushes are
part of this workflow.
