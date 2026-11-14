# Entry points

The current method is W36 Windowed CUSUM with first-post dating, followed by
Temporal Detection Aggregation (TDA) and conditional Spatial Timing Partitioning
(STP). Read [the algorithm guide](../docs/ALGORITHM.md) for callable interfaces,
input tables and the boundary between supplied-table execution and replay of
private frozen results.

| Current task | Implementation |
|---|---|
| First-post W36 pixel detector | [`detector_engine.py`](../tooling/diagnostics/balsa-national-firstpost-20261030/detector_engine.py) |
| TDA over accepted pixel detections | [`tda.py`](../tooling/diagnostics/harvest-inventory-20261104/tda.py), function `aggregate_frames` |
| STP over supplied report tables | [`star_segmentation.py`](../src/lr_cusum/star_segmentation.py), function `segment` |
| STP replay with frozen calibration and continued random stream | [`stp.py`](../tooling/diagnostics/harvest-inventory-20261104/stp.py), function `run_stp` |
| Spatial products from existing reports and local inventory | [`products.py`](../tooling/diagnostics/harvest-inventory-20261104/products.py) |
| Seven approved manuscript figure renderers | [`source_spec.json`](../provenance/figures/source_spec.json) |

Selected implementations retain their original `tooling/diagnostics/` paths
because existing imports, tests and scientific provenance refer to them. The
public source includes their required Python dependencies; private inputs and
run evidence remain local.

| Historical task | Entry |
|---|---|
| Earlier W36 pixel detection and B20 capsule replay | `reproduce.py` |
| Earlier frozen probability precision and catalogue recall | `evaluate_standard.py` |
| Earlier polarization/null summary | `summarize_w36_extensions.py` |
| Earlier eight-figure diagnostic builder | `figures/build_all.py` |

`windowed-cusum paper` and `reproduce.py` do not invoke the current first-post
adapter or the TDA/STP chain. `evaluate_standard.py` does not automatically use
later reference revisions. `figures/build_all.py` does not regenerate the current
seven manuscript figures.

The installed package lives in `src/lr_cusum`. Other author-workflow and review
scripts may require private inputs and are not all included in the public
allowlist. Do not add sibling archive directories to `PYTHONPATH`.
