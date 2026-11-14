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
| STP over supplied report tables | [`star_segmentation.py`](../src/windowed_cusum/stand/star_segmentation.py), function `segment` |
| STP replay with frozen calibration and continued random stream | [`stp.py`](../tooling/diagnostics/harvest-inventory-20261104/stp.py), function `run_stp` |
| Spatial products from existing reports and local inventory | [`products.py`](../tooling/diagnostics/harvest-inventory-20261104/products.py) |
| Seven approved manuscript figure renderers | [`source_spec.json`](../provenance/figures/source_spec.json) |

Selected implementations retain their original `tooling/diagnostics/` paths
because existing imports, tests and scientific provenance refer to them. The
public source includes their required Python dependencies; private inputs and
run evidence remain local.

The installed numerical package is organized under `src/windowed_cusum/`:
`detector`, `stand`, `evaluation` and `benchmarks`. See [the source guide](../src/README.md).
The old W36/B20 command-line entries are archived locally and are no longer
installed or exported. The current entry is `scripts/run_current_pipeline.py`.

Other author-workflow and review scripts may require private inputs and are not
all included in the public allowlist. Do not add archives to `PYTHONPATH`.
