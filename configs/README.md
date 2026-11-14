# Configuration entry point

[`current_pipeline.json`](current_pipeline.json) is used by
`scripts/run_current_pipeline.py`. It records the frozen W36/D12/S12 first-post
detector and current TDA/STP workflow. Geometry and aggregation fields describe
that frozen implementation; editing them does not select a new experimental
configuration. Use the [experiment index](../scripts/experiments/README.md) for
configuration sensitivity runs.

[`firstpost_w36_d12_s12_b20_20261030_v2.json`](firstpost_w36_d12_s12_b20_20261030_v2.json)
provides the frozen detector settings and first-post date contract used by
[`detector_engine.py`](../tooling/diagnostics/balsa-national-firstpost-20261030/detector_engine.py).
Its filename retains the historical B20 identifier; current TDA and STP rules
are implemented separately, as described in the
[algorithm guide](../docs/ALGORITHM.md).

[`standard_w36.json`](standard_w36.json) is retained for the historical
`windowed-cusum paper` synthetic demonstration and earlier W36+B20 interfaces.
Its evaluation fields are historical and do not define current TDA/STP scoring.

[`requirements-reproduce.txt`](requirements-reproduce.txt) records dependency
constraints. [`public_export.json`](public_export.json) defines the explicit
public file selection and approved PDF hashes. Other author-workflow
configurations remain local; they are not alternative public defaults.
