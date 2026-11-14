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

[`benchmarks/legacy_w36.json`](benchmarks/legacy_w36.json) preserves the earlier
W36 settings needed by the benchmark replay helper. It is not the default for
the current pipeline; the old B20 command-line entry is archived locally.

[`requirements-reproduce.txt`](requirements-reproduce.txt) records dependency
constraints. [`public_export.json`](public_export.json) defines the explicit
public file selection and approved PDF hashes. Other author-workflow
configurations remain local; they are not alternative public defaults.
