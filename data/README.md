# Public examples and data contracts

The public data directory contains synthetic demonstration sequences and
schemas. It contains no real pixel series, inventory geometry, review records
or row-level research results.

`demo/monthly_s1_synthetic.csv` has one invented stand-month per row, with `ID`,
`time`, `VV_dB` and `VH_dB`. Blank values represent missing observations and must
remain missing. `demo/reference_events_synthetic.csv` contains invented event
months. Neither file is derived from the study plantations or reproduces paper
statistics. The existing generator is `demo/generate_synthetic_demo.py`.

The current first-post W36 → TDA → STP interfaces and required arrays/tables are
described in the [algorithm guide](../docs/ALGORITHM.md). The synthetic CLI
example in the [root README](../README.md) exercises the historical `paper.py`
interface for compatibility. It is not a demonstration of the full current
pipeline. Older W36 capsule and sample-collection schemas describe their own
contracts; including a schema does not distribute the corresponding records.

The author workspace retains raw Sentinel-1/S1-GRiTS stores, monthly extraction
caches, exact-location inventories, optical interpretation material and private
frozen outputs. These remain local and are excluded from the export. No public
download, deposited capsule or permission to redistribute them is implied.

The seven approved manuscript figure PDFs are supplied as figures, not as a
release of their underlying data. See the
[release policy](../provenance/PUBLIC_RELEASE_POLICY.md).
