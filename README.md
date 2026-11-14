# Mapping rotational harvesting with windowed CuSum

A retrospective framework for reconstructing recurrent balsa harvest histories
from monthly Sentinel-1 SAR observations: **fixed pixel trajectories -> windowed
CuSum -> temporal detection aggregation (TDA) -> spatial timing partitioning (STP)**.
TDA defines stand events; STP describes timing groups within an event.

**Paper:** *Mapping rotational harvesting from decadal Sentinel-1 time series
with windowed CuSum: A case study of Ecuadorian balsa plantations*.
[Manuscript PDF](manuscript_JAG/main.pdf) | [Supplement](manuscript_JAG/supplementary.pdf) ?
[Workflow figure](manuscript_JAG/figures/fig02_workflow.pdf) | [Citation](CITATION.cff)

## Start here

| Reviewer question | Entry |
|---|---|
| How does the method work? | [Algorithm and actual implementation](docs/ALGORITHM.md) |
| Can I run the current complete pipeline? | Quick start below; [input/output guide](docs/REPRODUCIBILITY.md) |
| Where are Table 3-5 experiments and results? | [Experiment index](scripts/experiments/README.md) and [aggregate CSVs](results/summary/) |
| What generates each of the seven figures? | [Figure index](docs/FIGURES.md) and [source manifest](provenance/figures/source_spec.json) |
| What data are available? | [Data sources and access](data/README.md), [reproducibility levels](docs/REPRODUCIBILITY.md) |
| What is excluded from publication? | [Privacy boundary](docs/PRIVACY.md) |

## Quick start: current method, invented data

Use Python 3.12 in a fresh environment, from the repository root:

```sh
python -m pip install ".[io,validation]" -c configs/requirements-reproduce.txt
python scripts/run_current_pipeline.py --output .test-tmp/current-demo
python scripts/experiments/check_published_results.py
```

The pipeline command runs the real first-post detector, TDA and STP on an invented
6x6 stand. It needs no imagery, account or private records. The deterministic
example produces **72 pixel detections, 2 TDA events and 4 STP timing groups**.
Inspect `summary.json`, `pixel_detections.csv`, `tda_events.csv`, `stp_groups.csv`,
and the TDA/STP provenance tables. Choose a new output directory for another run.
The aggregate-check script checks the arithmetic and consistency of the published
aggregate tables; it does not rerun real-data detection or bootstrap intervals.

Use `--input-dir` to supply your own monthly pixels; see the
[data contract](docs/REPRODUCIBILITY.md#supplied-pixel-inputs).
The frozen calendar is January 2016-June 2026, with reports during 2017-2025.

## Paper configuration and reference agreement

The baseline uses VV, a 36-month context, central 12-month decision interval,
12-month stride, and a window-specific permutation acceptance threshold of
0.025. [Current settings](configs/current_pipeline.json) separate TDA from the
historical B20 configuration. The candidate date is the first valid observation
after the cumulative maximum.

Evaluation uses 1,049 reference records from 459 stands. Within each stand,
reports are matched one-to-one when their first-last detection range overlaps
the reference month +/-1 month. Precision, recall and F1 measure agreement with
these reference records. They are not based on representative-month ?1 matching.

| Baseline result | Value |
|---|---:|
| TDA reports / matched references | 1,208 / 945 |
| Reference precision / recall / F1 | 0.782 / 0.901 / 0.837 |
| Consecutive reference pairs recovered | 495 / 590 (83.9%) |
| National application: evaluable stands / TDA events | 3,364 / 6,303 |

The V1.0.0 dataset is published on Zenodo:
[10.5281/zenodo.24223182](https://doi.org/10.5281/zenodo.24223182).
It contains inventory boundaries, harvest references, monthly VV/VH trajectories
and frozen products. The [dataset guide](dataset/README.md) describes the four
ZIP archives and direct pipeline commands; [dataset/CITATION.cff](dataset/CITATION.cff)
provides the dataset citation. Data use CC BY-NC-SA 4.0; the software remains MIT.
See the [reproducibility guide](docs/REPRODUCIBILITY.md) for the baseline replay
and the additional inputs required by historical experiments and figures.

## Code organization

| Directory | Responsibility |
|---|---|
| `src/lr_cusum/` | Numerical methods, matching, temporal nulls and STP |
| `scripts/run_current_pipeline.py` | Current detector -> TDA -> STP entry |
| `scripts/experiments/` | Experiment index, supplied-report scoring and aggregate checks |
| `tooling/diagnostics/` | Selected frozen experiment implementations and dependencies |
| `scripts/figures/` | Current renderers and retained figure helpers |
| `configs/experiments/` | Paper-to-code experiment catalogue |
| `results/summary/` | Aggregate results with source hashes; no individual records |
| `data/demo/`, `data/schemas/` | Invented examples and input contracts |
| `manuscript_JAG/` | Paper, supplement, bibliography and seven final figures |
| `tests/` | Tests of actual detector, aggregation, matching and release boundaries |

Original implementation paths are retained to preserve imports and provenance.
Use the experiment/figure indexes instead of choosing the newest dated filename.
The old `python -m lr_cusum paper` CLI is a historical B20 compatibility interface,
not the current pipeline.

## Tests, publication and citation

```sh
python -m pip install pytest -c configs/requirements-reproduce.txt
python -m pytest tests -q -p no:cacheprovider --basetemp .test-tmp/public-check
python tooling/release/export_public.py --verify .
```

Run the last verification command from an exported public snapshot containing
`PUBLIC_FILES.json`. [Release instructions](docs/PUBLIC_RELEASE.md) explain how
the author workspace produces a clean snapshot without old Git history.
Original code is [MIT licensed](LICENSE); see [DATA_LICENSE](DATA_LICENSE) for
separate data conditions and [CITATION.cff](CITATION.cff) for software citation.
