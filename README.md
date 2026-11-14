# Mapping rotational harvesting with windowed CuSum

[??](README_ZH.md) ? [Paper](manuscript_JAG/main.pdf) ? [Supplement](manuscript_JAG/supplementary.pdf) ? [Data on Zenodo](https://doi.org/10.5281/zenodo.24223182)

How can we follow repeated harvests in the same plantation when cloud cover interrupts optical observations? This repository accompanies *Mapping rotational harvesting from decadal Sentinel-1 time series with windowed CuSum: A case study of Ecuadorian balsa plantations*.

The method follows fixed 30 m pixels through monthly radar observations. **Windowed CuSum** detects changes against a moving local context. **Temporal detection aggregation (TDA)** brings staggered pixel detections into stand-level reports. **Spatial timing partitioning (STP)** describes timing differences within a report without adding harvest events to the stand history.

**Monthly Sentinel-1 VV ? pixel detections ? TDA stand reports ? STP timing groups**

## Read, run, or reproduce

| What you want to do | Where to start |
|---|---|
| Understand the method | [Algorithm](docs/ALGORITHM.md) and [workflow figure](manuscript_JAG/figures/fig02_workflow.pdf) |
| Try it without downloading data | Run the example below |
| Reproduce the reference evaluation | Download the Zenodo data and run the reference commands below |
| Inspect ablations and benchmarks | [Experiment index](scripts/experiments/README.md) and [result tables](results/README.md) |
| Find a figure's code and inputs | [Seven-figure index](docs/FIGURES.md) |

## Try the complete pipeline

Use Python 3.12 in a fresh environment. From the repository root:

```sh
python -m pip install ".[io,validation]" -c configs/requirements-reproduce.txt
python scripts/run_current_pipeline.py --output .test-tmp/first-demo
```

This runs the actual detector, TDA and STP on a synthetic plantation. No account or imagery is needed. It produces **72 pixel detections, 2 TDA reports and 4 STP groups**. Open `summary.json`, `pixel_detections.csv`, `tda_events.csv` and `stp_groups.csv` in the output folder. Use a new output folder for each run.

## Reproduce the balsa reference results

Download the four V1.0.0 archives from [Zenodo](https://doi.org/10.5281/zenodo.24223182) and extract them together into `data/local/zenodo/`. They create a shared `dataset/` folder containing stand boundaries, references, monthly VV/VH arrays and frozen results. Then run:

```sh
python scripts/run_current_pipeline.py --input-dir data/local/zenodo/dataset/timeseries/reference_stands --output runs/reference-replay --workers 4 --stp-replay data/local/zenodo/dataset/reproduction/stp_replay
python scripts/experiments/evaluate_reports.py --references data/local/zenodo/dataset/references/harvest_events.csv --reports current runs/reference-replay/tda_events.csv --draws data/local/zenodo/dataset/reproduction/bootstrap_draws.npy --output runs/reference-score
```

The baseline uses VV, a 36-month context, a central 12-month decision interval, a 12-month stride and a permutation threshold of 0.025. The array calendar spans January 2016?June 2026; the reporting period is 2017?2025. Within each stand, one-to-one matching tests whether a report's first?last detection range overlaps the reference month ?1 month.

| Reference evaluation | Expected result |
|---|---:|
| Stands / reference harvest records | 459 / 1,049 |
| TDA reports / matched references | 1,208 / 945 |
| Precision / recall / F1 | 0.782 / 0.901 / 0.837 |
| Consecutive reference pairs recovered | 495 / 590 |

Nationwide application covers 3,364 evaluable stands and yields 6,303 TDA reports. See the [dataset guide](dataset/README.md) for the files and the [reproduction guide](docs/REPRODUCIBILITY.md) for input formats, replay settings and tests. Some historical experiments and exact figure redraws require additional archived inputs; their requirements are listed with the scripts.

## Find your way around

| Folder | Contents |
|---|---|
| `src/windowed_cusum/` | Detector, stand aggregation, evaluation and benchmark methods |
| `scripts/` | Pipeline, evaluation and figure entry points |
| `configs/` | Current settings and experiment catalogue |
| `results/summary/` | Paper tables and national/climate summaries, with source hashes |
| `dataset/` | Guide to the separately published Zenodo data |
| `data/demo/`, `tests/` | Synthetic examples and regression tests |
| `manuscript_JAG/` | Paper, supplement and seven final figures |
| `provenance/`, `docs/`, `tooling/` | Source links, instructions and supporting scripts |

Some dated helper scripts are still used by the current pipeline. Their paths are retained so that imports and recorded experiment versions stay traceable. [Repository guide](docs/REPOSITORY_GUIDE.md) explains what is current, archived or temporary.

To check the published summary tables without rerunning detection:

```sh
python scripts/experiments/check_published_results.py
```

Code is [MIT licensed](LICENSE). The [V1.0.0 dataset](https://doi.org/10.5281/zenodo.24223182) uses **CC BY-NC-SA 4.0**. Please cite the paper and the data version you use; citation files are provided for the [software](CITATION.cff) and [dataset](dataset/CITATION.cff).
