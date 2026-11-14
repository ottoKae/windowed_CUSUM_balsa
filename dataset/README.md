# Ecuadorian balsa stands, harvest references and monthly Sentinel-1 spatial time series

Dataset version: **V1.0.0**, published on 14 November 2026 at [Zenodo](https://doi.org/10.5281/zenodo.24223182). Dataset licence: **CC BY-NC-SA 4.0**, consistent with FC001 source terms. The accompanying software remains MIT-licensed. See `LICENSES.md`.

This dataset follows the same plantation pixels through time. It connects mapped stands, monthly VV/VH backscatter, dated harvest references, and the paper's pixel detections and stand reports. It contains numerical SAR derivatives, not PlanetScope imagery or raw Sentinel-1 bursts.

## What is included?

| Population | Stands | Native pixels | Dated references |
|---|---:|---:|---:|
| Retained FC001 balsa inventory | 3,498 | — | — |
| Inference population | 3,364 | 151,549 | — |
| Dated subset of the inference population | 459 | 25,555 | 1,049 |

The remaining 134 inventory stands have no selected native cell centres. Their harvest counts are missing, not zero. The reference subset retains all fixed pixels, including pixels with no detections.

The calendar has 126 monthly columns, January 2016–June 2026. Missing early observations remain missing columns. The reporting period is January 2017–December 2025. Observations use track-specific native 30 m cells. A cell is identified within its original grid and track; cells from different tracks are not merged merely because their footprints overlap.

## Download and open

The four ZIP files have complementary paths under `dataset/`; extract them into the same working directory.

| ZIP file | Contents |
|---|---|
| `balsa_stands_references_V1.0.0.zip` | Inventory attributes and GeoPackage boundaries, inference and reference cohorts, dated references, readable examples and documentation |
| `balsa_reference_STS_V1.0.0.zip` | VV/VH matrices, validity and observation counts, coordinates and membership for 459 stands |
| `balsa_national_STS_V1.0.0.zip` | The corresponding complete data for 3,364 stands |
| `balsa_results_replay_V1.0.0.zip` | Frozen pixel detections, TDA/STP products, bootstrap draws and STP calibration replay data |

Open `inventory/stands.gpkg` in QGIS and join `inventory/stands.csv` on `sample_id`. Open the CSV reference table in a spreadsheet. The six example trajectories are in `examples/pixel_month_examples.csv` and `examples/pixel_month_examples.pdf`. Full arrays use NumPy `.npy` to avoid millions of repetitive text rows.

```python
from pathlib import Path
import numpy as np
import pandas as pd

p = Path('dataset/timeseries/reference_stands')
pixels = pd.read_csv(p / 'units.csv')
membership = pd.read_csv(p / 'membership.csv')
months = pd.read_csv(p / 'calendar.csv')
vv = np.load(p / 'monthly_vv.npy', mmap_mode='r')
vh = np.load(p / 'monthly_vh.npy', mmap_mode='r')
# vv[i, j] belongs to pixels.iloc[i] and months.iloc[j].
```

Do not sort `units.csv` without applying the identical row permutation to all four arrays. Preserve `pixel_track_id`, which supplies the original detector's permutation identity. `sample_id` links all stand tables; `unit_id` links pixels and membership; `node_id` identifies a pixel–month detection; `component_id` identifies a TDA parent; `round_id` identifies an STP child.

## Run the paper pipeline

Use the accompanying repository and its reproduction environment. Run these commands from the repository root after extraction; output folders must not already exist.

```sh
python scripts/run_current_pipeline.py --input-dir dataset/timeseries/reference_stands --output runs/replay_reference_V1.0.0 --workers 4 --stp-replay dataset/reproduction/stp_replay
python scripts/experiments/evaluate_reports.py --references dataset/references/harvest_events.csv --reports current runs/replay_reference_V1.0.0/tda_events.csv --draws dataset/reproduction/bootstrap_draws.npy --output runs/score_reference_V1.0.0
```

For nationwide inference, replace `reference_stands` with `national_stands` and use a new output directory. The default frozen detector uses VV, W=36 months, D=S=12 months and alpha=0.025. Exported VH supports comparison; it is not silently added to the baseline. TDA uses fixed stand membership and 20% distinct-pixel support. STP partitions a parent without adding stand-level events.

The STP replay directory preserves the original 1,216-parent calibration sequence, including nine legacy stands outside the current 459-stand reference subset. Those records preserve historical random-number ordering, not additional reference labels. The provided reference and national runs replay this sequence before processing additional parents. Running arbitrary new subsets is supported, but STP permutation results for added parents may differ when their preceding sequence changes. Descriptive BH q-values also depend on the returned family; the frozen split gate uses p<=0.05.

The reference baseline is 1,208 TDA reports, 945 one-to-one matches to 1,049 references, and 495/590 recovered adjacent reference pairs. Matching tests overlap between a report's first–last detection range and reference month ±1 month. It does not use representative-month ±2-month matching. Nationwide frozen products contain 6,303 parents and 6,375 STP child groups. Expected numbers are in `reproduction/expected_summary.json`; accompanying acceptance files report which replays actually passed.

## Interpret the dates

Keep the reference `date_basis`: 1,045 records retain `historical_reported_month`, and four explicitly use `optical_first_visible_removal`. The export does not reinterpret all references as exact harvest onset. A TDA range describes detected responses, not independently verified operation duration. Representative months are rounded medians of pixel–month detection dates; adjacent-start intervals use consecutive parent first-detection months. STP child dates use the same contributing-detection definition.

See `DATA_DICTIONARY.md` for key fields and `schema_inventory.csv` for every exported table column. `manifest.json` records file hashes; the attachment-level `checksums.sha256` verifies downloads. Code file hashes in `reproduction/code_manifest.json` identify the frozen implementation. Cite version DOI `10.5281/zenodo.24223182` when reproducing V1.0.0. The deposited files retain their pre-publication wording; their contents and checksums have not been changed.
