# Authorized full-data reproduction

## Scope

This workflow separates a small public code repository from an authorized
local research package. It does not disguise private data as open data, and it
does not duplicate the S1-GRiTS Zarr archive.

```text
official inventory + dated references        S1-GRiTS catalog + Zarr
                 |                                      |
                 +------ frozen polygon roles ----------+
                                      |
                         polygon-interior pixel cache
                                      |
                    +-----------------+-----------------+
                    |                                   |
           polygon-first median                 pixel-first detection
                    |                                   |
                    +------- orbit/track fusion --------+
                                      |
                   reference matching + spatial blocks
                                      |
                 F0-F4 / sensitivity / national history
```

## One configuration, one runner

No workflow framework is required. Copy `configs/data.example.json` to the
Git-ignored `configs/data.local.json`, edit the seven source paths, and use
`scripts/reproduce.py`.

```powershell
python scripts/reproduce.py prepare --config configs/data.local.json
python scripts/reproduce.py audit --config configs/data.local.json
```

`prepare` is idempotent: unchanged files are reused, and new files are copied
through a temporary path before replacement. `audit` fails if
any paper-defining count changes. The local manifest records relative paths,
sizes, and SHA-256 hashes; workstation paths remain in ignored local metadata.

To rebuild the native-pixel cache from the mounted Zarr stores, run all tracks
or first verify a small track. The extractor reads every native pixel once and
preserves the full 126-month calendar, validity, and `n_obs` arrays.

```powershell
python scripts/reproduce.py extract --track SAMPLE_TRACK
python scripts/reproduce.py extract
```

For historical DOCC 9×9 point/patch shards, the same runner provides the
separate adapter path:

```powershell
python scripts/reproduce.py convert-shards PATH/TO/COHORT OUTPUT --cohort balsa
```

## Dataset freeze

The current authorized package freezes:

- 3,498 validated BALSA polygons and 392 excluded/pending polygons;
- immutable train, validation, guard, and test roles assigned by whole polygon;
- 1,093 dated ground events from 462 polygons in 57 spatial blocks;
- 4,001 polygon-track series;
- 176,935 polygon-to-pixel associations representing 176,907 unique
  native pixel-track sequences;
- 40 BALSA-supporting tiles drawn from 56 S1-GRiTS track groups;
- an explicit 2016-01--2026-06 calendar, with missing observations retained as
  missing rather than replaced by physical zero; and
- the frozen row-level N1 outputs, including candidate scores, track and ground
  events, F0--F4 ablation ledgers, and spatial-block bootstrap replicates.

The raw archive contains 40 tile directories, 56 monthly orbit/track stores,
and their static counterparts. Three stores stop before June 2026, but this
does not truncate the formal 2017--2025 reporting interval.

## Detection routes

### Polygon-first

For each month and orbit/track, take the median of all valid polygon-interior
pixels, then run N1 Windowed CUSUM once on the stand trajectory. This is the
frozen primary analysis.

```powershell
python scripts/reproduce.py detect --route polygon-first --components
```

### Pixel-first

Run N1 independently for each polygon-interior native pixel, form temporal and
eight-neighbour spatial support, then fuse the resulting polygon-track events.

```powershell
python scripts/reproduce.py detect --route pixel-first --workers 16
```

Both commands write one file per track to `runs/`. Completed files are skipped
on restart. After every available track finishes, the runner writes track-level
events, ground-level events, the VV-primary chronology, and the one-to-one
VH-supported subset. For a quick real-data check:

```powershell
python scripts/reproduce.py detect --route polygon-first --track SAMPLE_TRACK
```

The current paper's pixel-first comparison is a frozen sensitivity route from
the historical experiment. A new all-pixel N1 replay is a new analysis version
and must not silently replace manuscript counts.

After the all-pixel run completes, apply the connected-support cutoff selected
independently in every outer spatial fold (`r=0.20`) before cross-track fusion:

```powershell
python scripts/finalize_pixel_first_n1.py
```

This preserves the unscreened polygon-track candidates and writes the formal
N1 Pixel-first chronology, validation metrics, paired spatial-block bootstrap,
and comparisons with Polygon-first N1 and the historical Pixel-first L0 route
under `runs/windowed_cusum_balsa_paper_v1/pixel-first/formal_n1_connected_r0.20/`.

## Evaluation and F0--F4

The local `frozen_results/` package contains the complete accepted analysis:

- `component_ablation/scores/`: global and sliding candidate scores;
- `component_ablation/events/`: F0--F4 and removal-control track/ground events;
- `component_ablation/metrics/`: event matches, fold metrics, block metrics,
  and recurrence metrics;
- `component_ablation/bootstrap/`: all spatial-block bootstrap replicates and
  paired contrasts;
- `geometry/`: 24/36/48-month and 6/12-month displacement comparisons;
- `truncation/`: temporal truncation and first-detection-delay experiments; and
- `national/`: band candidates, orbit-specific events, fused events, annual
  counts, and adjacent-event intervals.

The public modules implement those algorithms independently of private paths.
With `--components`, detection writes per-track global and sliding candidate
ledgers under `runs/<dataset>/component_ablation/candidates/by_track/` and
atomically combines them after all tracks finish. These raw detector outputs,
including their N1 permutation scores, are the direct input to F0--F4. The
checked-in compact tables under `results/` are the non-identifying report layer.
The ignored row-level package remains the audit and frozen-comparison layer.

Recompute spatially independent F0--F4 metrics and the shared-index 5,000-draw
paired block bootstrap, then rebuild national event-count and interval tables:

```powershell
python scripts/reproduce.py evaluate --source recomputed
python scripts/reproduce.py summarize
```

Use `--source frozen` only to audit the archived manuscript ledgers. The
default `auto` mode prefers recomputed ledgers when they are complete and falls
back to the frozen package otherwise. Every run records the selected source in
`evaluation_receipt.json`.

## Figures

Each active manuscript figure has one canonical builder. Build all figures to
an isolated review directory (the default behavior) with:

```powershell
python scripts/figures/build_all.py --output-dir runs/figures/rebuilt
```

Pass `--figure N` for a single figure. `--publish` is deliberately explicit and
is the only mode that overwrites `manuscript/figures/`. Figures 1--5, 7, and 8
use repository results or the authorized local package. Figure 6 also needs
separately licensed optical scenes configured in the ignored
`configs/figure_sources.local.json`; the repository includes only an example
configuration and never redistributes those source images.

## Release gates

Before publishing `data/local/paper_v1`, the authors must separately approve:

1. MAG inventory redistribution and disclosure of exact plantation locations;
2. release of dated reference-event locations and spatial blocks;
3. redistribution terms for derived Sentinel-1 pixel series;
4. S1-GRiTS catalog/Zarr archival terms; and
5. basemap and optical-image attribution.

Esri cached tiles are local-use inputs and must not be committed. The license
decision for source code is also still open (`LICENSE_PENDING.md`).
