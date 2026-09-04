# Windowed CUSUM for recurrent stand disturbances

This repository contains the manuscript-facing implementation and compact
evidence package for detecting recurrent stand disturbances in Ecuadorian balsa
plantations from near-decadal monthly Sentinel-1 SAR time series.

The method evaluates overlapping local temporal contexts instead of comparing
every date with one permanent global baseline. The frozen paper protocol uses
48-month windows displaced by 12 months, admits candidates only inside the
central 24-month decision interval, retains negative shifts, and screens them
with a finite-sample-corrected IID permutation p-value.

National detections are interpreted as **stand disturbances**. They are not all
independently verified harvests, and the software does not infer disturbance
cause from SAR alone.

## Repository contents

| Path | Purpose |
|---|---|
| `src/lr_cusum/` | Core detector, N1 paper protocol, aggregation, validation, ablation, and optional data adapters |
| `configs/paper_v1.json` | Frozen detector, consolidation, and validation semantics |
| `data/demo/` | Synthetic, non-geographic monthly example |
| `data/schemas/`, `data/manifests/` | Input contracts and real-data release rules |
| `results/` | Compact aggregate tables and count ledger; no exact locations |
| `scripts/figures/` | Public builder for the methodological workflow figure |
| `manuscript/` | Portable LaTeX source and the eight active figures |
| `tests/` | Unit tests for algorithmic and evaluation semantics |
| `provenance/` | Source inventory, redistribution boundaries, and hashes |

The authorized workstation also has an ignored package at
`data/local/paper_v1/`. It contains the 3,498 validated geometries, frozen
polygon roles and spatial blocks, 1,093 dated reference events, complete
polygon-interior pixel caches, the S1-GRiTS catalog, local administrative and
basemap assets, and row-level frozen outputs. The multi-terabyte Zarr archive is
mounted in place rather than copied into Git.

Historical notebooks, private review workbooks, raw geospatial data, caches,
intermediate candidates, and versioned figure wrappers are intentionally absent.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

Parquet input, validation baselines, geospatial adapters, figures, and tests are
optional:

```bash
python -m pip install -e ".[io,validation,geo,figures,test]"
```

No software license has yet been selected. See `LICENSE_PENDING.md`; the authors
must add an approved `LICENSE` before public release.

## Run the frozen paper detector

The input is a long table with one row per object and calendar month. At minimum
it contains `ID`, `time`, and one or both of `VV_dB` and `VH_dB`. Calendar months
must not be omitted: retain a row and store an unavailable observation as NaN.

```bash
windowed-cusum paper \
  data/demo/monthly_s1_synthetic.csv \
  demo_events.csv \
  --config configs/paper_v1.json
```

The output contains accepted **window candidates**. Overlapping-window and
multi-track consolidation is deliberately explicit: the paper subsequently
fuses same-band detections within one month, treats VV as the primary chronology,
and matches VH support one-to-one within two months. This separation prevents a
candidate score from being confused with a final national event.

The archived empirical-confidence implementation is available only for
historical comparisons:

```bash
windowed-cusum legacy input.csv legacy_candidates.csv
```

Do not use `legacy` to claim reproduction of the final N1 manuscript protocol.

## Verify

```bash
python -m pytest
python scripts/figures/build_workflow.py workflow.pdf
```

Aggregate manuscript values can be checked directly against `results/`; a full
raw-to-paper replay additionally requires the authorized plantation inventory,
reference-event records, S1-GRiTS Zarr stores, and GRiTS-Resolver. These external
inputs are not redistributed here. See `docs/REPRODUCIBILITY.md` and
`data/README.md`.

The release-candidate audit covers the complete repository, verifies all
aggregate source hashes and manuscript figure references, and scans for exact
locations, workstation paths, non-English code text, and common secret formats:

```bash
python provenance/audit_public_package.py
python provenance/build_release_manifest.py
```

## Authorized full replay

Copy `configs/data.example.json` to the ignored `configs/data.local.json`, set
the source paths, then run:

```bash
python scripts/reproduce.py prepare --config configs/data.local.json
python scripts/reproduce.py audit --config configs/data.local.json
python scripts/reproduce.py extract --track SAMPLE_TRACK
python scripts/reproduce.py detect --route polygon-first
python scripts/reproduce.py detect --route pixel-first
python scripts/reproduce.py evaluate
python scripts/reproduce.py summarize
```

Detection writes one atomic Parquet file per track under `runs/`, so an
interrupted national run resumes without recomputing completed tracks. Use
`--track TILE_DIRECTION_TKNN` for a small integration test. The frozen paper
chronology is polygon-first VV; pixel-first is an explicit sensitivity route,
not a replacement for the published count ledger. See
`docs/FULL_REPRODUCTION.md` for data lineage, stages, and publication gates.

Legacy 9×9 S1-GRiTS sample shards can be converted to long Parquet tables with
the same runner:

```bash
python scripts/reproduce.py convert-shards PATH/TO/COHORT OUTPUT --cohort balsa
```

The detector was also compared with the frozen private N1 ledgers on sampled
VV and VH trajectories, including candidates that triggered the 19,999-draw
near-threshold refinement. Candidate identities and p-values matched exactly;
the non-identifying receipt is `provenance/PRIVATE_INTEGRATION_QA.json`.

## Frozen method in brief

- Temporal context: January 2016--June 2026.
- Event reporting interval: January 2017--December 2025.
- Stand representation: polygon-first monthly median.
- Window length / displacement: 48 / 12 months.
- Minimum finite observations per window: 12.
- N1 screen: 3,999 IID permutations and finite-sample correction.
- Near-threshold refinement: 19,999 permutations for
  `0.015 <= p_perm <= 0.035`.
- Admission: `p_perm <= 0.025`.
- Same-band track/event consolidation: within one month.
- Reference matching and VV--VH support: one-to-one within two months.
- Uncertainty: resampling complete spatial blocks, not individual events.

The authoritative machine-readable specification is
`configs/paper_v1.json`. Detailed boundary and index semantics are documented in
`docs/METHOD_SEMANTICS.md`.

## Citation and data availability

A permanent repository URL, archival DOI, and `CITATION.cff` should be added
after the release owner and software license are confirmed. Third-party data and
imagery retain their original licenses and are not covered by the future code
license.
