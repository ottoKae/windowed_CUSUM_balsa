# Restricted capsule data dictionary

| Subdirectory | Unit and purpose | Release risk |
|---|---|---|
| `inventory/` | One row and exact geometry per validated plantation polygon | Exact locations and official identifiers |
| `splits/` | Polygon roles, spatial-block geometry, and frozen 57-block/5-fold map | Evaluation geography and linkage risk |
| `references/` | Dated ground events and their observed-track expansion | Exact historical labels |
| `polygon_pixels/` | Polygon-to-pixel membership and 126-month VV/VH/n_obs arrays by track | Large exact-location remote-sensing samples |
| `frozen_protocol/` | Per-fold parameter choices, OOF predictions, and source ledger | Model-evaluation evidence |
| `raw_to_result/` | Rebuilt caches, Polygon-first/Pixel-first events, F0--F4 ledgers, bootstrap outputs, and receipts | Detailed reproducibility output |
| `prospective_2026/` | Right-censored N1 snapshots and review-overlap audit | Candidate locations requiring controlled review |

Missing Sentinel-1 months remain null/NaN. Physical zero is never used as an
imputation value. `track_key` is the resolver-defined observation group; data
remain separate by track until event-level fusion.

