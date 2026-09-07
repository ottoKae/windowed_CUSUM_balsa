# Sentinel-1 burst-acquisition provenance audit

Date: 2026-09-05. Scope: local archive metadata and retained local production logs.

## Finding

An exact count of the raw burst acquisitions used to produce the frozen monthly
archive could not be established from the inspected local sources. The relevant
catalogue columns exist, but their values are empty. A source-count column must
remain unavailable until a complete input manifest is recovered and reconciled.
No manuscript number, source catalogue, Zarr array or detector was modified.

## Sources inspected

Primary archive:

the author-side S1-GRiTS archive supplied with `--archive`

| Evidence | Inspected | Finding |
|---|---:|---|
| Primary catalogue | 112 rows | All `opera_ids` and `jpl_burst_ids` values empty |
| Tile catalogues | 40 files | All source-ID fields empty |
| STAC collection Parquets | 2 files | No burst-acquisition ID fields |
| Current Zarr group metadata | 112 groups | 56 monthly and 56 static groups; no acquisition manifest |
| Current and backup metadata together | 46 Parquets, 123 Zarr groups, 28 JSON files | No retained populated OPERA-ID lists; no metadata read errors |
| Local S1-GRiTS logs | 145 files | 230 distinct burst/time identifiers mentioned in 21 files |

All 21 logs containing these identifiers also mention those same identifiers on
warning/failure/retry lines. Consequently the 230 identifiers are evidence of log
mentions, not a count of successful or retained inputs. They are neither an
estimate nor a certified lower bound for the formal input count.

The scan included QC backups, merge backups, time-order repair backups, remote
staging metadata and the local cache next to the primary archive. Numerical Zarr
chunks were not read. The current monthly groups expose `VV_dB`, `VH_dB`, `n_obs`,
`time`, `x`, `y` and `spatial_ref`; no separate acquisition-ID array is present.
The available archive-level `run_summary.json` describes a one-tile June repair,
not the complete multi-year production run.

## Why the field name is misleading

The current source code explains the difference between two catalogue forms:

- `data_pipeline/S1-GRiTS/src/s1grits/scenes/smonthly_writer.py`, approximately
  lines 1240--1334, builds per-month provenance from source data, including
  `opera_ids` and `jpl_burst_ids`.
- `data_pipeline/S1-GRiTS/src/s1grits/analysis/catalog.py`, approximately
  lines 772--812, reconstructs one record per Zarr store. It assigns
  `n_scenes = len(time_steps)` and does not populate those source-ID lists.
- `data_pipeline/S1-GRiTS/src/s1grits/scenes/pipeline.py`, approximately
  lines 570--575, disables per-workflow STAC writing in the current implementation.

Thus, in this archive, `n_scenes=118` for 17MPV describes 118 stored monthly
positions. It does not mean 118 original burst images. The inspected catalogue
structure is consistent with store-based resynchronization. This code evidence
does not prove when or where historical source records were discarded.

Likewise, `n_bursts=20` in the 17MPV store is not its multi-year acquisition count.
The writer notes that this group attribute can change between processing batches;
it should not be assumed to be the lifetime union of all burst footprints.
Multiplying it by months or a nominal revisit rate cannot recover the actual total.
Summing `n_obs` would count pixel observations, not unique burst acquisitions.

## Consequences for the manuscript table

The intended compact table can omit `Store tokens`, `Months/group`, `Last month`
and `Unique burst footprints`. The requested `Burst acquisitions` column needs
actual source identities. There is currently no supported value to insert.

The existing, independently checked totals remain 31 unique MGRS tiles, 40 used
tile--track groups and 4,001 polygon--track trajectories. None is a raw-image count.
Avoid summing the per-family tile counts as national unique tiles because a tile
may occur in several track families.

## What would establish the requested count

Recover one of the following from the original production archive or a backup:

1. The pre-resynchronization per-month catalogue containing populated `opera_ids`.
2. A raw burst input manifest with product ID, burst ID, acquisition time,
   tile/track group, and retained/failed status.
3. Per-month source manifests recording inputs actually admitted to compositing.

Then match those records to the 40 used tile--track groups, distinguish source
context (2016-01 through 2026-06) from the reporting period (2017--2025), and
deduplicate by burst ID and acquisition time. VV/VH bands and copies in multiple
MGRS tiles must not multiply the count. Resolve grouped orbit labels using the
actual source identities. Confirm processing/admission status before reporting
the result as inputs used.

A new ASF search may provide acquisition availability, but it cannot by itself
prove which historical products were downloaded successfully and admitted to this
particular frozen archive. Such a reconstruction must be labelled accordingly.

## Reproduction and artifacts

Script: `scripts/audit_s1_burst_provenance.py`.

```powershell
python scripts/audit_s1_burst_provenance.py --output results/data_inventory/burst_provenance_NEW_RUN
```

Use a new output directory for each run. Source data are read-only.

This run: `results/data_inventory/burst_provenance_20260905/`:

- `metadata_inventory.csv`: metadata paths, fields and source-ID availability.
- `log_inventory.csv`: log coverage and identifier-mention counts.
- `log_mentions_NOT_verified_inputs.csv`: sanitized burst/time mentions only.
- `audit.json`: result and explicit null formal acquisition count.
- `COMPLETE.json`: completion status and output hashes; `formal_count_available=false`.

No signed download URLs or raw log messages are exported. This report does not
claim that the missing provenance is absent from an uninspected remote server.
