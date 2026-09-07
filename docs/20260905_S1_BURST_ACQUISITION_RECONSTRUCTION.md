# Burst-acquisition counts reconstructed from the MGRS lookup and ASF

The requested spatial/temporal reconstruction is feasible. On 2026-09-05, the
query returned **71,134 unique VV+VH RTC burst acquisitions for 2016-01 through
2026-06**, including **64,345 for the reporting period 2017-01 through 2025-12**.
These describe available observations for the burst footprints overlapping the
frozen tile/track cohort. They do not certify historical processing admission.

| Primary orbit family | MGRS tiles | Burst acquisitions, 2016-01 to 2026-06 | Burst acquisitions, 2017-2025 | Polygon-track trajectories |
|---|---:|---:|---:|---:|
| 18 | 19 | 36,006 | 32,924 | 3,467 |
| 47 | 2 | 3,664 | 3,297 | 16 |
| 91 | 5 | 7,648 | 6,830 | 61 |
| 120 | 14 | 23,816 | 21,294 | 457 |
| National union | 31 | **71,134** | **64,345** | **4,001** |

The table deliberately omits a unique-footprint column, store tokens and monthly
group counts. Tile counts overlap between orbit families; their national union
is 31. Families preserve the archive's grouping of adjacent relative orbits
(18/19, 47/48, 120/121) without counting a burst in two families.

## Calculation

1. Read `data_pipeline/S1-GRiTS/src/s1grits/data/mgrs_burst_lookup_table.parquet`.
   This file has 1,000,609 rows and seven columns describing spatial membership
   and orbit groups. It does not contain acquisition dates.
2. Match the 40 groups in the frozen `polygon_pixels/cache_manifest.parquet`
   by MGRS tile, ascending direction and the relative-orbit IDs in each group.
   This yields 524 tile/group/burst memberships and 258 distinct burst IDs.
3. Query the public ASF RTC metadata for each distinct burst, from
   `2016-01-01T00:00:00Z` through `2026-06-30T23:59:59Z`.
   Use the same `operaBurstID`, `processingLevel=RTC`, `start`, and `end`
   query parameters used by S1-GRiTS's `asf_tiles.py`.
4. Of 78,733 returned product records, retain 75,003 with VV+VH polarization.
   Exclude 3,730 other-polarization records. Deduplicate on burst ID and actual
   acquisition time, removing 3,869 repeated product records and retaining 71,134
   distinct acquisitions. VV and VH are not counted separately.
5. Intersect the acquisition months with the actual native Zarr time coordinates:
   4,432 distinct group-months across the 40 used groups. Both national totals
   remain unchanged after this restriction. The equality is an observed result,
   not an assumption that every month has the same acquisition count.
6. For each orbit family and month, count unique burst/time pairs. Sum monthly
   counts to obtain the family and national totals, deduplicating overlaps
   between MGRS tiles before summation.

Equivalently, sum `n(burst, month)` across distinct selected bursts and months.
Using the actual per-burst counts is preferable to multiplying all footprints
by a shared track-level monthly count: coverage can differ among bursts and dates.

## Verification and interpretation

- All 258 burst queries completed; two returned no RTC products in the queried
  time span. Empty responses are retained as query evidence, not replaced by
  nominal revisit counts.
- Unique burst/time keys have no duplicates in the final acquisition table.
- Four-family totals, monthly counts and national unique acquisitions reconcile.
- No selected burst belongs to more than one reporting orbit family.
- All output checksums passed verification.
- Query products and processing versions reflect ASF availability at query time.
  Native monthly timestamps are composite timestamps, not a list of all original
  acquisition dates. Matching a stored month does not prove every returned source
  product was successfully read or used in that composite.

The preceding provenance audit correctly found no preserved original input
manifest locally. It does not prevent this independently reconstructed
availability count. Avoid labelling it `number of images actually processed`.

Suggested short table note:

> Counts represent unique VV+VH RTC acquisitions available for lookup-matched
> bursts in the archive's stored months; historical input admission was not audited.

## Reproduction

Script: `scripts/reconstruct_burst_acquisition_counts.py`.

```powershell
python scripts/reconstruct_burst_acquisition_counts.py --output results/data_inventory/asf_burst_counts_20260905
```

The existing directory resumes from metadata caches if source hashes and query
parameters match. Use a new directory to obtain a new ASF snapshot. No raster
imagery is downloaded, and original Zarr/catalogue files remain unchanged.

Artifacts under `results/data_inventory/asf_burst_counts_20260905/`:

- `query_spec.json`: exact time range and input hashes.
- `tile_track_burst_mapping.csv`: spatial/orbit membership.
- `archive_group_months.csv`: actual stored monthly support.
- `asf_metadata/`: 258 query receipts and product metadata, with no signed URLs.
- `asf_products.parquet`: all returned product records and polarization flag.
- `unique_available_acquisitions.parquet`: unique dual-polarization acquisitions.
- `available_in_stored_group_months.parquet`: acquisition/group-month matches.
- `orbit_month_counts.csv`, `orbit_family_counts.csv`: monthly and family counts.
- `SUMMARY.json`, `COMPLETE.json`: scope, status and hashes.

The live manuscript table has not been changed by this reconstruction.
