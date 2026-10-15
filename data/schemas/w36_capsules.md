# W36 replay inputs

`units.parquet`: unique `unit_id`, unique `pixel_track_id`, integer `series_index` into the array. Optional `rng_offset_vv`, `rng_offset_vh` preserve numerical RNG identity after deidentification; each actual series seed is `(offset + candidate_seed) % (2**31 - 1)`.

`membership.parquet`: unique `(sample_id, unit_id)` pairs, all fixed master pixels, not just responding pixels. Every included stand must be complete. The capsule contains frozen centre-in-polygon assignments; this temporal replay does not recompute geographic membership.

`series.npz`: `values` of shape `(pixels, 2, months)`, VV then VH, float64 dB; `calendar` is a full monthly datetime array. Missing values stay NaN. The W36 capsule has 126 months from 2016-01 through 2026-06.

`manifest.json`: mapping of each required input filename to SHA256. The replay verifies these before computation.

Expected outputs and pilot selections are checks only, never inputs to detection or window selection. Exact geometry and original identity bridges are excluded from public capsules.
