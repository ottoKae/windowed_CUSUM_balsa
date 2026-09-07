#!/usr/bin/env python3
"""Compare a raw-Zarr cache rebuild with the frozen paper pixel cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuilt", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    detail_rows = []
    for rebuilt_array in sorted(args.rebuilt.glob("*.npz")):
        track = rebuilt_array.stem
        frozen_array = args.frozen / rebuilt_array.name
        rebuilt_pixels = pd.read_parquet(args.rebuilt / f"{track}.pixels.parquet")
        frozen_pixels = pd.read_parquet(args.frozen / f"{track}.pixels.parquet")
        left = rebuilt_pixels.sort_values("cache_index").reset_index(drop=True)
        right = frozen_pixels.sort_values("cache_index").reset_index(drop=True)
        pixel_ids_equal = left.pixel_track_id.equals(right.pixel_track_id)
        rebuilt = np.load(rebuilt_array)
        frozen = np.load(frozen_array)
        arrays_equal = {}
        for name in ("values", "valid", "n_obs", "calendar"):
            arrays_equal[name] = bool(
                rebuilt[name].shape == frozen[name].shape
                and np.array_equal(rebuilt[name], frozen[name], equal_nan=True)
            )
        rebuilt_values = rebuilt["values"]
        frozen_values = frozen["values"]
        finite_pair = np.isfinite(rebuilt_values) & np.isfinite(frozen_values)
        numeric_difference = finite_pair & ~np.isclose(
            rebuilt_values, frozen_values, rtol=0.0, atol=0.0
        )
        missingness_difference = np.isfinite(rebuilt_values) != np.isfinite(frozen_values)
        differing_positions = numeric_difference | missingness_difference
        differing_month_indexes = np.flatnonzero(differing_positions.any(axis=(0, 1)))
        differing_months = [
            str(pd.Timestamp(rebuilt["calendar"][index]).to_period("M"))
            for index in differing_month_indexes
        ]
        for band_index, band in enumerate(("VV_dB", "VH_dB")):
            for month_index in differing_month_indexes:
                local_numeric = numeric_difference[:, band_index, month_index]
                local_missing = missingness_difference[:, band_index, month_index]
                if local_numeric.any() or local_missing.any():
                    detail_rows.append(
                        {
                            "track_key": track,
                            "band": band,
                            "month": str(pd.Timestamp(rebuilt["calendar"][month_index]).to_period("M")),
                            "different_finite_cells": int(local_numeric.sum()),
                            "different_missingness_cells": int(local_missing.sum()),
                        }
                    )
        absolute = np.abs(rebuilt_values[finite_pair] - frozen_values[finite_pair])
        rows.append(
            {
                "track_key": track,
                "pixels_rebuilt": len(left),
                "pixels_frozen": len(right),
                "pixel_ids_equal": pixel_ids_equal,
                **{f"{name}_equal": value for name, value in arrays_equal.items()},
                "different_value_cells": int(differing_positions.sum()),
                "different_finite_cells": int(numeric_difference.sum()),
                "different_missingness_cells": int(missingness_difference.sum()),
                "maximum_absolute_difference": float(absolute.max()) if len(absolute) else 0.0,
                "differing_months": ";".join(differing_months),
                "rebuilt_sha256": sha256(rebuilt_array),
                "frozen_sha256": sha256(frozen_array),
            }
        )
    report = pd.DataFrame(rows)
    comparison_columns = [
        "pixel_ids_equal", "values_equal", "valid_equal", "n_obs_equal", "calendar_equal"
    ]
    complete = (
        len(report) == 40
        and all(report[column].all() for column in comparison_columns)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output / "track_cache_equivalence.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(args.output / "difference_detail.csv", index=False)
    receipt = {
        "status": "complete" if complete else "difference_detected",
        "tracks": len(report),
        "exact_tracks": int(report[comparison_columns].all(axis=1).sum()),
        "comparison": "pixel IDs and values/valid/n_obs/calendar arrays; NaNs compare equal",
    }
    (args.output / "COMPLETE.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if not complete:
        print(report.loc[~report[comparison_columns].all(axis=1)].to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
