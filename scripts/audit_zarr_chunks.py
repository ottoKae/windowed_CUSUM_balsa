#!/usr/bin/env python3
"""Report unreadable S1-GRiTS chunks touched by one polygon-track cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("associations", type=Path)
    parser.add_argument("track_key")
    parser.add_argument("--store", type=Path, help="Optional alternate copy of the same Zarr store.")
    args = parser.parse_args()

    associations = pd.read_parquet(args.associations)
    local = associations.loc[associations.track_key.eq(args.track_key)]
    if local.empty:
        raise SystemExit(f"unknown track_key: {args.track_key}")
    store = args.store or Path(local.dynamic_zarr_path.iloc[0])
    group = zarr.open_group(str(store), mode="r")
    times = pd.DatetimeIndex(np.asarray(group["time"][:]))
    chunk_height, chunk_width = group["VV_dB"].chunks[-2:]
    chunks = sorted(
        map(
            tuple,
            local[["chunk_y", "chunk_x"]].drop_duplicates().to_numpy(dtype=int),
        )
    )
    failures = []
    for chunk_y, chunk_x in chunks:
        y0, x0 = chunk_y * chunk_height, chunk_x * chunk_width
        y1 = min(y0 + chunk_height, group["VV_dB"].shape[1])
        x1 = min(x0 + chunk_width, group["VV_dB"].shape[2])
        for source_index, timestamp in enumerate(times):
            for band in ("n_obs", "VV_dB", "VH_dB"):
                try:
                    np.asarray(group[band][source_index, y0:y1, x0:x1])
                except Exception as error:  # preserve the store-level decoder message
                    failures.append(
                        {
                            "track_key": args.track_key,
                            "store": str(store),
                            "band": band,
                            "source_index": source_index,
                            "month": str(timestamp.to_period("M")),
                            "chunk_y": chunk_y,
                            "chunk_x": chunk_x,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    report = pd.DataFrame(failures)
    if report.empty:
        print(f"OK track={args.track_key} chunks={len(chunks)} months={len(times)}")
        return
    print(report.to_csv(index=False), end="")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
