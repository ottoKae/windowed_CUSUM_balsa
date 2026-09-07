#!/usr/bin/env python3
"""Extract one frozen polygon-track index from an explicitly selected Zarr copy."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lr_cusum.polygon_workflow import extract_track_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("associations", type=Path)
    parser.add_argument("track_key")
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calendar-start", default="2016-01")
    parser.add_argument("--calendar-end", default="2026-06")
    args = parser.parse_args()

    associations = pd.read_parquet(args.associations)
    local = associations.loc[associations.track_key.eq(args.track_key)].copy()
    if local.empty:
        raise SystemExit(f"unknown track_key: {args.track_key}")
    local["dynamic_zarr_path"] = str(args.store.resolve())
    calendar = pd.period_range(args.calendar_start, args.calendar_end, freq="M")
    report = extract_track_cache(local, args.output, calendar=calendar)
    print(report)


if __name__ == "__main__":
    main()
