"""Utilities for dated polygon-aware S1-GRiTS sample collections."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_CALENDAR = pd.period_range("2016-01", "2026-06", freq="M")


def temporal_reference_type(planting_date) -> str:
    """Describe how an official planting date relates to the S1 calendar."""
    if planting_date is None or pd.isna(planting_date):
        return "official_date_unavailable"
    year = pd.Timestamp(planting_date).year
    if year < 2016:
        return "pre_s1_left_censored"
    if year <= 2026:
        return "within_s1_reference"
    return "post_s1_right_censored"


def _safe_nan_stat(function, values: np.ndarray, axis: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(values, axis=axis)


def summarize_track_cache(
    cache_root: str | Path,
    track_key: str,
    *,
    inventory_release: str,
) -> pd.DataFrame:
    """Create one robust summary row per polygon-track-month.

    Missing observations remain NaN. A date or class label is intentionally not
    stored in the NPZ and must be joined from polygon metadata by ``sample_id``.
    """
    root = Path(cache_root)
    associations = pd.read_parquet(root / "cache" / f"{track_key}.associations.parquet")
    pixels = pd.read_parquet(root / "cache" / f"{track_key}.pixels.parquet")
    with np.load(root / "cache" / f"{track_key}.npz", allow_pickle=False) as cube:
        values = np.asarray(cube["values"], dtype=float)
        valid = np.asarray(cube["valid"], dtype=bool)
        n_obs = np.asarray(cube["n_obs"], dtype=float)
        calendar = pd.DatetimeIndex(cube["calendar"]).to_period("M")
    if not calendar.equals(LOCKED_CALENDAR):
        raise ValueError(f"Unexpected calendar for {track_key}: {calendar[0]}/{calendar[-1]}")
    mapping = pixels.set_index("pixel_track_id")["cache_index"]
    rows: list[pd.DataFrame] = []
    for polygon_track_id, group in associations.groupby("polygon_track_id", sort=False):
        indexes = mapping.loc[group["pixel_track_id"]].to_numpy(dtype="int64")
        local_values = values[indexes].copy()
        local_valid = valid[indexes]
        local_values = np.where(local_valid[:, None, :], local_values, np.nan)
        median = _safe_nan_stat(np.nanmedian, local_values, axis=0)
        q25 = _safe_nan_stat(lambda x, axis: np.nanpercentile(x, 25, axis=axis), local_values, axis=0)
        q75 = _safe_nan_stat(lambda x, axis: np.nanpercentile(x, 75, axis=axis), local_values, axis=0)
        local_obs = np.where(local_valid, n_obs[indexes], np.nan)
        obs_median = _safe_nan_stat(np.nanmedian, local_obs, axis=0)
        valid_pixels = local_valid.sum(axis=0)
        first = group.iloc[0]
        rows.append(
            pd.DataFrame(
                {
                    "inventory_release": inventory_release,
                    "polygon_track_id": str(polygon_track_id),
                    "sample_id": str(first["sample_id"]),
                    "polygon_id": str(first["polygon_id"]),
                    "role": str(first["role"]),
                    "spatial_block_id": str(first["spatial_block_id"]),
                    "tile_id": str(first["tile_id"]),
                    "track_key": str(track_key),
                    "month": calendar.astype(str),
                    "vv_median_db": median[0],
                    "vh_median_db": median[1],
                    "vv_iqr_db": q75[0] - q25[0],
                    "vh_iqr_db": q75[1] - q25[1],
                    "valid_pixels": valid_pixels.astype("int32"),
                    "extraction_pixels": int(len(indexes)),
                    "valid_fraction": valid_pixels / len(indexes),
                    "n_obs_median": obs_median,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_cache_root(cache_root: str | Path, *, inventory_release: str) -> pd.DataFrame:
    root = Path(cache_root)
    manifest = pd.read_parquet(root / "cache_manifest.parquet")
    frames = [
        summarize_track_cache(root, str(track_key), inventory_release=inventory_release)
        for track_key in manifest["track_key"].astype(str)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_polygon_track_series(
    monthly_summary: str | Path,
    sample_id: str,
    *,
    track_key: str | None = None,
) -> pd.DataFrame:
    """Load a dated sample's fast polygon-level monthly series."""
    frame = pd.read_parquet(monthly_summary, filters=[[('sample_id', '==', sample_id)]])
    if track_key is not None:
        frame = frame.loc[frame["track_key"].astype(str).eq(str(track_key))]
    return frame.sort_values(["track_key", "month"]).reset_index(drop=True)

