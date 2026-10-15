"""Polygon time-series aggregation utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def polygon_id_from_path(path) -> str:
    return Path(path).stem.split("_")[0]


def aggregate_polygon_timeseries(frame: pd.DataFrame, band: str, method="median",
                                 time_column="time") -> pd.DataFrame:
    """Aggregate all valid polygon pixels to one value per observation time."""
    if band not in frame or time_column not in frame:
        raise KeyError(f"required columns: {time_column}, {band}")
    local = frame[[time_column, band]].copy()
    local[time_column] = pd.to_datetime(local[time_column])
    local[band] = pd.to_numeric(local[band], errors="coerce")
    grouped = local.groupby(time_column)[band]
    if method == "median":
        values = grouped.median()
    elif method == "mean":
        values = grouped.mean()
    elif method == "mode":
        values = grouped.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    else:
        raise ValueError("method must be median, mean, or mode")
    return values.rename(band).reset_index().sort_values(time_column)

