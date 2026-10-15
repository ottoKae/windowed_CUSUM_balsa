"""Rotation-period and lifecycle summaries."""
from __future__ import annotations

import pandas as pd


def rotation_periods(events: pd.DataFrame, *, id_column="polygon_id",
                     time_column="change_time", band_column="Band") -> pd.DataFrame:
    """Calculate intervals in months between consecutive detected events."""
    required = {id_column, time_column, band_column}
    if not required.issubset(events.columns):
        raise KeyError(f"missing columns: {sorted(required - set(events.columns))}")
    frame = events.copy(); frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values([id_column, band_column, time_column])
    frame["previous_change_time"] = frame.groupby([id_column, band_column])[time_column].shift()
    current_month = frame[time_column].dt.year * 12 + frame[time_column].dt.month
    previous_month = (
        frame["previous_change_time"].dt.year * 12
        + frame["previous_change_time"].dt.month
    )
    frame["rotation_months"] = (current_month - previous_month).astype("Float64")
    frame["rotation_years"] = frame.rotation_months / 12
    return frame.loc[frame.previous_change_time.notna()].reset_index(drop=True)


def lifecycle_summary(events: pd.DataFrame, all_ids, *, id_column="polygon_id") -> pd.DataFrame:
    """Return event/generation counts including polygons with no detection."""
    counts = events.groupby(id_column).size().rename("detected_events")
    output = pd.DataFrame({id_column: pd.Series(list(all_ids), dtype="object")})
    output = output.join(counts, on=id_column)
    output["detected_events"] = output.detected_events.fillna(0).astype(int)
    output["generation_count"] = output.detected_events + 1
    return output
