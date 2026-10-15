"""Point and grouped-table detection without notebook global variables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import lr_cusum


@dataclass(frozen=True)
class DetectionConfig:
    bands: tuple[str, ...] = ("VV_dB", "VH_dB")
    window_length: int = 36
    step: int = 12
    confidence_threshold: float = .95
    bootstrap_iterations: int = 800
    seed: int = 20
    start_year: int | None = 2017
    end_year: int | None = 2025
    minimum_observations: int = 12


def stable_seed(identifier, base_seed):
    # Keep derived seeds in a portable signed 32-bit range. A SHA-256 prefix
    # plus the experiment seed can otherwise overflow the Numba RNG boundary.
    modulus = 2**31 - 1
    try:
        value = int(identifier) + int(base_seed)
    except (TypeError, ValueError):
        import hashlib
        digest = hashlib.sha256(str(identifier).encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "little") + int(base_seed)
    return value % modulus


def detect_group(frame: pd.DataFrame, *, id_column="ID", time_column="time",
                 config=DetectionConfig()) -> pd.DataFrame:
    """Detect changes independently for each configured band in one object."""
    if frame.empty:
        return pd.DataFrame()
    if id_column not in frame or time_column not in frame:
        raise KeyError(f"required columns: {id_column}, {time_column}")
    ordered = frame.sort_values(time_column).reset_index(drop=True)
    identifier = ordered[id_column].iloc[0]
    dates = pd.to_datetime(ordered[time_column]).to_numpy()
    rows = []
    for band in config.bands:
        if band not in ordered:
            continue
        events = lr_cusum(
            ordered[band].to_numpy(np.float64), dates,
            window_length=config.window_length, step=config.step,
            confidence_threshold=config.confidence_threshold,
            bootstrap_iterations=config.bootstrap_iterations,
            seed=stable_seed(identifier, config.seed), start_year=config.start_year,
            end_year=config.end_year, minimum_observations=config.minimum_observations,
        )
        for event in events:
            rows.append({
                id_column: identifier, "Band": band, **event.as_dict(),
                "window_length": config.window_length, "step": config.step,
            })
    return pd.DataFrame(rows)


def detect_table(frame: pd.DataFrame, *, id_column="ID", time_column="time",
                 config=DetectionConfig()) -> pd.DataFrame:
    """Run LR-CuSum for every object in a long time-series table."""
    outputs = [
        detect_group(group, id_column=id_column, time_column=time_column, config=config)
        for _, group in frame.groupby(id_column, sort=True)
    ]
    outputs = [output for output in outputs if not output.empty]
    if not outputs:
        return pd.DataFrame(columns=[id_column, "Band", "change_time"])
    return pd.concat(outputs, ignore_index=True).sort_values(
        [id_column, "Band", "change_time"]
    ).reset_index(drop=True)
