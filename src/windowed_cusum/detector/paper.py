"""Shared N1 configuration and candidate primitives.

The first-post detector adapter supplies the current W36/D12/S12 settings.
The defaults here also support historical experiments; this module alone is
not the current end-to-end detector, TDA and STP pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from windowed_cusum.detector.component_ablation import ComponentAblationConfig, ComponentLedgers, FORWARD_VARIANTS, REMOVAL_VARIANTS, build_component_ledgers, derive_variant_events
from windowed_cusum.detector.core import _calendar_align
from windowed_cusum.detector.detection import stable_seed
from windowed_cusum.detector.temporal_nulls import NullModelConfig, iid_permutation_score


@dataclass(frozen=True)
class PaperDetectionConfig:
    """Parameters frozen for the manuscript-facing N1 detector."""

    bands: tuple[str, ...] = ("VV_dB", "VH_dB")
    window_length: int = 48
    step: int = 12
    minimum_observations: int = 12
    alpha: float = 0.025
    permutations: int = 3_999
    refine_low: float = 0.015
    refine_high: float = 0.035
    refined_permutations: int = 19_999
    candidate_seed: int = 20260820
    start_year: int | None = 2017
    end_year: int | None = 2025

    def __post_init__(self) -> None:
        if self.window_length < 3 or self.step < 1:
            raise ValueError("window_length must be >= 3 and step must be >= 1")
        if self.window_length <= 2 * self.step:
            raise ValueError("window_length must be greater than twice the step")
        if self.minimum_observations < 3:
            raise ValueError("minimum_observations must be >= 3")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.permutations < 1 or self.refined_permutations < 1:
            raise ValueError("permutation counts must be positive")
        if not 0.0 <= self.refine_low <= self.refine_high <= 1.0:
            raise ValueError("refinement bounds must lie in [0, 1]")
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must not exceed end_year")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PaperDetectionConfig":
        local = dict(values)
        if "bands" in local:
            local["bands"] = tuple(local["bands"])
        return cls(**local)


def load_paper_config(path: str | Path) -> PaperDetectionConfig:
    """Load the ``detector`` section of a paper protocol JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    detector = payload.get("detector", payload)
    if not isinstance(detector, dict):
        raise ValueError("paper configuration must contain a detector object")
    return PaperDetectionConfig.from_mapping(detector)


def _validated_month_index(dates) -> pd.DatetimeIndex:
    dates = pd.to_datetime(np.asarray(dates), errors="raise")
    months = pd.PeriodIndex(dates, freq="M")
    if months.has_duplicates:
        raise ValueError("each trajectory must contain at most one row per calendar month")
    if not months.is_monotonic_increasing:
        raise ValueError("trajectory months must be sorted")
    if len(months):
        expected = pd.period_range(months[0], months[-1], freq="M")
        if not months.equals(expected):
            raise ValueError(
                "missing months must be explicit rows with NaN observations; "
                "do not omit calendar months"
            )
    return months.to_timestamp()


def _validated_months(frame: pd.DataFrame, time_column: str) -> pd.DatetimeIndex:
    return _validated_month_index(frame[time_column])


_OUTPUT_COLUMNS = [
    "Band",
    "change_time",
    "window_start",
    "window_end",
    "decision_start",
    "decision_end",
    "change_index",
    "cusum_amplitude",
    "pre_mean",
    "post_mean",
    "post_minus_pre",
    "null_p_value",
    "exceedance_count",
    "permutations",
    "near_threshold_refined",
    "candidate_id",
]


def scan_series_n1(
    series,
    dates,
    *,
    identifier: object,
    band: str,
    config: PaperDetectionConfig = PaperDetectionConfig(),
    all_component_candidates: bool = False,
    metadata: dict[str, object] | None = None,
    series_seed: int | None = None,
) -> ComponentLedgers:
    """Return N1-scored master ledgers for one monthly trajectory.

    Missing observations remain ``NaN`` at their calendar positions.  The
    reported change month is the first sample in the post-change segment.
    By default only candidates eligible for the final F4 detector are scored,
    preserving the cost of normal inference.  ``all_component_candidates``
    also scores valid global and sliding candidates required by F0--F4 and
    removal-control evaluation.
    """

    validated_dates = _validated_month_index(dates)
    values, aligned_dates = _calendar_align(series, validated_dates)
    values = np.asarray(values, dtype=np.float64)
    aligned_dates = pd.DatetimeIndex(aligned_dates)
    series_key = f"{identifier}:{band}"
    # Published pseudonyms must not change the frozen permutation stream.
    if series_seed is None:
        series_seed = stable_seed(series_key, config.candidate_seed)
    elif not isinstance(series_seed, (int, np.integer)) or series_seed < 0:
        raise ValueError("series_seed must be a non-negative integer")
    series_seed = int(series_seed)
    scan_config = ComponentAblationConfig(
        window_length=config.window_length,
        step=config.step,
        confidence_threshold=1.0,
        bootstrap_iterations=1,
        minimum_observations=config.minimum_observations,
        seed=config.candidate_seed,
        start_year=config.start_year,
        end_year=config.end_year,
    )
    candidate_metadata = dict(metadata or {})
    candidate_metadata["Band"] = band
    ledgers = build_component_ledgers(
        values,
        aligned_dates,
        config=scan_config,
        series_key=series_key,
        series_seed=series_seed,
        metadata=candidate_metadata,
    )

    base = NullModelConfig(
        method="N1_IID_P025",
        replicates=config.permutations,
        alpha=config.alpha,
    )
    refined = NullModelConfig(
        method="N1_IID_P025",
        replicates=config.refined_permutations,
        alpha=config.alpha,
    )
    frames = (ledgers.global_candidates, ledgers.sliding_candidates)
    for ledger in frames:
        ledger["passes_permutation"] = False
        ledger["null_p_value"] = np.nan
        ledger["exceedance_count"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
        ledger["permutations"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
        ledger["near_threshold_refined"] = False
        eligible = (
            ledger["passes_window_calendar"].fillna(False)
            & ledger["passes_reporting_calendar"].fillna(False)
            & ledger["candidate_valid"].fillna(False)
        )
        if not all_component_candidates:
            eligible &= ledger["candidate_scope"].eq("sliding")
            eligible &= ledger["passes_central"].fillna(False)
            eligible &= ledger["passes_decrease"].fillna(False)
        for index, candidate in ledger.loc[eligible].iterrows():
            window = values[
                int(candidate["window_start_index"]) : int(candidate["window_end_index"])
            ]
            seed = int(candidate["permutation_seed"])
            score = iid_permutation_score(
                window,
                float(candidate["cusum_amplitude"]),
                config=base,
                seed=seed,
            )
            refine = config.refine_low <= score.null_p_value <= config.refine_high
            if refine:
                score = iid_permutation_score(
                    window,
                    float(candidate["cusum_amplitude"]),
                    config=refined,
                    seed=seed,
                )
            ledger.at[index, "passes_permutation"] = score.admitted
            ledger.at[index, "null_p_value"] = score.null_p_value
            ledger.at[index, "exceedance_count"] = score.exceedance_count
            ledger.at[index, "permutations"] = score.replicates
            ledger.at[index, "near_threshold_refined"] = refine

    # Keep the persisted Boolean audit columns synchronized with the N1 score.
    for definition in FORWARD_VARIANTS + REMOVAL_VARIANTS:
        source = (
            ledgers.global_candidates
            if definition.source_ledger == "global"
            else ledgers.sliding_candidates
        )
        accepted = derive_variant_events(
            ledgers.global_candidates,
            ledgers.sliding_candidates,
            definition.variant_id,
            reporting_only=False,
        )
        source[f"accepted_{definition.variant_id.lower()}"] = source[
            "candidate_id"
        ].isin(set(accepted["candidate_id"]))
    return ledgers


def detect_series_n1(
    series,
    dates,
    *,
    identifier: object,
    band: str,
    config: PaperDetectionConfig = PaperDetectionConfig(),
    series_seed: int | None = None,
) -> pd.DataFrame:
    """Return accepted N1 candidates for one band of one monthly trajectory."""

    ledger = scan_series_n1(
        series,
        dates,
        identifier=identifier,
        band=band,
        config=config,
        series_seed=series_seed,
    ).sliding_candidates
    if ledger.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    selected = derive_variant_events(
        pd.DataFrame(), ledger, "F4", reporting_only=True
    )
    if selected.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return selected.loc[:, _OUTPUT_COLUMNS].reset_index(drop=True)


def detect_table_n1(
    frame: pd.DataFrame,
    *,
    id_column: str = "ID",
    time_column: str = "time",
    config: PaperDetectionConfig = PaperDetectionConfig(),
) -> pd.DataFrame:
    """Run the frozen N1 candidate detector for every trajectory in a table."""

    required = {id_column, time_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    if frame[id_column].isna().any():
        raise ValueError(f"{id_column} must not contain missing values")

    outputs: list[pd.DataFrame] = []
    for identifier, group in frame.groupby(id_column, sort=True):
        ordered = group.sort_values(time_column, kind="mergesort").reset_index(drop=True)
        dates = _validated_months(ordered, time_column)
        for band in config.bands:
            if band not in ordered:
                continue
            detected = detect_series_n1(
                ordered[band].to_numpy(np.float64),
                dates,
                identifier=identifier,
                band=band,
                config=config,
            )
            if not detected.empty:
                detected.insert(0, id_column, identifier)
                outputs.append(detected)
    if not outputs:
        return pd.DataFrame(columns=[id_column, *_OUTPUT_COLUMNS])
    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values([id_column, "Band", "change_time"], kind="mergesort")
        .reset_index(drop=True)
    )


__all__ = [
    "PaperDetectionConfig",
    "detect_series_n1",
    "detect_table_n1",
    "load_paper_config",
    "scan_series_n1",
]
