"""Recurrent-disturbance interval utilities for frozen N1 event ledgers.

The functions in this module intentionally do not run or alter the detector.
They construct an adjacent Polygon-first VV chronology, use Polygon-first VH
only as endpoint confirmation, and preserve complete spatial blocks during
uncertainty estimation.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


INTERVAL_BINS = (-np.inf, 12, 24, 48, np.inf)
INTERVAL_BIN_LABELS = ("<12", "12--23", "24--47", ">=48")


def month_number(values: pd.Series) -> np.ndarray:
    """Convert datetimes to monotonically increasing integer calendar months."""
    dates = pd.to_datetime(values)
    return (dates.dt.year * 12 + dates.dt.month).to_numpy(dtype=np.int64)


def select_polygon_first_vv(
    events: pd.DataFrame,
    *,
    start: str = "2017-01",
    end: str = "2025-12",
) -> pd.DataFrame:
    """Select the frozen Polygon-first VV chronology and validate uniqueness."""
    required = {"sample_id", "polygon_id", "Band", "route", "change_time"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Primary event ledger is missing columns: {missing}")

    result = events.loc[
        events["route"].eq("polygon_first") & events["Band"].eq("VV_dB")
    ].copy()
    result["change_time"] = (
        pd.to_datetime(result["change_time"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if result[["sample_id", "polygon_id", "change_time"]].isna().any().any():
        raise ValueError("Primary VV events contain missing identifiers or timestamps")

    first = pd.Period(start, freq="M").to_timestamp()
    last = pd.Period(end, freq="M").to_timestamp()
    result = result.loc[result.change_time.between(first, last)].copy()
    duplicates = result.duplicated(["sample_id", "change_time"], keep=False)
    if duplicates.any():
        example = result.loc[duplicates, ["sample_id", "change_time"]].head()
        raise ValueError(
            "Primary VV chronology contains duplicate polygon-month events:\n"
            + example.to_string(index=False)
        )
    return result.sort_values(
        ["sample_id", "change_time"], kind="mergesort"
    ).reset_index(drop=True)


def match_polygon_polarizations(
    ground_events: pd.DataFrame,
    *,
    tolerance_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Greedily match Polygon-first VV and VH events one-to-one by month.

    Candidate pairs are ordered by absolute lag and then by chronological row
    index, reproducing the frozen ``annotate_vv_with_vh`` rule.  The returned
    tuple contains annotated VV events, unmatched VH events, and matched pairs.
    """
    if tolerance_months < 0:
        raise ValueError("tolerance_months must be non-negative")
    required = {"sample_id", "polygon_id", "Band", "route", "change_time"}
    missing = sorted(required - set(ground_events.columns))
    if missing:
        raise ValueError(f"Ground event ledger is missing columns: {missing}")

    local = ground_events.loc[ground_events.route.eq("polygon_first")].copy()
    local["change_time"] = (
        pd.to_datetime(local.change_time, errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if local.change_time.isna().any():
        raise ValueError("Ground event ledger contains invalid timestamps")

    vv_rows: list[dict] = []
    vh_only_rows: list[dict] = []
    match_rows: list[dict] = []
    for sample_id, group in local.groupby("sample_id", sort=True):
        vv = group.loc[group.Band.eq("VV_dB")].sort_values(
            "change_time", kind="mergesort"
        ).reset_index(drop=True)
        vh = group.loc[group.Band.eq("VH_dB")].sort_values(
            "change_time", kind="mergesort"
        ).reset_index(drop=True)
        vv_month = month_number(vv.change_time) if len(vv) else np.array([], dtype=int)
        vh_month = month_number(vh.change_time) if len(vh) else np.array([], dtype=int)
        candidates = sorted(
            (abs(int(vm - hm)), vi, hi)
            for vi, vm in enumerate(vv_month)
            for hi, hm in enumerate(vh_month)
            if abs(int(vm - hm)) <= tolerance_months
        )
        matches: dict[int, tuple[int, int]] = {}
        used_vv: set[int] = set()
        used_vh: set[int] = set()
        for absolute_lag, vv_index, vh_index in candidates:
            if vv_index in used_vv or vh_index in used_vh:
                continue
            used_vv.add(vv_index)
            used_vh.add(vh_index)
            matches[vv_index] = (vh_index, absolute_lag)

        for vv_index, item in vv.iterrows():
            output = item.to_dict()
            match = matches.get(vv_index)
            output["vh_supported"] = match is not None
            output["vh_delta_months"] = match[1] if match else pd.NA
            output["vh_signed_delta_months"] = (
                int(vh_month[match[0]] - vv_month[vv_index]) if match else pd.NA
            )
            output["vh_change_time"] = (
                vh.iloc[match[0]].change_time if match else pd.NaT
            )
            output["vh_track_support"] = (
                int(vh.iloc[match[0]].get("track_support", 0)) if match else 0
            )
            vv_rows.append(output)
            if match:
                vh_item = vh.iloc[match[0]]
                match_rows.append({
                    "sample_id": sample_id,
                    "polygon_id": item.polygon_id,
                    "vv_change_time": item.change_time,
                    "vh_change_time": vh_item.change_time,
                    "signed_delta_months": int(vh_month[match[0]] - vv_month[vv_index]),
                    "absolute_delta_months": int(match[1]),
                })

        for vh_index, item in vh.iterrows():
            if vh_index not in used_vh:
                vh_only_rows.append(item.to_dict())

    annotated = pd.DataFrame(vv_rows)
    vh_only = pd.DataFrame(vh_only_rows)
    matched = pd.DataFrame(match_rows)
    if len(annotated):
        annotated = annotated.sort_values(
            ["sample_id", "change_time"], kind="mergesort"
        ).reset_index(drop=True)
    if len(vh_only):
        vh_only = vh_only.sort_values(
            ["sample_id", "change_time"], kind="mergesort"
        ).reset_index(drop=True)
    return annotated, vh_only, matched


def build_adjacent_intervals(primary_events: pd.DataFrame) -> pd.DataFrame:
    """Build adjacent intervals before applying any VH-support restriction."""
    required = {"sample_id", "polygon_id", "change_time", "vh_supported"}
    missing = sorted(required - set(primary_events.columns))
    if missing:
        raise ValueError(f"Annotated primary events are missing columns: {missing}")

    events = primary_events.sort_values(
        ["sample_id", "change_time"], kind="mergesort"
    ).copy()
    events["event_ordinal"] = events.groupby("sample_id").cumcount() + 1
    grouped = events.groupby("sample_id", sort=False)
    events["previous_change_time"] = grouped.change_time.shift(1)
    events["previous_event_ordinal"] = grouped.event_ordinal.shift(1)
    events["previous_vh_supported"] = grouped.vh_supported.shift(1)
    for column in (
        "track_support",
        "vh_change_time",
        "vh_track_support",
        "vh_signed_delta_months",
    ):
        if column in events:
            events[f"previous_{column}"] = grouped[column].shift(1)

    intervals = events.loc[events.previous_change_time.notna()].copy()
    intervals["interval_months"] = (
        month_number(intervals.change_time)
        - month_number(intervals.previous_change_time)
    )
    if (intervals.interval_months <= 0).any():
        raise ValueError("Every adjacent interval must have positive duration")
    intervals["both_endpoints_vh_supported"] = (
        intervals.previous_vh_supported.astype(bool)
        & intervals.vh_supported.astype(bool)
    )
    intervals["interval_bin"] = pd.cut(
        intervals.interval_months,
        bins=INTERVAL_BINS,
        labels=INTERVAL_BIN_LABELS,
        right=False,
    ).astype(str)
    intervals["interval_id"] = (
        intervals.sample_id.astype(str)
        + "|"
        + intervals.previous_change_time.dt.strftime("%Y-%m")
        + "|"
        + intervals.change_time.dt.strftime("%Y-%m")
    )
    return intervals.reset_index(drop=True)


def add_nearest_vv_distance(
    vh_only: pd.DataFrame,
    vv_events: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate each VH-only event with its nearest VV event in the polygon."""
    output = vh_only.copy()
    nearest_signed: list[object] = []
    nearest_time: list[object] = []
    vv_groups = {
        key: group.sort_values("change_time")
        for key, group in vv_events.groupby("sample_id", sort=False)
    }
    for item in output.itertuples(index=False):
        vv = vv_groups.get(item.sample_id)
        if vv is None or vv.empty:
            nearest_signed.append(pd.NA)
            nearest_time.append(pd.NaT)
            continue
        vh_month = int(pd.Timestamp(item.change_time).year * 12 + pd.Timestamp(item.change_time).month)
        vv_months = month_number(vv.change_time)
        signed = vv_months - vh_month
        index = int(np.argmin(np.abs(signed)))
        nearest_signed.append(int(signed[index]))
        nearest_time.append(pd.Timestamp(vv.iloc[index].change_time))
    output["nearest_vv_change_time"] = nearest_time
    output["nearest_vv_signed_months"] = nearest_signed
    output["nearest_vv_absolute_months"] = pd.to_numeric(
        output.nearest_vv_signed_months, errors="coerce"
    ).abs()
    return output


def assign_area_class(area_ha: pd.Series) -> pd.Series:
    """Apply the manuscript's frozen small/medium/large stand classes."""
    values = pd.to_numeric(area_ha, errors="coerce")
    result = pd.Series("missing", index=values.index, dtype="object")
    result.loc[values.le(1)] = "small (<=1 ha)"
    result.loc[values.gt(1) & values.le(5)] = "medium (>1--5 ha)"
    result.loc[values.gt(5)] = "large (>5 ha)"
    return result


def summarize_interval_cohort(
    intervals: pd.DataFrame,
    *,
    cohort: str,
    events: int,
) -> dict:
    """Return manuscript-facing descriptive statistics for one cohort."""
    values = intervals.interval_months.to_numpy(dtype=float)
    polygon_medians = intervals.groupby("sample_id").interval_months.median()
    result = {
        "cohort": cohort,
        "events": int(events),
        "polygons": int(intervals.sample_id.nunique()),
        "intervals": int(len(intervals)),
        "median_months": float(np.median(values)) if len(values) else np.nan,
        "q10_months": float(np.quantile(values, 0.10)) if len(values) else np.nan,
        "q25_months": float(np.quantile(values, 0.25)) if len(values) else np.nan,
        "q75_months": float(np.quantile(values, 0.75)) if len(values) else np.nan,
        "q90_months": float(np.quantile(values, 0.90)) if len(values) else np.nan,
        "polygon_weighted_median_months": (
            float(np.median(polygon_medians)) if len(polygon_medians) else np.nan
        ),
    }
    for label in INTERVAL_BIN_LABELS:
        result[f"proportion_{label}"] = (
            float(intervals.interval_bin.eq(label).mean()) if len(intervals) else np.nan
        )
    return result


def _metrics(values: np.ndarray, polygon_values: np.ndarray) -> dict[str, float]:
    result = {
        "median_months": float(np.median(values)) if len(values) else np.nan,
        "polygon_weighted_median_months": (
            float(np.median(polygon_values)) if len(polygon_values) else np.nan
        ),
    }
    for lower, upper, label in zip(
        INTERVAL_BINS[:-1], INTERVAL_BINS[1:], INTERVAL_BIN_LABELS
    ):
        result[f"proportion_{label}"] = (
            float(((values >= lower) & (values < upper)).mean()) if len(values) else np.nan
        )
    return result


def spatial_block_bootstrap(
    intervals: pd.DataFrame,
    *,
    replicates: int = 5_000,
    seed: int = 20260830,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired complete-spatial-block bootstrap for primary and supported cohorts."""
    required = {
        "sample_id",
        "spatial_block_id",
        "interval_months",
        "both_endpoints_vh_supported",
    }
    missing = sorted(required - set(intervals.columns))
    if missing:
        raise ValueError(f"Intervals are missing bootstrap columns: {missing}")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if intervals.spatial_block_id.isna().any():
        raise ValueError("Every interval must belong to a spatial block")
    if intervals.groupby("sample_id").spatial_block_id.nunique().max() != 1:
        raise ValueError("A polygon appears in more than one spatial block")

    blocks = np.array(sorted(intervals.spatial_block_id.astype(str).unique()))
    interval_arrays: dict[str, dict[str, np.ndarray]] = {}
    polygon_arrays: dict[str, dict[str, np.ndarray]] = {}
    for cohort, mask in (
        ("primary_vv", pd.Series(True, index=intervals.index)),
        ("both_endpoints_vh_supported", intervals.both_endpoints_vh_supported.astype(bool)),
    ):
        subset = intervals.loc[mask].copy()
        interval_arrays[cohort] = {
            block: subset.loc[
                subset.spatial_block_id.astype(str).eq(block), "interval_months"
            ].to_numpy(dtype=float)
            for block in blocks
        }
        polygon_medians = (
            subset.groupby(["spatial_block_id", "sample_id"], as_index=False)
            .interval_months.median()
        )
        polygon_arrays[cohort] = {
            block: polygon_medians.loc[
                polygon_medians.spatial_block_id.astype(str).eq(block), "interval_months"
            ].to_numpy(dtype=float)
            for block in blocks
        }

    rng = np.random.default_rng(seed)
    replicate_rows: list[dict] = []
    contrast_rows: list[dict] = []
    metric_names = None
    for replicate in range(replicates):
        selected = rng.choice(blocks, size=len(blocks), replace=True)
        cohort_metrics: dict[str, dict[str, float]] = {}
        for cohort in interval_arrays:
            values = np.concatenate([interval_arrays[cohort][block] for block in selected])
            polygons = np.concatenate([polygon_arrays[cohort][block] for block in selected])
            metrics = _metrics(values, polygons)
            cohort_metrics[cohort] = metrics
            replicate_rows.append({
                "replicate": replicate,
                "cohort": cohort,
                "sampled_blocks": int(len(selected)),
                "unique_sampled_blocks": int(len(set(selected))),
                "intervals": int(len(values)),
                "polygons": int(len(polygons)),
                **metrics,
            })
            metric_names = list(metrics)
        contrast_rows.append({
            "replicate": replicate,
            **{
                metric: (
                    cohort_metrics["both_endpoints_vh_supported"][metric]
                    - cohort_metrics["primary_vv"][metric]
                )
                for metric in metric_names or []
            },
        })
    return pd.DataFrame(replicate_rows), pd.DataFrame(contrast_rows)


def summarize_bootstrap(
    replicates: pd.DataFrame,
    *,
    group_column: str = "cohort",
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize bootstrap distributions without treating rows as raw data."""
    if metrics is None:
        metrics = [
            column
            for column in replicates.columns
            if column
            not in {"replicate", group_column, "sampled_blocks", "unique_sampled_blocks", "intervals", "polygons"}
        ]
    groups = [("paired_difference", replicates)] if group_column not in replicates else list(
        replicates.groupby(group_column, sort=False)
    )
    rows = []
    for group, frame in groups:
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            rows.append({
                group_column: group,
                "metric": metric,
                "bootstrap_median": float(values.median()) if len(values) else np.nan,
                "ci95_low": float(values.quantile(0.025)) if len(values) else np.nan,
                "ci95_high": float(values.quantile(0.975)) if len(values) else np.nan,
                "replicates": int(len(values)),
            })
    return pd.DataFrame(rows)
