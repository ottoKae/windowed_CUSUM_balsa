"""Replay cached cloud/monitoring outputs into the common monthly TDA contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import ROOT, SOURCE, MONTHS, load_inputs, save_detector, digest

sys.path.insert(0, str(ROOT / "src"))
from lr_cusum.ccdc_adapter import segment_candidates
from lr_cusum.structural_breakpoint_adapter import structural_events
from lr_cusum.ewmacd_adapter import ewma_trace

OLD_INPUTS = ROOT / "runs/validation/existing_labels_w36_20260927_v1/inputs"
CLOUD = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_cloud_v1"
DATES = MONTHS.to_timestamp()
DATES_MS = DATES.asi8 / 1_000_000


def main():
    units, membership, values = load_inputs()
    cached_units = pd.read_parquet(OLD_INPUTS / "units.parquet").sort_values(
        ["track_key", "cache_index"]
    ).reset_index(drop=True)
    aligned = units.set_index("unit_id").loc[cached_units.unit_id]
    for column in ("pixel_track_id", "track_key", "cache_index"):
        np.testing.assert_array_equal(aligned[column], cached_units[column])
    pd.testing.assert_frame_equal(
        membership.sort_values("unit_id").reset_index(drop=True),
        pd.read_parquet(OLD_INPUTS / "membership.parquet").sort_values("unit_id").reset_index(drop=True),
    )
    unit_positions = pd.Series(np.arange(len(units)), index=units.unit_id)
    cached_positions = unit_positions.loc[cached_units.unit_id].to_numpy(int)
    cache_sources = verify_cached_values(cached_units, cached_positions, values)
    common_sources = [
        Path(__file__), OLD_INPUTS / "units.parquet", OLD_INPUTS / "membership.parquet",
        SOURCE / "master_values.npy", SOURCE / "master_units.parquet", *cache_sources,
    ]
    cloud_config_path = CLOUD / "config/cloud_configuration_manifest.json"
    cloud_config = json.loads(cloud_config_path.read_text(encoding="utf-8"))
    statuses = pd.read_parquet(CLOUD / "cloud/pixel_status.parquet")
    frozen_events = pd.read_parquet(CLOUD / "cloud/pixel_events.parquet")
    for method, parameters in cloud_config["configurations"].items():
        run_cloud_method(method, parameters, units, values, cached_units, cached_positions,
                         statuses, frozen_events, cloud_config, [cloud_config_path, *common_sources])
    for family in ("bfast", "ewmacd"):
        run_monitor(family, units, values, cached_units, cached_positions, common_sources)


def verify_cached_values(cached_units, cached_positions, values):
    """The old cloud jobs read native caches; compare every supplied value and NaN."""
    cache_root = ROOT / "data/local/paper_v1/polygon_pixels/cache"
    sources = []
    for track, rows in cached_units.groupby("track_key", sort=False):
        path = cache_root / f"{track}.npz"
        with np.load(path) as arrays:
            positions = rows.cache_index.to_numpy(int)
            original = np.where(arrays["valid"][positions].astype(bool),
                                arrays["values"][positions, 0, :], np.nan)
            np.testing.assert_array_equal(original, values[cached_positions[rows.index]])
            np.testing.assert_array_equal(pd.DatetimeIndex(arrays["calendar"]), DATES)
        sources.append(path)
    return sources


def ccdc_fitted_support(record, finite):
    """Retrospective fitted support, not a reconstructed online monitoring log."""
    covered = np.zeros(len(DATES), dtype=bool)
    for start, end, count, coefficients in zip(
        record.get("tStart") or [], record.get("tEnd") or [],
        record.get("numObs") or [], record.get("VV_coefs") or [], strict=True,
    ):
        if (np.isfinite([start, end, count]).all() and start <= end and count > 0
                and len(coefficients) and np.isfinite(coefficients).all()):
            covered |= (DATES_MS >= start) & (DATES_MS <= end)
    return finite & covered


def structural_search_support(record, finite, spacing=12):
    """Searchable first-new-segment months require enough samples on both sides."""
    if not record.get("tStart") or not np.isfinite(record.get("VV_coefs", [])).all():
        return np.zeros(len(finite), dtype=bool)
    before = np.cumsum(finite) - finite
    including_after = int(finite.sum()) - before
    return finite & (before >= spacing) & (including_after >= spacing)


def run_cloud_method(method, parameters, units, values, cached_units, cached_positions,
                     statuses, frozen_events, cloud_config, common_sources):
    local_status = statuses.loc[statuses.method.eq(method)].set_index("unit_id").loc[units.unit_id]
    if not local_status.status.eq("completed").all():
        raise ValueError(f"{method}: cached execution contains unavailable or failed pixels")
    np.testing.assert_array_equal(local_status.pixel_track_id, units.pixel_track_id)
    eligible = np.zeros(values.shape, dtype=bool)
    raw_files = sorted((CLOUD / "cloud/raw" / method).glob("*.json"))
    replay = []
    covered_ids = []
    for path in raw_files:
        start = int(path.stem)
        records = json.loads(path.read_text(encoding="utf-8"))
        if [int(row["trajectory_index"]) for row in records] != list(range(len(records))):
            raise ValueError(f"Unordered cloud batch: {path}")
        rows = cached_units.iloc[start:start + len(records)]
        positions = cached_positions[start:start + len(records)]
        covered_ids.extend(rows.unit_id.tolist())
        for record, pos in zip(records, positions, strict=True):
            finite = np.isfinite(values[pos])
            eligible[pos] = (ccdc_fitted_support(record, finite) if parameters["family"] == "ccdc"
                             else structural_search_support(record, finite, parameters["spacing"]))
        if parameters["family"] == "ccdc":
            frame = segment_candidates(records, direction_band="VV")
            frame = frame.loc[frame.confirmed & frame.decrease].copy()
        else:
            frame = structural_events(records, DATES)
            frame = frame.loc[frame.decrease].copy()
        frame = frame.loc[frame.change_time.dt.year.between(2017, 2025)].copy()
        frame["unit_id"] = rows.unit_id.to_numpy()[frame.trajectory_index.to_numpy(int)]
        replay.append(frame[["unit_id", "change_time", "magnitude"]])
    if covered_ids != cached_units.unit_id.tolist():
        raise ValueError(f"{method}: cached batches do not cover the complete ordered domain")
    events = frozen_events.loc[frozen_events.method.eq(method)].copy()
    check_columns = ["unit_id", "change_time", "magnitude"]
    recomputed = pd.concat(replay, ignore_index=True).sort_values(check_columns).reset_index(drop=True)
    expected = events[check_columns].sort_values(check_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(recomputed, expected, check_dtype=False, check_exact=True)
    finite = np.isfinite(values)
    event_positions = pd.Series(np.arange(len(units)), index=units.unit_id).loc[events.unit_id].to_numpy(int)
    event_months = pd.to_datetime(events.change_time).dt.to_period("M").astype("int64") - MONTHS[0].ordinal
    outside = ~eligible[event_positions, event_months.to_numpy(int)]
    if parameters["family"] == "ccdc":
        contract = (
            "Union of raw fitted-segment tStart..tEnd intervals with finite coefficients "
            "and positive finite numObs, intersected with finite original monthly observations. "
            "Includes all modeled segments, independent of accepted breakpoint direction or "
            "confirmation. Finite observations in inter-segment gaps remain unknown. "
            "This is retrospective fitted-support eligibility, not exact online monitoring opportunity."
        )
        coverage_type = "retrospective_fitted_support"
        adapter = ROOT / "src/lr_cusum/ccdc_adapter.py"
    else:
        contract = (
            "Successful full-record fit and finite current observation; at least 12 valid "
            "samples strictly before the potential new-segment month and at least 12 "
            "including and after it. Spacing counts samples, not calendar months."
        )
        coverage_type = "retrospective_structural_search_support"
        adapter = ROOT / "src/lr_cusum/structural_breakpoint_adapter.py"
    frozen_adapter_hash = next(v for k, v in cloud_config["files"].items() if Path(k).name == adapter.name)
    if digest(adapter) != frozen_adapter_hash:
        raise ValueError(f"Cached {method} algorithm adapter no longer matches its frozen source")
    metadata = dict(
        family=parameters["family"], label=method, parameters=parameters,
        status="cached_official_fit_replayed_exactly", new_cloud_requests=0,
        date_contract=("native confirmed CCDC tBreak month; negative VV magnitude"
                       if parameters["family"] == "ccdc" else "first observed new-segment month; negative fitted jump"),
        eligibility_contract=contract, coverage_type=coverage_type,
        cache_values_exact=True, raw_output_event_replay_exact=True,
        finite_months_full_calendar=int(finite.sum()),
        finite_unknown_months_full_calendar=int((finite & ~eligible).sum()),
        fitted_or_search_supported_months_full_calendar=int(eligible.sum()),
        finite_unknown_months_reporting=int((finite & ~eligible)[:, 12:120].sum()),
        accepted_events_outside_quiet_support=int(outside.sum()),
        active_precedence="Accepted detections remain active, even at a fitted-support boundary.",
        algorithm_adapter_matches_frozen_source=True,
    )
    save_detector(method, events, eligible, metadata,
                  [*common_sources, CLOUD / "cloud/pixel_events.parquet", CLOUD / "cloud/pixel_status.parquet",
                   adapter, *raw_files])


def run_monitor(family, units, values, cached_units, cached_positions, common_sources):
    run = ROOT / f"runs/validation/pixel_b20_benchmark_window_20261004_{family}_v1"
    config_path = run / "config/configuration_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = config["method"]
    support = pd.read_parquet(run / "monitoring_support.parquet")
    events = pd.read_parquet(run / "pixel_events.parquet")
    eligible = np.zeros(values.shape, dtype=bool)
    unit_positions = pd.Series(np.arange(len(units)), index=units.unit_id)
    for year, rows in support.groupby("monitor_year", sort=True):
        positions = unit_positions.loc[rows.unit_id].to_numpy(int)
        history = (DATES.year >= year - 2) & (DATES.year < year)
        monitor = DATES.year == year
        finite = np.isfinite(values[positions])
        np.testing.assert_array_equal(finite[:, history].sum(axis=1), rows.history_valid_months)
        np.testing.assert_array_equal(finite[:, monitor].sum(axis=1), rows.monitor_valid_months)
        np.testing.assert_array_equal(rows.eligible, (finite[:, history].sum(axis=1) >= 18)
                                      & (finite[:, monitor].sum(axis=1) >= 6))
        eligible[positions] |= rows.eligible.to_numpy()[:, None] & finite & monitor[None, :]
    expected_jobs = set(zip(support.loc[support.eligible, "trajectory_index"],
                            support.loc[support.eligible, "monitor_year"]))
    completed = set()
    replay = []
    if family == "bfast":
        raw_files = sorted((run / "raw/results").glob("*/results.json"))
        for path in raw_files:
            for record in json.loads(path.read_text(encoding="utf-8")):
                if record["status"] != "success":
                    raise ValueError(f"BFAST fit failed in {path}")
                key = (int(record["trajectory_index"]), int(record["monitor_year"]))
                completed.add(key)
                boundary, magnitude = record.get("breakpoint"), record.get("alarm_residual_mean")
                if boundary is not None and magnitude is not None and float(magnitude) < 0:
                    year = int(np.floor(boundary))
                    month = int(round((float(boundary) - year) * 12)) + 1
                    if month == 13:
                        year, month = year + 1, 1
                    date = pd.Timestamp(year=year, month=month, day=1)
                    if date.year != key[1]:
                        raise ValueError("BFAST alarm outside scheduled monitoring year")
                    replay.append(dict(unit_id=cached_units.iloc[key[0]].unit_id,
                                       change_time=date, magnitude=float(magnitude)))
        comparison_columns = ["unit_id", "change_time", "magnitude"]
        extra_sources = [ROOT / "scripts/run_bfast_monitor.R", ROOT / "scripts/run_pixel_b20_bfast_monitor.py"]
    else:
        raw_files = sorted((run / "raw").glob("*/*.json"))
        for path in raw_files:
            year, start = int(path.parent.name), int(path.stem)
            identities = support.loc[(support.monitor_year == year) & support.eligible,
                                     "trajectory_index"].to_numpy(int)
            records = json.loads(path.read_text(encoding="utf-8"))
            ids = identities[start:start + len(records)]
            mask = (DATES.year >= year - 2) & (DATES.year <= year)
            x = values[cached_positions[ids]][:, None, :][:, :, mask]
            trace = ewma_trace(records, x, DATES[mask])
            trace["trajectory_index"] = ids[trace.trajectory_index.to_numpy(int)]
            completed.update((int(i), year) for i in ids)
            alarm = trace.loc[(trace.month.dt.year == year) & trace.score.lt(0)].sort_values("month")
            for record in alarm.groupby("trajectory_index", as_index=False).head(1).itertuples():
                replay.append(dict(unit_id=cached_units.iloc[int(record.trajectory_index)].unit_id,
                                   change_time=record.month, score=float(record.score)))
        comparison_columns = ["unit_id", "change_time", "score"]
        extra_sources = [ROOT / "src/lr_cusum/ewmacd_adapter.py", ROOT / "scripts/run_pixel_b20_ewmacd_monitor.py"]
    if completed != expected_jobs:
        raise ValueError(f"{family}: completed cache jobs differ from eligibility records")
    actual = pd.DataFrame(replay).sort_values(comparison_columns).reset_index(drop=True)
    expected = events[comparison_columns].sort_values(comparison_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    parameter_keys = ["history_years", "minimum_history_fraction", "minimum_monitor_observations", "monitor_years"]
    parameter_keys += ["order", "h", "level"] if family == "bfast" else ["lambda", "persistence"]
    metadata = dict(
        family=family, label=method,
        parameters={key: config[key] for key in parameter_keys},
        status="cached_monitoring_outputs_replayed_exactly", new_remote_or_r_jobs=0,
        date_contract=("first BFAST alarm in the scheduled year with negative residual mean"
                       if family == "bfast" else "first negative EWMACD score in each scheduled monitor year"),
        eligibility_contract="Finite month in a completed eligible monitoring year; >=18 history observations "
                             "in preceding two calendar years and >=6 in monitor year. 2017--2018 are unknown, not quiet.",
        coverage_type="completed_annual_monitoring_support", completed_jobs=len(completed),
        cache_values_exact=True, raw_output_event_replay_exact=True,
        no_observation_eligibility_2017_2018=True,
    )
    save_detector(method, events, eligible, metadata,
                  [*common_sources, config_path, run / "completed.json", run / "monitoring_support.parquet",
                   run / "pixel_events.parquet", *extra_sources, *raw_files])


if __name__ == "__main__":
    main()
