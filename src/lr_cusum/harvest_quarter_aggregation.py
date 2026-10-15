"""Calendar-quarter aggregation of frozen W36 accepted pixel events.

No reference dates enter this adapter. Annual execution receipts determine
opportunity; missing/failed receipts never become zero responses. Q20 is a
literal subset of Q_any and M0 keeps its original prediction identity/month.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

DETECTOR_METHOD = "windowed_w36_d12_s12"
YEARS = tuple(range(2017, 2026))
QUARTERS = tuple(range(4 * YEARS[0], 4 * (YEARS[-1] + 1)))
COMPLETED_WINDOWS = {
    "completed", "completed_event", "completed_no_event",
    "reused_completed_event", "reused_completed_no_event",
}
SOURCE_COLUMNS = [
    "source_response_key", "source_unit_ids", "source_pixel_track_ids",
    "source_event_ids", "source_event_count",
]


def quarter_number(value):
    """Return 4*year + zero-based natural quarter for a date or YYYYQn."""
    match = re.fullmatch(r"(\d{4})Q([1-4])", str(value))
    if match:
        return 4 * int(match[1]) + int(match[2]) - 1
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("A missing date has no calendar quarter")
    return 4 * date.year + (date.month - 1) // 3


def quarter_label(number):
    """Format a quarter index from :func:`quarter_number` as YYYYQn."""
    number = int(number)
    return f"{number // 4:04d}Q{number % 4 + 1}"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _identifier(prefix, *values):
    return prefix + hashlib.sha256(_json(values).encode("utf-8")).hexdigest()[:24]


def _require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _selected(frame, detector_method=DETECTOR_METHOD):
    """Select one detector stream, or keep all rows for a preselected stream.

    ``None`` is used by the common pixel-to-B20 adapter after the caller has
    already isolated and validated a method/configuration.  The historical
    default remains byte-for-byte compatible with the frozen W36 workflow.
    """
    if detector_method is not None and "method" in frame:
        return frame.loc[frame.method.eq(detector_method)].copy()
    return frame.copy()


def _validate_native(frame, units, name):
    _require(frame, ["unit_id", "pixel_track_id"], name)
    if frame.unit_id.isna().any() or not set(frame.unit_id).issubset(units.unit_id):
        raise ValueError(f"{name} contains unknown master cells")
    lookup = units.set_index("unit_id")
    for column in ("pixel_track_id", "track_key", "native_row", "native_col"):
        if column in frame and column in lookup:
            expected = frame.unit_id.map(lookup[column])
            # Fresh execution receipts omit optional native coordinates,
            # while legacy receipts include them as floats. Native pixel IDs
            # remain mandatory and exact for every row in either source.
            known = frame[column].notna() if column in {"native_row", "native_col"} else pd.Series(True, index=frame.index)
            if not expected[known].eq(frame.loc[known, column]).all():
                raise ValueError(f"{name} selected native identity mismatch: {column}")


def _add_decision_year(frame, name):
    if "decision_start" in frame:
        derived = pd.to_datetime(frame.decision_start, errors="raise").dt.year
        if "decision_year" in frame and not derived.eq(frame.decision_year).all():
            raise ValueError(f"{name} decision year disagrees with decision_start")
        frame["decision_year"] = derived
    else:
        _require(frame, ["decision_year"], name)
    if frame.decision_year.isna().any():
        raise ValueError(f"{name} has unknown annual decision identity")
    frame["decision_year"] = frame.decision_year.astype(int)
    return frame


def _source_summary(group):
    if group.empty:
        return {"representative_month": "", "monthly_counts": "{}", "response_months": "[]",
                "source_response_key": "", "source_unit_ids": "[]", "source_pixel_track_ids": "[]",
                "source_event_ids": "[]", "source_event_count": 0, "response_pixels": 0}
    # Exact duplicate pixel/month records were removed before membership joins.
    counts = group.groupby("response_month", sort=True).unit_id.nunique()
    month_numbers = pd.PeriodIndex(group.response_month, freq="M").asi8
    median = int(np.floor(np.median(month_numbers) + .5))
    ids = sorted(group.source_event_id.unique().tolist())
    return {
        "representative_month": str(pd.Period(ordinal=median, freq="M")),
        "monthly_counts": _json({month: int(count) for month, count in counts.items()}),
        "response_months": _json(counts.index.tolist()),
        "source_response_key": _identifier("responses_", ids),
        "source_unit_ids": _json(sorted(group.unit_id.unique().tolist())),
        "source_pixel_track_ids": _json(sorted(group.pixel_track_id.unique().tolist())),
        "source_event_ids": _json(ids), "source_event_count": len(group),
        "response_pixels": int(group.unit_id.nunique()),
    }


def _prepare_events(events, units, *, allow_multiple_months_per_year=False,
                    detector_method=DETECTOR_METHOD,
                    validate_frozen_gates=True,
                    require_annual_decision=True):
    ev = _selected(events, detector_method=detector_method)
    _require(ev, ["unit_id", "pixel_track_id", "change_time"], "events")
    _validate_native(ev, units, "events")
    ev["change_time"] = pd.to_datetime(ev.change_time, errors="raise")
    if ev.change_time.isna().any():
        raise ValueError("Accepted event is missing its original change_time")
    ev["response_month"] = ev.change_time.dt.strftime("%Y-%m")
    ev["event_year"] = ev.change_time.dt.year
    if not ev.event_year.isin(YEARS).all():
        raise ValueError("Accepted events outside frozen 2017-2025 reporting period")
    if require_annual_decision and ("decision_year" in ev or "decision_start" in ev):
        ev = _add_decision_year(ev, "events")
        if not ev.decision_year.eq(ev.event_year).all():
            raise ValueError("Accepted event outside its annual decision interval")
    if validate_frozen_gates:
        for column, condition in (("Band", lambda x: x.eq("VV_dB")),
                                  ("null_p_value", lambda x: x.le(.025)),
                                  ("post_minus_pre", lambda x: x.lt(0))):
            if column in ev and not condition(ev[column]).all():
                raise ValueError(f"Accepted event violates frozen gate: {column}")
    # Different months in one cell/year violate the frozen detector. Do not
    # invent a policy for weighting or selecting these contradictory records.
    if not allow_multiple_months_per_year and ev.groupby(["unit_id", "event_year"]).response_month.nunique().gt(1).any():
        raise ValueError("Conflicting accepted months within one master-cell annual decision window")
    ev["source_event_id"] = [
        str(row.candidate_id) if hasattr(row, "candidate_id") and pd.notna(row.candidate_id)
        and str(row.candidate_id) else _identifier("pixel_", row.unit_id, row.pixel_track_id, row.response_month)
        for row in ev.itertuples(index=False)
    ]
    identity_counts = ev.groupby("source_event_id").agg(units=("unit_id", "nunique"), months=("response_month", "nunique"))
    if identity_counts.gt(1).any().any():
        raise ValueError("A source event identity refers to multiple cell/month records")
    ev = ev.sort_values(["unit_id", "response_month", "source_event_id"], kind="stable")
    duplicate_rows = ev.loc[ev.duplicated(["unit_id", "response_month"], keep="first")]
    duplicates = duplicate_rows[["unit_id", "response_month", "source_event_id"]].to_dict("records")
    ev = ev.drop_duplicates(["unit_id", "response_month"]).copy()
    ev["quarter_index"] = ev.event_year * 4 + (ev.change_time.dt.month - 1) // 3
    return ev, duplicates


def _annual_status(units, pixel_status, window_status, events):
    status, windows = _selected(pixel_status), _selected(window_status)
    _require(status, ["status"], "pixel_status")
    _validate_native(status, units, "pixel_status")
    if status.unit_id.duplicated().any():
        raise ValueError("Duplicate master-cell execution receipts")
    _require(windows, ["execution_status", "passes_minimum_observations"], "window_status")
    _validate_native(windows, units, "window_status")
    windows = _add_decision_year(windows, "window_status")
    windows = windows.loc[windows.decision_year.isin(YEARS)].copy()
    keys = ["unit_id", "decision_year"]
    if windows.duplicated(keys).any():
        raise ValueError("Duplicate annual window receipts")
    flags = windows.passes_minimum_observations
    if not flags.dropna().isin([True, False]).all():
        raise ValueError("Window opportunity must be a boolean or unknown")
    if "valid_observations" in windows:
        known = flags.notna() & windows.valid_observations.notna()
        if not flags[known].eq(windows.loc[known, "valid_observations"].ge(12)).all():
            raise ValueError("Window opportunity differs from frozen >=12-month rule")
    table = pd.MultiIndex.from_product([units.unit_id, YEARS], names=keys).to_frame(index=False)
    columns = keys + ["execution_status", "passes_minimum_observations"]
    if "accepted_event" in windows:
        columns.append("accepted_event")
    table = table.merge(windows[columns], on=keys, how="left", validate="one_to_one")
    table = table.merge(status[["unit_id", "status"]], on="unit_id", how="left", validate="many_to_one")
    table["eligible_pixels"] = table.passes_minimum_observations.eq(True).fillna(False)
    completed = table.status.eq("completed")
    table["completed_eligible_pixels"] = (table.eligible_pixels & completed
                                           & table.execution_status.isin(COMPLETED_WINDOWS))
    table["insufficient_opportunity_pixels"] = (
        table.passes_minimum_observations.eq(False).fillna(False)
        & table.status.isin(["completed", "input_insufficient"])
        & table.execution_status.isin(COMPLETED_WINDOWS | {"input_insufficient"}))
    table["technical_gap_pixels"] = ~(table.completed_eligible_pixels | table.insufficient_opportunity_pixels)
    actual = events.assign(decision_year=events.event_year)[keys].drop_duplicates()
    actual["has_event"] = True
    table = table.merge(actual, on=keys, how="left", validate="one_to_one")
    has_event = table.has_event.eq(True)
    if (has_event & table.passes_minimum_observations.eq(False)).any():
        raise ValueError("Accepted event has no eligible annual window")
    if "accepted_event" in table:
        if (has_event & table.accepted_event.eq(False)).any():
            raise ValueError("Accepted event contradicts its annual window receipt")
        table.loc[table.accepted_event.eq(True) & ~has_event, "technical_gap_pixels"] = True
    # A completed_event receipt is also evidence of an omitted output row.
    table.loc[table.execution_status.isin(["completed_event", "reused_completed_event"])
              & ~has_event, "technical_gap_pixels"] = True
    count_mismatches = []
    if "event_count" in status:
        actual_counts = events.groupby("unit_id").size()
        reported = pd.to_numeric(status.event_count, errors="raise")
        mismatched = reported.notna() & reported.ne(status.unit_id.map(actual_counts).fillna(0))
        count_mismatches = sorted(status.loc[mismatched, "unit_id"].tolist())
        table.loc[table.unit_id.isin(count_mismatches), "technical_gap_pixels"] = True
    table.loc[table.technical_gap_pixels, "completed_eligible_pixels"] = False
    table.loc[table.technical_gap_pixels, "insufficient_opportunity_pixels"] = False
    return table, count_mismatches


def build_quarter_aggregations(units, membership, events, pixel_status, window_status,
                               sample_ids, m0_predictions):
    """Return (support, predictions, JSON-serializable diagnostics).

    ``sample_ids`` may be a one-column frame or iterable of all cohort IDs,
    including stands without a master cell. Fixed annual receipts apply to
    all four quarters. Insufficient opportunity preserves the fixed denominator;
    technical gaps suppress the support fraction and quarter predictions.
    """
    _require(units, ["unit_id", "pixel_track_id"], "units")
    if units[["unit_id", "pixel_track_id"]].isna().any().any():
        raise ValueError("Master/native identities must be present")
    if units.unit_id.duplicated().any() or units.pixel_track_id.duplicated().any():
        raise ValueError("Master/native identities must be unique")
    _require(membership, ["unit_id", "sample_id"], "membership")
    member = membership[["sample_id", "unit_id"]].copy()
    if member.duplicated().any() or member.isna().any().any():
        raise ValueError("Duplicate or missing polygon membership")
    if isinstance(sample_ids, pd.DataFrame):
        _require(sample_ids, ["sample_id"], "sample_ids")
        sample_ids = sample_ids.sample_id
    cohort = sorted(set(sample_ids))
    if not set(member.sample_id).issubset(cohort) or not set(member.unit_id).issubset(units.unit_id):
        raise ValueError("Membership contains an unknown cohort/master-cell identity")
    ev, duplicate_rows = _prepare_events(events, units)
    cell_year, count_mismatches = _annual_status(units, pixel_status, window_status, ev)
    status_columns = ["eligible_pixels", "completed_eligible_pixels", "insufficient_opportunity_pixels",
                      "technical_gap_pixels"]
    annual = member.merge(cell_year, on="unit_id", validate="many_to_many")
    annual = annual.groupby(["sample_id", "decision_year"], sort=True)[status_columns].sum().reset_index()
    support = pd.MultiIndex.from_product([cohort, QUARTERS], names=["sample_id", "quarter_index"]).to_frame(index=False)
    support["decision_year"] = support.quarter_index // 4
    support = support.merge(annual, on=["sample_id", "decision_year"], how="left", validate="many_to_one")
    for column in status_columns:
        support[column] = support[column].fillna(0).astype(int)
    support["total_pixels"] = support.sample_id.map(member.groupby("sample_id").unit_id.nunique()).fillna(0).astype(int)
    support["predicted_quarter"] = support.quarter_index.map(quarter_label)
    support["quarter_start"] = pd.to_datetime(support.predicted_quarter.map(lambda q: f"{q[:4]}-{3 * (int(q[-1]) - 1) + 1:02d}-01"))
    support["quarter_end"] = support.quarter_start + pd.offsets.QuarterEnd()
    support["short_context_2025"] = support.decision_year.eq(2025)
    support["complete_calculation"] = support.total_pixels.gt(0) & support.technical_gap_pixels.eq(0)
    support["opportunity_status"] = np.select(
        [support.total_pixels.eq(0), support.technical_gap_pixels.gt(0), support.insufficient_opportunity_pixels.gt(0)],
        ["no_spatial_support", "incomplete_execution", "insufficient"], default="sufficient")
    joined = member.merge(ev, on="unit_id", how="inner", validate="many_to_many")
    summaries = [{"sample_id": sample, "quarter_index": quarter, **_source_summary(group)}
                 for (sample, quarter), group in joined.groupby(["sample_id", "quarter_index"], sort=True)]
    empty_summary = _source_summary(joined.iloc[:0])
    summary_frame = pd.DataFrame(summaries, columns=["sample_id", "quarter_index", *empty_summary])
    support = support.merge(summary_frame, on=["sample_id", "quarter_index"], how="left", validate="one_to_one")
    for column, default in empty_summary.items():
        if isinstance(default, int):
            support[column] = pd.to_numeric(support[column], errors="raise").fillna(default)
        else:
            support[column] = support[column].fillna(default)
    support["response_pixels"] = support.response_pixels.astype(int)
    support["source_event_count"] = support.source_event_count.astype(int)
    support["predicted_month"] = support.representative_month
    support["support_fraction"] = (support.response_pixels / support.total_pixels).where(support.complete_calculation)
    for column, condition in [("passes_q_any", support.response_pixels.ge(1)),
                              ("passes_q20", support.response_pixels.ge(1) & (5 * support.response_pixels).ge(support.total_pixels))]:
        support[column] = condition.astype("boolean").where(support.complete_calculation, pd.NA)
    any_predictions = support.loc[support.passes_q_any.fillna(False)].copy()
    any_predictions["method"] = "Q_any"
    any_predictions["prediction_id"] = [_identifier("Q_any_", row.sample_id, row.predicted_quarter)
                                          for row in any_predictions.itertuples(index=False)]
    q20 = any_predictions.loc[any_predictions.passes_q20.fillna(False)].copy()
    q20["method"] = "Q20"
    q20["prediction_id"] = [_identifier("Q20_", row.sample_id, row.predicted_quarter)
                              for row in q20.itertuples(index=False)]

    # Preserve every annual-median prediction, including multiple original
    # identities in a quarter; its annual support remains explicitly labeled.
    m0 = m0_predictions.copy()
    if "method" in m0:
        m0 = m0.loc[m0.method.isin(["M0", "M0_year_median"])].copy()
    _require(m0, ["sample_id", "prediction_id", "predicted_month"], "m0_predictions")
    if m0.prediction_id.duplicated().any() or not set(m0.sample_id).issubset(cohort):
        raise ValueError("M0 has duplicate prediction IDs or unknown stand identities")
    annual_sources = {key: _source_summary(group)
                      for key, group in joined.groupby(["sample_id", "event_year"], sort=True)}
    lookup = support.set_index(["sample_id", "quarter_index"])
    inherited = []
    for row in m0.to_dict("records"):
        qindex = quarter_number(row["predicted_month"])
        if qindex not in QUARTERS:
            raise ValueError("M0 prediction outside frozen reporting period")
        base = lookup.loc[(row["sample_id"], qindex)].to_dict()
        if not base["total_pixels"]:
            raise ValueError("M0 prediction has no spatial support")
        source = annual_sources.get((row["sample_id"], qindex // 4), empty_summary)
        if not source["response_pixels"]:
            raise ValueError("M0 prediction has no original annual response source")
        if "total_pixels" in row and row["total_pixels"] != base["total_pixels"]:
            raise ValueError("M0 denominator differs from frozen master-cell membership")
        if "unique_support_pixels" in row and row["unique_support_pixels"] != source["response_pixels"]:
            raise ValueError("M0 annual support differs from original pixel events")
        result = {**row, **base, **source, "sample_id": row["sample_id"], "quarter_index": qindex,
                  "prediction_id": row["prediction_id"], "method": "M0_year_median",
                  "representative_month": row["predicted_month"], "predicted_month": row["predicted_month"],
                  "support_scope": "annual", "source_prediction_id": row["prediction_id"]}
        result["support_fraction"] = (source["response_pixels"] / base["total_pixels"]
                                      if base["complete_calculation"] else np.nan)
        inherited.append(result)
    for frame in (any_predictions, q20):
        frame["support_scope"] = "quarter"
        frame["source_prediction_id"] = ""
    predictions = pd.concat([pd.DataFrame(inherited), any_predictions, q20], ignore_index=True, sort=False)
    predictions = predictions.sort_values(["method", "sample_id", "quarter_index", "prediction_id"], kind="stable").reset_index(drop=True)
    if predictions.duplicated(["method", "prediction_id"]).any():
        raise AssertionError("Quarter prediction identity collision")
    diagnostics = {
        "master_cells": len(units), "cohort_stands": len(cohort), "spatially_supported_stands": int(member.sample_id.nunique()),
        "quarter_rows": len(support), "input_event_rows": len(_selected(events)), "unique_pixel_month_events": len(ev),
        "duplicate_pixel_month_rows": len(duplicate_rows), "duplicate_sources": duplicate_rows,
        "conflicting_annual_event_policy": "raise_without_repair",
        "event_count_mismatch_units": count_mismatches,
        "technical_gap_stand_quarters": int(support.technical_gap_pixels.gt(0).sum()),
        "opportunity_status_counts": support.opportunity_status.value_counts().to_dict(),
        "prediction_counts": {method: int(predictions.method.eq(method).sum()) for method in ["M0_year_median", "Q_any", "Q20"]},
        "m0_support_scope": "annual; original representative month and prediction identity retained",
        "source_trace": "source_event_ids use frozen candidate_id or deterministic master-cell/native/month identity; source_unit_ids and source_pixel_track_ids identify frozen membership and original event rows",
    }
    return support, predictions, diagnostics
