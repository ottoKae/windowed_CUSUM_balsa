"""Inputs and detector-output contract for the fixed-configuration TDA benchmark."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OUT = ROOT / "runs/validation/detector_tda_benchmark_20261109_v1"
MONTHS = pd.period_range("2016-01", "2026-06", freq="M")
REFERENCES = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"


def load_inputs():
    membership = pd.read_parquet(SOURCE / "validation_membership.parquet")
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    units = units.loc[units.unit_id.isin(membership.unit_id)].sort_values("unit_id").reset_index(drop=True)
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")[units.series_index.to_numpy()]
    if len(units) != 26890 or membership.sample_id.nunique() != 468 or not membership.unit_id.is_unique:
        raise ValueError("Frozen validation membership differs from the benchmark domain")
    if values.shape != (26890, 126) or set(units.unit_id) != set(membership.unit_id):
        raise ValueError("Pixel identity/calendar mismatch")
    return units, membership, values


def save_detector(method_id, events, eligible, metadata, sources):
    units, _, _ = load_inputs()
    if eligible.dtype != bool or eligible.shape != (len(units), len(MONTHS)):
        raise ValueError("Monthly eligibility must be boolean in the common unit/calendar order")
    if not {"unit_id", "change_time"} <= set(events.columns):
        raise ValueError("Detector events require unit_id and native change_time")
    frame = events.copy()
    frame["change_time"] = pd.to_datetime(frame.change_time)
    if frame.change_time.isna().any() or not frame.unit_id.isin(units.unit_id).all():
        raise ValueError("Detector events contain unknown identities or dates")
    frame["response_month"] = frame.change_time.dt.strftime("%Y-%m")
    frame["month_index"] = frame.change_time.dt.to_period("M").astype("int64") - MONTHS[0].ordinal
    if not frame.month_index.between(12, 119).all():
        raise ValueError("Detector events must be restricted to the frozen 2017--2025 reporting period")
    target = OUT / "detectors" / method_id
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "pixel_events.parquet", index=False)
    np.savez_compressed(target / "monthly_eligibility.npz", eligibility=eligible,
                        unit_id=units.unit_id.to_numpy(dtype=str), months=MONTHS.astype(str).to_numpy(dtype=str))
    write_json(target / "detector_receipt.json", {
        **metadata, "method_id": method_id, "adapter_completed": True,
        "n_pixels": len(units), "n_event_rows": len(frame),
        "n_unique_pixel_months": len(frame.drop_duplicates(["unit_id", "response_month"])),
        "n_eligible_pixel_months": int(eligible[:, 12:120].sum()),
        "input_sha256": {str(Path(p).resolve()): digest(p) for p in sources},
        "output_sha256": {name: digest(target / name) for name in
                          ("pixel_events.parquet", "monthly_eligibility.npz")},
    })
    print(f"{method_id}: {len(frame)} nominations; {len(frame.drop_duplicates(['unit_id', 'response_month']))} pixel-month nodes", flush=True)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x:
        x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
