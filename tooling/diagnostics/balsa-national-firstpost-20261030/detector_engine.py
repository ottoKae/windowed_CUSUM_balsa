"""One first-post-valid detector/aggregation implementation for both study scopes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.detector.core import cusum_amplitude
from windowed_cusum.detector.paper import load_paper_config
from windowed_cusum.detector.temporal_nulls import iid_permutation_score
from windowed_cusum.stand.harvest_bounded_event_aggregation import build_bounded_events
import windowed_cusum.detector.window_geometry as geometry

METHOD = "windowed_w36_d12_s12_firstpost_v2"
AGGREGATION = "B20_FIRSTPOST_V2"
STATE = {}
COUNTS = [0, 0]


def first_post_statistic(clean):
    amplitude, peak = cusum_amplitude(clean)
    return amplitude, peak + 1 if 0 <= peak < len(clean)-1 else -1


def counted_score(window, observed, *, config, seed):
    COUNTS[int(config.replicates != STATE["config"].permutations)] += 1
    return iid_permutation_score(window, observed, config=config, seed=seed)


def initialize(values_path, config_path):
    protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if protocol["date_policy"]["candidate"] != "first_valid_observation_after_first_CUSUM_maximum":
        raise ValueError("This engine implements only the frozen first-post date contract")
    config = load_paper_config(config_path)
    STATE.update(values=np.load(values_path, mmap_mode="r"), config=config,
                 dates=pd.date_range("2016-01-01", "2026-06-01", freq="MS"))
    # shortcut: process-local adapters; this entry is used in single-threaded worker processes.
    geometry.cusum_amplitude = first_post_statistic
    geometry.iid_permutation_score = counted_score


def detect(values, pixel_track_id):
    COUNTS[:] = [0, 0]
    return geometry.detect_series(values, STATE["dates"], pixel_track_id=pixel_track_id,
                                  geometry=geometry.Geometry(36, 12, 12), config=STATE["config"])


def batch(rows):
    output = []
    for unit, identity, index in rows:
        values = STATE["values"][index]
        result = detect(values, identity)
        base_calls, refined_calls = COUNTS
        opportunity = 0
        for bit, (ws, we, _, _) in enumerate(geometry.windows(len(values), geometry.Geometry(36, 12, 12))):
            opportunity |= int(np.isfinite(values[ws:we]).sum() >= 12) << bit
        events = result.to_dict("records")
        for event in events:
            ws = STATE["dates"].get_loc(event["window_start"])
            we = STATE["dates"].get_loc(event["window_end"]) + 1
            valid = np.flatnonzero(np.isfinite(values[ws:we]))
            _, peak = cusum_amplitude(values[ws:we][valid])
            expected = ws + int(valid[peak+1])
            assert int(event["change_index"]) == expected
            event.update(unit_id=unit, pixel_track_id=identity, method=METHOD,
                         peak_marker_month=STATE["dates"][ws+int(valid[peak])].strftime("%Y-%m"),
                         peak_to_firstpost_gap_months=expected-ws-int(valid[peak]))
            for col in ("change_time", "window_start", "window_end", "decision_start", "decision_end"):
                event[col] = pd.Timestamp(event[col]).strftime("%Y-%m-%d")
        payload = dict(unit_id=unit, pixel_track_id=identity, series_index=index,
            valid_months=int(np.isfinite(values).sum()), eligible_year_mask=opportunity,
            pre_null_candidates=base_calls, refined_candidates=refined_calls, events=events)
        output.append((identity, json.dumps(payload, allow_nan=False)))
    return output


def aggregate(units, membership, events, protocol):
    reports, nodes, receipt = build_bounded_events(units, membership, events,
        source_method=METHOD, output_method=AGGREGATION,
        protocol_version=protocol["protocol_id"], validate_frozen_gates=True,
        require_annual_decision=True)
    reports["interval_time_basis"] = "actual_pixel_response_months"
    return reports, nodes, receipt
