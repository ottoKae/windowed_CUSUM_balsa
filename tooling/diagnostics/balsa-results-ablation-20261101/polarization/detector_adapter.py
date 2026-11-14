"""Run the frozen scan code with real band identity and an optional as-of limit."""
from pathlib import Path
import sys
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from lr_cusum import window_geometry as geometry
from lr_cusum.core import cusum_amplitude
from lr_cusum.detection import stable_seed
from lr_cusum.paper import load_paper_config
from lr_cusum.temporal_nulls import iid_permutation_score

CONFIG_PATH = ROOT / "configs/firstpost_w36_d12_s12_b20_20261030_v2.json"
DATES = pd.date_range("2016-01-01", "2026-06-01", freq="MS")
DEFAULT_CONFIG = load_paper_config(CONFIG_PATH)
SCAN = geometry.detect_series
SCHEDULE = tuple(geometry.windows(len(DATES), geometry.Geometry(36, 12, 12)))


def detect(series, pixel_track_id, *, band="VV_dB", config=None,
           cutoff_index=None, score_fn=None, permutation_identity=None):
    """Return accepted events and all scheduled-window execution records.

    The scan's code object is unchanged. Its private globals replace only the
    First-post statistic, band-specific seed key, schedule, and score recorder.
    The source's literal VV output labels are corrected at this boundary.
    """
    config = DEFAULT_CONFIG if config is None else config
    scorer = iid_permutation_score if score_fn is None else score_fn
    values = np.asarray(series, dtype=np.float64)
    if values.shape != (len(DATES),) or band not in (
            "VV_dB", "VH_dB", "CR_POWER", "CR_DB", "RVI_DP"):
        raise ValueError("Expected one frozen 126-month supported input trajectory")
    cutoff = len(DATES)-1 if cutoff_index is None else int(cutoff_index)
    if not 0 <= cutoff < len(DATES):
        raise ValueError("Cutoff is outside the frozen calendar")
    if cutoff < len(DATES)-1:
        values = values.copy()
        values[cutoff+1:] = np.nan
    key = f"{pixel_track_id}:{band}"
    random_key = key if permutation_identity is None else permutation_identity
    seed_base = stable_seed(random_key, config.candidate_seed)
    records = []
    current = None

    def schedule(length, unused_geometry):
        nonlocal current
        for logical, (ws, nominal_we, ds, de) in enumerate(SCHEDULE):
            if ds > cutoff:
                break
            we = min(nominal_we, cutoff+1)
            current = dict(pixel_track_id=pixel_track_id, Band=band,
                logical_window=logical, candidate_id=f"{key}|sliding|w{logical}",
                window_start_index=ws, actual_window_end_exclusive=we,
                nominal_window_end_exclusive=nominal_we,
                nominal_decision_start=ds, nominal_decision_end=de,
                cutoff_index=cutoff, n_valid=int(np.isfinite(values[ws:we]).sum()),
                permutation_seed=int(seed_base+ws), candidate_month=None,
                change_index=None, peak_marker_month=None,
                cusum_amplitude=None, pre_mean=None, post_mean=None,
                post_minus_pre=None, initial_p=None, final_p=None,
                initial_exceedance_count=None, exceedance_count=None,
                permutations=0, near_threshold_refined=False,
                pre_null_eligible=False, accepted=False, outcome="not_scored")
            records.append(current)
            yield ws, we, ds, de

    def first_post(clean):
        amplitude, peak = cusum_amplitude(clean)
        current["cusum_amplitude"] = float(amplitude)
        if not 0 <= peak < len(clean)-1:
            return amplitude, -1
        ws, we = current["window_start_index"], current["actual_window_end_exclusive"]
        window = values[ws:we]
        valid = np.flatnonzero(np.isfinite(window))
        local = int(valid[peak+1])
        position = ws+local
        pre, post = float(np.nanmean(window[:local])), float(np.nanmean(window[local:]))
        current.update(candidate_month=DATES[position].strftime("%Y-%m"),
            change_index=position, peak_marker_month=DATES[ws+int(valid[peak])].strftime("%Y-%m"),
            peak_to_firstpost_gap_months=local-int(valid[peak]),
            pre_mean=pre, post_mean=post, post_minus_pre=post-pre)
        return amplitude, peak+1

    def score(window, observed, *, config, seed):
        result = scorer(window, observed, config=config, seed=seed)
        current.update(pre_null_eligible=True, final_p=float(result.null_p_value),
            exceedance_count=int(result.exceedance_count), permutations=int(result.replicates),
            outcome="permutation_rejected")
        if current["initial_p"] is None:
            current.update(initial_p=float(result.null_p_value),
                initial_exceedance_count=int(result.exceedance_count))
        else:
            current["near_threshold_refined"] = True
        return result

    namespace = dict(SCAN.__globals__, windows=schedule, cusum_amplitude=first_post,
        stable_seed=lambda unused_key, base: stable_seed(random_key, base),
        iid_permutation_score=score)
    scan = FunctionType(SCAN.__code__, namespace, SCAN.__name__, SCAN.__defaults__)
    events = scan(values, DATES, pixel_track_id=pixel_track_id,
        geometry=geometry.Geometry(36, 12, 12), config=config)
    events["Band"] = band
    events["candidate_id"] = events.candidate_id.str.replace(
        f"{pixel_track_id}:VV_dB|", f"{key}|", regex=False)
    accepted = set(events.candidate_id)
    for row in records:
        if row["candidate_id"] in accepted:
            row.update(accepted=True, outcome="accepted")
        elif row["n_valid"] < config.minimum_observations:
            row["outcome"] = "insufficient_observations"
        elif row["candidate_month"] is None:
            row["outcome"] = "no_first_post_candidate"
        elif not row["pre_null_eligible"]:
            row["outcome"] = "pre_null_gate_rejected"
    trace = pd.DataFrame(records)
    if not events.empty:
        extra = trace.set_index("candidate_id")
        for name in ("peak_marker_month", "peak_to_firstpost_gap_months", "permutation_seed"):
            events[name] = events.candidate_id.map(extra[name])
    return events, trace
