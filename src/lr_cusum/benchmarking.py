"""Fresh legacy/Windowed candidate grids and training-fold-only selection.

Grids, seeds and tie breaks follow the archived notebook-comparison, validation
closure and N1 geometry drivers. Numerical detectors remain in their own modules.
This module neither reads frozen predictions nor submits Earth Engine requests.
"""
from dataclasses import replace

import pandas as pd

from .baselines import fpol_bocd_detect, is_decrease, recusum_detect
from .detection import stable_seed
from .paper import detect_series_n1
from .validation_closure import (
    classical_lower_cusum_events, match_events_detailed, maximum_drop_events,
)


def local_grids():
    """Return ordered, explicit grids; do not tune them on held-out outcomes."""
    return {
        "windowed": {
            f"w{w}_s{s:02d}": {"window_length": w, "step": s}
            for w, s in ((24, 6), (36, 6), (48, 6), (36, 12), (48, 12))
        },
        "recusum": {
            f"recusum_c{c:g}_m{m}": {"confidence": c, "minimum_segment": m,
                                    "maximum_depth": 10}
            for c in (.90, .95, .975) for m in (12, 18, 24)
        },
        "bocd": {
            f"fpol_bocd_h{h}_d{d}": {"hazard": 1 / h, "delta_m": d, "min_obs": 12}
            for h in (24, 50, 100) for d in (6, 10, 12)
        },
        "maximum_drop": {
            f"maxdrop_t{t:g}": {"threshold_db": t}
            for t in (.75, 1., 1.25, 1.5, 2., 2.5, 3., 4.)
        },
        "classical_cusum": {
            f"cusum_k{k:g}_h{h:g}": {"allowance": k, "decision_interval": h}
            for k in (.25, .50, .75) for h in (3., 4., 5., 6., 8.)
        },
    }


def detect_local_grid(task):
    """One parameter configuration over all tracks, safe for a process pool.

NaNs keep their calendar positions; detections are never shifted to make them
agree with references. BOCD uses both bands, with a VV direction filter. ReCuSum
uses the historical 800 draws and per-polygon-track seed (not the Windowed seed).
"""
    family, name, params, meta, values, dates, paper_config = task
    rows = []
    for i, item in enumerate(meta.itertuples(index=False)):
        vv, vh = values[i]
        if family == "windowed":
            events = detect_series_n1(
                vv, dates, identifier=item.polygon_track_id, band="VV_dB",
                config=replace(paper_config, **params),
                series_seed=getattr(item, "seed_VV_dB", None),
            ).to_dict("records")
        elif family in ("recusum", "bocd"):
            if family == "recusum":
                seed = getattr(item, "seed_recusum", stable_seed(item.polygon_track_id, 20260820))
                points = recusum_detect(vv, **params, bootstrap=800, seed=int(seed))
            else:
                points = fpol_bocd_detect(vv, vh, **params)
            events = [{"change_time": dates[int(j)], "decrease": is_decrease(vv, int(j)),
                       "Band": "VV+VH" if family == "bocd" else "VV_dB"}
                      for j in points if 0 <= int(j) < len(dates)]
        elif family == "maximum_drop":
            events = maximum_drop_events(vv, dates, **params)
        elif family == "classical_cusum":
            events = classical_lower_cusum_events(vv, dates, **params)
        else:
            raise ValueError(family)
        rows.extend({**event, "trajectory_index": i} for event in events)
    # Typed columns also cover a successful run with zero detections.
    raw = pd.DataFrame(rows) if rows else pd.DataFrame({
        "trajectory_index": pd.Series(dtype="int64"),
        "change_time": pd.Series(dtype="datetime64[ns]"),
        "decrease": pd.Series(dtype="bool"),
    })
    return family, name, raw


def select_training_oof(refs, candidates, *, matcher_version):
    """Historical legacy tie break: training F1, training precision, name.

Each polygon and all its events/tracks belong to one outer fold. Its dates never
enter its own parameter selection. Persist every training-grid score, including
zero-event cases; the held-out score is computed only by the later evaluator.
"""
    if not candidates:
        raise ValueError("No complete parameter configurations")
    identities = refs[["sample_id", "spatial_block_id", "outer_fold"]].drop_duplicates()
    if identities.isna().any().any() or identities.sample_id.duplicated().any():
        raise ValueError("Every polygon must belong to exactly one block and fold")
    if identities.groupby("spatial_block_id").outer_fold.nunique().max() != 1:
        raise ValueError("A spatial block crosses outer folds")
    if refs.outer_fold.nunique() < 2:
        raise ValueError("At least two folds required")
    rows, outputs, selections = [], [], []
    for fold in sorted(refs.outer_fold.unique()):
        train = refs.loc[refs.outer_fold.ne(fold)]
        test_ids = set(refs.loc[refs.outer_fold.eq(fold), "sample_id"])
        scores = []
        for name, events in candidates.items():
            metrics, _ = match_events_detailed(
                train, events.loc[events.sample_id.isin(train.sample_id)],
                tolerance_months=2, matcher_version=matcher_version,
            )
            rows.append({"outer_fold": int(fold), "parameter_id": name,
                         "selection_matcher": matcher_version, **metrics})
            scores.append((metrics["f1"], metrics["precision"], name))
        f1, precision, name = max(scores)
        local = candidates[name].loc[candidates[name].sample_id.isin(test_ids)].copy()
        local["outer_fold"] = int(fold)
        local["parameter_id"] = name
        outputs.append(local)
        selections.append({"outer_fold": int(fold), "parameter_id": name,
                           "training_f1": f1, "training_precision": precision,
                           "selection_matcher": matcher_version,
                           "tie_break": "training precision, parameter name"})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(selections), pd.DataFrame(rows)
