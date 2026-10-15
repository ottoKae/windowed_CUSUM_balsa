"""Explicit calendar-year aggregation, matching and spatial OOF evaluation.

Integer years remain years. No fake monthly dates or annual-to-month conversion.
"""
import numpy as np
import pandas as pd

from .validation_closure import _optimal_ordered_pairs


def aggregate_annual(values, dates, years, minimum_months=6):
    dates = pd.DatetimeIndex(dates)
    x = np.asarray(values)
    if x.ndim != 2 or x.shape[1] != len(dates) or dates.to_period("M").has_duplicates:
        raise ValueError("Expected distinct monthly track observations")
    result = np.full((len(x), len(years)), np.nan)
    rows = []
    for j, year in enumerate(years):
        mask = dates.year == year
        for i, series in enumerate(x):
            selected = np.flatnonzero(mask & np.isfinite(series))
            months = dates.month[selected]
            h1, h2 = bool((months <= 6).any()), bool((months >= 7).any())
            ok = len(selected) >= minimum_months and h1 and h2
            if ok:
                result[i, j] = np.median(series[selected])
            rows.append({"trajectory_index": i, "year": year, "valid_months": len(selected),
                         "first_month": int(months.min()) if len(months) else None,
                         "last_month": int(months.max()) if len(months) else None,
                         "half1": h1, "half2": h2, "eligible": ok,
                         "annual_vv": result[i, j]})
    return result, pd.DataFrame(rows)


def annual_events(monthly, years=(2018, 2025)):
    frame = monthly[["sample_id", "change_time"]].copy()
    frame["event_year"] = pd.to_datetime(frame.change_time).dt.year
    return frame.loc[frame.event_year.between(*years), ["sample_id", "event_year"]].drop_duplicates()


def counts(tp, fp, fn):
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r,
            "f1": 2 * p * r / max(1e-12, p + r)}


def match_years(refs, events, tolerance=0):
    if tolerance < 0:
        raise ValueError("Negative year tolerance")
    r = refs[["sample_id", "event_year"]].drop_duplicates()
    e = events[["sample_id", "event_year"]].drop_duplicates()
    if r.event_year.isna().any() or e.event_year.isna().any():
        raise ValueError("Missing event year")
    rg = {k: np.sort(g.event_year.to_numpy(int)) for k, g in r.groupby("sample_id")}
    eg = {k: np.sort(g.event_year.to_numpy(int)) for k, g in e.groupby("sample_id")}
    rows = []
    for sample in sorted(set(rg) | set(eg)):
        a, b = rg.get(sample, np.array([], int)), eg.get(sample, np.array([], int))
        for ri, ei in _optimal_ordered_pairs(a, b, tolerance):
            rows.append({"sample_id": sample, "reference_year": int(a[ri]), "detected_year": int(b[ei]),
                         "signed_error_years": int(b[ei] - a[ri]), "absolute_error_years": abs(int(b[ei] - a[ri]))})
    matched = pd.DataFrame(rows, columns=["sample_id", "reference_year", "detected_year",
                                         "signed_error_years", "absolute_error_years"])
    metrics = counts(len(rows), len(e) - len(rows), len(r) - len(rows))
    metrics["matched_year_mae"] = float(matched.absolute_error_years.mean()) if len(rows) else np.nan
    return metrics, matched


def validate_identity(identity):
    if identity.isna().any().any() or identity.sample_id.duplicated().any():
        raise ValueError("Each polygon must have one complete block/fold identity")
    if identity.groupby("spatial_block_id").outer_fold.nunique().max() != 1:
        raise ValueError("Block crosses outer folds")


def block_scores(refs, events, identity, tolerance=0):
    validate_identity(identity)
    if not set(events.sample_id).issubset(set(identity.sample_id)):
        raise ValueError("Unknown polygon in detections")
    rows = []
    for block, group in identity.groupby("spatial_block_id", sort=True):
        metrics, _ = match_years(refs.loc[refs.sample_id.isin(group.sample_id)],
                                 events.loc[events.sample_id.isin(group.sample_id)], tolerance)
        rows.append({"spatial_block_id": block, "outer_fold": int(group.outer_fold.iloc[0]),
                     "tolerance_years": tolerance, **metrics})
    return pd.DataFrame(rows)


def select_annual_oof(refs, candidates, identity, complexity=None):
    """Exact-year training counts select each fold, without consulting its labels."""
    validate_identity(identity)
    if not candidates or identity.outer_fold.nunique() < 2:
        raise ValueError("Need candidate grids and at least two folds")
    complexity = complexity or {}
    scored = {name: block_scores(refs, frame, identity) for name, frame in candidates.items()}
    outputs, selection, grid = [], [], []
    for fold in sorted(identity.outer_fold.unique()):
        choices = []
        for rank, (name, frame) in enumerate(scored.items()):
            numbers = frame.loc[frame.outer_fold.ne(fold), ["tp", "fp", "fn"]].sum()
            m = counts(*numbers)
            grid.append({"outer_fold": int(fold), "parameter_id": name, **m})
            choices.append((m["f1"], m["precision"], -complexity.get(name, 0), -rank, name))
        best = max(choices)
        name = best[-1]
        ids = identity.loc[identity.outer_fold.eq(fold), "sample_id"]
        frame = candidates[name].loc[candidates[name].sample_id.isin(ids)].copy()
        frame["outer_fold"], frame["parameter_id"] = int(fold), name
        outputs.append(frame)
        selection.append({"outer_fold": int(fold), "parameter_id": name,
                          "training_f1": best[0], "training_precision": best[1]})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(selection), pd.DataFrame(grid)
