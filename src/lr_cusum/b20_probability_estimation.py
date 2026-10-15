"""S4 estimates for the frozen B20 spatial probability validation plan v2.1.

No sampling or review is performed here. ``summarize_probability_results``
requires an explicit result row for every selected report at every requested tolerance.
Pending reviews never become reviewed unknowns or partial precision estimates.

The public total/ratio functions also accept small synthetic designs, allowing
the two-stage formulas to be checked independently of the production data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


FRAME_COLUMNS = (
    "prediction_id", "selected", "geo_stratum", "block_id", "candidate_class",
    "M_h", "m_h", "N_bc", "n_bc", "inclusion_probability", "sampling_weight",
)
RESULT_COLUMNS = ("prediction_id", "tolerance_quarters", "review_status", "judgement")
ESTIMATE_COLUMNS = (
    "precision_hajek", "precision_ht", "lower_hajek", "upper_hajek",
    "unknown_hajek", "lower_ht", "upper_ht", "unknown_ht", "se_lower",
    "se_upper", "se_unknown", "variance_lower", "variance_upper",
    "variance_unknown", "ci95_lower", "ci95_upper", "t_critical",
)
CELL_KEYS = ["geo_stratum", "block_id", "candidate_class"]


def tolerance_spec(tolerance_unit="quarter", tolerances=None):
    """Resolve explicit units without relabelling quarter distances as months.

    Legacy quarter calls retain (0, 1); the monthly sensitivity analysis uses
    (1, 2, 3). Custom tolerances must be unique, nonnegative integer counts.
    """
    if tolerance_unit not in {"quarter", "month"}:
        raise ValueError("tolerance_unit must be 'quarter' or 'month'")
    if tolerances is None:
        tolerances = (0, 1) if tolerance_unit == "quarter" else (1, 2, 3)
    try:
        values = tuple(tolerances)
    except TypeError as exc:
        raise ValueError("tolerances must be a nonempty sequence of integers") from exc
    if not values or any(isinstance(value, (bool, np.bool_))
                         or not isinstance(value, (int, np.integer)) or value < 0
                         for value in values):
        raise ValueError("tolerances must be nonempty, nonnegative integer counts")
    if len(set(values)) != len(values):
        raise ValueError("tolerances must be unique")
    return f"tolerance_{tolerance_unit}s", tuple(int(value) for value in values)


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if not table.columns.is_unique:
        raise ValueError(f"{name} has duplicate columns")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _validated_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a complete population frame or all rows of the formal sample.

    If any row is unselected, the input is treated as a complete population
    frame, so population cell sizes and block counts are checked as well.
    A sample-only input cannot detect an entirely absent population cell; the
    production caller should therefore pass the full frozen sampling frame.
    """
    _require_columns(frame, FRAME_COLUMNS, "frame")
    if frame.empty:
        raise ValueError("frame is empty")
    data = frame.loc[:, FRAME_COLUMNS].copy()
    for col in ("prediction_id", "geo_stratum", "block_id", "candidate_class"):
        if data[col].isna().any() or data[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"frame has missing {col}")
    if data.prediction_id.duplicated().any():
        raise ValueError("frame has duplicate prediction_id")
    if not data.candidate_class.isin(("C1", "C2", "C3")).all():
        raise ValueError("candidate_class must be C1, C2, or C3")
    if data.selected.isna().any() or not data.selected.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("selected must contain explicit booleans")
    for col in ("M_h", "m_h", "N_bc", "n_bc", "inclusion_probability", "sampling_weight"):
        try:
            data[col] = pd.to_numeric(data[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric {col}") from exc
        if not np.isfinite(data[col].to_numpy(dtype=float)).all() or (data[col] <= 0).any():
            raise ValueError(f"{col} must be finite and positive")
    for col in ("M_h", "m_h", "N_bc", "n_bc"):
        if (data[col] % 1 != 0).any():
            raise ValueError(f"{col} must contain integers")
        data[col] = data[col].astype(int)
    if (data.m_h > data.M_h).any() or (data.n_bc > data.N_bc).any():
        raise ValueError("sample quota exceeds population size")
    if ((data.m_h < data.M_h) & (data.m_h < 2)).any():
        raise ValueError("noncensus stratum requires at least two sampled blocks")
    if (data.n_bc < np.minimum(2, data.N_bc)).any():
        raise ValueError("noncensus cell requires at least two sampled reports")
    if (data.groupby("block_id").geo_stratum.nunique() != 1).any():
        raise ValueError("block_id belongs to multiple geo_stratum values")
    for col in ("M_h", "m_h"):
        if (data.groupby("geo_stratum")[col].nunique() != 1).any():
            raise ValueError(f"inconsistent {col} within stratum")
    for col in ("N_bc", "n_bc"):
        if (data.groupby(CELL_KEYS)[col].nunique() != 1).any():
            raise ValueError(f"inconsistent {col} within cell")
    expected_pi = (data.m_h / data.M_h) * (data.n_bc / data.N_bc)
    if not np.allclose(data.inclusion_probability, expected_pi, rtol=1e-10, atol=1e-12):
        raise ValueError("inclusion_probability contradicts two-stage quotas")
    if not np.allclose(data.sampling_weight, 1 / expected_pi, rtol=1e-10, atol=1e-12):
        raise ValueError("sampling_weight contradicts two-stage quotas")

    sample = data.loc[data.selected].copy()
    if sample.empty:
        raise ValueError("frame has no selected reports")
    strata = data.groupby("geo_stratum", sort=True).first()
    selected_blocks = sample.groupby("geo_stratum").block_id.nunique().reindex(strata.index, fill_value=0)
    if not selected_blocks.eq(strata.m_h).all():
        raise ValueError("selected block count contradicts m_h or a selected block is missing")
    cells = data.groupby(CELL_KEYS, sort=True).first()
    selected_sizes = sample.groupby(CELL_KEYS).size().reindex(cells.index, fill_value=0)
    selected_block_ids = set(sample.block_id)
    expected_sizes = cells.n_bc.where(
        cells.index.get_level_values("block_id").isin(selected_block_ids), 0
    )
    if not selected_sizes.eq(expected_sizes).all():
        raise ValueError("selected cell count contradicts n_bc or a selected report is missing")
    if (~data.selected).any():
        if not data.groupby(CELL_KEYS).size().eq(cells.N_bc).all():
            raise ValueError("complete frame cell count contradicts N_bc")
        if not data.groupby("geo_stratum").block_id.nunique().eq(strata.M_h).all():
            raise ValueError("complete frame block count contradicts M_h")
    return sample


def _values(sample: pd.DataFrame, values) -> np.ndarray:
    # Require explicit index agreement for Series to prevent silent misalignment.
    if isinstance(values, pd.Series) and not values.index.equals(sample.index):
        raise ValueError("values index must match sample row order")
    result = np.asarray(values, dtype=float)
    if result.shape != (len(sample),) or not np.isfinite(result).all():
        raise ValueError("values must be one finite number per sampled report")
    return result


def _total(sample: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    working = sample.assign(_z=values)
    block_rows = []
    for (stratum, block), block_data in working.groupby(["geo_stratum", "block_id"], sort=True):
        block_total = block_variance = 0.0
        for _, cell in block_data.groupby("candidate_class", sort=True):
            N, n = int(cell.N_bc.iloc[0]), int(cell.n_bc.iloc[0])
            block_total += N * float(cell._z.mean())
            # A census contributes zero, including a singleton: do not form 0/0.
            if n < N:
                block_variance += N * N * (1 - n / N) * float(cell._z.var(ddof=1)) / n
        block_rows.append({
            "geo_stratum": stratum, "block_id": block,
            "M_h": int(block_data.M_h.iloc[0]), "m_h": int(block_data.m_h.iloc[0]),
            "total": block_total, "variance": block_variance,
        })
    total = between = within = 0.0
    for _, blocks in pd.DataFrame(block_rows).groupby("geo_stratum", sort=True):
        M, m = int(blocks.M_h.iloc[0]), int(blocks.m_h.iloc[0])
        total += (M / m) * float(blocks.total.sum())
        if m < M:
            between += M * M * (1 - m / M) * float(blocks.total.var(ddof=1)) / m
        within += (M / m) * float(blocks.variance.sum())
    return {
        "total": total, "variance": between + within,
        "variance_between": between, "variance_within": within,
    }


def estimate_two_stage_total(sample: pd.DataFrame, values) -> dict[str, float]:
    """Estimate a total and its two-stage design variance (plan section 6.3).

    ``sample`` must contain every formal sample row, all with selected=True;
    values are positional, or a Series whose index equals the sample index.
    Both variance components are returned for audit and independent checks.
    """
    checked = _validated_sample(sample)
    if len(checked) != len(sample):
        raise ValueError("total estimation requires only selected sample rows")
    return _total(checked, _values(checked, values))


def _ratio(sample: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    weight_sum = float(sample.sampling_weight.sum())
    # Preserve exact constants even when dot() and sum() accumulate differently.
    if np.ptp(values) == 0:
        estimate = float(values[0])
        weighted_total = estimate * weight_sum
    else:
        weighted_total = float(np.dot(sample.sampling_weight, values))
        estimate = weighted_total / weight_sum
    linearized = _total(sample, values - estimate)
    variance = linearized["variance"] / weight_sum**2
    return {
        "estimate": estimate, "variance": variance, "se": float(np.sqrt(variance)),
        "variance_between": linearized["variance_between"] / weight_sum**2,
        "variance_within": linearized["variance_within"] / weight_sum**2,
        "weighted_total": weighted_total,
    }


def estimate_two_stage_ratio(sample: pd.DataFrame, values) -> dict[str, float]:
    """Hájek ratio and first-order two-stage variance using z = y - R_hat."""
    checked = _validated_sample(sample)
    if len(checked) != len(sample):
        raise ValueError("ratio estimation requires only selected sample rows")
    return _ratio(checked, _values(checked, values))


def summarize_probability_results(
    frame: pd.DataFrame, results: pd.DataFrame, *, population_size: int = 1266,
    tolerance_unit: str = "quarter", tolerances=None,
) -> pd.DataFrame:
    """Summarize requested tolerances without manufacturing incomplete results.

    Pass the complete frozen frame whenever possible. Sample-only input is
    accepted, but must include every formal sampled row and correct design
    counts. ``population_size`` defaults to the frozen D=1266 and may be
    overridden only for reusable synthetic-design checks.

    ``results`` must contain exactly one row per selected prediction_id for
    each requested tolerance; pending rows must be explicit. Quarter defaults
    are (0, 1), monthly defaults (1, 2, 3). Month mode requires tolerance_months
    and never chooses a primary monthly tolerance from these observed results.
    All estimates remain NaN until *all* requested tolerances are reviewed. Reviewed U
    remains in the denominator, yields scenario endpoints, and suppresses a
    unique precision. HT fields are raw, including values outside [0, 1].
    """
    tolerance_column, tolerances = tolerance_spec(tolerance_unit, tolerances)
    sample = _validated_sample(frame)
    if (isinstance(population_size, (bool, np.bool_)) or
            not isinstance(population_size, (int, np.integer)) or population_size <= 0):
        raise ValueError("population_size must be a positive integer")
    if len(sample) > population_size:
        raise ValueError("sample size exceeds population_size")
    if (~frame.selected).any() and len(frame) != population_size:
        raise ValueError("complete frame size differs from population_size")
    strata = sample.groupby("geo_stratum").first()
    first_stage_census = bool(strata.M_h.eq(strata.m_h).all())
    if first_stage_census:
        known_size = int(sample.groupby(CELL_KEYS).N_bc.first().sum())
        if known_size != population_size:
            raise ValueError("all-block population total differs from population_size")
    is_census = first_stage_census and bool(sample.N_bc.eq(sample.n_bc).all())
    degrees_of_freedom = int((strata.loc[strata.m_h < strata.M_h, "m_h"] - 1).sum())

    result_columns = tuple(tolerance_column if column == "tolerance_quarters" else column
                           for column in RESULT_COLUMNS)
    _require_columns(results, result_columns, "results")
    if "tolerance_unit" in results and not results.tolerance_unit.eq(tolerance_unit).all():
        raise ValueError("results tolerance_unit contradicts the requested unit")
    reviews = results.loc[:, result_columns].copy()
    if reviews.prediction_id.isna().any():
        raise ValueError("results has missing prediction_id")
    if not reviews[tolerance_column].isin(tolerances).all():
        raise ValueError(f"{tolerance_column} must be one of {tolerances}")
    if reviews.duplicated(["prediction_id", tolerance_column]).any():
        raise ValueError(f"results has duplicate prediction_id/{tolerance_column}")
    selected_ids = set(sample.prediction_id)
    for delta in tolerances:
        ids = set(reviews.loc[reviews[tolerance_column].eq(delta), "prediction_id"])
        if ids != selected_ids:
            raise ValueError(f"results for tolerance {delta} must cover all and only selected prediction_id values")
    if not reviews.review_status.isin(("pending", "reviewed")).all():
        raise ValueError("review_status must be pending or reviewed")
    valid_judgement = reviews.judgement.isin(("T", "F", "U"))
    blank_judgement = reviews.judgement.isna() | reviews.judgement.eq("")
    if (reviews.review_status.eq("reviewed") & ~valid_judgement).any():
        raise ValueError("reviewed results require a T/F/U judgement")
    if (~valid_judgement & ~blank_judgement).any():
        raise ValueError("judgement must be T/F/U or blank for pending reviews")
    complete = bool(reviews.review_status.eq("reviewed").all())
    summaries = []
    for delta in tolerances:
        joined = sample.merge(
            reviews.loc[reviews[tolerance_column].eq(delta)],
            on="prediction_id", how="left", validate="one_to_one", sort=False,
        )
        reviewed = joined.review_status.eq("reviewed")
        n_unknown = int((reviewed & joined.judgement.eq("U")).sum())
        row = {
            tolerance_column: delta, "is_primary_result": tolerance_unit == "quarter" and delta == 1,
            "review_status": "reviewed" if complete else "pending",
            "population_size": population_size, "n_selected": len(sample),
            "n_reviewed": int(reviewed.sum()), "n_pending": int((~reviewed).sum()),
            "n_T": int((reviewed & joined.judgement.eq("T")).sum()),
            "n_F": int((reviewed & joined.judgement.eq("F")).sum()), "n_U": n_unknown,
            "sample_weight_sum": float(sample.sampling_weight.sum()),
            "degrees_of_freedom": degrees_of_freedom, "is_population_census": is_census,
            **dict.fromkeys(ESTIMATE_COLUMNS, np.nan),
            "ci_status": "pending",
            "ci_note": ("Both tolerances must be fully reviewed before S4 estimation."
                        if tolerance_unit == "quarter" and tolerances == (0, 1)
                        else "All requested tolerances must be fully reviewed before estimation."),
        }
        if tolerance_unit == "month":
            row["tolerance_unit"] = tolerance_unit
        if complete:
            estimates = {}
            for label, indicator in (
                ("lower", joined.judgement.eq("T")),
                ("upper", joined.judgement.isin(("T", "U"))),
                ("unknown", joined.judgement.eq("U")),
            ):
                estimated = _ratio(joined, indicator.to_numpy(dtype=float))
                estimates[label] = estimated
                row[f"{label}_hajek"] = estimated["estimate"]
                row[f"{label}_ht"] = estimated["weighted_total"] / population_size
                row[f"se_{label}"] = estimated["se"]
                row[f"variance_{label}"] = estimated["variance"]
            if n_unknown == 0:
                row["precision_hajek"] = row["lower_hajek"]
                row["precision_ht"] = row["lower_ht"]
            if is_census:
                row.update(ci95_lower=row["lower_hajek"], ci95_upper=row["upper_hajek"],
                           ci_status="population_census", ci_note="Census scenario endpoints; no sampling uncertainty.")
            # Residual totals that should cancel can retain machine-roundoff
            # noise. This threshold is on the squared proportion scale and is
            # far below a meaningful sampling variance in frozen D=1266.
            elif any(estimates[label]["variance"] <= 64 * np.finfo(float).eps**2
                     for label in ("lower", "upper")):
                row.update(ci_status="degenerate_variance", ci_note=(
                    "A scenario endpoint has zero linearized variance while population units remain unobserved; "
                    "the approximate interval is unavailable."
                ))
            elif degrees_of_freedom <= 0:
                row.update(ci_status="insufficient_degrees_of_freedom", ci_note=(
                    "No noncensus first-stage stratum contributes degrees of freedom for the prespecified t interval."
                ))
            else:
                q = float(t.ppf(0.975, degrees_of_freedom))
                row.update(
                    t_critical=q,
                    ci95_lower=max(0.0, row["lower_hajek"] - q * row["se_lower"]),
                    ci95_upper=min(1.0, row["upper_hajek"] + q * row["se_upper"]),
                    ci_status="approximate_95_percent",
                    ci_note="Per-tolerance approximate outer interval; not simultaneous coverage across tolerances.",
                )
        summaries.append(row)
    return pd.DataFrame(summaries)
