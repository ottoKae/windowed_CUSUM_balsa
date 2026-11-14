"""Direct agreement with the frozen, incomplete Balsa reference catalogue.

Unmatched reports are not verified false positives. Precision_label and
F1_label describe catalogue agreement, not population harvest precision/F1.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lr_cusum.catalogue_evaluation import bootstrap_ratios, frozen_recall


REPORT_COLUMNS = ["prediction_id", "sample_id", "representative_month", "response_months"]
SCOPE = "incomplete_month_catalogue_agreement_not_population_precision_or_F1"


def evaluate_labels(method, reports, references, eligibility, stands):
    """Match monthly labels one-to-one in their fixed stand cohort.

    The ten interval labels use a separate call and never enter the monthly
    recall, precision_label, or F1_label denominators.
    """
    reports = reports.copy()
    if reports.empty:
        reports = reports.reindex(columns=sorted(set(reports.columns) | set(REPORT_COLUMNS)))
    if not set(REPORT_COLUMNS).issubset(reports.columns):
        raise ValueError("reports require prediction identity, stand, month, and response months")
    if (reports.prediction_id.isna().any() or reports.prediction_id.duplicated().any()
            or reports.prediction_id.eq("").any() or reports.sample_id.isna().any()):
        raise ValueError("report identities must be present and unique")
    if (references.event_id.duplicated().any() or eligibility.event_id.duplicated().any()
            or set(references.event_id) != set(eligibility.event_id)):
        raise ValueError("reference and eligibility identities must agree one-to-one")
    if len(reports):
        months = pd.PeriodIndex(reports.representative_month, freq="M")
        if months.isna().any() or (months < "2017-01").any() or (months > "2025-12").any():
            raise ValueError("reports must retain the frozen 2017-2025 report period")
    eligible_ids = set(eligibility.loc[eligibility.eligibility.eq("eligible"), "event_id"])
    qualified = references.loc[references.event_id.isin(eligible_ids)]
    month_refs = qualified.loc[qualified.timing_type.eq("month")]
    if month_refs.empty:
        raise ValueError("at least one eligible monthly reference is required")
    cohort = sorted(month_refs.sample_id.unique())
    if stands.sample_id.duplicated().any() or not set(cohort).issubset(stands.sample_id):
        raise ValueError("stand information must uniquely cover the fixed reference cohort")

    cohort_reports = reports.loc[reports.sample_id.isin(cohort)]
    month_eligibility = eligibility.loc[eligibility.event_id.isin(month_refs.event_id)]
    matches, _ = frozen_recall(month_refs, month_eligibility, cohort_reports)
    matches["method"] = method
    matches["absolute_month_error"] = np.nan
    found = matches.found
    if found.any():
        actual = pd.PeriodIndex(matches.loc[found, "representative_month"], freq="M").asi8
        expected = pd.PeriodIndex(matches.loc[found, "reference_month"], freq="M").asi8
        matches.loc[found, "absolute_month_error"] = np.abs(actual - expected)
    n_predictions = cohort_reports.groupby("sample_id").size().reindex(cohort, fill_value=0)
    summary, per_stand = [], []
    for tolerance, part in matches.groupby("tolerance_months", sort=True):
        counts = part.groupby("sample_id").agg(
            n_reference_months=("event_id", "size"), matched_labels=("found", "sum"))
        counts["n_reports_cohort"] = n_predictions
        counts["unmatched_labels"] = counts.n_reference_months - counts.matched_labels
        counts["unmatched_reports"] = counts.n_reports_cohort - counts.matched_labels
        counts["method"], counts["tolerance_months"] = method, tolerance
        per_stand.append(counts.reset_index())
        n_ref, n_pred, tp = len(part), len(cohort_reports), int(part.found.sum())
        errors = part.loc[part.found, "absolute_month_error"]
        summary.append(dict(method=method, tolerance_months=int(tolerance),
            is_primary_result=tolerance == 2, n_reference_months=n_ref,
            n_reference_stands=len(cohort), n_reports_cohort=n_pred, n_reports_all=len(reports),
            n_reports_outside_cohort=len(reports) - n_pred, matched_labels=tp,
            unmatched_labels=n_ref - tp, unmatched_reports=n_pred - tp,
            recall_label=tp / n_ref, precision_label=tp / n_pred if n_pred else np.nan,
            F1_label=2 * tp / (n_ref + n_pred),
            mean_abs_month_error=float(errors.mean()), median_abs_month_error=float(errors.median()),
            metric_scope=SCOPE))

    interval_refs = qualified.loc[qualified.timing_type.eq("interval")]
    interval_summary = pd.DataFrame(columns=["method", "tolerance_months",
        "n_interval_references", "n_interval_compatible", "metric_scope"])
    if len(interval_refs):
        interval_eligibility = eligibility.loc[eligibility.event_id.isin(interval_refs.event_id)]
        _, raw_interval_summary = frozen_recall(interval_refs, interval_eligibility, reports)
        interval_summary = raw_interval_summary[["tolerance_months", "frozen_interval_references",
            "frozen_interval_compatible"]].rename(columns={
                "frozen_interval_references": "n_interval_references",
                "frozen_interval_compatible": "n_interval_compatible"})
        interval_summary["method"] = method
        interval_summary["metric_scope"] = "separate_actual_response_month_interval_compatibility"
    return dict(summary=pd.DataFrame(summary), matches=matches,
                per_stand=pd.concat(per_stand, ignore_index=True), interval_summary=interval_summary)


def paired_summary(per_stand_all, stands, baseline="B0"):
    """Paired spatial-block differences using the existing frozen resampling."""
    if baseline not in set(per_stand_all.method):
        raise ValueError("baseline is absent")
    if per_stand_all.duplicated(["method", "tolerance_months", "sample_id"]).any():
        raise ValueError("per-stand results must be unique within method and tolerance")
    rows = []
    for tolerance, group in per_stand_all.groupby("tolerance_months", sort=True):
        base = group.loc[group.method.eq(baseline)].set_index("sample_id").sort_index()
        info = stands.loc[stands.sample_id.isin(base.index)]
        base_draws, base_values = {}, {}
        for method, part in group.groupby("method", sort=True):
            part = part.set_index("sample_id").sort_index()
            if (not part.index.equals(base.index)
                    or not part.n_reference_months.equals(base.n_reference_months)):
                raise ValueError("paired methods must retain identical reference stands and counts")
            for name, numerator, denominator in (
                ("recall_label", part.matched_labels, part.n_reference_months),
                ("precision_label", part.matched_labels, part.n_reports_cohort),
                ("F1_label", 2 * part.matched_labels, part.n_reference_months + part.n_reports_cohort),
            ):
                if name not in base_draws:
                    base_num = 2 * base.matched_labels if name == "F1_label" else base.matched_labels
                    base_den = (base.n_reference_months if name == "recall_label" else
                                base.n_reports_cohort if name == "precision_label" else
                                base.n_reference_months + base.n_reports_cohort)
                    base_draws[name] = bootstrap_ratios(base_num, base_den, info)
                    base_values[name] = base_num.sum() / base_den.sum() if base_den.sum() else np.nan
                draws = bootstrap_ratios(numerator, denominator, info)
                differences = draws - base_draws[name]
                finite = differences[np.isfinite(differences)]
                lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
                value = numerator.sum() / denominator.sum() if denominator.sum() else np.nan
                rows.append(dict(method=method, baseline=baseline, tolerance_months=tolerance,
                    metric=name, estimate=float(value), baseline_estimate=float(base_values[name]),
                    difference=float(value - base_values[name]), ci_lower=float(lower), ci_upper=float(upper),
                    n_bootstrap_valid=len(finite), n_bootstrap_total=len(draws),
                    metric_scope=SCOPE, ci_scope="paired_conditional_spatial_block_not_population"))
    return pd.DataFrame(rows)
