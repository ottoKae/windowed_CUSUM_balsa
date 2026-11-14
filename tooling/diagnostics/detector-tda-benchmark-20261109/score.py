"""Evaluate fixed detector operating points after the production TDA adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import ROOT, OUT, REFERENCES, digest, load_module, write_json

METHODS = {
    "windowed_cusum_w36_firstpost": "windowed CuSum W36",
    "maximum_drop_t125": "Maximum monthly decline",
    "classical_cusum_k075_h3": "Classical lower CuSum",
    "bocd_h24_d6_decrease": "BOCD prototype (VV+VH)",
    "nrt_b12_a6_tc98_raw": "Monthly CuSum NRT (raw)",
    "nrt_b12_a6_tc98_decrease": "Monthly CuSum NRT (decrease)",
    "recusum_decrease_c09_m12": "ReCuSum legacy (decrease)",
    "recusum_raw_c09_m18": "ReCuSum legacy (raw)",
    "recusum_notebook_firstpost_c09_m12_adapted": "ReCuSum first-post adaptation",
    "ccdc_p95_n3_l01": "CCDC original configuration",
    "ccdc_expanded_p95_n4_l001": "CCDC expanded configuration",
    "structural_h1_n12_b3": "Structural breakpoints",
    "bfast_hist2_h025_a005": "BFAST Monitor adaptation",
    "ewmacd_hist2_l03_p1": "EWMACD adaptation",
}
BASELINE = next(iter(METHODS))
PRIMARY = ("nrt_b12_a6_tc98_raw", "ccdc_p95_n3_l01")
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain"
LEGACY = ROOT / "tooling/diagnostics/balsa-results-optimization-20261105"
sys.path.insert(0, str(LEGACY))
EVAL = load_module("benchmark_frozen_evaluator", LEGACY / "evaluate.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    protocol = OUT / "evaluation_protocol.json"
    if args.freeze:
        if protocol.exists():
            raise FileExistsError("The evaluation contract is already frozen")
        write_json(protocol, dict(
            frozen_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS,
            input_reference_sha256=digest(REFERENCES),
            primary_contrasts=list(PRIMARY), primary_rule="pm2", primary_ci=.975,
            exploratory_ci=.95, rules=["pm1", "pm2", "pm3"],
            bootstrap_unit="stand", bootstrap_draws=5000, seed=20261105,
            parameter_selection="historical_frozen_operating_points_no_new_tuning",
            prior_results_seen=True, out_of_fold=False, report_level="TDA_parent",
            reference_stands=459, reference_events=1049, original_pairs=590,
            code_sha256={str(p): digest(p) for p in [Path(__file__), LEGACY / "evaluate.py"]},
        ))
        print("Evaluation contract frozen before new matching", flush=True)
        return
    contract = json.loads(protocol.read_text(encoding="utf-8"))
    if (contract["methods"] != METHODS or contract["primary_contrasts"] != list(PRIMARY)
            or contract["input_reference_sha256"] != digest(REFERENCES)):
        raise ValueError("Frozen evaluation contract changed")
    refs = pd.read_csv(REFERENCES, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - EVAL.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = EVAL.original_pairs(refs)
    if (len(refs), len(cohort), len(pairs)) != (1049, 459, 590):
        raise ValueError("Reference cohort changed")
    axes = json.loads((FROZEN / "paired_stand_axes.json").read_text(encoding="utf-8"))
    if axes["sample_ids"] != cohort:
        raise ValueError("Paired bootstrap stand order changed")
    index = np.load(FROZEN / "paired_stand_indices.npy")
    if index.shape != (5000, 459):
        raise ValueError("Paired bootstrap dimensions changed")
    metrics = OUT / "metrics"
    metrics.mkdir(exist_ok=True)
    summaries, all_matches, all_counts, all_pairs = [], {}, {}, {}
    yearly, multiplicity, diagnostics, manifest = [], [], [], {}
    for method, label in METHODS.items():
        path = OUT / "methods" / method
        aggregation = json.loads((path / "aggregation_receipt.json").read_text(encoding="utf-8"))
        detector_path = OUT / "detectors" / method / "detector_receipt.json"
        if aggregation["detector_receipt_sha256"] != digest(detector_path) or not aggregation["node_conservation"]:
            raise ValueError(f"Aggregation input integrity failed: {method}")
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        reports = pd.read_parquet(path / "reports.parquet")
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        if not reports.prediction_id.is_unique:
            raise ValueError(f"Duplicate scientific reports: {method}")
        target = path / "metrics"
        target.mkdir(exist_ok=True)
        for rule in contract["rules"]:
            key = (method, rule)
            matched = EVAL.match_reports(refs, reports, rule)
            counts = EVAL.stand_counts(refs, reports, matched, cohort)
            history = EVAL.pair_matches(pairs, matched)
            row = EVAL.summary_row((method, "TDA_parent", rule), counts, history, matched)
            row.update(method_label=label, own_matched_date_MAE=float(matched.absolute_error.mean()),
                       own_matched_date_MedAE=float(matched.absolute_error.median()),
                       original_pair_retention=float(history.found.mean()),
                       n_multiref_stands=int(counts.n.ge(2).sum()))
            row.update(individual_intervals(counts, index))
            summaries.append(row)
            all_matches[key], all_counts[key], all_pairs[key] = matched, counts, history
            matched.to_parquet(target / f"{rule}_matches.parquet", index=False)
            history.to_parquet(target / f"{rule}_original_pairs.parquet", index=False)
            counts.reset_index().to_csv(target / f"{rule}_stand_counts.csv", index=False)
            joined = matched.merge(refs[["event_id", "reference_month"]], on="event_id", validate="one_to_one")
            joined["year"] = joined.reference_month.str[:4].astype(int)
            joined["reference_multiplicity"] = joined.sample_id.map(refs.groupby("sample_id").size())
            for grouping, destination in [("year", yearly), ("reference_multiplicity", multiplicity)]:
                for group, part in joined.groupby(grouping):
                    destination.append(dict(method_id=method, matching_rule=rule, **{grouping: int(group)},
                        references=len(part), matched=int(part.found.sum()), recall=float(part.found.mean())))
        monthly = pd.read_parquet(path / "monthly_activity.parquet")
        monthly = monthly.loc[monthly.sample_id.isin(cohort)]
        diagnostics.append(dict(method_id=method, n_reports=len(reports),
            median_span_months=float(reports.span_months.median()), max_span_months=int(reports.span_months.max()),
            n_span_over12=int(reports.span_months.gt(12).sum()), fraction_span_over12=float(reports.span_months.gt(12).mean()),
            n_left_censored=int(reports.left_censored.sum()), n_right_censored=int(reports.right_censored.sum()),
            **{f"stand_months_{s}": int(n) for s, n in monthly.state.value_counts().items()},
            n_raw_nominations=detector["n_event_rows"], n_unique_pixel_months=detector["n_unique_pixel_months"]))
        manifest[method] = {"detector": detector, "aggregation": aggregation,
            "reports_sha256": digest(path / "reports.parquet")}
        print(f"Scored {method}: {len(reports)} reference-cohort parents", flush=True)
    baseline = next(r for r in summaries if r["variant_id"] == BASELINE and r["matching_rule"] == "pm2")
    if (baseline["n_reports"], baseline["n_matches"]) != (1208, 925):
        raise ValueError("Primary TDA baseline does not reproduce the frozen 1208/925 anchor")
    effects, changes, common_members = paired_comparisons(cohort, index, all_matches, all_counts, all_pairs)
    pd.DataFrame(summaries).to_csv(metrics / "summary.csv", index=False)
    pd.DataFrame(effects).to_csv(metrics / "paired_effects.csv", index=False)
    pd.DataFrame(effects).query("primary_contrast").to_csv(metrics / "primary_contrasts.csv", index=False)
    pd.DataFrame(yearly).to_csv(metrics / "recall_by_reference_year.csv", index=False)
    pd.DataFrame(multiplicity).to_csv(metrics / "recall_by_reference_multiplicity.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(metrics / "report_and_monitoring_diagnostics.csv", index=False)
    pd.concat(changes, ignore_index=True).to_parquet(metrics / "reference_outcome_changes.parquet", index=False)
    pd.concat(common_members, ignore_index=True).to_parquet(metrics / "common_reference_and_pair_sets.parquet", index=False)
    write_json(OUT / "execution_manifest.json", manifest)
    write_json(OUT / "evaluation_receipt.json", dict(status="completed", methods=list(METHODS), n_configurations=len(METHODS),
        references=1049, reference_stands=459, original_pairs=590, bootstrap=5000,
        protocol_sha256=digest(protocol), source_sha256={str(p): digest(p) for p in [Path(__file__), LEGACY / "evaluate.py"]},
        output_sha256={p.name: digest(p) for p in metrics.iterdir() if p.is_file()},
        baseline_anchor_passed=True, no_new_tuning=True, report_level="TDA_parent"))
    print(pd.DataFrame(summaries).query("matching_rule == 'pm2'")[["method_label", "n_reports", "n_matches", "catalogue_precision", "recall", "catalogue_f1"]].to_string(index=False))


def individual_intervals(counts, index):
    hits = counts.hit.to_numpy()[index].sum(axis=1)
    refs = counts.n.to_numpy()[index].sum(axis=1)
    reports = counts.pred.to_numpy()[index].sum(axis=1)
    out = {}
    for metric, draws in [("catalogue_precision", hits / reports), ("recall", hits / refs),
                          ("catalogue_f1", 2 * hits / (refs + reports))]:
        low, high = np.quantile(draws, [.025, .975])
        out[f"{metric}_ci95_low"], out[f"{metric}_ci95_high"] = float(low), float(high)
    return out


def paired_comparisons(cohort, index, matches, counts, histories):
    effects, changes, members = [], [], []
    for (method, rule), found in matches.items():
        if method == BASELINE:
            continue
        key, base = (method, rule), (BASELINE, rule)
        current = found.set_index("event_id")
        main = matches[base].set_index("event_id").reindex(current.index)
        shared = current.found & main.found
        gained, lost = current.found & ~main.found, ~current.found & main.found
        primary = method in PRIMARY and rule == "pm2"
        effect = EVAL.bootstrap_effect(counts[base], counts[key], index, "catalogue_f1", .975 if primary else .95)
        row = dict(comparator=method, method_label=METHODS[method], matching_rule=rule,
            primary_contrast=primary, effect_direction="windowed_CuSum_minus_comparator", **effect,
            comparator_gained=int(gained.sum()), comparator_lost=int(lost.sum()),
            n_common_references=int(shared.sum()), n_common_stands=int(current.loc[shared, "sample_id"].nunique()))
        pair_current = histories[key].set_index("pair_id")
        pair_main = histories[base].set_index("pair_id").reindex(pair_current.index)
        shared_pairs = pair_current.found & pair_main.found
        row["n_common_original_pairs"] = int(shared_pairs.sum())
        for kind, a, b, mask in [("date", main, current, shared), ("interval", pair_main, pair_current, shared_pairs)]:
            left, right = a.loc[mask, ["sample_id", "absolute_error"]], b.loc[mask, ["sample_id", "absolute_error"]]
            lo, hi, valid = EVAL.common_error_ci(left, right, cohort, index)
            row.update({f"main_common_{kind}_MAE": float(left.absolute_error.mean()),
                f"comparator_common_{kind}_MAE": float(right.absolute_error.mean()),
                f"main_common_{kind}_MedAE": float(left.absolute_error.median()),
                f"comparator_common_{kind}_MedAE": float(right.absolute_error.median()),
                f"common_{kind}_MAE_difference": float(left.absolute_error.mean() - right.absolute_error.mean()),
                f"common_{kind}_MAE_difference_ci95_low": lo, f"common_{kind}_MAE_difference_ci95_high": hi,
                f"common_{kind}_bootstrap_valid": valid})
            members.append(pd.DataFrame(dict(comparator=method, matching_rule=rule, kind=kind, member_id=a.index[mask])))
        effects.append(row)
        changes.append(pd.DataFrame(dict(event_id=current.index, sample_id=current.sample_id,
            comparator=method, matching_rule=rule, main_found=main.found, comparator_found=current.found,
            comparator_gained=gained, comparator_lost=lost,
            main_report_month_index=main.report_index, comparator_report_month_index=current.report_index)).reset_index(drop=True))
    return effects, changes, members


if __name__ == "__main__":
    main()
