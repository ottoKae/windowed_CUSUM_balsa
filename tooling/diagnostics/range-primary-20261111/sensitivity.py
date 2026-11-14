"""Frozen configuration sensitivity under reference +/-1-month range overlap."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain"
OUT = ROOT / "runs/validation/range_primary_20261111_v1/sensitivity"
CODE = ROOT / "tooling/diagnostics/balsa-results-optimization-20261105"
sys.path.insert(0, str(CODE))
import evaluate as evaluation

VARIANTS = ("full", "W48", "W60", "VH", "VV_union_VH", "VV_intersection_VH",
            "alpha_0.005", "alpha_0.01", "alpha_0.05", "alpha_0.10", "block3")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference_path = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    draws_path, axes_path = FROZEN / "paired_stand_indices.npy", FROZEN / "paired_stand_axes.json"
    refs = pd.read_csv(reference_path, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = evaluation.original_pairs(refs)
    draws = np.load(draws_path)
    assert (len(refs), len(cohort), len(pairs)) == (1049,459,590)
    assert json.loads(axes_path.read_text())["sample_ids"] == cohort and draws.shape == (5000,459)
    paths = [reference_path,draws_path,axes_path,Path(__file__),Path(evaluation.__file__),
             ROOT / "src/windowed_cusum/evaluation/event_scoring.py"]
    summaries, counts, matches_by, histories, stand_rows = [], {}, {}, {}, []
    old_path = FROZEN / "summary_by_stage.csv"
    paths.append(old_path)
    old = pd.read_csv(old_path)
    for variant in VARIANTS:
        source = FROZEN / "variants" / variant / "octa/reports.parquet"
        paths.append(source)
        reports = pd.read_parquet(source)
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        assert reports.prediction_id.is_unique
        # Expanding a copy of the envelope gives the reference +/-1 intersection edges.
        admissible = reports.copy()
        admissible["first_response_month"] = (pd.PeriodIndex(reports.first_response_month,freq="M")-1).astype(str)
        admissible["last_response_month"] = (pd.PeriodIndex(reports.last_response_month,freq="M")+1).astype(str)
        matches = evaluation.match_reports(refs,admissible,"envelope")
        bounds = reports.set_index("prediction_id")
        for name in ("first_response_month","last_response_month","span_months"):
            matches[name] = matches.prediction_id.map(bounds[name])
        found = matches.loc[matches.found]
        first = pd.PeriodIndex(found.first_response_month,freq="M").asi8 - evaluation.MONTH0
        last = pd.PeriodIndex(found.last_response_month,freq="M").asi8 - evaluation.MONTH0
        assert found.prediction_id.is_unique and found.event_id.is_unique
        assert (first <= found.reference_index.to_numpy()+1).all()
        assert (last >= found.reference_index.to_numpy()-1).all()
        stand = evaluation.stand_counts(refs,reports,matches,cohort)
        history = evaluation.pair_matches(pairs,matches)
        counts[variant], matches_by[variant], histories[variant] = stand,matches,history
        summary = evaluation.summary_row((variant,"octa","reference_pm1_interval_overlap"),stand,history,matches)
        summary.update(median_span_months=float(reports.span_months.median()),
            span_q25=float(reports.span_months.quantile(.25)),span_q75=float(reports.span_months.quantile(.75)),
            reports_over_12_months=int(reports.span_months.gt(12).sum()),
            proportion_over_12_months=float(reports.span_months.gt(12).mean()),max_span_months=int(reports.span_months.max()))
        prior = old.loc[(old.variant_id==variant)&(old.output_level=="octa")]
        pm2 = prior.loc[prior.matching_rule=="pm2"].iloc[0]
        assert summary["n_reports"] == pm2.n_reports
        assert summary["n_matches"] >= prior.loc[prior.matching_rule=="pm1","n_matches"].iloc[0]
        for key in ("n_matches","catalogue_precision","recall","catalogue_f1"):
            summary["secondary_representative_pm2_"+key] = pm2[key]
        summaries.append(summary)
        stand_rows.append(stand.assign(variant_id=variant).reset_index())
        target = OUT / "variants" / variant
        target.mkdir(parents=True,exist_ok=True)
        matches.to_parquet(target / "matches.parquet",index=False)
        history.to_parquet(target / "pairs.parquet",index=False)
    summary = pd.DataFrame(summaries)
    baseline_path = ROOT / "runs/validation/cumulative_components_20261111_v1/range_overlap_pm1/summary.csv"
    paths.append(baseline_path)
    expected = pd.read_csv(baseline_path).query("variant_id == 'F4_permutation'").iloc[0]
    baseline = summary.iloc[0]
    for key in ("n_reports","n_matches","catalogue_precision","recall","catalogue_f1","n_original_pairs_retained"):
        np.testing.assert_allclose(baseline[key],expected[key],rtol=0,atol=1e-14)
    effects = []
    for variant in VARIANTS[1:]:
        left = matches_by[variant].set_index("event_id")
        right = matches_by["full"].set_index("event_id").reindex(left.index)
        common = left.found & right.found
        left_pairs = histories[variant].set_index("pair_id")
        right_pairs = histories["full"].set_index("pair_id").reindex(left_pairs.index)
        common_pairs = left_pairs.found & right_pairs.found
        effect = dict(variant_id=variant,effect_direction="variant_minus_full",
            **evaluation.bootstrap_effect(counts[variant],counts["full"],draws,"catalogue_f1",.95),
            n_matched_new=int((left.found & ~right.found).sum()),n_matched_lost=int((~left.found & right.found).sum()),
            n_common_references=int(common.sum()),n_common_original_pairs=int(common_pairs.sum()))
        for kind,a,b,mask in (("date",left,right,common),("interval",left_pairs,right_pairs,common_pairs)):
            lo,hi,valid = evaluation.common_error_ci(a.loc[mask,["sample_id","absolute_error"]],
                b.loc[mask,["sample_id","absolute_error"]],cohort,draws)
            effect.update({f"variant_common_{kind}_MAE":float(a.loc[mask,"absolute_error"].mean()),
                f"full_common_{kind}_MAE":float(b.loc[mask,"absolute_error"].mean()),
                f"common_{kind}_MAE_difference":float((a.loc[mask,"absolute_error"]-b.loc[mask,"absolute_error"]).mean()),
                f"common_{kind}_MAE_difference_ci_low":lo,f"common_{kind}_MAE_difference_ci_high":hi,
                f"common_{kind}_bootstrap_valid":valid})
        effects.append(effect)
    effects = pd.DataFrame(effects)
    summary.to_csv(OUT / "summary.csv",index=False)
    effects.to_csv(OUT / "paired_effects.csv",index=False)
    pd.concat(stand_rows,ignore_index=True).to_csv(OUT / "per_stand_counts.csv",index=False)
    write_results(summary,effects)
    receipt = dict(status="passed",reference_stands=459,reference_records=1049,bootstrap_replicates=5000,
        matching_rule="same stand; [reference-1,reference+1] intersects original [first_detection,last_detection]",
        one_to_one=True,tie_break="record identifiers",detector_rerun=False,aggregation_rerun=False,
        baseline_cumulative_parity=True,original_report_spans_unchanged=True,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(summary[["variant_id","n_reports","n_matches","catalogue_precision","recall","catalogue_f1","n_original_pairs_retained"]].to_string(index=False))
    print(effects[["variant_id","difference","ci_low","ci_high"]].to_string(index=False))


def write_results(summary,effects):
    s,e = summary.set_index("variant_id"),effects.set_index("variant_id")
    metric_cells = {
        metric: s[metric].map(lambda value: rf"\textbf{{{value:.3f}}}" if value == s[metric].max() else f"{value:.3f}")
        for metric in ("catalogue_precision", "recall", "catalogue_f1")
    }
    groups = [
        ("Context length",[("W48",r"$W=48$ months"),("W60",r"$W=60$ months")]),
        ("Polarization",[("VH","VH"),("VV_union_VH","VV--VH union"),("VV_intersection_VH","VV--VH intersection")]),
        ("Acceptance threshold",[(f"alpha_{a}",rf"$\alpha_{{\mathrm{{ts}}}}={a}$") for a in ("0.005","0.01","0.05","0.10")]),
        ("Temporal permutation",[("block3","Three-month calendar blocks")]),
    ]
    lines = [r"\subsubsection{Configuration sensitivity and temporal robustness}",
        r"\label{sec:cusum_configuration_sensitivity}", "",
        r"Increasing the context length from 36 to 48 and 60 months, with $D=S=12$, increased reference-based precision but reduced recall (Table~\ref{tab:cusum_configuration_sensitivity}). "
        f"F1 was {s.loc['W48','catalogue_f1']:.3f} for W48 and {s.loc['W60','catalogue_f1']:.3f} for W60, compared with {s.loc['full','catalogue_f1']:.3f} for W36. "
        r"The paired 95\% confidence interval for W48 included zero, whereas W60 had lower F1 "
        rf"($\Delta F_{{1,\mathrm{{ref}}}}={e.loc['W60','difference']:+.3f}$; 95\% CI: ${e.loc['W60','ci_low']:+.3f}$ to ${e.loc['W60','ci_high']:+.3f}$). "
        f"Recovery of adjacent reference pairs declined from {s.loc['full','n_original_pairs_retained']:.0f}/590 to {s.loc['W48','n_original_pairs_retained']:.0f}/590 and {s.loc['W60','n_original_pairs_retained']:.0f}/590, respectively.", "",
        r"\begin{table*}[pos=!htbp]",
        r"    \centering\small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{Configuration sensitivity of windowed CuSum under common TDA rules and one-to-one matching between each reference month $\pm1$ month and the first--last detection range. Each alternative changed the indicated setting relative to the baseline: VV, $W=36$, $D=S=12$, $\alpha_{\mathrm{ts}}=0.025$, and unrestricted permutation. $\Delta F_{1,\mathrm{ref}}$ denotes the alternative minus the baseline, with 95\% percentile intervals from 5,000 paired stand-level bootstrap resamples. Bold values mark the maximum in each accuracy column.}",
        r"    \label{tab:cusum_configuration_sensitivity}",
        r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrc@{}}",
        r"        \toprule",
        r"        Configuration & $N_{\mathrm{det}}$ & $M$ & $P_{\mathrm{ref}}$ & $R_{\mathrm{ref}}$ & $F_{1,\mathrm{ref}}$ & $\Delta F_{1,\mathrm{ref}}$ (95\% CI) \\",
        r"        \midrule",
        f"        Baseline & {s.loc['full','n_reports']:,.0f} & {s.loc['full','n_matches']:,.0f} & {metric_cells['catalogue_precision']['full']} & {metric_cells['recall']['full']} & {metric_cells['catalogue_f1']['full']}" + r" & -- \\",
    ]
    for heading,rows in groups:
        lines.extend([r"        \midrule",r"        \multicolumn{7}{@{}l}{\textit{"+heading+r"}} \\"])
        for variant,label in rows:
            row,delta = s.loc[variant],e.loc[variant]
            lines.append(f"        {label} & {row.n_reports:,.0f} & {row.n_matches:,.0f} & {metric_cells['catalogue_precision'][variant]} & {metric_cells['recall'][variant]} & {metric_cells['catalogue_f1'][variant]} & "
                rf"${delta.difference:+.3f}$ [${delta.ci_low:+.3f}$, ${delta.ci_high:+.3f}$] \\")
    lines.extend([r"        \bottomrule",r"    \end{tabular*}",r"\end{table*}","",
        f"VV alone achieved the highest F1 among the polarization strategies. VH yielded F1 of {s.loc['VH','catalogue_f1']:.3f}. "
        f"The VV--VH union increased recall to {s.loc['VV_union_VH','recall']:.3f} but reduced precision to {s.loc['VV_union_VH','catalogue_precision']:.3f}, yielding F1 of {s.loc['VV_union_VH','catalogue_f1']:.3f}. "
        f"The intersection increased precision to {s.loc['VV_intersection_VH','catalogue_precision']:.3f} but reduced recall to {s.loc['VV_intersection_VH','recall']:.3f}, yielding F1 of {s.loc['VV_intersection_VH','catalogue_f1']:.3f}.","",
        rf"The highest threshold-specific F1 estimate was {s.loc['alpha_0.01','catalogue_f1']:.3f} at $\alpha_{{\mathrm{{ts}}}}=0.01$, "
        rf"with a paired difference of ${e.loc['alpha_0.01','difference']:+.3f}$ from the baseline (95\% CI: ${e.loc['alpha_0.01','ci_low']:+.3f}$ to ${e.loc['alpha_0.01','ci_high']:+.3f}$). "
        rf"Increasing $\alpha_{{\mathrm{{ts}}}}$ to 0.05 and 0.10 raised recall to {s.loc['alpha_0.05','recall']:.3f} and {s.loc['alpha_0.10','recall']:.3f}, "
        f"while precision declined to {s.loc['alpha_0.05','catalogue_precision']:.3f} and {s.loc['alpha_0.10','catalogue_precision']:.3f}. "
        f"The resulting F1 values were {s.loc['alpha_0.05','catalogue_f1']:.3f} and {s.loc['alpha_0.10','catalogue_f1']:.3f}.","",
        rf"Replacing unrestricted permutation with three-month calendar blocks at the same $\alpha_{{\mathrm{{ts}}}}=0.025$ "
        f"increased precision to {s.loc['block3','catalogue_precision']:.3f} but reduced recall to {s.loc['block3','recall']:.3f}. "
        f"The resulting {s.loc['block3','n_reports']:,.0f} TDA events matched {s.loc['block3','n_matches']:,.0f} reference records, yielding F1 of {s.loc['block3','catalogue_f1']:.3f} "
        rf"($\Delta F_{{1,\mathrm{{ref}}}}={e.loc['block3','difference']:+.3f}$; 95\% CI: ${e.loc['block3','ci_low']:+.3f}$ to ${e.loc['block3','ci_high']:+.3f}$). "
        f"Recovery of adjacent reference pairs decreased to {s.loc['block3','n_original_pairs_retained']:.0f}/590.", ""])
    (OUT / "results_sensitivity.tex").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
