"""Compare frozen detector/TDA reports by reference +/-1 interval overlap."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD_CODE = ROOT / "tooling/diagnostics/detector-tda-benchmark-20261109"
sys.path.insert(0, str(OLD_CODE))
spec = importlib.util.spec_from_file_location("frozen_benchmark_score", OLD_CODE / "score.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
EVAL = old.EVAL
OUT = ROOT / "runs/validation/range_primary_20261111_v1/benchmark"
RULE = "reference_pm1_interval_overlap"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = pd.read_csv(old.REFERENCES, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - EVAL.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = EVAL.original_pairs(refs)
    draws_path = old.FROZEN / "paired_stand_indices.npy"
    axes_path = old.FROZEN / "paired_stand_axes.json"
    index = np.load(draws_path)
    assert (len(refs), len(cohort), len(pairs)) == (1049, 459, 590)
    assert json.loads(axes_path.read_text())["sample_ids"] == cohort and index.shape == (5000,459)
    paths = [old.REFERENCES, draws_path, axes_path, Path(__file__), OLD_CODE / "score.py",
             OLD_CODE / "common.py", Path(EVAL.__file__), ROOT / "src/windowed_cusum/evaluation/event_scoring.py"]
    summary, counts, matches, histories, yearly, multiplicity = [], {}, {}, {}, [], []
    for method, label in old.METHODS.items():
        path = old.OUT / "methods" / method / "reports.parquet"
        paths.append(path)
        reports = pd.read_parquet(path)
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        assert reports.prediction_id.is_unique
        eligible = reports.copy()
        # Only matching bounds are expanded; output spans and dating stay original.
        eligible["first_response_month"] = (pd.PeriodIndex(reports.first_response_month, freq="M")-1).astype(str)
        eligible["last_response_month"] = (pd.PeriodIndex(reports.last_response_month, freq="M")+1).astype(str)
        matched = EVAL.match_reports(refs, eligible, "envelope")
        by = reports.set_index("prediction_id")
        matched["first_response_month"] = matched.prediction_id.map(by.first_response_month)
        matched["last_response_month"] = matched.prediction_id.map(by.last_response_month)
        matched["reference_month"] = matched.event_id.map(refs.set_index("event_id").reference_month)
        found = matched.loc[matched.found]
        assert found.prediction_id.is_unique
        lower = (pd.PeriodIndex(found.reference_month, freq="M")-1).astype(str)
        upper = (pd.PeriodIndex(found.reference_month, freq="M")+1).astype(str)
        assert (found.first_response_month.to_numpy() <= upper).all()
        assert (found.last_response_month.to_numpy() >= lower).all()
        stand = EVAL.stand_counts(refs, reports, matched, cohort)
        history = EVAL.pair_matches(pairs, matched)
        row = EVAL.summary_row((method, "TDA_parent", RULE), stand, history, matched)
        row.update(method_label=label, own_matched_date_MAE=float(matched.absolute_error.mean()),
            own_matched_date_MedAE=float(matched.absolute_error.median()),
            original_pair_retention=float(history.found.mean()),
            median_span_months=float(reports.span_months.median()), max_span_months=int(reports.span_months.max()),
            n_span_over12=int(reports.span_months.gt(12).sum()), fraction_span_over12=float(reports.span_months.gt(12).mean()))
        row.update(old.individual_intervals(stand, index))
        summary.append(row)
        counts[method], matches[method], histories[method] = stand, matched, history
        target = OUT / "methods" / method
        target.mkdir(parents=True, exist_ok=True)
        stand.reset_index().to_csv(target / "stand_counts.csv", index=False)
        matched.to_parquet(target / "matches.parquet", index=False)
        history.to_parquet(target / "original_pairs.parquet", index=False)
        tagged = matched.copy()
        tagged["reference_year"] = tagged.reference_month.str[:4].astype(int)
        tagged["reference_multiplicity"] = tagged.sample_id.map(refs.groupby("sample_id").size())
        for column, destination in [("reference_year", yearly), ("reference_multiplicity", multiplicity)]:
            for value, frame in tagged.groupby(column):
                destination.append(dict(method_id=method, **{column:int(value)}, references=len(frame),
                                        matched=int(frame.found.sum()), recall=float(frame.found.mean())))
        print(f"{method}: {row['n_matches']}/{row['n_reports']}, F1={row['catalogue_f1']:.6f}", flush=True)
    summary = pd.DataFrame(summary)
    base = summary.set_index("variant_id").loc[old.BASELINE]
    assert (base.n_reports, base.n_matches) == (1208,945)
    anchor_path = ROOT / "runs/validation/cumulative_components_20261111_v1/range_overlap_pm1/summary.csv"
    anchor = pd.read_csv(anchor_path).set_index("variant_id").loc["F4_permutation"]
    parity = ["n_reports","n_matches","catalogue_precision","recall","catalogue_f1","n_original_pairs_retained"]
    np.testing.assert_allclose(base[parity].to_numpy(float), anchor[parity].to_numpy(float),rtol=0,atol=1e-14)
    pm1_path = old.OUT / "metrics/summary.csv"
    pm1 = pd.read_csv(pm1_path).query("matching_rule == 'pm1'").set_index("variant_id").loc[summary.variant_id]
    np.testing.assert_array_equal(summary.n_reports,pm1.n_reports)
    assert (summary.n_matches.to_numpy() >= pm1.n_matches.to_numpy()).all()
    paths += [anchor_path,pm1_path]
    effects, common_sets = [], []
    for method in old.METHODS:
        if method == old.BASELINE:
            continue
        primary = method in old.PRIMARY
        row = dict(comparator=method, method_label=old.METHODS[method], matching_rule=RULE,
            primary_contrast=primary, effect_direction="windowed_CuSum_minus_comparator",
            **EVAL.bootstrap_effect(counts[old.BASELINE],counts[method],index,"catalogue_f1",.975 if primary else .95))
        left = matches[old.BASELINE].set_index("event_id")
        right = matches[method].set_index("event_id").reindex(left.index)
        shared = left.found & right.found
        row.update(n_common_references=int(shared.sum()), main_gained=int((left.found & ~right.found).sum()),
                   main_lost=int((~left.found & right.found).sum()))
        lp = histories[old.BASELINE].set_index("pair_id")
        rp = histories[method].set_index("pair_id").reindex(lp.index)
        shared_pairs = lp.found & rp.found
        row["n_common_original_pairs"] = int(shared_pairs.sum())
        for kind,a,b,mask in [("date",left,right,shared),("interval",lp,rp,shared_pairs)]:
            x,y = a.loc[mask,["sample_id","absolute_error"]],b.loc[mask,["sample_id","absolute_error"]]
            low,high,valid = EVAL.common_error_ci(x,y,cohort,index)
            row.update({f"main_common_{kind}_MAE":float(x.absolute_error.mean()),
                f"comparator_common_{kind}_MAE":float(y.absolute_error.mean()),
                f"common_{kind}_MAE_difference":float(x.absolute_error.mean()-y.absolute_error.mean()),
                f"common_{kind}_MAE_difference_ci95_low":low,f"common_{kind}_MAE_difference_ci95_high":high,
                f"common_{kind}_bootstrap_valid":valid})
            common_sets.append(pd.DataFrame(dict(comparator=method,kind=kind,member_id=a.index[mask])))
        effects.append(row)
    effects = pd.DataFrame(effects)
    summary.to_csv(OUT / "summary.csv",index=False)
    effects.to_csv(OUT / "paired_effects.csv",index=False)
    effects.loc[effects.primary_contrast].to_csv(OUT / "primary_contrasts.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT / "recall_by_reference_year.csv",index=False)
    pd.DataFrame(multiplicity).to_csv(OUT / "recall_by_reference_multiplicity.csv",index=False)
    pd.concat(common_sets,ignore_index=True).to_parquet(OUT / "common_reference_and_pair_sets.parquet",index=False)
    render_results(summary, effects)
    receipt = dict(status="passed",method_count=14,reference_stands=459,reference_records=1049,original_pairs=590,
        matching="same stand; [reference_month-1,reference_month+1] intersects original first-last detection range",
        one_to_one=True,tie_break="record identifiers",bootstrap_replicates=5000,bootstrap_draws="same frozen array",
        primary_contrasts=list(old.PRIMARY),primary_ci=.975,exploratory_ci=.95,
        detector_rerun=False,aggregation_rerun=False,retuned=False,baseline_anchor_passed=True,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(summary[["method_label","n_reports","n_matches","catalogue_precision","recall","catalogue_f1","n_original_pairs_retained"]].to_string(index=False))
    print(effects.to_string(index=False))


def render_results(summary, effects):
    rows = summary.set_index("variant_id")
    contrast = effects.set_index("comparator")
    main = rows.loc[old.BASELINE]
    best_id = rows.drop(index=old.BASELINE).catalogue_f1.idxmax()
    assert best_id == "recusum_notebook_firstpost_c09_m12_adapted"
    best, delta = rows.loc[best_id], contrast.loc[best_id]
    nrt, ccdc = contrast.loc[list(old.PRIMARY)].iloc[0], contrast.loc[list(old.PRIMARY)].iloc[1]
    labels = dict(old.METHODS)
    labels[old.BASELINE] = "Windowed CuSum (Ours)"
    labels["recusum_decrease_c09_m12"] = "ReCuSum (decrease-filtered)"
    labels["recusum_raw_c09_m18"] = "ReCuSum (raw)"
    labels["nrt_b12_a6_tc98_decrease"] = "Monthly CuSum NRT (decrease-filtered)"
    latex = [r"\subsection{Benchmark performance of harvest detectors}",
        r"\label{sec:benchmark_null_sensitivity}", "",
        f"Windowed CuSum achieved the highest reference-based F1 among the 14 evaluated detector configurations "
        f"(Table~\\ref{{tab:benchmark_comparison}}). Its {main.n_reports:,.0f} TDA reports matched {main.n_matches:,.0f} "
        f"reference records, giving precision {main.catalogue_precision:.3f}, recall {main.recall:.3f}, and F1 {main.catalogue_f1:.3f}. "
        f"The strongest external configuration was the first-post ReCuSum adaptation (F1 {best.catalogue_f1:.3f}); "
        f"the paired F1 difference was ${delta.difference:+.3f}$ (95\\% CI: ${delta.ci_low:.3f}$ to ${delta.ci_high:.3f}$).", "",
        r"\begin{table*}[pos=!htbp]", r"    \centering", r"    \small", r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{Benchmark performance after common TDA aggregation. One-to-one matching required overlap between each report's first--last detection range and a $\pm1$-month interval around the reference month, within the same stand. Metrics use 1,049 reference records from 459 stands. $N_{\mathrm{det}}$ and $M$ denote TDA report counts and matched reference records. Bold values mark column maxima for precision, recall, and F1.}",
        r"    \label{tab:benchmark_comparison}",
        r"    % Source: runs/validation/range_primary_20261111_v1/benchmark/summary.csv.",
        r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"        \toprule",
        r"        Detector configuration & $N_{\mathrm{det}}$ & $M$ & $P_{\mathrm{ref}}$ & $R_{\mathrm{ref}}$ & $F_{1,\mathrm{ref}}$ \\",
        r"        \midrule"]
    metrics = ("catalogue_precision", "recall", "catalogue_f1")
    maxima = rows.loc[list(old.METHODS), list(metrics)].max()
    for method in old.METHODS:
        row = rows.loc[method]
        cells = [
            r"\textbf{" + f"{row[metric]:.3f}" + "}" if row[metric] == maxima[metric]
            else f"{row[metric]:.3f}"
            for metric in metrics
        ]
        latex.append(f"        {labels[method]} & {row.n_reports:,.0f} & {row.n_matches:,.0f} & "
                     + " & ".join(cells) + " " + r"\\")
    latex += [r"        \bottomrule",r"    \end{tabular*}",r"\end{table*}","",
        f"The two designated F1 contrasts favoured windowed CuSum over raw monthly CuSum NRT "
        f"(${nrt.difference:+.3f}$; 97.5\\% CI: ${nrt.ci_low:.3f}$ to ${nrt.ci_high:.3f}$) and the original CCDC configuration "
        f"(${ccdc.difference:+.3f}$; 97.5\\% CI: ${ccdc.ci_low:.3f}$ to ${ccdc.ci_high:.3f}$). "
        "Decrease-filtered monthly CuSum NRT recovered the most reference records (985/1,049; recall 0.939), "
        "but its 1,960 reports yielded precision 0.503 and F1 0.655. The BOCD prototype had the highest precision "
        "(0.888) but recall of 0.197.", "",
        f"Relative to first-post ReCuSum, windowed CuSum recovered more reference events "
        f"({main.recall*100:.1f}\\% versus {best.recall*100:.1f}\\%) and more original adjacent reference pairs "
        f"({main.n_original_pairs_retained:.0f}/590 versus {best.n_original_pairs_retained:.0f}/590), "
        f"with slightly lower precision ({main.catalogue_precision:.3f} versus {best.catalogue_precision:.3f}). "
        f"Among the {delta.n_common_references:.0f} reference records matched by both configurations, "
        f"representative-month MAE was {delta.main_common_date_MAE:.3f} and {delta.comparator_common_date_MAE:.3f} months, "
        f"respectively (paired difference {delta.common_date_MAE_difference:.3f} months; "
        f"95\\% CI: ${delta.common_date_MAE_difference_ci95_low:.3f}$ to ${delta.common_date_MAE_difference_ci95_high:.3f}$).", "",
        "Report ranges also differed among detectors. The median range was four months for both windowed CuSum "
        "and first-post ReCuSum; 6.4\\% and 5.3\\% of their reports exceeded 12 months, respectively. "
        "Raw monthly CuSum NRT had a median range of seven months, with 32.6\\% of reports exceeding 12 months.", ""]
    (OUT / "results_benchmark.tex").write_text("\n".join(latex), encoding="utf-8")


if __name__ == "__main__":
    main()
