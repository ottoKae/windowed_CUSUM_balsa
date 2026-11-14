"""Strict one-to-one month-in-response-range evaluation of frozen reports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/release"),
               str(ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101")]
from build_sample_release import original_tables
from lr_cusum.event_scoring import _canonical_matching
from label_evaluation import paired_summary

SOURCE = ROOT / "runs/validation/balsa_sts_episode_v4_1_20261102_v1"
OUT = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1"
RULES = ("envelope", "actual_months")
MAIN = ["R0_B20", "R2_TIME", "M13_NORM_Q50", "M13_DB_Q50", "M13_CONTEXT_Q50", "M13_RELIABLE_Q50", "SPACE_ON"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalize_reports(reports):
    if reports.prediction_id.duplicated().any() or reports.prediction_id.isna().any() or reports.prediction_id.eq("").any():
        raise ValueError("Prediction identities must be unique and nonempty")
    result = reports.copy()
    calendars, spans = [], []
    for report in result.itertuples():
        raw = json.loads(report.response_months)
        if not raw or raw != sorted(set(raw)):
            raise ValueError("Actual response months must be nonempty, unique and sorted")
        periods = pd.PeriodIndex(raw, freq="M")
        if periods.isna().any() or list(periods.astype(str)) != raw or (periods < "2017-01").any() or (periods > "2025-12").any():
            raise ValueError("Actual responses must retain valid frozen calendar months")
        first, last = raw[0], raw[-1]
        span = int(periods[-1].ordinal-periods[0].ordinal+1)
        for column, value in [("first_response_month", first), ("last_response_month", last), ("span_months", span)]:
            if hasattr(report, column) and getattr(report, column) != value:
                raise ValueError(f"Stored response range disagrees with actual months: {column}")
        calendars.append(raw)
        spans.append(span)
    result["actual_month_list"] = calendars
    result["first_response_month"] = [x[0] for x in calendars]
    result["last_response_month"] = [x[-1] for x in calendars]
    result["span_months"] = spans
    result["n_actual_months"] = [len(x) for x in calendars]
    result["n_gap_months"] = result.span_months-result.n_actual_months
    result["span_over_12"] = result.span_months.gt(12)
    return result


def match_rule(references, reports, rule):
    """Only temporal compatibility; equal costs never imply verified identity."""
    if rule not in RULES or references.event_id.duplicated().any():
        raise ValueError("One rule and unique reference events are required")
    groups = {s: g.to_dict("records") for s, g in reports.groupby("sample_id")}
    details, opportunities = [], []
    for sample, refs in references.groupby("sample_id", sort=True):
        candidates = groups.get(sample, [])
        lookup = {r["prediction_id"]: r for r in candidates}
        edges = {}
        for ref in refs.itertuples():
            for report in candidates:
                compatible = (report["first_response_month"] <= ref.reference_month <= report["last_response_month"]
                              if rule == "envelope" else ref.reference_month in report["actual_month_list"])
                if compatible:
                    edges[ref.event_id, report["prediction_id"]] = (1, 0)
        paired = _canonical_matching(edges)
        winner = {event: prediction for prediction, event in paired.items()}
        counts_ref = Counter(event for event, _ in edges)
        counts_report = Counter(prediction for _, prediction in edges)
        for ref in refs.itertuples():
            prediction = winner.get(ref.event_id, "")
            report = lookup.get(prediction)
            on_actual = bool(report is not None and ref.reference_month in report["actual_month_list"])
            details.append(dict(matching_rule=rule, event_id=ref.event_id, sample_id=sample,
                date_basis=ref.date_basis, reference_month=ref.reference_month, found=bool(prediction),
                prediction_id=prediction, compatible_reports=counts_ref[ref.event_id],
                first_response_month=report["first_response_month"] if report else "",
                last_response_month=report["last_response_month"] if report else "",
                representative_month=report["representative_month"] if report else "",
                span_months=report["span_months"] if report else np.nan,
                matched_on_actual_response_month=on_actual, matched_in_gap=bool(prediction and not on_actual)))
        for report in candidates:
            opportunities.append(dict(matching_rule=rule, prediction_id=report["prediction_id"], sample_id=sample,
                compatible_references=counts_report[report["prediction_id"]], matched=report["prediction_id"] in paired,
                matched_reference_id=paired.get(report["prediction_id"], "")))
    detail = pd.DataFrame(details)
    if detail.loc[detail.found, "prediction_id"].duplicated().any() or detail.event_id.duplicated().any():
        raise AssertionError("One-to-one reference/report capacity violated")
    return detail, pd.DataFrame(opportunities, columns=["matching_rule", "prediction_id", "sample_id",
        "compatible_references", "matched", "matched_reference_id"])


def evaluate_method(method, references, reports, states):
    reports = normalize_reports(reports)
    cohort = set(references.sample_id)
    selected = reports.loc[reports.sample_id.isin(cohort)].copy()
    ref_counts = references.groupby("sample_id").size()
    pred_counts = selected.groupby("sample_id").size().reindex(ref_counts.index, fill_value=0)
    summaries, all_details, all_opportunities, per_stand = [], [], [], []
    for rule in RULES:
        details, opportunities = match_rule(references, selected, rule)
        details["method"] = method
        details["reference_month_stand_state"] = [states.get((r.sample_id, r.reference_month), "") for r in details.itertuples()]
        opportunities["method"] = method
        tp, n, p = int(details.found.sum()), len(references), len(selected)
        summaries.append(dict(method=method, matching_rule=rule, boundary_tolerance_months=0,
            n_reference_months=n, n_reference_stands=len(cohort), n_reports_cohort=p, n_reports_all=len(reports),
            matched_labels=tp, recall_label=tp/n, precision_label=tp/p if p else np.nan,
            F1_label=2*tp/(n+p), n_gap_matches=int(details.matched_in_gap.sum()),
            n_references_with_compatible_report=int(details.compatible_reports.gt(0).sum()),
            n_references_with_multiple_compatible_reports=int(details.compatible_reports.gt(1).sum()),
            n_compatible_pairs=int(opportunities.compatible_references.sum()),
            n_compatible_but_unmatched=int((details.compatible_reports.gt(0) & ~details.found).sum()),
            n_reports_covering_multiple_references=int(opportunities.compatible_references.gt(1).sum()),
            metric_scope="strict_temporal_catalogue_compatibility_not_verified_harvest_identity"))
        counts = pd.DataFrame({"n_reference_months": ref_counts, "n_reports_cohort": pred_counts})
        counts["matched_labels"] = details.groupby("sample_id").found.sum().reindex(counts.index, fill_value=0)
        counts["method"], counts["matching_rule"], counts["tolerance_months"] = method, rule, 0
        per_stand.append(counts.reset_index())
        all_details.append(details)
        all_opportunities.append(opportunities)
    assert summaries[0]["matched_labels"] >= summaries[1]["matched_labels"]
    selected["method"] = method
    return pd.DataFrame(summaries), pd.concat(all_details), pd.concat(all_opportunities), pd.concat(per_stand), selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    directories = sorted([p for base in [SOURCE / "baselines/methods", SOURCE / "methods", SOURCE / "space/methods"]
                          for p in base.iterdir() if p.is_dir()])
    inputs = [p / name for p in directories for name in ["reports.parquet", "source_members.parquet"]]
    inputs += [SOURCE / "evaluation/label_metrics.csv", SOURCE / "evaluation/reference_matches.parquet",
               ROOT / "runs/validation/balsa_harvest_episode_20261101_v3/monthly_activity.parquet",
               ROOT / "src/lr_cusum/event_scoring.py",
               ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101/label_evaluation.py"]
    inputs += list((ROOT / "manuscript_JAG").rglob("*.tex"))
    tables, config = original_tables()
    inputs += [Path(record["path"]) for record in config["inputs"].values()]
    protected = {str(p): digest(p) for p in inputs}
    write_json(OUT / "protocol_frozen.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        rules=list(RULES), boundary_tolerance_months=0, same_stand=True, one_to_one=True,
        matching="existing_canonical_maximum_cardinality_then_lexicographic; equal_edge_cost=(1,0)",
        span="inclusive_months_from_actual_response_months", long_chain="span>12_months_diagnostic_not_filter",
        reference_semantics="retain_original_date_basis; no_onset_or_duration_ground_truth_assumed",
        point_plusminus3="separate_existing_comparator; edge_sets_not_nested", methods=[p.name for p in directories]))
    write_json(OUT / "input_manifest.json", dict(protected_sha256=protected))
    qualified = set(tables["eligibility"].loc[tables["eligibility"].eligibility.eq("eligible"), "event_id"])
    references = tables["references"].loc[tables["references"].event_id.isin(qualified)
        & tables["references"].timing_type.eq("month")].copy()
    assert len(references) == 1049 and references.sample_id.nunique() == 459
    references.to_csv(OUT / "reference_semantics.csv", index=False)
    monthly = pd.read_parquet(ROOT / "runs/validation/balsa_harvest_episode_20261101_v3/monthly_activity.parquet")
    states = {(r.sample_id, r.month): r.state for r in monthly.itertuples()}
    metrics, matches, opportunities, counts, report_tables = [], [], [], [], []
    for directory in directories:
        reports = pd.read_parquet(directory / "reports.parquet")
        sources = pd.read_parquet(directory / "source_members.parquet")
        actual = sources.loc[sources.prediction_id.isin(reports.prediction_id)].groupby("prediction_id").response_month.agg(
            lambda values: sorted(set(values)))
        assert set(actual.index) == set(reports.prediction_id)
        for r in reports.itertuples():
            assert json.loads(r.response_months) == actual.loc[r.prediction_id]
        result = evaluate_method(directory.name, references, reports, states)
        for target, frame in zip([metrics, matches, opportunities, counts, report_tables], result):
            target.append(frame)
        a, b = result[0].matched_labels
        print(f"{directory.name}: strict envelope={a}; exact actual month={b}; reports={len(result[4])}", flush=True)
    metrics, matches, opportunities, counts, reports = [pd.concat(x, ignore_index=True) for x in
        [metrics, matches, opportunities, counts, report_tables]]
    metrics.to_csv(OUT / "range_metrics.csv", index=False)
    matches.to_parquet(OUT / "reference_matches.parquet", index=False)
    opportunities.to_parquet(OUT / "report_reference_opportunities.parquet", index=False)
    counts.to_parquet(OUT / "per_stand_counts.parquet", index=False)
    reports.drop(columns="actual_month_list").to_parquet(OUT / "report_range_diagnostics.parquet", index=False)
    span_rows = []
    for method, group in reports.groupby("method"):
        span_rows.append(dict(method=method, n_reports=len(group), median_span=float(group.span_months.median()),
            p90_span=float(group.span_months.quantile(.9)), max_span=int(group.span_months.max()),
            n_span_over_12=int(group.span_over_12.sum()), proportion_span_over_12=float(group.span_over_12.mean()),
            median_actual_months=float(group.n_actual_months.median()), n_reports_with_gaps=int(group.n_gap_months.gt(0).sum()),
            n_calendar_month_slots=int(group.span_months.sum()), n_actual_month_slots=int(group.n_actual_months.sum()),
            n_gap_slots=int(group.n_gap_months.sum())))
    spans = pd.DataFrame(span_rows)
    spans.to_csv(OUT / "range_length_summary.csv", index=False)
    wide = matches.pivot(index=["method", "event_id", "sample_id", "reference_month", "date_basis"],
                         columns="matching_rule", values=["found", "prediction_id", "matched_in_gap"])
    wide.columns = ["_".join(x) for x in wide.columns]
    wide.reset_index().to_parquet(OUT / "range_vs_actual_reference_changes.parquet", index=False)
    changes = []
    for rule in RULES:
        selected = matches.loc[matches.matching_rule.eq(rule)]
        for baseline in ["R0_B20", "R2_TIME"]:
            before = set(selected.loc[selected.method.eq(baseline) & selected.found, "event_id"])
            for method, group in selected.groupby("method"):
                after = set(group.loc[group.found, "event_id"])
                gained = group.loc[group.event_id.isin(after-before)]
                changes.append(dict(method=method, baseline=baseline, matching_rule=rule,
                    gained_labels=len(after-before), lost_labels=len(before-after), net_change=len(after)-len(before),
                    gains_in_spans_over_12=int(gained.span_months.gt(12).sum()),
                    gains_in_spans_at_most_12=int(gained.span_months.le(12).sum())))
    pd.DataFrame(changes).to_csv(OUT / "method_reference_changes.csv", index=False)
    semantics = matches.groupby(["method", "matching_rule", "date_basis"]).agg(
        n_references=("event_id", "size"), matched_labels=("found", "sum")).reset_index()
    semantics.to_csv(OUT / "date_basis_summary.csv", index=False)
    paired = []
    for rule, group in counts.groupby("matching_rule"):
        for baseline in ["R0_B20", "R2_TIME"]:
            comparison = paired_summary(group, tables["stands"], baseline=baseline)
            comparison["matching_rule"] = rule
            paired.append(comparison)
    pd.concat(paired, ignore_index=True).to_csv(OUT / "paired_differences.csv", index=False)
    point = pd.read_csv(SOURCE / "evaluation/label_metrics.csv").query("tolerance_months == 3")
    compare = metrics.merge(point[["method", "matched_labels", "recall_label", "precision_label", "F1_label"]],
        on="method", suffixes=("_range", "_point_pm3"), validate="many_to_one")
    compare.to_csv(OUT / "range_vs_point_pm3.csv", index=False)
    write_report(metrics, spans)
    assert all(digest(path) == expected for path, expected in protected.items())
    write_json(OUT / "execution_receipt.json", dict(passed=True, methods=len(directories), matching_rules=2,
        reference_count=1049, reference_stands=459, verified_identity_claim=False,
        original_files_and_tex_unchanged=True, source_sha256={str(Path(__file__)): digest(__file__)},
        matching_source=str(sys.modules[_canonical_matching.__module__].__file__),
        output_sha256={str(p.relative_to(OUT)): digest(p) for p in OUT.iterdir()
                       if p.is_file() and p.name != "execution_receipt.json"}))


def write_report(metrics, spans):
    table = ["| 方法 | 包络命中 | 实际月命中 | 包络R | 实际月R | 跨度中位/P90/最大（月） | >12月报告 |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for method in MAIN:
        a = metrics.loc[metrics.method.eq(method) & metrics.matching_rule.eq("envelope")].iloc[0]
        b = metrics.loc[metrics.method.eq(method) & metrics.matching_rule.eq("actual_months")].iloc[0]
        s = spans.loc[spans.method.eq(method)].iloc[0]
        table.append(f"| {method} | {int(a.matched_labels)} | {int(b.matched_labels)} | {a.recall_label:.2%} | "
                     f"{b.recall_label:.2%} | {s.median_span:g}/{s.p90_span:g}/{s.max_span:g} | {int(s.n_span_over_12)} ({s.proportion_span_over_12:.2%}) |")
    text = "\n".join(["# Balsa 冻结结果的严格时间范围评价", "", "日期：2026-11-02。所有方法固定1049月标签、459月标签林分。", "",
        "## 评价合同", "", "同林分，参考月落入首末实际响应包络为envelope；参考月恰为一个实际响应月为actual_months。"
        "均不增加±3月容差，复用既有最大一对一匹配和平局规则，所有合资格边成本统一为(1,0)，不偏爱长区间或更近代表月。"
        "这些边表示时间相容，不表示已独立核验同次采伐。", "",
        "每个报告和每条参考最多配一次。源节点账本逐报告核对实际月份，未改变候选、分区、支持或日期。"
        "跨度按末月−首月+1计算。>12月仅诊断，不删除长报告。1045条历史报告月、4条光学首次可见月保持原date_basis；"
        "另外10条光学不确定区间不进入本1049分母，也不被当作作业持续期。", "",
        "## 主配置结果", "", *table, "",
        "完整36方法×2规则的R、P_label、F1_label及配对区间见range_metrics.csv和paired_differences.csv。"
        "P/F1描述与不完整参考目录的相容，不是总体真实precision/F1。", "",
        "## 对当前结果的判断", "",
        "时间组织R2的包络命中871/1049（83.03%），实际响应月命中850/1049（81.03%）；B20分别792（75.50%）和780（74.36%）。"
        "实际月口径R2相对B20新增71、丢失1，净增70；新增71中58来自跨度不超过12个月的报告，13来自更长报告。"
        "因此增加的覆盖不全是包络内无响应空档，也不全由超长报告贡献；更完整的来源组织有可检验价值。", "",
        "同时，B20报告跨度中位/P90/最大均为3个月，R2为4/10/35个月，77/1208（6.37%）超12个月。"
        "输出范围更宽会增加命中机会；35个月不能直接叫作一次实际作业的持续期，长链仍是优先核验对象。"
        "两种范围指标下R2相对B20召回差的条件配对95%区间分别为包络+6.03至+9.15百分点、实际月+5.19至+8.18百分点，"
        "只说明当前队列下的时间相容覆盖差异，不包含历次探索或事件身份的不确定性。", "",
        "STS归一化切分把实际月命中从850降为845；可靠性要求恢复到850，但主域报告1219条，多于R2的1208条，"
        "相应P/F1没有超过R2。硬空间接触规则实际月仅654条。当前没有足够证据把这些切分规则升级为默认过程组织方法。", "",
        "七组主配置均无单报告包络同时覆盖两条参考月。对一个只含一个参考月的原报告，切分通常得不到更多匹配名额，"
        "却可能使子组件跌破20%空间支持。因此范围评价可以检验覆盖，但这批参考不足以充分检验‘正确拆开两次作业’的收益。"
        "未出现多参考冲突不能证明没有误合并，尤其未记录事件仍可能存在。", "",
        "下一步应保留代表月±1/2/3、严格范围及实际月三类结果，优先核验R2相对B20新增/丢失参考、77个长报告及新增分组身份。"
        "现有结果支持继续研究林分过程组织，尚不能把STS形态相容性或硬空间接触本身写成已验证的事件身份增益。", "",
        "## 如何解读", "",
        "包络命中越高不自动说明方法越好；同时查看实际月命中、跨度、长链和多参考竞争。"
        "一个报告覆盖多个参考月时仍只有一个匹配名额。report_reference_opportunities.parquet保留每份报告的原始可配参考数与实际获配参考。", "",
        "range_metrics.csv的n_gap_matches表示包络规则最终配对中、参考月没有本报告接受响应的数量；"
        "这与包络总命中减实际月总命中不同，因为两个最大匹配可重新分配参考。reference_matches.parquet记录逐条配对与该月全林分观测状态。", "",
        "实际月边是包络边的子集，故总匹配数应不超过包络；逐参考found可以因平局和竞争重新分配。"
        "range_vs_actual_reference_changes.parquet保留两向变化。范围规则和代表月±3的合资格边集合不嵌套，"
        "不得把改变评价规则产生的差值称为算法精度增益；并列结果见range_vs_point_pm3.csv。", "",
        "此评价可以判断已有标签是否被响应过程覆盖，不能确认实际作业起止、持续期或正确事件身份。"
        "跨度中位数低也不能掩盖少数很长的报告，因此同时提供P90、最大值和>12月比例。", "",
        "## 交付与复现", "", "代码：tooling/diagnostics/balsa-range-evaluation-20261102/evaluate_ranges.py；测试在tests/test_episode_range_evaluation.py。"
        "来源指纹、协议与执行证明为input_manifest.json、protocol_frozen.json、execution_receipt.json。旧v4.1结果、标签及TeX保持只读。", ""])
    (OUT / "RANGE_EVALUATION_REPORT_ZH.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
