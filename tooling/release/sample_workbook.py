"""Render deidentified release tables as a three-sheet, human-readable workbook.

This is a presentation of supplied evidence and frozen scores, not a review form
or a scoring implementation. It never opens the author's private source files.
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


JUDGEMENTS = {"T": "正确（T）", "F": "不符合（F）", "U": "未判定（U）"}
REASONS = {
    "verified_one_to_one_match": "事件、位置、时间符合，取得唯一匹配信用",
    "time_mismatch": "确认有对应事件，但月份超出容差",
    "duplicate_report": "同一次采伐的重复报告",
    "no_event": "无对应目标采伐或位置不符",
    "insufficient_optical_evidence": "光学证据不足",
    "competition_context_incomplete": "相关报告的竞争关系尚未核清",
    "unresolved_identity_space_time_or_competition": "事件、位置、时间或竞争关系仍不明确",
    "no_confirmed_identity_and_spatial_link": "尚未确认事件身份与空间对应",
    "pending_optical_review": "尚未核验",
}
ELIGIBILITY = {
    "time_unknown": "不参与：时间未知",
    "outside_reporting_period": "不参与：超出研究期",
    "no_true_interior": "不参与：无地块内有效像元中心",
    "input_insufficient": "不参与：输入观测不足",
    "partial_interval_input_support": "不参与：区间仅有部分输入支持",
}
DATE_BASIS = {
    "first_visible_month": "光学首次可见月份",
    "optical_first_visible_month": "光学首次可见月份",
    "first_visible_harvest_month": "光学首次可见采伐月份",
    "first_visible_removal": "光学首次可见移除",
    "reported_month": "原记录所报月份",
    "historical_reported_month": "历史资料记录月份",
    "optical_observation_interval": "光学观测限定区间",
    "unknown": "时间含义未知",
    "optical_first_visible_removal": "光学首次可见采伐痕迹",
    "optical_first_observed_month_not_operation_month": "光学首次可见月份（不等于作业开始）",
    "latest_user_optical_first_visible_month_not_operation_start": "光学首次可见月份（不等于作业开始）",
    "latest_human_stated_harvest_month_in_optical_review": "最新人工核验记录月份",
}
KINDS = {
    "identity": "事件关系确认", "identity_group": "事件关系确认",
    "identity_groups": "事件关系确认", "followup": "后续关系确认",
    "relationship_followup": "后续关系确认", "repeat": "重复判读",
    "repeat_interpretation": "重复判读", "history_domain": "连续历史检查",
    "history_observation": "连续历史中的事件线索",
}
NOT_SELECTED = "未抽中（不计入正式评分）"


def _text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _rows(table, key, columns, name):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if key and (table[key].isna().any() or table[key].astype(str).eq("").any()
                or table[key].duplicated().any()):
        raise ValueError(f"{name}: {key} must be nonempty and unique")
    return table.to_dict("records")


def _selected(value):
    value = _text(value).lower().strip()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"Invalid selected flag: {value!r}")
    return value in {"true", "1", "yes"}


def _time(event):
    if event["timing_type"] == "month":
        return _text(event.get("reference_month")) or "月份未确定"
    if event["timing_type"] == "interval":
        lower = event.get("interval_start", event.get("interval_lower", ""))
        upper = event.get("interval_end", event.get("interval_upper", ""))
        flags = {"true": "包含", "1": "包含", "yes": "包含",
                 "false": "不含", "0": "不含", "no": "不含"}
        start = flags.get(_text(event.get("start_inclusive")).lower())
        end = flags.get(_text(event.get("end_inclusive")).lower())
        bounds = (f"（{start}起月、{end}止月）" if start and end else
                  f"（{_text(event.get('bound_semantics'))}）" if _text(event.get("bound_semantics")) else "")
        return f"区间：{_text(lower)} 至 {_text(upper)}{bounds}"
    return "时间未知"


def _put(cell, value):
    # Explicit string cells preserve text without executing formula-like notes.
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        value = ""
    cell.value = value
    if isinstance(value, str):
        cell.data_type = "s"


def _sheet(workbook, name, note, legend, headers, rows, widths, comments=None):
    sheet = workbook.create_sheet(name)
    for number, text in [(1, note), (2, legend)]:
        sheet.merge_cells(start_row=number, start_column=1,
                          end_row=number, end_column=len(headers))
        _put(sheet.cell(number, 1), text)
        sheet.cell(number, 1).alignment = Alignment(wrap_text=True, vertical="center")
        sheet.cell(number, 1).font = Font(name="Microsoft YaHei", size=11,
                                         bold=number == 1, color="243B53")
        sheet.row_dimensions[number].height = 34
    for column, label in enumerate(headers, 1):
        cell = sheet.cell(4, column, label)
        cell.fill = PatternFill("solid", fgColor="244B64")
        cell.font = Font(name="Microsoft YaHei", color="FFFFFF", bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        if comments and label in comments:
            cell.comment = Comment(comments[label], "Data release")
    for number, row in enumerate(rows, 5):
        for column, value in enumerate(row, 1):
            cell = sheet.cell(number, column)
            _put(cell, value)
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            color = {"正确（T）": "E1F0DF", "不符合（F）": "FBE4E2",
                     "未判定（U）": "FFF1C7", NOT_SELECTED: "EEEEEE"}.get(value)
            if color:
                cell.fill = PatternFill("solid", fgColor=color)
            elif number % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F3F6F8")
        sheet.row_dimensions[number].height = 34
    sheet.row_dimensions[4].height = 34
    for column, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "C5"
    sheet.auto_filter.ref = f"A4:{get_column_letter(len(headers))}{max(4, len(rows) + 4)}"
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 85
    sheet.print_title_rows = "1:4"
    return sheet


def build_workbook(output_path, references, eligibility, frame, scores, events, links,
                   supplementary):
    """Write three sheets from deidentified DataFrames; refuse to overwrite.

    ``frame.selected`` defines formal sample membership. Scores are long-form
    (prediction_id, tolerance_months, judgement, reason) and must cover each
    selected report at 1, 2 and 3 months. Extra context scores cannot turn an
    unselected report into a formal sample. No inference or matching is done.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    reference_rows = _rows(references, "event_id", ["event_id", "sample_id", "timing_type"], "references")
    eligible_rows = _rows(eligibility, "event_id", ["event_id", "eligibility"], "eligibility")
    report_rows = _rows(frame, "prediction_id", ["prediction_id", "sample_id", "selected", "representative_month"], "frame")
    event_rows = _rows(events, "verified_event_id", ["verified_event_id", "timing_type"], "events")
    score_rows = _rows(scores, None, ["prediction_id", "tolerance_months", "judgement", "reason"], "scores")
    link_rows = _rows(links, None, ["prediction_id", "verified_event_id"], "links")
    supplemental_rows = _rows(supplementary, "supplement_id", ["supplement_id", "kind", "sample_id", "linked_reports", "conclusion", "detail"], "supplementary")
    eligibility_by_id = {row["event_id"]: row["eligibility"] for row in eligible_rows}
    if set(eligibility_by_id) != {row["event_id"] for row in reference_rows}:
        raise ValueError("References and eligibility have different event IDs")
    events_by_id = {row["verified_event_id"]: row for row in event_rows}
    reports_by_id = {row["prediction_id"]: row for row in report_rows}
    scores_by_key = {}
    for row in score_rows:
        delta = float(row["tolerance_months"])
        if delta not in (1, 2, 3):
            raise ValueError("Workbook scores require tolerances 1, 2 or 3 months")
        key = (row["prediction_id"], int(delta))
        if key in scores_by_key or row["prediction_id"] not in reports_by_id:
            raise ValueError(f"Duplicate or unknown report score: {key}")
        scores_by_key[key] = row
    report_links = {}
    for row in link_rows:
        if row["verified_event_id"] not in events_by_id or row["prediction_id"] not in reports_by_id:
            raise ValueError("A report-to-event link refers to an unknown identifier")
        report_links.setdefault(row["prediction_id"], []).append(row)

    reference_view = []
    for row in reference_rows:
        status = eligibility_by_id[row["event_id"]]
        if status == "eligible":
            status = "目录 recall（月级）" if row["timing_type"] == "month" else "区间参考（单列）"
        else:
            status = ELIGIBILITY.get(status, f"未纳入：{status}")
        basis = _text(row.get("date_basis"))
        reference_view.append([row["sample_id"], row["event_id"], _time(row),
                               DATE_BASIS.get(basis, basis), status,
                               _text(row.get("bound_semantics"))])

    report_view = []
    for row in report_rows:
        pid = row["prediction_id"]
        selected = _selected(row["selected"])
        linked_times = []
        for link in report_links.get(pid, []):
            eid = link["verified_event_id"]
            identity = {"confirmed": "已确认", "uncertain": "待确认"}.get(
                _text(link.get("identity_status")), "见对应证据")
            spatial = {"confirmed": "位置对应", "uncertain": "位置待确认"}.get(
                _text(link.get("spatial_status")), "见空间证据")
            linked_times.append(f"{eid}：{_time(events_by_id[eid])}（{identity}；{spatial}）")
        score_labels = []
        for delta in (3, 1, 2):
            score = scores_by_key.get((pid, delta), {})
            if selected and score.get("judgement") not in JUDGEMENTS:
                raise ValueError(f"Selected report {pid} lacks a final T/F/U score at ±{delta} months")
            score_labels.append(JUDGEMENTS[score["judgement"]] if selected else NOT_SELECTED)
        primary = scores_by_key.get((pid, 3), {}) if selected else {}
        reason = primary.get("reason", "")
        report_view.append([row["sample_id"], pid, row["representative_month"],
                            "是（正式概率样本）" if selected else NOT_SELECTED,
                            "\n".join(linked_times) or "未建立核验事件关联",
                            score_labels[0], REASONS.get(reason, reason),
                            primary.get("verified_event_id", ""), score_labels[1], score_labels[2],
                            row.get("window_start_month", row.get("window_start", "")),
                            row.get("window_end_month", row.get("window_end", "")),
                            row.get("support_fraction", "")])

    workbook = Workbook()
    workbook.remove(workbook.active)
    _sheet(workbook, "参考事件", "原有采伐目录：一行是一条来源事件；同一林分可以有多次采伐。无需填写。",
           "月份参考用于目录 recall；区间参考单列，未知或不合格记录保留，不当作无采伐。日期表达不代表置信度。",
           ["林分编号", "目录事件编号", "参考时间", "日期含义", "评价用途", "区间端点含义"],
           reference_view, [18, 20, 30, 36, 34, 28])
    _sheet(workbook, "检测与核验", "全部 W36＋B20 检测报告：筛选“正式核验”可查看概率样本。这里只汇总已有结果，无需填写。",
           "主展示为±3月：正确T／不符合F／未判定U。±1、±2月为敏感性结果。未抽中不等于U或误报；三个月聚合与±3月容差是两回事。",
           ["林分编号", "报告编号", "检测代表月", "正式核验", "关联事件及参考时间（含待确认）",
            "±3月结论（主展示）", "±3月原因", "±3月计分关联事件", "±1月结论", "±2月结论",
            "聚合窗口起月", "聚合窗口止月", "响应像元比例"],
           report_view, [18, 20, 16, 28, 52, 25, 42, 24, 25, 25, 16, 16, 18],
           {"检测代表月": "冻结候选响应节点的中位月份，不一定是采伐开始月份。",
            "关联事件及参考时间（含待确认）": "保留全部已有关联；同一报告可有多个关联，不代表每个关联均取得正确信用。",
            "±3月计分关联事件": "评分器记录的事件。必须结合T/F/U与原因；重复报告也可关联同一事件，但不能再计一次正确报告。",
            "响应像元比例": "三个月内响应的不同固定主格数／该林分固定主格数；不是核验precision。"})
    for cell in workbook["检测与核验"]["M"][4:]:
        if isinstance(cell.value, (int, float)):
            cell.number_format = "0.0%"
    _sheet(workbook, "补充核验", "已有补充判读记录：用于解释事件身份、重复报告和可见性，不作为新的独立样本相加。",
           "连续历史中看不清的时段继续保留未知；重复判读不自动成为独立验证。详情为脱敏后的结构化结论。无需填写。",
           ["核验编号", "检查类型", "林分编号", "关联报告", "核验结论", "说明"],
           [[row["supplement_id"], KINDS.get(row["kind"], row["kind"]), row["sample_id"],
             row["linked_reports"], row["conclusion"], row["detail"]] for row in supplemental_rows],
           [20, 26, 18, 34, 42, 70])
    workbook.properties.title = "巴沙木采伐参考与核验样本"
    workbook.properties.subject = "Frozen evidence and scores; read-only presentation"
    workbook.properties.creator = "Windowed CUSUM sample release"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    return {"参考事件": len(reference_view), "检测与核验": len(report_view),
            "补充核验": len(supplemental_rows)}
