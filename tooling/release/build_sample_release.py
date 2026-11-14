"""Build the small, deidentified sample package from the October 15 freeze.

This author-side exporter reads existing evidence. It does not relabel samples,
change scientific code, or publish anything. Original-ID bridges stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from sample_workbook import build_workbook

V = ROOT / "runs/validation"
SAMPLING = V / "b20_spatial_probability_review_20260928_v1"
FREEZE = V / "jag_reference_score_closeout_20261015_v1"
IDENTITY = V / "b20_month_tolerance_finalization_20260928_v1"
HISTORY = V / "common_domain_prf_closure_20261008_v1/reference_library_20261015_v1"
FOLLOWUP = V / "jag_event_accuracy_recalculation_20260928_v1/precision_after_relations_20261002_v1"
FRAME_COLUMNS = [
    "prediction_id", "sample_id", "representative_month", "response_months",
    "window_start_month", "window_end_month", "response_pixels", "total_pixels",
    "support_fraction", "selected", "geo_stratum", "block_id", "candidate_class",
    "M_h", "m_h", "N_bc", "n_bc", "inclusion_probability", "sampling_weight",
]
REFERENCE_COLUMNS = ["event_id", "sample_id", "timing_type", "reference_month",
                     "date_basis", "interval_lower", "interval_upper", "bound_semantics"]
EVENT_COLUMNS = ["verified_event_id", "sample_id", "timing_type", "reference_month",
                 "interval_start", "interval_end", "start_inclusive", "end_inclusive",
                 "date_basis", "evidence_notes"]
LINK_COLUMNS = ["prediction_id", "verified_event_id", "identity_status", "spatial_status", "evidence_notes"]
REVIEW_COLUMNS = ["prediction_id", "review_status", "occurrence", "context_complete", "evidence_notes"]
METHOD = "windowed_w36_d12_s12"
ENGINE_FILES = ("event_scoring.py", "b20_probability_estimation.py", "catalogue_evaluation.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")


def ordered_map(values, prefix):
    """Preserve original lexical order, including deterministic matching ties."""
    keys = sorted({str(v) for v in values if str(v)})
    return {key: f"{prefix}{i:06d}" for i, key in enumerate(keys, 1)}


def remap(table, bridges):
    result = table.copy()
    for col in result.columns:
        key = "prediction_id" if col == "competing_prediction_id" else col
        if key not in bridges:
            continue
        values = result[col].astype(str)
        unknown = set(values) - set(bridges[key]) - {""}
        if unknown:
            raise ValueError(f"Unmapped {col}: {len(unknown)} identities")
        result[col] = values.map({"": "", **bridges[key]})
    return result


def sanitize_notes(table, label):
    table = table.copy()
    table["evidence_notes"] = [
        f"{label} evidence record {i:06d}; original evidence retained under restricted access."
        if str(value).strip() else ""
        for i, value in enumerate(table.evidence_notes, 1)
    ]
    return table


def verify_freeze():
    frozen = json.loads((FREEZE / "frozen_results_manifest.json").read_text(encoding="utf-8"))
    for key, base in [("outputs_sha256", FREEZE), ("scientific_source_hashes", ROOT)]:
        for name, expected in frozen[key].items():
            if digest(base / name) != expected:
                raise ValueError(f"Frozen file changed: {name}")
    return frozen


def supplementary_tables(frame):
    """Structured evidence summaries only; raw notes and source identities stay local."""
    by_review = frame.set_index("review_id").prediction_id.to_dict()
    by_report = frame.set_index("prediction_id").sample_id.to_dict()
    records = []

    def add(kind, sample, reports, conclusion, detail):
        records.append(dict(supplement_id=f"A{len(records)+1:06d}", kind=kind,
                            sample_id=sample, linked_reports=reports,
                            conclusion=conclusion, detail=detail))

    for row in read_csv(IDENTITY / "identity_relation_answers.csv").to_dict("records"):
        review_ids = [v for v in re.split(r"[;,|\s]+", row["formal_review_ids"]) if v]
        if any(v not in by_review for v in review_ids):
            raise ValueError("Identity answer contains an unknown formal review")
        add("identity", row["sample_id"], [by_review[v] for v in review_ids],
            {"同一次": "同一次采伐", "不同次": "不同次采伐"}[row["relation"]],
            "已完成关系确认；最终计分使用最新冻结事件表，关系记录不另增加样本。")
    actions = {"same_existing_month": "同一事件，沿用已确认月份",
               "same_correct_month": "同一事件，已修订月份",
               "no_target_event": "无对应目标采伐或位置不符",
               "retain_unknown": "保留未判定", "same_time_uncertain": "同一过程，但月份仍不明确"}
    for row in read_csv(FOLLOWUP / "answer_import_audit.csv").to_dict("records"):
        pid = row["prediction_id"]
        add("followup", by_report[pid], [pid], actions[row["action"]],
            "补充确认已纳入评分；这条记录不是新增独立样本。")
    for row in read_csv(IDENTITY / "repeat_comparison.csv").to_dict("records"):
        add("repeat", row["sample_id"], [by_review[row["review_id"]]],
            "已录入月份，尚未确认两次判读对应同一事件",
            f"原录入：{row['first_month']}；再次录入：{row['repeat_month']}。不能据此计算判读一致率。")
    domains = read_csv(HISTORY / "domain_evidence_status.csv")
    issues = {"": "部分时段证据仍有限。", "geometry_scope_conflict": "地块范围对应仍不明确。",
              "possible_prolonged_non_regeneration": "可能存在持续较久的清地或未恢复过程。"}
    for row in domains.to_dict("records"):
        add("history_domain", row["sample_id"], [], "尚不能作为完整历史真值",
            "检查范围为2018—2022年；已提交判读。" + issues[row["issue_code"]])
    for row in read_csv(HISTORY / "harvest_observations.csv").to_dict("records"):
        spatial = "地块范围仍不明确。" if row["spatial_status"] == "unresolved_polygon_scope" else "位置为人工描述，非像元边界真值。"
        add("history_observation", row["sample_id"], [], "已观察到采伐线索，独立事件身份尚未核清",
            f"首次可见月份：{row['first_visible_month']}；不等于作业开始。{spatial}")
    result = pd.DataFrame(records)
    if len(result) != 117:
        raise ValueError("Supplementary source counts changed")
    return result


def original_tables():
    config = json.loads((SAMPLING / "sampling_config.json").read_text(encoding="utf-8-sig"))
    for key in ("references", "event_comparison", "units", "membership", "old_blocks"):
        item = config["inputs"][key]
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen sampling input changed: {key}")
    frame = read_csv(SAMPLING / "sampling_frame.csv")
    refs = read_csv(config["inputs"]["references"]["path"])[REFERENCE_COLUMNS]
    comparison = read_csv(config["inputs"]["event_comparison"]["path"])
    eligibility = comparison.loc[comparison.method.eq("B20") & comparison.tolerance_quarters.eq(1),
                                 ["event_id", "eligibility"]]
    # Use the original all-master-cell stand assignment, including stands without
    # reports. Keeping only the 462 report-bearing stands can change bootstrap CIs.
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_b20_probability_validation import assign_stands
    units = pd.read_parquet(config["inputs"]["units"]["path"])
    membership = pd.read_parquet(config["inputs"]["membership"]["path"])
    blocks = read_csv(config["inputs"]["old_blocks"]["path"])
    stands = assign_stands(membership, units).merge(blocks[["block_id", "geo_stratum"]],
                                                 on="block_id", validate="many_to_one")
    tables = {"sampling_frame": frame[FRAME_COLUMNS].copy(), "references": refs,
              "eligibility": eligibility, "stands": stands[["sample_id", "block_id", "geo_stratum"]],
              "report_scores": read_csv(FREEZE / "scoring/report_scores.csv")}
    for name, cols in [("verified_events", EVENT_COLUMNS), ("candidate_event_links", LINK_COLUMNS),
                       ("candidate_reviews", REVIEW_COLUMNS)]:
        tables[name] = sanitize_notes(read_csv(FREEZE / f"evidence/{name}.csv")[cols], name)
    tables["supplementary"] = supplementary_tables(frame)
    expected_catalogue = read_csv(FREEZE / "frozen_comparisons/catalogue_metrics.csv")
    tables["expected_catalogue"] = expected_catalogue.loc[expected_catalogue.method.eq(METHOD)].drop(
        columns=["source_is_primary_result", "source_file", "source_sha256", "processing"])
    tables["expected_precision"] = read_csv(FREEZE / "scoring/weighted_precision.csv")
    return tables, config


def build(output, private_bridge):
    output, private_bridge = Path(output).resolve(), Path(private_bridge).resolve()
    if output.exists() or private_bridge.exists():
        raise FileExistsError("Choose new output and private bridge directories; existing records are never overwritten")
    if output == private_bridge or output in private_bridge.parents or private_bridge in output.parents:
        raise ValueError("Private bridge must be outside the release")
    if ROOT / "data/private" not in private_bridge.parents:
        raise ValueError("Keep the original-ID bridge in this repository's ignored data/private directory")
    frozen = verify_freeze()
    tables, config = original_tables()
    bridges = {}
    for column, prefix in [("prediction_id", "P"), ("sample_id", "S"), ("event_id", "R"),
                           ("verified_event_id", "E"), ("block_id", "B"), ("geo_stratum", "H")]:
        values = [str(v) for table in tables.values() if column in table for v in table[column]]
        bridges[column] = ordered_map(values, prefix)
    private_bridge.mkdir(parents=True)
    write_json(private_bridge / "id_bridge.json", bridges)
    write_json(private_bridge / "source_record.json", {
        "freeze_manifest_sha256": digest(FREEZE / "frozen_results_manifest.json"),
        "sampling_frame_sha256": digest(SAMPLING / "sampling_frame.csv"),
        "sampling_inputs": config["inputs"], "output": str(output),
    })
    for name, table in list(tables.items()):
        tables[name] = remap(table, bridges)
    tables["supplementary"]["linked_reports"] = tables["supplementary"].linked_reports.map(
        lambda reports: "; ".join(bridges["prediction_id"][pid] for pid in reports))
    counts = {name: len(table) for name, table in tables.items()}
    counts["formal_probability_reports"] = int(tables["sampling_frame"].selected.sum())
    counts["qualified_complete_domains"] = 0
    expected_counts = {"sampling_frame": 1266, "references": 1081, "report_scores": 921,
                       "verified_events": 300, "candidate_event_links": 363,
                       "candidate_reviews": 717, "supplementary": 117, "formal_probability_reports": 307}
    if any(counts[k] != n for k, n in expected_counts.items()):
        raise ValueError(f"Frozen counts changed: {counts}")
    replay_dir = output / "reproducibility"
    replay_dir.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(replay_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    engine = replay_dir / "engine/lr_cusum"
    engine.mkdir(parents=True)
    (engine / "__init__.py").write_text('"""Frozen scoring engine for the sample release."""\n', encoding="utf-8")
    for name in ENGINE_FILES:
        shutil.copyfile(ROOT / "src/lr_cusum" / name, engine / name)
    shutil.copyfile(ROOT / "LICENSE", replay_dir / "CODE_LICENSE.txt")
    shutil.copyfile(Path(__file__).with_name("replay_sample_release.py"), replay_dir / "replay.py")
    (replay_dir / "requirements.txt").write_text("numpy==2.4.3\npandas==2.3.3\nscipy==1.17.1\n", encoding="utf-8")
    build_workbook(output / "samples.xlsx", tables["references"], tables["eligibility"],
                   tables["sampling_frame"], tables["report_scores"], tables["verified_events"],
                   tables["candidate_event_links"], tables["supplementary"])
    (output / "README.md").write_text(README, encoding="utf-8")
    (replay_dir / "DATA_DICTIONARY.md").write_text(DICTIONARY, encoding="utf-8")
    manifest = {
        "freeze_version": "jag_reference_score_closeout_20261015_v1",
        "package_version": "balsa_samples_v1", "status": "local_release_candidate_not_uploaded",
        "primary_tolerance_months": 3, "primary_display_selected_after_viewing_results": True,
        "sensitivity_tolerances_months": [1, 2], "counts": counts,
        "code_license": "MIT", "data_redistribution_license": "not_assigned_by_this_export",
        "population_F1": None,
        "sampling": {"design": "geographically_stratified_two_stage_probability_sample",
                     "strata": 8, "eligible_blocks": 67, "selected_blocks": 50,
                     "seed": config["seed"], "original_order_preserved": True,
                     "estimator": "Hajek scenario endpoints; two-stage variance; approximate t interval; df26"},
        "id_mapping": "consistent order-preserving surrogate IDs; private bridge excluded",
        "source_freeze_manifest_sha256": digest(FREEZE / "frozen_results_manifest.json"),
        "file_hashes": {},
    }
    manifest["file_hashes"] = {p.relative_to(output).as_posix(): digest(p)
                              for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(replay_dir / "manifest.json", manifest)
    verify_freeze()
    return {"output": str(output), "counts": counts,
            "frozen_outputs_verified": len(frozen["outputs_sha256"]),
            "scientific_sources_verified": len(frozen["scientific_source_hashes"])}


README = """# 巴沙木采伐样本集 v1

**先打开 `samples.xlsx`。这是已有样本和核验结果的整理，无需重新填写。**

| 工作表 | 一行是什么 | 怎么查 |
|---|---|---|
| 参考事件 | 原采伐目录的一条记录，共1,081条 | 按林分编号或评价用途筛选；1,049条合格月度参考用于目录recall，10条区间参考另列 |
| 检测与核验 | W36＋B20的一条检测报告，共1,266条 | 筛选“正式核验”查看307条概率样本；主列为±3月结论及原因，后面保留±1/2月 |
| 补充核验 | 一次关系确认、重复判读或历史检查，共117条 | 按林分编号或关联报告查找；这些记录不另算独立样本 |

相同林分编号代表同一地块，事件编号区分不同采伐。`R`是原目录事件，`E`是报告核验事件；两者不能相加或仅因月份相同就合并。编号均已替换，原编号和位置不在本包中。

**读结果时只记住三点：** T表示符合事件、位置、时间和一对一规则；F表示不符合，具体看原因；U表示现有证据仍不足。“未抽中”不等于U或误报。三个月聚合窗口与±3月匹配容差是两件事。

当前主展示（±3月）：307条中T=202、F=91、U=14；加权precision情景端点为69.62%—72.34%，计入抽样误差的近似95%外侧区间为65.17%—76.88%。目录recall为955/1,049=91.04%。这些指标分母不同，不能拼成总体F1。±3月主展示是在查看既有结果后选择的，Excel同时保留±1/2月。

**需要复算时才打开 `reproducibility/`：**

```bash
python -m pip install -r reproducibility/requirements.txt
python reproducibility/replay.py --output ../balsa_samples_replay
```

复算包括全部307条的一对一评分、加权precision及不确定性、冻结目录recall及条件性区间；结果逐项与冻结版本核对。这里复现的是已有判读的计分，不能代替重新查看光学影像。

资料范围：Sentinel-1月度检测、已有采伐目录及PlanetScope等光学人工判读；报告期2017—2025，7个历史核验林分只检查2018—2022，均尚未形成完整历史真值。原始影像、地块边界、坐标、自由文本备注和身份对照表留在私有库。冻结日期为2026-10-15。本包是本地发布候选，尚未上传；代码为MIT，数据再分发许可未由本次导出代定。其他历史样本和旧实验仍可在作者完整样本索引中检索。
"""


DICTIONARY = """# Machine tables and evaluation scope

The workbook is a readable view; CSVs are the replay inputs. UTF-8 CSVs retain
blank values as blank, not zero or a negative observation. All IDs are opaque,
consistent and order preserving. Spatial grouping is retained without geometry.

| File | Unit and important fields |
|---|---|
| sampling_frame.csv | All1,266 frozen W36+B20 reports. prediction_id links every report table; sample_id links stands. representative_month is the frozen median response-node month. response_months is a JSON list of actual response months. window_start_month/end and response_pixels/total_pixels/support_fraction describe the frozen B20 support. selected identifies307 formal probability samples. |
| references.csv | Original R0 catalogue1081 rows; event_id is a catalogue identity, not a verified_event_id. timing_type, reference_month, interval_lower/upper and bound_semantics retain original timing. date_basis records whether the time is reported, first visible, interval-based or unknown. |
| eligibility.csv | One row per catalogue event; eligible monthly1049 and interval10 are separate. Other states retain all excluded/unknown records. |
| stands.csv | All master-cell-assigned stands, including stands without reports. Block is the original most-fixed-master-cells assignment, sorted-block tie rule. No spatial clipping or response-based reassignment. |
| verified_events.csv | 300 precision-evidence identities with time semantics; verified_event_id is shared by all linked reports. Five events have unknown time. No implied merge with R0. |
| candidate_event_links.csv | All363 links with identity_status and spatial_status, including unresolved links and nonsampled competitors. |
| candidate_reviews.csv | All717 context/review records. Only307 selected reports enter the estimate; pending contextual records are not additional unfilled formal samples. occurrence and context_complete are required by the frozen scorer. |
| report_scores.csv | Expected921 rows (307 reports ×3 tolerances). judgement T/F/U; reason preserves timing failures, duplicate reports, no target event/space failures and unknown causes. competing_prediction_id links the report receiving unique credit when applicable. |
| supplementary.csv | 40 identity decisions +28 followups +31 repeat readings +7 history domains +11 observations. supplement_id is an evidence-operation identifier, not an event. linked_reports uses semicolon-separated report IDs. Summaries are structured and deidentified; raw notes remain private. |
| expected_precision.csv | Frozen weighted scenario endpoints, variances and approximate outer confidence interval for1/2/3 months. |
| expected_catalogue.csv | Frozen W36 R0 recall, interval compatibility and conditional block-bootstrap interval for1/2/3 months. |

## Sampling and uncertainty

The fixed design has8 geographical strata,67 eligible blocks,50 selected blocks
and307 selected reports. geo_stratum/block_id are opaque design keys;
candidate_class is the original C1/C2/C3 sampling category, not a new truth label.
C1 comprises973 reports matched to the earlier catalogue; C2 comprises106
unmatched reports competing for an existing reference; C3 comprises187 reports
without a legal earlier-reference match. These historical sampling categories
are retained even when later optical adjudication changes the judgement.
M_h/m_h are total/selected blocks; N_bc/n_bc are total/selected reports in each
block/category. inclusion_probability=(m_h/M_h)*(n_bc/N_bc), sampling_weight=1/pi.
The original unequal probabilities, all frame rows and both variance stages remain.
An unselected report has no formal judgement, even if its context is reviewed.

With unresolved selected reports, lower_hajek treats U as0 and upper_hajek as1;
the weighted denominator includes all selected reports. These are scenario
endpoints, not confidence limits. ci95_lower/upper add sampling uncertainty to
the outside of these endpoints using the existing two-stage variance and t26.
The optimistic endpoint does not prove every unresolved report could receive
simultaneous one-to-one credit. Ordinary precision_hajek remains blank when U
prevents a single fully determined estimate. HT fields are auxiliary estimates.

Catalogue bootstrap intervals are conditional on the selected R0 reference
cohort; they do not measure population recall. Matching uses the full report
history before grouping. Eligible interval references use actual response-month
compatibility and are not silently counted as monthly references.

## Limits and provenance

The code in engine/lr_cusum is copied byte-for-byte from the frozen implementation.
Only the result-display flag is set to±3 months after replay; the original engine
contains a historical±2 default. Scores at all three tolerances remain unchanged.
Deterministic ID order, sample membership and the original bootstrap seed20260929
are preserved. manifest.json records every packaged file checksum; replay rejects
modified listed inputs/code. This package does not rerun detection or original
sampling, and does not estimate benchmark population precision.

Evidence notes are replaced with nonempty restricted-source pointers. The private
original-ID bridge and original evidence archive allow author-side traceability;
they are intentionally absent here. Independent optical re-interpretation needs
access to the source imagery. No imagery redistribution rights are implied.
Seven incomplete history domains and31 unaligned repeat readings cannot be used
as complete ground truth or a repeatability estimate. Population F1 is undefined.
The catalogue and probability-review event collections have different selection
mechanisms, so their records or estimates must not be added together.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/balsa_samples_v1")
    parser.add_argument("--private-bridge", type=Path, default=ROOT / "data/private/sample_release_20261016_v1")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.private_bridge), ensure_ascii=True, indent=2))
