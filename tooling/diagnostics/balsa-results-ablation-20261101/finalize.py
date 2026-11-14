"""Verify the completed phases and assemble manuscript-facing result evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd

from preflight import ROOT, OUT, digest

LABELS = dict(full_firstpost="完整 First-post W36+B20", no_local_window="单个全记录 CUSUM+B20",
    no_central_gate="去中心决策区间", no_permutation_gate="去置换筛选", no_direction_gate="去下降方向约束",
    vh_firstpost="VH", vv_union_vh_firstpost="VV∪VH（VV日期优先）", vv_intersection_vh_firstpost="VV∩VH（VV日期优先）")


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers),
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows]]


def main():
    preflight = json.loads((OUT / "preflight_receipt.json").read_text(encoding="utf-8"))
    assert preflight["passed"]
    for path, expected in preflight["protected_sha256"].items():
        assert digest(path) == expected, path
    assert digest(OUT / "plan_at_execution.md") == preflight["plan_sha256"]
    for name, expected in preflight["output_sha256"].items():
        assert digest(OUT / name) == expected, name
    for evidence in preflight["reused_receipts"].values():
        assert digest(evidence["path"]) == evidence["sha256"], evidence["path"]
    stage_paths = [OUT / "components/verification_receipt.json", OUT / "polarization/verification_receipt.json",
                   OUT / "truncation/verification_receipt.json", OUT / "seed/receipt.json"]
    checked_outputs = 0
    for path in stage_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        assert receipt["passed"], path
        if path.parent.name == "polarization":
            output_receipt = path.parent / "receipt.json"
            assert digest(output_receipt) == receipt["output_receipt_sha256"]
            receipt = json.loads(output_receipt.read_text(encoding="utf-8"))
        if path.parent.name == "seed":
            assert digest(Path(__file__).with_name("run_seed.py")) == receipt["script_sha256"]
        if path.parent.name == "truncation":
            completion = json.loads((path.parent / "completion_receipt.json").read_text(encoding="utf-8"))
            assert completion["status"] == "completed_and_hash_verified"
            assert digest(path) == completion["verification_receipt_sha256"]
        for name, expected in receipt.get("output_sha256", receipt.get("outputs_sha256", {})).items():
            assert digest(path.parent / name) == expected, name
            checked_outputs += 1
        for key in ("source_sha256", "helper_sha256", "scientific_source_sha256"):
            for name, expected in receipt.get(key, {}).items():
                assert digest(name) == expected, name
    component = pd.read_csv(OUT / "components/catalogue_recall.csv")
    polar = pd.read_csv(OUT / "polarization/evaluation/catalogue_recall.csv")
    component = component[component.tolerance_months.eq(2)].copy()
    polar = polar[polar.tolerance_months.eq(2)].copy()
    assert component.frozen_month_references.eq(1049).all() and polar.frozen_month_references.eq(1049).all()
    assert component.loc[component.method.eq("full_firstpost"), "frozen_month_found"].item() == 930
    assert polar.loc[polar.method.eq("full_firstpost"), "frozen_month_found"].item() == 930
    combined = pd.concat([component.assign(phase="P1"), polar.loc[~polar.method.eq("full_firstpost")].assign(phase="P2")], ignore_index=True)
    combined["display_name"] = combined.method.map(LABELS)
    assert combined.display_name.notna().all()
    combined.to_csv(OUT / "primary_ablation_results.csv", index=False, encoding="utf-8-sig")
    histories = []
    for folder in (OUT / "components", OUT / "polarization/evaluation"):
        history = pd.read_csv(folder / "registered_history_recovery.csv")
        history = history[history.tolerance_months.eq(2)].copy()
        if folder.name == "evaluation":
            history = history[~history.method.eq("full_firstpost")]
        pairs = pd.read_csv(folder / "registered_pair_summary.csv")
        pairs = pairs[pairs.tolerance_months.eq(2)].drop(columns="tolerance_months")
        strata = pd.read_csv(folder / "catalogue_strata.csv")
        strata = strata[strata.tolerance_months.eq(2) & strata.stratum.isin(["first", "subsequent"])]
        assert strata.loc[strata.stratum.eq("first"), "n_references"].eq(459).all()
        assert strata.loc[strata.stratum.eq("subsequent"), "n_references"].eq(590).all()
        history = history.merge(pairs, on="method", validate="one_to_one").merge(
            strata.pivot(index="method", columns="stratum", values="n_found").rename(
                columns={"first":"first_found", "subsequent":"subsequent_found"}), on="method", validate="one_to_one")
        histories.append(history)
    history = pd.concat(histories, ignore_index=True)
    assert history.n_multi_stands.eq(403).all() and history.registered_pairs.eq(590).all()
    assert set(history.method) == set(combined.method)
    history.to_csv(OUT / "primary_registered_history_results.csv", index=False, encoding="utf-8-sig")
    horizons = pd.read_csv(OUT / "truncation/summary.csv")
    assert horizons.horizon_months.tolist() == [0, 3, 6, 12] and horizons.references.eq(1021).all()
    stability = pd.read_csv(OUT / "truncation/common_success_date_stability.csv")
    stability = stability[stability.earlier_horizon.eq(6) & stability.later_horizon.eq(12)].iloc[0]
    trajectories = pd.read_csv(OUT / "truncation/event_horizon_trajectories.csv")
    seed = pd.read_csv(OUT / "seed/summary.csv")
    assert seed.base_seed.tolist() == [20260820, 20260821, 20260822] and seed.pixels.eq(512).all()
    baseline = component.set_index("method").loc["full_firstpost"]
    no_central = component.set_index("method").loc["no_central_gate"]
    no_perm = component.set_index("method").loc["no_permutation_gate"]
    rows = [(LABELS[r.method], int(r.n_reports), f"{int(r.frozen_month_found)}/1049", f"{r.frozen_recall:.2%}") for r in component.itertuples()]
    lines = ["# Balsa Results 消融补齐：执行结果", "", "2026-11-01。P0—P5已完成；原生输入、冻结科学实现、原标签和TeX保持不变。全国推理继续作为既有应用结果，本轮没有新增全国推理或判读任务。", "",
        "主方法为VV、First-post、W36/D12/S12、IID α=0.025、3999/临界19999次置换及固定成员B20。全期比较固定468林分、26890像元、2017—2025报告期，主容差±2月。", "",
        "## 1. 核心组件", "", *markdown_table(["方法", "报告数", "参考找回", "条件找回率"], rows), "",
        f"去掉中心区间多找回{int(no_central.frozen_month_found-baseline.frozen_month_found)}条登记参考，同时多出{int(no_central.n_reports-baseline.n_reports)}份报告；去掉置换筛选多找回{int(no_perm.frozen_month_found-baseline.frozen_month_found)}条，同时多出{int(no_perm.n_reports-baseline.n_reports)}份报告。这是检出与报告数量的取舍，新增报告不自动等于误报。",
        "方向约束在全部79301个有效中心内候选中没有额外拒绝；解除该规则后，像元响应、林分报告及来源内容与主方法等价。零效应按实际结果保留。全记录对照同时减少局部参考更新及候选机会，不能把全部差异归因于单一机制。", "",
        "来源：[组件报告](components/README_ZH.md)、[配对差异](components/paired_comparisons.csv)、[共同找回日期误差](components/common_found_date_error.csv)。", "",
        "## 2. 极化", "", *markdown_table(["策略", "报告数", "参考找回", "条件找回率"],
        [("VV" if r.method=="full_firstpost" else LABELS[r.method], int(r.n_reports), f"{int(r.frozen_month_found)}/1049", f"{r.frozen_recall:.2%}") for r in polar.itertuples()]), "",
        "VH使用同一原生成员、联合有效掩膜和真实VH值；VV基准精确复现。融合在同像元接受响应层以±2月一对一配对，交集保留配对VV日期，并集保留全部VV与未配对VH，再使用相同B20。融合容差与评价容差分别冻结。", "",
        "这里比较的是指定的VV日期优先策略；没有对应的完整precision/F1证据时，不宣称VV或融合的整体精度最佳。来源：[极化报告](polarization/README_ZH.md)、[完整比较](polarization/evaluation/EVALUATION_REPORT_ZH.md)。", "",
        "### 登记事件序列的恢复", "", *markdown_table(["方法", "首次/459", "后续/590", "全部登记事件找回/403林分", "相邻参考两端找回/590对"],
        [(LABELS[r.method], int(r.first_found), int(r.subsequent_found), int(r.n_all_found), int(r.both_found)) for r in history.itertuples()]), "",
        "这组结果对应重复采伐历史的研究目标，但仅覆盖已登记事件。全部登记事件找回不要求无额外报告，也不证明全部真实采伐历史正确。相邻参考两端均被检出后，两个对应报告未必仍相邻；相关计数保留在[序列恢复表](primary_registered_history_results.csv)，不能把所有报告间隔直接视为真实轮伐期。", "",
        "## 3. 采后观测前缀", "", "主队列固定1021条月份参考；28条仅因参考月+12超出2026-06档案终点而排除。观测不足和未检出留在主分母。已完成4084个唯一林分—截止月上下文、224892个完整成员像元前缀。", "",
        *markdown_table(["相对参考月", "找回/1021", "条件找回率", "95%空间块区间"],
        [(f"+{int(r.horizon_months)}个月", int(r.found), f"{r.recall:.2%}", f"{r.ci_lower:.2%}—{r.ci_upper:.2%}") for r in horizons.itertuples()]), "",
        f"h6与h12共同找回{int(stability.common_found_references)}条参考，日期绝对变化中位数{stability.median_absolute_date_change:g}月；变化不超过2个月的比例为{stability.proportion_within_two_months:.2%}。这是相同事件在两个时点均被找回条件下的日期稳定性。",
        f"同一共同成功子集的月份MAE在h6/h12分别为{stability.earlier_MAE:.3f}/{stability.later_MAE:.3f}月；找回更多不等于日期误差更小。首次检出后又在较长前缀消失的参考有{int(trajectories.disappeared_after_first_detection.sum())}条。当前配置依赖后续观测支持，不能依据全档案回顾结果宣称参考当月即可稳定检出。",
        "本实验先截断输入，再检测、置换和B20；未结束的三个月窗口不能参加分配。不同截止点报告没有累加为一个总体，各事件的找回也没有拼成完整历史。h相对光学参考月份，不是已知物理采伐日的实时延迟。", "",
        "来源：[前缀报告](truncation/README_ZH.md)、[固定队列曲线](truncation/fixed_cohort_curve.csv)、[日期稳定性](truncation/common_success_date_stability.csv)。", "",
        "## 4. 随机种子", "", *markdown_table(["Base seed", "接受像元月", "新增", "丢失", "与基准Jaccard"],
        [(int(r.base_seed), int(r.accepted_nodes), int(r.added), int(r.lost), f"{r.jaccard:.4f}") for r in seed.itertuples()]), "",
        "固定512个身份哈希像元，全部置换前候选均纳入；其他两种子各自进行3999/19999次评分与精化。候选日期、幅度及方向完全相同，差异来自Monte Carlo评分。该子集不是完整林分样本，不计算部分成员B20或林分精度。来源：[种子汇总](seed/summary.csv)。", "",
        "## 5. 验收与写作边界", "",
        "基准复现已核对242010条窗口轨迹、44311个接受响应及1267条报告/来源；新增组件、极化、前缀和种子诊断均有实际执行及独立验收记录。前缀通过真实序列未来注入不变、完整档案复现及逐任务完整成员检查。", "",
        "主结果使用登记参考的条件找回，不把未匹配报告当FP，不报告没有依据的总体precision/F1，不继承旧Polygon-first五折OOF。各比较完整展示，配对95%区间属于逐比较而非全族同时覆盖。", "",
        "原Results的旧W48、旧F1及旧日期版本数字应由作者后续替换或移出当前方法结论；本轮只交付更新依据。见[Results更新说明](RESULTS_UPDATE_NOTES_ZH.md)、[主张对应表](manuscript_claim_matrix.csv)及[总验收](acceptance_receipt.json)。", ""]
    lines += ["## 6. 执行记录", "",
        "执行范围以[冻结计划](plan_at_execution.md)为准；阶段验收分别为[基准](preflight_receipt.json)、[组件](components/verification_receipt.json)、[极化](polarization/verification_receipt.json)、[截断](truncation/verification_receipt.json)和[种子](seed/receipt.json)。[执行状态](execution_status.json)记录本轮完成情况；计划原文保留编制时状态，不覆盖冻结快照。", "",
        "实验入口与适配位于`tooling/diagnostics/balsa-results-ablation-20261101/`，使用仓库现有统计量、置换器、B20及匹配器；各阶段保存输入/源码签名、实际候选或来源账本。总验收同时复核原始受保护文件、复用实验收据及新结果哈希。", ""]
    (OUT / "README_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    claims = [
        ("4.2", "表征顺序", "既有同版实验复用", "../balsa_firstpost_result_analysis_20261030_v1/contributions/README_ZH.md", "Pixel/Polygon组合效应；不外推独立像元精度"),
        ("4.2", "W长度", "既有同版实验复用", "../balsa_research_execution_20261031_v3/windows/README_ZH.md", "固定D12/S12；不称全几何调优"),
        ("4.2", "极化", "P2已完成", "polarization/evaluation/catalogue_recall.csv", "VV日期优先；条件找回和报告负担；非F1排名"),
        ("4.3", "组件贡献", "P1已完成", "components/catalogue_recall.csv", "当前单组件移除；方向零效应；新增报告不是FP"),
        ("aggregation", "B20/Q20", "既有同节点实验复用", "../balsa_research_execution_20261031_v3/controls/CONTROLS_EXECUTION_REPORT_ZH.md", "汇总组合效应；非物理事件身份独立验证"),
        ("4.4", "核心基准", "既有同版实验复用", "../balsa_research_execution_20261031_v3/controls/CONTROLS_EXECUTION_REPORT_ZH.md", "保留Notebook适配身份，不移植旧OOF"),
        ("4.4", "时间依赖", "既有同版实验复用", "../balsa_research_execution_20261031_v3/simulation/README_ZH.md", "区分模拟无采伐误报与真实域条件找回"),
        ("4.4", "种子稳定性", "P4已完成", "seed/summary.csv", "固定512像元层；不外推林分稳定性"),
        ("4.5", "后续观测长度", "P3已完成", "truncation/summary.csv", "1021固定参考；回顾性前缀，不是部署前瞻精度"),
        ("4.6", "全国应用", "既有结果分析", "../balsa_firstpost_result_analysis_20261030_v1/national_findings_ZH.md", "本轮没有新增全国推理或判读任务")]
    pd.DataFrame(claims, columns=["section", "claim", "status", "evidence", "interpretation_limit"]).to_csv(
        OUT / "manuscript_claim_matrix.csv", index=False, encoding="utf-8-sig")
    notes = ["# Results 后续更新依据", "", "本轮不修改TeX。当前旧稿的数字和方法身份不能自动移植。", "",
        "1. §4.2保留已经完成的Pixel/Polygon及W36/48/60比较，补入本次极化四臂；删除继续保留Polygon-first W48的当前方法叙述。若不独立测试D/S，将标题和结论限定为窗口长度。",
        "2. §4.3使用当前单组件移除表；保留方向约束零效应，中心/置换写成检出—报告数量取舍。不能继续用旧累计表的precision/F1增益表述当前方法。",
        "3. §4.4使用同版Notebook ReCuSum适配、最大月降幅及IID/block3证据；种子稳定性使用512像元新表。旧CCDC/NRT等五折结果若保留，应明确其历史方法版本。",
        "4. §4.5换用1021条共同参考的前缀曲线及共同找回日期稳定性，保留不单调或消失事件；旧2026前瞻精度没有在本轮重建，不能移接。",
        "5. §4.6只使用既有同版应用结果。报告间隔仍是检测序列间隔，不因本轮消融而成为独立验证的生物轮伐期。", "",
        "全期1049参考、前缀1021参考和512像元数值诊断是三个不同分析单位与范围，表和图中分别标明。配对效应及不确定性见各阶段CSV。", ""]
    (OUT / "RESULTS_UPDATE_NOTES_ZH.md").write_text("\n".join(notes), encoding="utf-8")
    for claim in claims:
        assert (OUT / claim[3]).exists(), claim[3]
    checked_links = len(claims)
    for name in ("README_ZH.md", "RESULTS_UPDATE_NOTES_ZH.md"):
        for target in re.findall(r"\]\(([^)]+)\)", (OUT / name).read_text(encoding="utf-8")):
            if target == "acceptance_receipt.json":
                continue
            assert (OUT / target).exists(), target
            checked_links += 1
    status = dict(updated_utc=datetime.now(timezone.utc).isoformat(), **{f"P{i}":"completed" for i in range(6)})
    (OUT / "execution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    receipt = dict(passed=True, completed_utc=status["updated_utc"], protected_files_unchanged=len(preflight["protected_sha256"]),
        phase_receipts={str(p.relative_to(OUT)): digest(p) for p in stage_paths}, checked_phase_output_hashes=checked_outputs,
        truncation_completion_sha256=digest(OUT / "truncation/completion_receipt.json"),
        checked_document_links=checked_links, all_phases_completed=True, tex_modified=False, national_inference_run=False,
        checked_reused_receipts=len(preflight["reused_receipts"]),
        full_period_references=1049, prefix_references=1021, seed_pixels=512,
        output_sha256={n:digest(OUT/n) for n in ("README_ZH.md", "RESULTS_UPDATE_NOTES_ZH.md", "primary_ablation_results.csv", "primary_registered_history_results.csv", "manuscript_claim_matrix.csv", "execution_status.json")},
        generator_sha256=digest(Path(__file__)))
    (OUT / "acceptance_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:receipt[k] for k in ("passed", "protected_files_unchanged", "checked_phase_output_hashes", "all_phases_completed")}))


if __name__ == "__main__":
    main()
