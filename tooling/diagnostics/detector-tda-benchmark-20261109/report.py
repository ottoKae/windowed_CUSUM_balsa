"""Render the completed common-TDA benchmark tables without editing manuscript TeX."""
from pathlib import Path

import pandas as pd

from common import ROOT, OUT


def main():
    metrics = OUT / "metrics"
    summary = pd.read_csv(metrics / "summary.csv")
    effects = pd.read_csv(metrics / "paired_effects.csv")
    diagnostics = pd.read_csv(metrics / "report_and_monitoring_diagnostics.csv")
    table = summary.loc[summary.matching_rule.eq("pm2")]
    rows = ["| 方法/配置 | TDA报告 | 匹配参考 | Reference precision | Recall | Reference F1 | 保留原相邻对 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r in table.itertuples():
        rows.append(f"| {r.method_label} | {r.n_reports:,} | {r.n_matches:,} | {100*r.catalogue_precision:.2f}% | "
                    f"{100*r.recall:.2f}% | {100*r.catalogue_f1:.2f}% | {r.n_original_pairs_retained}/590 |")
    primary = ["| 对照（主方法减对照） | F1差值/百分点 | 配对97.5%区间/百分点 |",
               "|---|---:|---:|"]
    for r in effects.loc[effects.primary_contrast].itertuples():
        primary.append(f"| {r.method_label} | +{100*r.difference:.2f} | [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}] |")
    sensitive = ["| 容差 | 主方法匹配数/1049 | Reference F1 |", "|---|---:|---:|"]
    for r in summary.loc[summary.variant_id.eq("windowed_cusum_w36_firstpost")].itertuples():
        sensitive.append(f"| ±{r.matching_rule[2:]}月 | {r.n_matches} | {100*r.catalogue_f1:.2f}% |")
    long = ["| 方法 | 中位跨度/月 | 最大跨度/月 | 超过12月报告 |", "|---|---:|---:|---:|"]
    labels = table.set_index("variant_id").method_label
    for r in diagnostics.itertuples():
        long.append(f"| {labels[r.method_id]} | {r.median_span_months:g} | {r.max_span_months} | {r.n_span_over12}/{r.n_reports} |")
    text = """# 现有月度检测器统一TDA基准实验结果

日期：2026-11-09。本轮全部14配置已完成，包含windowed CuSum主方法和9类外部月度方法。未改变主方法、TDA门槛、参考标签、全国推理产品或任何TeX；没有比较BWA或评价TDA/STP独立贡献。

## 1. 本轮回答的问题

在当前固定像元输入及共同TDA规则下，现有检测器配置对参考采伐的检出、报告数量和重复采伐记录保留有何差异？主创新集中于windowed CuSum；TDA为共同报告层，STP为报告内部的空间时间表达。统一TDA不意味着各原始算法具有相同的监测起点、输入通道、报警日期或重置方式。

这是一轮**冻结历史参数的回顾比较**，不是新完成的五折调参、空间外折检验或全国概率抽样精度估计。各实现及资格差异见[详细实验模块](../../../docs/experiments/detector_tda_benchmark_20261109/EXPERIMENT_MODULE_ZH.md)。

## 2. 数据和评价范围

- 执行域：26,890个原生像元，468个林分，2016-01—2026-06月历；正式报告期2017—2025。
- 评价域：1,049条固定参考，459个林分；403个多参考林分，590个原始相邻参考对。
- 所有precision/F1的预测分母均限制在这459个林分。另9个执行林分的报告不进入主评价。
- 主匹配：同林分、代表月±2月、最大基数且最小日期误差的一对一匹配。敏感性为±1月和±3月。
- 参考未匹配报告不自动等于真实误报；下表为reference/catalogue-based precision和F1。Reference recall也只描述现有目录的找回。
- 主方法使用W36、D12、S12、alpha_ts=0.025和冻结First-post v2日期。TDA使用原固定成员20%支持、G1、观测约束和H12仅标记；STP子组不增加事件计数。

## 3. 全部配置主结果

""" + "\n".join(rows) + """

所有行均为同一次完整时期匹配，分母1049不因方法表现或监测起点调整。报告数为459参考林分中的TDA父报告，而不是像元检测数。

## 4. 可支持的主要结论

1. **当前配置下，windowed CuSum的reference-F1和recall最高。** 其1208个父报告找回925条参考，precision 76.57%、recall 88.18%、F1 81.97%。这里是TDA父报告结果；旧版加入STP日期见证后的930不能混入本表。
2. **最接近的外部配置是ReCuSum first-post适配。** 它以1024个父报告找回804条，precision 78.52%、recall 76.64%、F1 77.57%。主方法多找回121条，代价是更多报告与略低的参考precision。F1差4.40个百分点，探索性95%配对区间为[2.59,6.23]个百分点。
3. **对重复采伐记录的保留更充分。** 主方法保留473/590个原始相邻参考对，并完整找回301/403个多参考林分的已记录事件；ReCuSum first-post分别为372/590和230/403。这里的“完整”仅指现有目录，不表示完整真实经营历史。
4. **不能声称主方法对所有已找回事件都定位更准。** 在与ReCuSum first-post共同找回的783条参考上，主方法与对照MAE分别为0.693和0.701月；差值95%区间[-0.042,0.027]月跨零。主优势是更完整的事件找回与平衡表现，不是普遍更低的日期误差。各方法仅在自身成功样本上的MAE不能直接横向证明优劣。
5. **方向筛选和输出密度是外部配置的重要差异。** NRT下降版找回900条、recall85.80%，但1960个报告对应reference precision45.92%；不能仅用原NRT(raw)较低的结果代表整个NRT家族。原始/下降/递归日期变体全部列出，不根据结果挑选最弱对照。

## 5. 统计对照和容差

评分前冻结两项主要对照；使用相同5000次459林分簇重采样，seed=20261105。以下为两比较Bonferroni 97.5%区间，其他差异为探索性95%区间。旧结果曾被查看，不能称盲态预注册；这是stand bootstrap，未显式消除林分间空间相关性。

""" + "\n".join(primary) + "\n\n" + "\n".join(sensitive) + """

主方法在±3月下953条匹配与原冻结R2结果一致；本轮没有为外部方法追加平移日期或扩大时间包络的匹配规则。

## 6. 必须随结果呈现的适配边界

**年度监测适配。** BFAST Monitor和EWMACD的既有调度仅执行2019—2025，每像元每年最多一个候选；2017—2018保持unknown。这两年包含277条参考，占26.41%。不能把全期低recall全部归为原算法本身的能力不足，也不能笼统解释为这两年都绝对没有足够训练数据。BFAST先取该年度原生首个alarm，再检验负方向，不搜索替代的后续负alarm。实际按年度统计保存在`metrics/recall_by_reference_year.csv`。

**CCDC支持范围。** 两个CCDC配置使用全部有效拟合segment的回顾性tStart—tEnd支持，包括没有接受负断点的segment；不是只用接受事件推资格。未建模区间保持未知。缓存未包含精确逐月在线资格，故不能声称重建了它。接受断点处active优先于quiet资格。

**输入和日期。** BOCD是VV+VH原型；其余主配置是VV。经典CuSum的报警月、NRT原峰值月、CCDC断点月以及ReCuSum的真实递归日期定义均保留。First-post ReCuSum进行了真正递归重放，不是最后统一加一个月。

**密集检测进入共同TDA后可能形成长链。** 最大月下降和NRT raw最长报告均达到108个月；主方法最长35个月。它说明检测密度、TDA连接和代表月会共同影响最终参考匹配，不代表真实采伐持续这么久。相同TDA保证报告规则统一，但这轮衡量的是“检测器接入当前报告层后的表现”，不能独立归因为每个检测器的纯像元能力。未经对应参考核验，不额外判定长报告为错误事件，也不以新切分规则改善名次。

""" + "\n".join(long) + """

年度LandTrendr现有输出没有经验证的月份，因此不进入该月度协议，不虚构1月或7月。W24/W48等属于主方法已有配置敏感性，未重复作为外部家族计数。

## 7. 完成和复现记录

- 主方法44311个像元—月份节点→1216个执行域父报告；报告、组件、progress、monthly activity与冻结生产基线逐项一致。其中459参考林分为1208报告。
- 最大月下降、经典CuSum、BOCD及三个ReCuSum配置完成全域实际算法重放，接受日期/相关分数与冻结输出一致。
- NRT每配置重建2653789个真实扫描窗口，逐项核对观测Q、首峰、日期和前后均值；保留已冻结置信度，未重复500次置换。
- CCDC、结构断点、BFAST、EWMACD复放原始缓存并核对当前实际输入及接受输出；未启动新EE/R任务。
- 30项相关测试通过；生产TDA基线和评价1208/925锚点通过。每个输入、适配输出、TDA链及评价表均有来源记录。
- `report_and_monitoring_diagnostics.csv`的报告/跨度/月状态为459林分域；raw nominations与unique pixel-months是468执行林分域。两种范围不能直接用于计算压缩比。

主要文件：

| 文件 | 用途 |
|---|---|
| `evaluation_protocol.json` | 本轮新匹配之前冻结的配置和统计合同 |
| `evaluation_receipt.json`、`execution_manifest.json` | 全14配置完成记录、来源及输出hash |
| `metrics/summary.csv` | 14配置×3容差，P/R/F1、日期及历史指标 |
| `metrics/primary_contrasts.csv`、`metrics/paired_effects.csv` | 主对比、探索性差异、共同成功日期/间隔误差 |
| `metrics/reference_outcome_changes.parquet` | 相对主方法新增、丢失及共同匹配来源 |
| `methods/<id>/reports.parquet` | 各方法完整TDA父报告 |
| `methods/<id>/source_members.parquet` | 每个像元检测节点的去向，含未解析/不足支持 |
| `methods/<id>/metrics/pm2_matches.parquet` | 固定1049条参考的逐条匹配 |
| `detectors/<id>/detector_receipt.json` | 原参数、日期、资格定义、重放范围和来源hash |

入口代码位于`tooling/diagnostics/detector-tda-benchmark-20261109/`。`local.py`、`recusum.py full --workers 4`和`cloud.py`负责适配；`aggregate.py`使用生产TDA并跳过已完成方法；`score.py`评分；`report.py`生成本报告。已冻结的`score.py --freeze`不重复执行。使用新协议/参数时应新建版本，不能覆盖此冻结结果来筛选参数。

## 8. 写作落实

3.4增加`\\subsubsection{Benchmark detector comparison}`，位于事件匹配定义后、检测器消融前。统计段统一说明两项主要97.5%与其他95%，不再笼统声称全部比较都是95%。英文可用稿及14配置补充表见[英文Methods草稿](../../../docs/experiments/detector_tda_benchmark_20261109/BENCHMARK_METHODS_DRAFT_EN.md)。

Results建议依次为：4.1参考采伐的VV/VH响应；4.2共同TDA下检测器比较；4.3局部上下文与候选筛选的消融贡献；4.4配置敏感性与时间稳健性；4.5采伐历史制图与林分内部时间格局。TDA/STP作为统一结果组织步骤保留在Methods3.3，不单独声称精度增益。

当前TeX的旧4.2—4.6仍含Polygon-first、W48、五折外推、旧benchmark及全国数字，不能与本表直接拼接。本轮未改TeX；后续写作应按当前输入、TDA和实际实验版本逐段替换。
"""
    (OUT / "EXPERIMENT_REPORT_ZH.md").write_text(text, encoding="utf-8")
    print(OUT / "EXPERIMENT_REPORT_ZH.md")


if __name__ == "__main__":
    main()
