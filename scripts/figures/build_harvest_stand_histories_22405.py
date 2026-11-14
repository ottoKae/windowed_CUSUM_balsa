"""Replace the middle Figure 6 neighbourhood and retain unfilled cell outlines."""
import json
from pathlib import Path

import pandas as pd

import build_harvest_stand_histories_panel_centered as previous

integrated = previous.previous.integrated
figure, spans = integrated.figure, integrated.spans
callouts = integrated.previous.callouts
SOURCE = previous.OUT
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v18_22405'
OPTIONS = figure.ROOT / 'runs/figures/harvest_stp_case_options_20261112_v1'
CASES = {716: [719, 717, 718, 716, 715],
         22405: [22824, 22808, 22405, 22806, 22805],
         22301: [22885, 22883, 22605, 22302, 22301]}


def main():
    inputs = OUT / 'inputs'
    (inputs / 'imagery').mkdir(parents=True, exist_ok=True)
    records = json.loads((SOURCE / 'imagery/selected_input_imagery.json').read_text(encoding='utf-8'))
    records['records'] = [r for r in records['records'] if r['source_row_original'] != 728]
    candidates = json.loads((OPTIONS / 'imagery/candidate_input_imagery.json').read_text(encoding='utf-8'))
    record = next(r for r in candidates['records'] if r['source_row_original'] == 22405)
    records['records'].append(record)
    (inputs / 'imagery/selected_input_imagery.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

    members = pd.read_parquet(figure.base.PRODUCT / 'national_membership.parquet')
    parents = pd.read_parquet(figure.PRODUCT / 'octa/reports.parquet')
    parents = parents.sort_values(['sample_id', 'representative_month', 'prediction_id'])
    latest = parents.groupby('sample_id', sort=False).tail(1).set_index('sample_id')
    labels = []
    for case, ids in CASES.items():
        for index, stand_id in enumerate(ids):
            sample_id = members.loc[members.source_row_original.eq(stand_id), 'sample_id'].iloc[0]
            labels.append(dict(case=case, letter=chr(65+index), stand_id=stand_id,
                               sample_id=sample_id, latest_prediction_id=latest.loc[sample_id, 'prediction_id']))
    labels = pd.DataFrame(labels)
    labels.to_csv(inputs / 'source_stands.csv', index=False)
    parents.loc[parents.sample_id.isin(labels.sample_id)].merge(
        labels[['case', 'letter', 'stand_id', 'sample_id']], on='sample_id', validate='many_to_one'
    ).to_csv(inputs / 'source_tda_reports.csv', index=False)
    (OUT / 'FIGURE_CONTRACT_ZH.md').write_text(
        '本图联系像元检出、林分多年TDA历史和最新TDA内部的STP分组。Python绘制，保留180×147 mm三行三列版式。\n'
        '用户指定将中行替换为22405邻域；该邻域用于展示多个STP，并非随机精度样本。首尾两行林分不变。\n'
        '左列无检出成员不填色，但保留黑色像元边界；全图月检出数量统一0–60（22405峰值53）。\n'
        '右列继续各林分最新TDA，保留跨年完整范围，不施加2024年选择或截断。STP编号在父报告内定义。\n'
        '保留2025-03空间背景、已批准图例和配色，不重跑算法；输出PDF/SVG/600dpi PNG及源数据。\n', encoding='utf-8')

    integrated.PREVIOUS = inputs
    spans.monthly.PREVIOUS = inputs
    spans.monthly.CASES = CASES
    spans.monthly.COUNT_MAX = spans.COUNT_MAX = 60
    figure.DRAW_EMPTY_CELL_BORDERS = True
    callouts.LABEL_POSITIONS[22405] = [(.16, .74), (.25, .44), (.92, .56), (.20, .31), (.54, .065)]
    previous.OUT = OUT
    previous.main()

    caption = (OUT / 'CAPTION_EN.md').read_text(encoding='utf-8').replace(
        'pixels without detections are transparent.',
        'pixels without detections are unfilled, with black cell outlines retained.').replace('0–40', '0–60')
    (OUT / 'CAPTION_EN.md').write_text(caption, encoding='utf-8')
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(
        selection='User-selected 22405 neighbourhood replaces the middle row to illustrate multiple STP groups; first and third rows retained.',
        count_axes='hidden; common linear 0–60 scale stated in caption',
        no_pixel_detection='unfilled with 0.22-pt black cell outlines retained; no legend swatch',
        plotting_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pixels = pd.read_parquet(OUT / 'source_pixels.parquet')
    assert (manifest['fixed_pixels'], manifest['parent_count'], manifest['latest_stp_group_count']) == (748, 34, 18)
    assert manifest['monthly_count_peak'] == 53
    (OUT / 'README_ZH.md').write_text(
        '# Figure 6：22405邻域与透明格网边界\n\n'
        '第二行A–E为22824、22808、22405、22806、22805；其中22824和22405的最新TDA均有两个STP。\n'
        f'共748个固定成员、34个TDA、18个最新STP组；左列{pixels.last_month.isna().sum()}个无检出成员保留黑边、不填色。\n'
        '柱状图统一0–60，实际峰值53；全部父报告和STP沿用冻结产品，右列仍为各林分最新TDA。\n'
        '配色、图例居中和180×147 mm排版保留。脚本：scripts/figures/build_harvest_stand_histories_22405.py。\n', encoding='utf-8')


if __name__ == '__main__':
    main()
