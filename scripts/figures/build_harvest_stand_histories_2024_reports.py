"""Display complete TDA reports containing actual 2024 detections in Figure 6."""
import json
from pathlib import Path

import geopandas as gpd
from matplotlib.patches import Rectangle
import pandas as pd

import build_harvest_stand_histories_22405 as previous

figure, spans, callouts = previous.figure, previous.spans, previous.callouts
SOURCE = previous.OUT
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v19_2024_reports'
ORIGINAL_MAP = callouts.draw_map
ORIGINAL_DECORATIONS = previous.previous.draw_decorations
SELECTED_PIXELS = None


def main():
    global SELECTED_PIXELS
    OUT.mkdir(parents=True, exist_ok=True)
    reports = pd.read_csv(SOURCE / 'source_tda_reports.csv')
    ownership = pd.read_parquet(figure.PRODUCT / 'cstr/node_ownership.parquet')
    ownership = ownership.loc[ownership.prediction_id.isin(reports.prediction_id)]
    selected_ids = set(ownership.loc[ownership.response_month.between('2024-01', '2024-12'), 'prediction_id'])
    selected = reports.loc[reports.prediction_id.isin(selected_ids)]
    assert not selected.duplicated('sample_id').any()
    selected_nodes = ownership.loc[ownership.prediction_id.isin(selected_ids)].copy()
    owners = selected_nodes[['unit_id', 'prediction_id', 'round_id']].drop_duplicates()
    assert not owners.duplicated('unit_id').any()
    rounds = pd.read_parquet(figure.PRODUCT / 'cstr/rounds.parquet')
    rounds = rounds.loc[rounds.parent_prediction_id.isin(selected_ids)]
    owners = owners.merge(rounds[['round_id', 'round_index']], on='round_id', validate='many_to_one')
    pixels = gpd.read_parquet(SOURCE / 'source_pixels.parquet')
    SELECTED_PIXELS = pixels.drop(columns=['prediction_id', 'round_id', 'round_index']).merge(
        owners, on='unit_id', how='left', validate='one_to_one')
    labels = pd.read_csv(SOURCE / 'source_stands.csv')
    rounds = rounds.merge(labels[['sample_id', 'stand_id', 'case', 'letter']], on='sample_id', validate='many_to_one')
    selected.to_csv(OUT / 'source_selected_tda_reports.csv', index=False)
    rounds.to_csv(OUT / 'source_selected_stp_groups.csv', index=False)
    selected_nodes.to_parquet(OUT / 'source_selected_nodes.parquet', index=False)
    SELECTED_PIXELS.to_parquet(OUT / 'source_display_pixels.parquet', index=False)
    assert (len(selected), len(rounds), len(owners)) == (14, 17, 435)

    callouts.draw_map = draw_map
    previous.integrated.previous.HEIGHT_MM = 150
    previous.previous.draw_decorations = draw_decorations
    previous.OUT = OUT
    previous.main()

    caption = (
        'Stand-level harvest histories and within-event timing in three balsa neighbourhoods. '
        '(a,d,g) Latest pixelwise detection months, 2017–2025. '
        '(b,e,h) Monthly detected-pixel counts (0–60 per stand), with shaded TDA detection ranges. '
        '(c,f,i) STP groups within complete TDA reports containing detections in 2024, including cross-year detections. '
        'Letters identify corresponding stands across each row. Basemaps are March 2025 PlanetScope images.\n')
    (OUT / 'CAPTION_EN.md').write_text(caption, encoding='utf-8')
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(
        right_map_selection='Complete TDA reports with at least one actual response_month in 2024; retain all their nodes and STP groups.',
        displayed_parent_count=14, displayed_stp_group_count=17,
        displayed_stp_pixels=435, outside_selected_pixels=313,
        selected_node_count=len(selected_nodes),
        source_latest_tables_role='Preserved latest-per-stand source tables support the unchanged history colours; source_selected_* and source_display_pixels describe right maps.',
        outside_latest_tda='superseded for right-map display by outside selected TDA reports',
        legends='subtitle above each column legend; second row retains range/bar or STP colour keys',
        legend_centre_from_canvas_top_mm=142.7, subtitle_centre_from_canvas_top_mm=138.4,
        footer_subtitles=['Latest pixelwise detections', 'Stand harvest histories', 'TDA reports with detections in 2024'],
        stp_legend_rows=[['STP 1', 'STP 2', 'Outside selected TDA']],
        plotting_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUT / 'README_ZH.md').write_text(
        '# Figure 6：2024有检出的完整TDA报告\n\n'
        'c/f/i按实际检出月份筛选父报告：至少一个检出月份落在2024年。保留入选报告全部年份的像元和STP。\n'
        '共14份报告、17个STP组、435个不同成员，313个成员在选中报告外（灰色）。\n'
        '相较v18，仅22605的2018报告不再显示；该林分18个成员全灰。左列和完整历史数值不变。\n'
        '435成员中31个只在2023或2025检出，仍属于所选完整报告；不能把本图解释为2024年度像元分类。\n'
        '三列图例上方各加一行subtitle，画布180×150mm，地图/历史面板大小与位置不变。\n'
        '中列继续Latest/Earlier TDA颜色，右列显示统一筛选范围。检测、TDA、STP结果均未重算。\n'
        'source_selected_*及source_display_pixels为右列实际数据；source_latest_*保留历史来源。\n', encoding='utf-8')
    (OUT / 'FIGURE_CONTRACT_ZH.md').write_text(
        'Python在已批准三列布局上展示：最近像元检出、完整林分历史、2024有检出的完整TDA之空间分组。\n'
        'archetype: image plate + quant。选择依据为实际response_month，不使用代表年或仅包络相交。\n'
        '保持成员、时间范围、跨年来源、共同0–60柱高尺度及2025-03背景。PDF/SVG/600dpi PNG。\n'
        '只变化右列选择与图例文字/两行布局，不改变检测或聚合。\n', encoding='utf-8')


def draw_map(ax, case, palette, panel, stp):
    if stp:
        case = {**case, 'pixels': SELECTED_PIXELS.loc[SELECTED_PIXELS.source_row_original.isin(case['ids'])]}
    ORIGINAL_MAP(ax, case, palette, panel, stp)


def draw_decorations(fig, palette, parents, rounds, history_handles):
    ORIGINAL_DECORATIONS(fig, palette, parents, rounds, history_handles)
    cax, hax, rax = fig.axes[-3:]
    for ax in (cax, hax, rax):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0-2.7/figure.HEIGHT_MM, pos.width, pos.height])
    cax.texts[0].set_text('Detection month')
    position = rax.get_position()
    rax.set_position([133/figure.WIDTH_MM, position.y0, 46/figure.WIDTH_MM, position.height])
    rax.clear()
    rax.set(xlim=(0, 46), ylim=(0, 4))
    rax.set_axis_off()
    square = 2.2
    for x, label, colour in [(0, 'STP 1', figure.STP_COLORS[0]),
                             (10, 'STP 2', figure.STP_COLORS[1]),
                             (20, 'Outside selected TDA', figure.OUTSIDE_LATEST)]:
        rax.add_patch(Rectangle((x, 2-square/2), square, square, facecolor=colour, edgecolor='none'))
        rax.text(x+square+.6, 2, label, va='center', fontsize=6.6, color='#202523')
    titles = []
    for centre, label in [(24, 'Latest pixelwise detections'),
                           (89.65, 'Stand harvest histories'),
                           (156, 'TDA reports with detections in 2024')]:
        titles.append(fig.text(centre/figure.WIDTH_MM, 1-138.4/figure.HEIGHT_MM, label,
                               ha='center', va='center', fontsize=6.6, weight='bold', color='#202523'))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in (cax, hax, rax):
        box = ax.get_window_extent(renderer)
        for text in ax.texts:
            tb = text.get_window_extent(renderer)
            assert box.contains(tb.x0, tb.y0) and box.contains(tb.x1, tb.y1), text.get_text()
    for title, limit in zip(titles, [40, 88.7, 40]):
        tb = title.get_window_extent(renderer)
        assert tb.width/fig.dpi*25.4 <= limit
    path = OUT / 'layout_measurements.json'
    layout = json.loads(path.read_text(encoding='utf-8'))
    layout.update(legend_centre_from_top_mm=142.7, subtitle_centre_from_top_mm=138.4,
                  legend_left_mm=[4, 51.65, 133], legend_width_mm=[40, 76, 46],
                  stp_key_mm=[2.2, 2.2], dimensions_mm=[180, 150])
    path.write_text(json.dumps(layout, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
