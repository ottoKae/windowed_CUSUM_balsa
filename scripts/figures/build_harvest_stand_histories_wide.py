"""Widen Figure 6 histories and align column legend blocks at their tops."""
import json
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import pandas as pd

import build_harvest_stand_histories_callouts as callouts

spans = callouts.spans
figure = callouts.figure
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v12_wide'


def main():
    spans.OUT = OUT
    spans.LATEST_PURPLE = True
    spans.SHOW_COUNT_AXIS = False
    spans.draw_span_bars = draw_history
    spans.monthly.draw_map = callouts.draw_map
    callouts.RIGHT_CALLOUTS = True
    figure.HEIGHT_MM = 153
    figure.TOP_MM, figure.ROW_STEP_MM = 3, 45
    figure.LEFT_MAP_MM, figure.RIGHT_MAP_MM = 4, 136
    figure.HISTORY_LEFT_MM, figure.HISTORY_WIDTH_MM = 58, 76
    figure.STP_COLORS = ['#3594BC', '#69429A']
    figure.draw_decorations = draw_decorations
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FIGURE_CONTRACT_ZH.md').write_text(
        '左右两列均标A–E，复用经核验的林分外引线位置。左右地图移至x4/136mm，'
        '中图x58宽76mm；原始地图范围与40mm大小不变。x轴2017至2026逐年，显示年度网格线。'
        '隐藏0–40文字及小刻度，caption保留共同0–40数量尺度，未归一化柱高。'
        '三列图例区顶部对齐，分类图例横排。Python180×153mm，最小6.6pt，PDF/SVG/600dpi PNG。',
        encoding='utf-8')
    spans.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(history_colors='latest TDA purple, previous dark blue, earliest light blue',
                    count_axes='hidden; common linear 0–40 scale stated in caption',
                    history_axis_alignment='bottom x axis coincides with last stand zero baseline',
                    history_panel_labels='lower right', history_plot_width_mm=76,
                    history_year_ticks=list(range(2017, 2027)), annual_grid=True,
                    map_left_positions_mm=[4, 136], right_map_stand_letters=True,
                    callout_positions=callouts.LABEL_POSITIONS, column_titles=False,
                    legends='column blocks aligned by their top text edges; categorical legends horizontal',
                    minimum_font_pt=6.6, date_colourbar_domain=['2017-01', '2026-01'],
                    stp_palette=figure.STP_COLORS,
                    wide_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    previous = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v11_callouts/CAPTION_EN.md'
    caption = previous.read_text(encoding='utf-8').replace(
        'external letter labels identify the stands in (b,e,h).',
        'matching external letter labels in both map columns identify the stands in (b,e,h).')
    caption = caption.replace('on a common 0–40 scale.',
        'on a common linear scale of 0–40 pixels per row.')
    (OUT / 'CAPTION_EN.md').write_text(caption, encoding='utf-8')


def draw_history(ax, case, panel):
    handles = callouts.draw_history(ax, case, panel)
    ax.set_xticks(pd.to_datetime([f'{y}-01-01' for y in range(2017, 2027)]))
    ax.grid(axis='x', color='#cdd6dd', linewidth=.45, zorder=0)
    return handles


def draw_decorations(fig, palette, parents, rounds, history_handles):
    width, height = figure.WIDTH_MM, figure.HEIGHT_MM
    cax = fig.add_axes([4 / width, 10 / height, 40 / width, 2.4 / height])
    cmap = ListedColormap([palette['month_hex'][m] for m in sorted(palette['month_hex'])])
    bar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(range(109), 108), cmap=cmap),
                      cax=cax, orientation='horizontal')
    bar.set_ticks([0, 36, 72, 108], labels=['2017', '2020', '2023', '2026'])
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=6.6, length=2, pad=1)
    bar.outline.set_visible(False)
    heading = cax.text(.5, 1.65, 'Latest pixel-detection month', transform=cax.transAxes,
                       ha='center', fontsize=6.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = heading.get_window_extent(renderer).y1 / fig.bbox.height
    common = dict(frameon=False, fontsize=6.6, handlelength=1.1, handletextpad=.3,
                  columnspacing=.35, borderaxespad=0, borderpad=0, loc='upper center')
    fig.legend(handles=[Patch(facecolor=figure.NO_DETECTION, label='No pixel detection')],
               bbox_to_anchor=(24 / width, 5.5 / height), **common)
    history = [
        (Patch(facecolor=spans.SPAN_COLORS[2]), Patch(facecolor=spans.BAR_COLORS[2])),
        (Patch(facecolor=spans.SPAN_COLORS[1]), Patch(facecolor=spans.BAR_COLORS[1])),
        Patch(facecolor=spans.UNASSIGNED)]
    fig.legend(handles=history, labels=['Latest TDA', 'Earlier TDA', 'Unassigned'], ncol=3,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.1)},
               bbox_to_anchor=(96 / width, top), **common)
    groups = int(rounds.round_index.max()) + 1
    right = [Patch(facecolor=figure.STP_COLORS[i], label=f'STP {i+1}') for i in range(groups)]
    right.append(Patch(facecolor=figure.OUTSIDE_LATEST, label='Not in report'))
    fig.legend(handles=right, ncol=3, bbox_to_anchor=(156 / width, top), **common)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tops = [heading.get_window_extent(renderer).y1] + [
        max(t.get_window_extent(renderer).y1 for t in legend.get_texts()) for legend in fig.legends[1:]]
    widths = [legend.get_window_extent(renderer).width / fig.dpi * 25.4 for legend in fig.legends]
    assert max(tops) - min(tops) < .1, tops
    assert all(w <= limit for w, limit in zip(widths, [40, 76, 40])), widths
    (OUT / 'layout_measurements.json').write_text(json.dumps(
        {'legend_text_top_pixels': tops, 'legend_widths_mm': widths,
         'legend_column_widths_mm': [40, 76, 40], 'history_width_mm': 76,
         'native_figure_width_mm': 180, 'current_manuscript_text_width_mm': 164.6,
         'minimum_printed_font_pt': 6.6 * 164.6 / 180}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
