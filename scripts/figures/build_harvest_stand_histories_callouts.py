"""Figure 6 with external stand callouts, aligned IDs and horizontal legends."""
import json
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from shapely.geometry import Point
from shapely.ops import nearest_points

import build_harvest_stand_histories_column_legends as compact

spans = compact.spans
figure = compact.figure
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v11_callouts'
RIGHT_CALLOUTS = False
LABEL_POSITIONS = {
    716: [(.40, .95), (.95, .78), (.045, .66), (.95, .35), (.46, .105)],
    728: [(.69, .87), (.79, .60), (.87, .34), (.16, .26), (.19, .16)],
    22301: [(.60, .83), (.13, .65), (.08, .40), (.90, .27), (.40, .20)],
}


def main():
    spans.OUT = OUT
    spans.LATEST_PURPLE = True
    spans.draw_span_bars = draw_history
    spans.monthly.draw_map = draw_map
    figure.HEIGHT_MM = 153
    figure.TOP_MM, figure.ROW_STEP_MM = 3, 45
    figure.HISTORY_LEFT_MM, figure.HISTORY_WIDTH_MM = 63, 67
    figure.STP_COLORS = ['#3594BC', '#69429A']
    figure.draw_decorations = draw_decorations
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FIGURE_CONTRACT_ZH.md').write_text(
        '仅调整Figure6表达。左列15个字母置于所有选定林分外，引线指向本林分最近边界。'
        '中列字母与ID分别左对齐，绘图宽64→67mm。三列图例在同一水平线各自横排。'
        '文字至少6.6pt，主稿164.6mm缩放后至少约6pt；地图保持40mm，全部冻结数据保留。'
        'Python绘制180×153mm，PDF/SVG/600dpi PNG；检查标注矩形与矢量、引线和图例间距。', encoding='utf-8')
    spans.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(history_colors='latest TDA purple, previous dark blue, earliest light blue',
                    history_axis_alignment='bottom x axis coincides with last stand zero baseline',
                    history_panel_labels='lower right', history_plot_width_mm=67,
                    history_labels='letters and stand IDs in separate left-aligned columns',
                    callout_positions=LABEL_POSITIONS, column_titles=False,
                    legends='single horizontal row, each beneath its column',
                    minimum_font_pt=6.6, date_colourbar_domain=['2017-01', '2026-01'],
                    stp_palette=figure.STP_COLORS,
                    callouts_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    previous = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v10_compact/CAPTION_EN.md'
    caption = previous.read_text(encoding='utf-8').replace(
        'letters identify the stands in (b,e,h).',
        'external letter labels identify the stands in (b,e,h).')
    caption = caption.replace('Maps within each row share the same extent;',
        'Grey cells in (c,f,i) lie outside the latest report. Maps within each row share the same extent;')
    (OUT / 'CAPTION_EN.md').write_text(caption, encoding='utf-8')


def draw_history(ax, case, panel):
    handles = compact.draw_history(ax, case, panel)
    ax.set_yticklabels([])
    for index, stand in enumerate(case['ids']):
        ax.text(-11.7 / figure.HISTORY_WIDTH_MM, index, chr(65 + index), transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=7.2, clip_on=False)
        ax.text(-8.4 / figure.HISTORY_WIDTH_MM, index, str(stand), transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=7.2, clip_on=False)
    ax.tick_params(axis='x', labelsize=6.8)
    for text in ax.texts:
        if text.get_text() in ('0', '40'):
            text.set_fontsize(6.6)
            text.set_color('#465660')
        elif text.get_text() == f'({panel})':
            text.set_fontsize(8.5)
    return handles


def draw_map(ax, case, palette, panel, stp):
    spans.monthly.ORIGINAL_MAP(ax, case, palette, panel, stp, show_labels=False)
    ax.tick_params(axis='both', labelsize=6.6)
    for text in ax.texts:
        if text.get_text() == f'({panel})':
            text.set_fontsize(8.5)
        else:
            text.set_fontsize(6.6)
    if stp and not RIGHT_CALLOUTS:
        return
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for index, (stand, fraction) in enumerate(zip(case['shapes'].itertuples(), LABEL_POSITIONS[case['focal']])):
        label_point = Point(xmin + fraction[0] * (xmax - xmin), ymin + fraction[1] * (ymax - ymin))
        assert not case['shapes'].geometry.covers(label_point).any()
        target = nearest_points(stand.geometry.boundary, label_point)[0]
        line = ax.annotate('', xy=(target.x, target.y), xytext=(label_point.x, label_point.y),
                           arrowprops=dict(arrowstyle='-', color='white', lw=.7,
                                           shrinkA=4.3, shrinkB=0), zorder=7)
        line.arrow_patch.set_path_effects([pe.Stroke(linewidth=1.8, foreground='#203028'), pe.Normal()])
        ax.text(label_point.x, label_point.y, chr(65 + index), color='white', weight='bold',
                fontsize=8.5, ha='center', va='center', zorder=8,
                path_effects=[pe.withStroke(linewidth=1.7, foreground='#203028')])


def draw_decorations(fig, palette, parents, rounds, history_handles):
    width, height = figure.WIDTH_MM, figure.HEIGHT_MM
    cax = fig.add_axes([9 / width, 10 / height, 40 / width, 2.4 / height])
    cmap = ListedColormap([palette['month_hex'][m] for m in sorted(palette['month_hex'])])
    bar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(range(109), 108), cmap=cmap),
                      cax=cax, orientation='horizontal')
    bar.set_ticks([0, 36, 72, 108], labels=['2017', '2020', '2023', '2026'])
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=6.6, length=2, pad=1)
    bar.outline.set_visible(False)
    cax.text(.5, 1.65, 'Latest pixel-detection month', transform=cax.transAxes,
             ha='center', fontsize=6.6)
    common = dict(frameon=False, fontsize=6.6, handlelength=1.1, handletextpad=.3,
                  columnspacing=.35, borderaxespad=0, loc='upper center')
    y = 5.5 / height
    fig.legend(handles=[Patch(facecolor=figure.NO_DETECTION, label='No pixel detection')],
               bbox_to_anchor=(29 / width, y), **common)
    history = [
        (Patch(facecolor=spans.SPAN_COLORS[2]), Patch(facecolor=spans.BAR_COLORS[2])),
        (Patch(facecolor=spans.SPAN_COLORS[1]), Patch(facecolor=spans.BAR_COLORS[1])),
        Patch(facecolor=spans.UNASSIGNED)]
    fig.legend(handles=history, labels=['Latest TDA', 'Earlier TDA', 'Unassigned'], ncol=3,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.1)},
               bbox_to_anchor=(96.5 / width, y), **common)
    groups = int(rounds.round_index.max()) + 1
    right = [Patch(facecolor=figure.STP_COLORS[i], label=f'STP {i+1}') for i in range(groups)]
    right.append(Patch(facecolor=figure.OUTSIDE_LATEST, label='Not in report'))
    fig.legend(handles=right, ncol=3, bbox_to_anchor=(154 / width, y), **common)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_widths = [legend.get_window_extent(renderer).width / fig.dpi * 25.4 for legend in fig.legends]
    assert all(w <= limit for w, limit in zip(legend_widths, [40, 67, 40])), legend_widths
    (OUT / 'layout_measurements.json').write_text(json.dumps(
        {'legend_widths_mm': legend_widths, 'legend_column_widths_mm': [40, 67, 40],
         'native_figure_width_mm': 180, 'current_manuscript_text_width_mm': 164.6,
         'minimum_printed_font_pt': 6.6 * 164.6 / 180}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
