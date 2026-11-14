"""Compact Figure 6 with column-specific legends and latest-report emphasis."""
import json
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

import build_harvest_stand_histories_spans as spans

figure = spans.figure
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v10_compact'
ORIGINAL_HISTORY = spans.draw_span_bars


def main():
    spans.OUT = OUT
    spans.LATEST_PURPLE = True
    spans.draw_span_bars = draw_history
    figure.HEIGHT_MM = 154
    figure.TOP_MM, figure.ROW_STEP_MM = 3, 45
    figure.STP_COLORS = ['#3594BC', '#69429A']
    figure.draw_decorations = draw_decorations
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FIGURE_CONTRACT_ZH.md').write_text(
        '保留三行三列、全部冻结数据及40mm空间图。删除列标题，行距9mm改5mm；'
        'b/e/h在图内右下，末行柱基线与x轴重合。图例分别置于所属列下方。'
        '最新TDA紫色，前一次深蓝，更早天蓝；STP1天蓝/STP2紫。'
        '日期色标2017-01至2026-01，108个月格保留原像元颜色，2026是末格的右边界。'
        'Python绘制180×154mm，PDF/SVG/600dpi PNG，所有文字至少5pt。', encoding='utf-8')
    spans.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(history_colors='latest TDA purple, previous dark blue, earliest light blue',
                    history_axis_alignment='last stand zero baseline coincides with x axis',
                    history_panel_labels='lower right inside each panel',
                    row_gap_mm=5, column_titles=False, legends='below their corresponding columns',
                    date_colourbar_domain=['2017-01', '2026-01'],
                    stp_palette=figure.STP_COLORS,
                    column_legends_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    caption = (
        'Pixel detections and reconstructed stand harvest histories in three balsa neighbourhoods. '
        '(a,d,g) Latest accepted pixel-detection months over PlanetScope basemaps; letters identify the stands in (b,e,h). '
        '(b,e,h) Monthly counts of distinct detected pixels on a common 0–40 scale. Shading spans each TDA report\'s first through last detection month. '
        'The latest report is purple, earlier reports are blue, and unassigned detections are grey. '
        '(c,f,i) Final STP footprints of each stand\'s latest report, with groups 1 and 2 in blue and purple; Stand 22301 contains two timing groups. '
        'Maps within each row share the same extent; basemap months are labelled. Shaded time ranges describe detection spread rather than verified harvest duration.')
    (OUT / 'CAPTION_EN.md').write_text(caption + '\n', encoding='utf-8')


def draw_history(ax, case, panel):
    handles = ORIGINAL_HISTORY(ax, case, panel)
    ax.set_ylim(4.35, -.4)
    label = next(t for t in ax.texts if t.get_text() == f'({panel})')
    label.set_position((.985, .025))
    label.set(ha='right', va='bottom')
    return handles


def draw_decorations(fig, palette, parents, rounds, history_handles):
    width, height = figure.WIDTH_MM, figure.HEIGHT_MM
    cax = fig.add_axes([9 / width, 12 / height, 40 / width, 2.4 / height])
    cmap = ListedColormap([palette['month_hex'][m] for m in sorted(palette['month_hex'])])
    bar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(range(109), 108), cmap=cmap),
                      cax=cax, orientation='horizontal')
    bar.set_ticks([0, 36, 72, 108], labels=['2017', '2020', '2023', '2026'])
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    bar.outline.set_visible(False)
    cax.text(.5, 1.65, 'Latest pixel-detection month', transform=cax.transAxes,
             ha='center', fontsize=6.1)
    common = dict(frameon=False, fontsize=5.7, handlelength=1.5, handletextpad=.5,
                  labelspacing=.55, borderaxespad=0)
    fig.legend(handles=[Patch(facecolor=figure.NO_DETECTION, label='No pixel detection')],
               loc='upper center', bbox_to_anchor=(29 / width, 6 / height), **common)
    history = [
        (Patch(facecolor=spans.SPAN_COLORS[2]), Patch(facecolor=spans.BAR_COLORS[2])),
        (Patch(facecolor=spans.SPAN_COLORS[1]), Patch(facecolor=spans.BAR_COLORS[1])),
        Patch(facecolor=spans.UNASSIGNED)]
    fig.legend(handles=history, labels=['Latest TDA report', 'Earlier TDA reports', 'Unassigned detection'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.15)},
               loc='upper center', bbox_to_anchor=(97 / width, 17 / height), **common)
    groups = int(rounds.round_index.max()) + 1
    right = [Patch(facecolor=figure.STP_COLORS[i], label=f'STP {i+1}') for i in range(groups)]
    right.append(Patch(facecolor=figure.OUTSIDE_LATEST, label='Outside latest report'))
    fig.legend(handles=right, loc='upper center', bbox_to_anchor=(154 / width, 17 / height), **common)


if __name__ == '__main__':
    main()
