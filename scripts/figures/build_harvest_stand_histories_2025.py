"""Use 2025 context basemaps and a compact, aligned footer for Figure 6."""
import json
from datetime import datetime
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

import build_harvest_stand_histories_wide as wide

spans, figure, callouts = wide.spans, wide.figure, wide.callouts
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v13_2025'
HEIGHT_MM = 147


def main():
    spans.OUT = OUT
    spans.LATEST_PURPLE = True
    spans.SHOW_COUNT_AXIS = False
    spans.draw_span_bars = draw_history
    spans.monthly.draw_map = callouts.draw_map
    spans.monthly.IMAGERY_SOURCE = OUT / 'imagery/selected_input_imagery.json'
    callouts.RIGHT_CALLOUTS = True
    figure.HEIGHT_MM = HEIGHT_MM
    figure.TOP_MM, figure.ROW_STEP_MM = 3, 45
    figure.LEFT_MAP_MM, figure.RIGHT_MAP_MM = 4, 136
    figure.HISTORY_LEFT_MM, figure.HISTORY_WIDTH_MM = 58, 76
    figure.STP_COLORS = ['#3594BC', '#69429A']
    figure.NO_DETECTION = 'none'
    figure.draw_decorations = draw_decorations
    spans.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(
        history_colors='latest TDA purple, previous dark blue, earliest light blue',
        count_axes='hidden; common linear 0-40 scale stated in caption',
        history_axis_alignment='bottom x axis coincides with last stand zero baseline',
        history_panel_labels='lower right', history_plot_width_mm=76,
        history_year_ticks=list(range(2017, 2027)), annual_grid=True,
        map_left_positions_mm=[4, 136], right_map_stand_letters=True,
        callout_positions=callouts.LABEL_POSITIONS, column_titles=False,
        legends='one shared footer with top-aligned column blocks; title inside date colourbar',
        no_pixel_detection='transparent, including cell edges; no legend swatch',
        outside_latest_tda='grey cells; legend label Not in TDA',
        history_panel_border='all four spines, 0.55 pt charcoal',
        minimum_font_pt=6.6, date_colourbar_domain=['2017-01', '2026-01'],
        date_colourbar_height_mm=4, stp_palette=figure.STP_COLORS,
        basemap_role='2025 monthly PlanetScope spatial context, not contemporaneous event evidence',
        basemap_months={k: v['month'] for k, v in manifest['image_records'].items()},
        plotting_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    months = sorted(set(manifest['basemap_months'].values()))
    background = (f'PlanetScope basemaps from {datetime.strptime(months[0], "%Y-%m"):%B %Y}' if len(months) == 1
                  else '2025 PlanetScope basemaps (months labelled)')
    caption = (
        'Pixel detections and reconstructed stand harvest histories in three balsa neighbourhoods. '
        f'(a,d,g) Latest accepted pixel-detection months over {background}; '
        'pixels without accepted detections are transparent. Matching letter labels in both map '
        'columns identify the stands in (b,e,h). (b,e,h) Monthly counts of distinct detected pixels '
        'on a common linear scale of 0–40 pixels per row. Shading spans each TDA report\'s first '
        'through last detection month. The latest report is purple, earlier reports are blue, '
        'and unassigned detections are grey. (c,f,i) Final STP footprints within each stand\'s latest '
        'TDA report, with groups 1 and 2 in blue and purple; Stand 22301 contains two timing groups. '
        'Grey cells lie outside the latest TDA report. Maps within each row share the same extent. '
        'Basemaps provide spatial context; shaded time ranges describe detection spread rather '
        'than verified harvest duration.\n')
    (OUT / 'CAPTION_EN.md').write_text(caption, encoding='utf-8')
    print(OUT)


def draw_history(ax, case, panel):
    handles = wide.draw_history(ax, case, panel)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#515b62')
        spine.set_linewidth(.55)
    return handles


def draw_decorations(fig, palette, parents, rounds, history_handles):
    width, height = figure.WIDTH_MM, figure.HEIGHT_MM
    footer_top_mm = 138
    bar_height_mm = 4
    top = 1 - footer_top_mm / height
    cax = fig.add_axes([4 / width, top - bar_height_mm / height, 40 / width,
                       bar_height_mm / height])
    cmap = ListedColormap([palette['month_hex'][m] for m in sorted(palette['month_hex'])])
    bar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(range(109), 108), cmap=cmap),
                      cax=cax, orientation='horizontal')
    bar.set_ticks([0, 36, 72, 108], labels=['2017', '2020', '2023', '2026'])
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=6.6, length=2, pad=1)
    bar.outline.set_visible(False)
    heading = cax.text(.5, .5, 'Latest pixel-detection month', transform=cax.transAxes,
                       ha='center', va='center', fontsize=6.6, color='#202523',
                       path_effects=[pe.withStroke(linewidth=.7, foreground='white', alpha=.75)])
    common = dict(frameon=False, fontsize=6.6, handlelength=1.1, handletextpad=.3,
                  columnspacing=.35, borderaxespad=0, borderpad=0, loc='upper center')
    history = [
        (Patch(facecolor=spans.SPAN_COLORS[2]), Patch(facecolor=spans.BAR_COLORS[2])),
        (Patch(facecolor=spans.SPAN_COLORS[1]), Patch(facecolor=spans.BAR_COLORS[1])),
        Patch(facecolor=spans.UNASSIGNED)]
    fig.legend(handles=history, labels=['Latest TDA', 'Earlier TDA', 'Unassigned'], ncol=3,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.1)},
               bbox_to_anchor=(96 / width, top), **common)
    groups = int(rounds.round_index.max()) + 1
    right = [Patch(facecolor=figure.STP_COLORS[i], label=f'STP {i+1}') for i in range(groups)]
    right.append(Patch(facecolor=figure.OUTSIDE_LATEST, label='Not in TDA'))
    fig.legend(handles=right, ncol=3, bbox_to_anchor=(156 / width, top), **common)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in fig.legends]
    tops = [cax.get_window_extent(renderer).y1] + [b.y1 for b in boxes]
    widths = [b.width / fig.dpi * 25.4 for b in boxes]
    assert max(tops) - min(tops) < .1, tops
    assert all(w <= limit for w, limit in zip(widths, [76, 40])), widths
    title = heading.get_window_extent(renderer)
    assert cax.get_window_extent(renderer).contains(title.x0, title.y0)
    assert cax.get_window_extent(renderer).contains(title.x1, title.y1)
    (OUT / 'layout_measurements.json').write_text(json.dumps({
        'legend_block_top_pixels': tops, 'categorical_legend_widths_mm': widths,
        'legend_column_widths_mm': [40, 76, 40], 'colourbar_height_mm': bar_height_mm,
        'footer_top_from_canvas_top_mm': footer_top_mm, 'history_width_mm': 76,
        'native_figure_width_mm': 180, 'current_manuscript_text_width_mm': 164.6,
        'minimum_printed_font_pt': 6.6 * 164.6 / 180}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
