"""Give Figure 6 two-row square-key legends and a concise caption."""
import json
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

import build_harvest_stand_histories_integrated as integrated

figure, spans = integrated.figure, integrated.spans
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v15_legends'
CAPTION = (
    'Pixel detections and 2017–2025 stand harvest histories in three balsa neighbourhoods. '
    '(a,d,g) Latest accepted pixel-detection months; pixels without detections are transparent. '
    'Letters identify corresponding stands within each row. '
    '(b,e,h) Monthly counts of distinct detected pixels on a common 0–40 scale per stand. '
    'Coloured bands span each TDA report\'s first to last detection month; purple denotes '
    'the latest report, blue earlier reports, and grey bars unassigned detections. '
    '(c,f,i) STP groups within each stand\'s latest TDA report; grey cells lie outside that report. '
    'Maps use March 2025 PlanetScope basemaps and share the same extent within each row. '
    'Temporal ranges describe detection spread rather than verified harvest duration.\n')


def main():
    integrated.OUT = OUT
    integrated.draw_decorations = draw_decorations
    integrated.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(
        legends='two-row square keys without background panels; date colourbar unchanged',
        history_legend_rows=[['Latest TDA', 'Earlier TDA'], ['Unassigned']],
        stp_legend_rows=[['STP 1', 'STP 2'], ['Not in TDA']],
        history_legend_left_mm=58, legend_key_size_mm=2.6,
        plotting_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUT / 'CAPTION_EN.md').write_text(CAPTION, encoding='utf-8')


def draw_decorations(fig, palette, parents, rounds, history_handles):
    width, height = figure.WIDTH_MM, figure.HEIGHT_MM
    top = 1 - 138 / height
    cax = fig.add_axes([4 / width, top - 4 / height, 40 / width, 4 / height])
    cmap = ListedColormap([palette['month_hex'][m] for m in sorted(palette['month_hex'])])
    bar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(range(109), 108), cmap=cmap),
                      cax=cax, orientation='horizontal')
    bar.set_ticks([0, 36, 72, 108], labels=['2017', '2020', '2023', '2026'])
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=6.6, length=2, pad=1)
    bar.outline.set_visible(False)
    cax.text(.5, .5, 'Latest pixel-detection month', transform=cax.transAxes,
             ha='center', va='center', fontsize=6.6, color='#202523',
             path_effects=[pe.withStroke(linewidth=.7, foreground='white', alpha=.75)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(t.get_window_extent(renderer).y0 for t in cax.get_xticklabels()) / fig.bbox.height
    band_height_mm = (top-bottom)*height
    square = 2.6
    rows = [band_height_mm-square, 0]
    legend_axes = []
    for x_mm, w_mm, items in [
        (58, 76, [(0, 0, 'Latest TDA', spans.BAR_COLORS[2]),
                  (23, 0, 'Earlier TDA', spans.BAR_COLORS[1]),
                  (0, 1, 'Unassigned', spans.UNASSIGNED)]),
        (136, 40, [(0, 0, 'STP 1', figure.STP_COLORS[0]),
                   (14, 0, 'STP 2', figure.STP_COLORS[1]),
                   (0, 1, 'Not in TDA', figure.OUTSIDE_LATEST)]),
    ]:
        ax = fig.add_axes([x_mm/width, bottom, w_mm/width, top-bottom])
        ax.set(xlim=(0, w_mm), ylim=(0, band_height_mm))
        ax.set_axis_off()
        for x, row, label, colour in items:
            y = rows[row]
            ax.add_patch(Rectangle((x, y), square, square, facecolor=colour, edgecolor='none'))
            ax.text(x+square+1.1, y+square/2, label, ha='left', va='center',
                    fontsize=6.6, color='#202523')
        legend_axes.append(ax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in legend_axes:
        box = ax.get_window_extent(renderer)
        for text in ax.texts:
            tb = text.get_window_extent(renderer)
            assert box.contains(tb.x0, tb.y0) and box.contains(tb.x1, tb.y1), text.get_text()
    (OUT / 'layout_measurements.json').write_text(json.dumps({
        'legend_rows': [['Latest TDA', 'Earlier TDA'], ['Unassigned']],
        'stp_legend_rows': [['STP 1', 'STP 2'], ['Not in TDA']],
        'legend_backgrounds': False, 'legend_key_mm': [square, square],
        'categorical_legend_height_mm': band_height_mm,
        'colourbar_with_ticks_height_mm': band_height_mm,
        'legend_left_mm': [4, 58, 136], 'history_2017_x_mm': 58,
        'minimum_font_pt': 6.6, 'minimum_printed_font_pt': 6.6*164.6/180,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
