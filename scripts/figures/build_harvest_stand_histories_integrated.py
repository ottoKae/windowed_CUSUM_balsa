"""Integrate stand labels, alternating rows and aligned legend strips in Figure 6."""
import json
from pathlib import Path
import shutil

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

import build_harvest_stand_histories_2025 as previous

figure, spans = previous.figure, previous.spans
PREVIOUS = previous.OUT
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v14_integrated'
FRAME_LEFT_MM = 45.3
FRAME_RIGHT_MM = 134


def main():
    (OUT / 'imagery').mkdir(parents=True, exist_ok=True)
    shutil.copy2(PREVIOUS / 'imagery/selected_input_imagery.json',
                 OUT / 'imagery/selected_input_imagery.json')
    previous.OUT = OUT
    previous.draw_history = draw_history
    previous.draw_decorations = draw_decorations
    figure.SHOW_BASEMAP_MONTH = False
    figure.SCALE_BAR_Y_FRACTION = .035
    previous.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(
        history_panel_border='0.55 pt frame includes A-E labels and stand IDs',
        history_panel_bounds_mm=[FRAME_LEFT_MM, FRAME_RIGHT_MM],
        history_plot_bounds_mm=[58, 134],
        alternate_row_background={'rows': ['A', 'C', 'E'], 'fill': '#F6F6F6',
                                  'extent': 'entire panel, including stand labels'},
        legends='three 4-mm strips, top and bottom aligned; categorical widths carry no data',
        map_basemap_month_labels=False, basemap_month_location='caption',
        scale_bar_y_fraction=.035,
        plotting_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    caption_path = OUT / 'CAPTION_EN.md'
    caption = caption_path.read_text(encoding='utf-8').replace('Shading spans', 'Coloured shading spans')
    caption_path.write_text(caption, encoding='utf-8')


def draw_history(ax, case, panel):
    handles = previous.wide.draw_history(ax, case, panel)
    left = (FRAME_LEFT_MM - figure.HISTORY_LEFT_MM) / figure.HISTORY_WIDTH_MM
    for low, high in [(-.4, .5), (1.5, 2.5), (3.5, 4.35)]:
        ax.add_patch(Rectangle((left, low), 1-left, high-low,
                              transform=ax.get_yaxis_transform(), clip_on=False,
                              facecolor='#F6F6F6', edgecolor='none', zorder=-2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((left, 0), 1-left, 1, transform=ax.transAxes,
                          facecolor='none', edgecolor='#515b62', linewidth=.55,
                          clip_on=False, zorder=5))
    return handles


def draw_decorations(fig, palette, parents, rounds, history_handles):
    width, height = figure.WIDTH_MM, figure.HEIGHT_MM
    top = 1 - 138 / height
    bar_height = 4 / height
    cax = fig.add_axes([4 / width, top - bar_height, 40 / width, bar_height])
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

    hwidth = FRAME_RIGHT_MM - FRAME_LEFT_MM
    hax = fig.add_axes([FRAME_LEFT_MM / width, top-bar_height, hwidth / width, bar_height])
    hax.set(xlim=(0, hwidth), ylim=(0, 4))
    hax.set_axis_off()
    gap = .6
    segment_width = (hwidth - 2*gap) / 3
    for index, (label, light, dark) in enumerate([
        ('Latest TDA', spans.SPAN_COLORS[2], spans.BAR_COLORS[2]),
        ('Earlier TDA', spans.SPAN_COLORS[1], spans.BAR_COLORS[1]),
        ('Unassigned', '#F0F2F4', spans.UNASSIGNED),
    ]):
        x = index*(segment_width+gap)
        hax.add_patch(Rectangle((x, 0), segment_width, 4, facecolor=light, edgecolor='none'))
        for j, bar_fraction in enumerate([.45, 1, .65]):
            hax.add_patch(Rectangle((x+2+j*.85, .65), .65, 2.7*bar_fraction,
                                   facecolor=dark, edgecolor='none'))
        hax.text(x+segment_width/2+1.5, 2, label, ha='center', va='center',
                 fontsize=6.6, color='#202523')

    rax = fig.add_axes([136 / width, top-bar_height, 40 / width, bar_height])
    rax.set(xlim=(0, 40), ylim=(0, 4))
    rax.set_axis_off()
    x = 0
    for segment_width, label, light, dark in [
        (11, 'STP 1', '#EAF4F8', figure.STP_COLORS[0]),
        (11, 'STP 2', '#EFE8F5', figure.STP_COLORS[1]),
        (16.8, 'Not in TDA', '#F2F3F2', figure.OUTSIDE_LATEST),
    ]:
        rax.add_patch(Rectangle((x, 0), segment_width, 4, facecolor=light, edgecolor='none'))
        rax.add_patch(Rectangle((x+.7, .65), 1.7, 2.7, facecolor=dark, edgecolor='none'))
        rax.text(x+segment_width/2+1, 2, label, ha='center', va='center',
                 fontsize=6.6, color='#202523')
        x += segment_width+gap

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(renderer) for ax in (cax, hax, rax)]
    assert max(b.y1 for b in boxes)-min(b.y1 for b in boxes) < .1
    assert max(b.y0 for b in boxes)-min(b.y0 for b in boxes) < .1
    for ax in (cax, hax, rax):
        box = ax.get_window_extent(renderer)
        for text in ax.texts:
            tb = text.get_window_extent(renderer)
            assert box.contains(tb.x0, tb.y0) and box.contains(tb.x1, tb.y1), text.get_text()
    (OUT / 'layout_measurements.json').write_text(json.dumps({
        'legend_top_pixels': [b.y1 for b in boxes],
        'legend_bottom_pixels': [b.y0 for b in boxes],
        'legend_widths_mm': [40, hwidth, 40], 'legend_height_mm': 4,
        'history_full_panel_bounds_mm': [FRAME_LEFT_MM, FRAME_RIGHT_MM],
        'history_data_bounds_mm': [58, 134], 'minimum_font_pt': 6.6,
        'native_figure_width_mm': 180, 'current_manuscript_text_width_mm': 164.6,
        'minimum_printed_font_pt': 6.6*164.6/180}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
