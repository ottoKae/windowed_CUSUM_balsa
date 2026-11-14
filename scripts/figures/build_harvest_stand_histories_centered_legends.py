"""Restore TDA range/bar keys and centre single-row legends on the date bar."""
import json
from pathlib import Path

from matplotlib.patches import Rectangle

import build_harvest_stand_histories_final_legends as approved

integrated = approved.integrated
figure = integrated.figure
ORIGINAL_DECORATIONS = integrated.draw_decorations
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v16_centered'


def main():
    integrated.OUT = OUT
    integrated.draw_decorations = draw_decorations
    integrated.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(
        legends='single-row keys centred vertically on the date colourbar',
        history_legend_style='light range backgrounds with dark miniature bars',
        history_legend_rows=[['Latest TDA', 'Earlier TDA', 'Unassigned']],
        stp_legend_style='square colour keys, no background strip',
        stp_legend_rows=[['STP 1', 'STP 2', 'Not in TDA']],
        history_legend_left_mm=58, history_legend_width_mm=76,
        legend_centre_from_canvas_top_mm=140, colourbar_title_weight='bold',
        plotting_script_sha256=figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUT / 'CAPTION_EN.md').write_text(approved.CAPTION, encoding='utf-8')


def draw_decorations(fig, palette, parents, rounds, history_handles):
    ORIGINAL_DECORATIONS(fig, palette, parents, rounds, history_handles)
    cax, hax, rax = fig.axes[-3:]
    cax.texts[0].set_fontweight('bold')
    # Keep the approved range/bar symbols; align their strip with the time axis.
    position = hax.get_position()
    hax.set_position([58/figure.WIDTH_MM, position.y0, 76/figure.WIDTH_MM, position.height])
    rax.clear()
    rax.set(xlim=(0, 40), ylim=(0, 4))
    rax.set_axis_off()
    square = 2.6
    for x, label, colour in [
        (0, 'STP 1', figure.STP_COLORS[0]),
        (12, 'STP 2', figure.STP_COLORS[1]),
        (24, 'Not in TDA', figure.OUTSIDE_LATEST),
    ]:
        rax.add_patch(Rectangle((x, 2-square/2), square, square,
                                facecolor=colour, edgecolor='none'))
        rax.text(x+square+1.1, 2, label, ha='left', va='center',
                 fontsize=6.6, color='#202523')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(renderer) for ax in (cax, hax, rax)]
    centres = [(b.y0+b.y1)/2 for b in boxes]
    assert max(centres)-min(centres) < .1
    for ax, box in zip((cax, hax, rax), boxes):
        for text in ax.texts:
            tb = text.get_window_extent(renderer)
            assert box.contains(tb.x0, tb.y0) and box.contains(tb.x1, tb.y1), text.get_text()
    (OUT / 'layout_measurements.json').write_text(json.dumps({
        'legend_centres_pixels': centres, 'legend_centre_from_top_mm': 140,
        'legend_left_mm': [4, 58, 136], 'legend_width_mm': [40, 76, 40],
        'history_legend_range_and_bar_symbols': True,
        'stp_legend_background': False, 'stp_key_mm': [square, square],
        'all_legends_single_row': True, 'colourbar_title_weight': 'bold',
        'minimum_font_pt': 6.6, 'minimum_printed_font_pt': 6.6*164.6/180,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
