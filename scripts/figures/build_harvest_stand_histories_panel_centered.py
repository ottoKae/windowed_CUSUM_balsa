"""Centre the TDA legend on the complete history panel, including stand IDs."""
import json
from pathlib import Path

import build_harvest_stand_histories_centered_legends as previous

OUT = previous.figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v17_panel_centered'
ORIGINAL_DECORATIONS = previous.draw_decorations


def main():
    previous.OUT = OUT
    previous.draw_decorations = draw_decorations
    previous.main()
    path = OUT / 'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(history_legend_left_mm=51.65,
                    history_legend_centre_mm=89.65,
                    history_legend_alignment='centred on full 45.3–134 mm history panel',
                    plotting_script_sha256=previous.figure.base.sha256(Path(__file__)))
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def draw_decorations(fig, palette, parents, rounds, history_handles):
    ORIGINAL_DECORATIONS(fig, palette, parents, rounds, history_handles)
    hax = fig.axes[-2]
    position = hax.get_position()
    centre_mm = (previous.integrated.FRAME_LEFT_MM + previous.integrated.FRAME_RIGHT_MM)/2
    width_mm = 76
    left_mm = centre_mm-width_mm/2
    hax.set_position([left_mm/previous.figure.WIDTH_MM, position.y0,
                      width_mm/previous.figure.WIDTH_MM, position.height])
    assert abs((left_mm+width_mm/2)-centre_mm) < 1e-9
    path = OUT / 'layout_measurements.json'
    layout = json.loads(path.read_text(encoding='utf-8'))
    layout['legend_left_mm'][1] = left_mm
    layout.update(history_legend_centre_mm=centre_mm,
                  history_panel_centre_mm=centre_mm,
                  history_legend_side_margins_mm=[6.35, 6.35])
    path.write_text(json.dumps(layout, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
