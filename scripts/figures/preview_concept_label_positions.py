"""Preview Figure 3 panel-label positions using the exact frozen input manifest.

Only (a) and (f) positions change. The manuscript PDF and source workflow stay
untouched. All data, windows, null draws, colours and other labels are inherited.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.workflows import build_windowed_cusum_concept_n1_current as source

ARCHIVE = ROOT.parent / 'LR-CuSum_v100/outputs/jag_20260821/method_figure/windowed_cusum_concept_20260903_n1_current_v22_terminology_final'
OUT = ROOT / 'runs/figures/concept_label_preview_20260911'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ARCHIVE/'windowed_cusum_concept_n1_current_manifest.json').read_text())
    asset = ROOT/'manuscript_JAG_v2/figures/fig04_windowed_cusum_concept.pdf'
    tracked = [asset, *sorted((ROOT/'manuscript_JAG_v2').rglob('*.tex'))]
    before = {str(path):sha(path) for path in tracked}
    assert sha(asset) == sha(ARCHIVE/'windowed_cusum_concept_n1_current.pdf')
    for entry in manifest['inputs'].values():
        assert sha(Path(entry['path'])) == entry['sha256'], entry['path']

    original_render = source.render_figure
    def preview_render(*args, **kwargs):
        figure, diagnostics = original_render(*args, **kwargs)
        positions = {'(a)':(0.018, 0.975, 'top'), '(f)':(0.018, 0.025, 'bottom')}
        moved = {}
        for axis in figure.axes:
            for label in axis.texts:
                if label.get_text() in positions:
                    x, y, vertical = positions[label.get_text()]
                    label.set_position((x,y))
                    label.set_horizontalalignment('left')
                    label.set_verticalalignment(vertical)
                    label.set_zorder(20)
                    moved[label.get_text()] = [x,y,vertical]
        assert set(moved) == set(positions)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            for label in axis.texts:
                if label.get_text() in positions:
                    box = label.get_window_extent(renderer)
                    assert axis.bbox.contains(box.x0,box.y0) and axis.bbox.contains(box.x1,box.y1)
                    legend = axis.get_legend()
                    assert legend is None or not box.overlaps(legend.get_window_extent(renderer))
        diagnostics['preview_label_positions_axes_fraction'] = moved
        diagnostics['preview_only'] = True
        return figure, diagnostics

    source.render_figure = preview_render
    options = []
    for flag, name in [('--metadata','polygon_series_metadata.parquet'),
                       ('--series','polygon_series.npz'),
                       ('--reference-events','reference_events.parquet'),
                       ('--candidates','sliding_candidates.parquet'),
                       ('--n1-scores','N1_IID_P025.parquet'),
                       ('--protocol','windowed_cusum_temporal_null_audit_40tile_v1.json')]:
        options.extend([flag,manifest['inputs'][name]['path']])
    options += ['--sample-id',manifest['input_semantics']['sample_id'],
                '--output-dir',str(OUT),'--dpi','600']
    result = source.main(options)
    assert result == 0
    assert all(sha(Path(path)) == digest for path,digest in before.items())
    (OUT/'label_preview_receipt.json').write_text(json.dumps({
        'change':'(a) inside upper-left; (f) inside lower-left',
        'preview_script':str(Path(__file__).resolve()),
        'preview_script_sha256':sha(Path(__file__)),
        'source_workflow_sha256':sha(Path(source.__file__)),
        'all_frozen_input_hashes_match':True,
        'manuscript_asset_and_tex_unchanged':True,
        'original_asset_sha256':before[str(asset)],
        'labels_inside_axes_and_clear_of_legends':True,
    },indent=2),encoding='utf-8')
    # A smaller preview is rendered by the same Python backend for inspection.
    from PIL import Image
    with Image.open(OUT/'windowed_cusum_concept_n1_current.png') as preview:
        preview.thumbnail((1600,1600))
        preview.save(OUT/'preview.png')


if __name__ == '__main__':
    main()
