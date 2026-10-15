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
OUT = ROOT / 'runs/figures/concept_label_preview_20260911_v2_review'


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
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        labels = {label.get_text(): label for axis in figure.axes for label in axis.texts
                  if label.get_text() in {'(a)', '(b)', '(d)', '(f)'}}
        target_left = labels['(b)'].get_window_extent(renderer).x0
        assert abs(labels['(d)'].get_window_extent(renderer).x0-target_left) < 1e-6
        positions = {'(a)':(0.018, 0.975, 'top'), '(f)':(0.018, 0.025, 'bottom')}
        moved = {}
        for axis in figure.axes:
            for label in axis.texts:
                if label.get_text() in positions:
                    x, y, vertical = positions[label.get_text()]
                    x = axis.transAxes.inverted().transform((target_left, axis.bbox.y0))[0]
                    y -= (3.0 / 72.0 * figure.dpi) / axis.bbox.height
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
                    assert abs(box.x0-target_left) < 1e-6
                    legend = axis.get_legend()
                    assert legend is None or not box.overlaps(legend.get_window_extent(renderer))
        diagnostics['preview_label_positions_axes_fraction'] = moved
        diagnostics['labels_a_f_downward_shift_pt'] = 3.0
        diagnostics['labels_a_b_d_f_share_left_edge'] = True
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
    assert sha(asset) == before[str(asset)]
    # The author may keep editing TeX in the IDE while this preview runs.
    externally_changed_tex = [path for path,digest in before.items()
                              if Path(path).suffix == '.tex' and sha(Path(path)) != digest]
    (OUT/'label_preview_receipt.json').write_text(json.dumps({
        'change':'(a) and (f) left edges align with (b)/(d); both shifted down 3 pt relative to preview v1',
        'preview_script':str(Path(__file__).resolve()),
        'preview_script_sha256':sha(Path(__file__)),
        'source_workflow_sha256':sha(Path(source.__file__)),
        'all_frozen_input_hashes_match':True,
        'manuscript_asset_unchanged':True,
        'tex_files_written_by_preview':[],
        'tex_files_changed_externally_during_render':externally_changed_tex,
        'original_asset_sha256':before[str(asset)],
        'labels_a_b_d_f_left_edges_aligned_and_clear_of_legends':True,
    },indent=2),encoding='utf-8')
    # A smaller preview is rendered by the same Python backend for inspection.
    from PIL import Image
    with Image.open(OUT/'windowed_cusum_concept_n1_current.png') as preview:
        preview.thumbnail((1600,1600))
        preview.save(OUT/'preview.png')


if __name__ == '__main__':
    main()
