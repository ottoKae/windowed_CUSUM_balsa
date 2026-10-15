"""Render mainland-wide ascending burst coverage in all four study-area panels.

Reuses the study-area renderer and cached basemap. All geometry is real;
national stand symbols encode area, while panels b--d retain true boundaries.
Source manuscript files and published figures are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_study_area_figure as source
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
GREEN, GREEN_EDGE, YELLOW = '#2F8F5B', '#246B45', '#F2C744'
BURST, BURST_FILL = '#557A99', '#9DBBD0'
FIGURE_WIDTH_MM = 180.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_bursts(geometry_path, lookup_path, ecuador):
    lookup = pd.read_parquet(lookup_path)
    metadata = lookup[['jpl_burst_id','orbit_pass','track_number']].drop_duplicates()
    if metadata.jpl_burst_id.duplicated().any():
        raise ValueError('Conflicting orbit metadata for a burst ID.')
    mainland = ecuador.geometry.union_all()
    geometry = gpd.read_parquet(geometry_path).to_crs(4326)
    geometry = geometry.loc[geometry.intersects(mainland)]
    selected = geometry.merge(metadata,on='jpl_burst_id',how='left',validate='one_to_one')
    if selected.orbit_pass.isna().any():
        raise ValueError('Missing orbit direction for a mainland-intersecting burst.')
    selected = selected.loc[selected.orbit_pass.eq('ASCENDING')].copy()
    if selected.empty or selected.is_empty.any() or not selected.is_valid.all():
        raise ValueError('Invalid ascending coverage geometry.')
    selected = selected.sort_values('jpl_burst_id').reset_index(drop=True)
    union = selected.geometry.union_all()
    mainland_area = gpd.GeoSeries([mainland],crs=4326).to_crs(6933).area.iloc[0]
    gap = mainland.difference(union)
    gap_area = gpd.GeoSeries([gap],crs=4326).to_crs(6933).area.iloc[0]
    if gap_area > 1:
        raise ValueError(f'Mainland coverage gap exceeds 1 square metre: {gap_area}')
    return selected, {
        'selection_rule':'ASCENDING in authoritative LUT; intersects supplied Ecuador mainland boundary',
        'selected_bursts':len(selected),
        'actual_tracks':selected.groupby('track_number').size().to_dict(),
        'mainland_area_km2':mainland_area/1e6,'uncovered_area_m2':gap_area,
        'mainland_covered_fraction':1-gap_area/mainland_area,
        'meaning':'Static ascending footprint coverage; not filtered by study dates or historical processing admission.',
    }


def draw_coverage(axis, selected, extent, *, fill_alpha, line_alpha, linewidth):
    view = box(extent[0],extent[2],extent[1],extent[3])
    local = selected.loc[selected.intersects(view)]
    if local.empty:
        return {'footprints':0,'crossing_boundaries':0}
    union = gpd.GeoSeries([local.geometry.union_all().intersection(view)],crs=4326)
    union.plot(ax=axis,color=BURST_FILL,alpha=fill_alpha,edgecolor='none',zorder=2.5)
    # Intersect actual boundary lines, not clipped polygons: a map-frame edge
    # must never be introduced as an apparent burst boundary.
    lines = local.boundary.intersection(view)
    lines = lines.loc[~lines.is_empty]
    if not lines.empty:
        lines.plot(ax=axis,color=BURST,alpha=line_alpha,linewidth=linewidth,zorder=4)
    axis.set_xlim(extent[:2]); axis.set_ylim(extent[2:])
    return {'footprints':len(local),'crossing_boundaries':len(lines),
            'fill_alpha':fill_alpha,'line_alpha':line_alpha,'linewidth_pt':linewidth,
            'burst_ids':local.jpl_burst_id.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'LR-CuSum_v100')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/figures/study_area_bursts_v5')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    protected = list((ROOT/'manuscript_JAG_v2').rglob('*.tex'))
    protected += list((ROOT/'manuscript_JAG_v2/figures').glob('*.pdf'))
    before = {p: digest(p) for p in protected}
    config_path = ROOT/'configs/study_area_figure_v1.json'
    config = source.load_config(config_path)
    config['figure']['width_mm'] = FIGURE_WIDTH_MM
    config['figure']['raster_dpi'] = args.dpi
    config['figure']['colors'].update(inventory=GREEN, reference=YELLOW)
    config['figure']['marker_area_pt2']['inventory_alpha'] = .60
    config['figure']['detail_polygon_style'].update(
        inventory_fill=GREEN, inventory_edge=GREEN_EDGE,
        inventory_alpha=.60, reference_alpha=.85)
    data_paths = {k: source.resolve_path(args.data_root, v) for k,v in config['data'].items()}
    inventory, points, inventory_audit = source.load_inventory(data_paths['inventory_shp'], config['expected'])
    _, reference_ids, reference_audit = source.load_reference_matches(
        data_paths['reference_events'], inventory, points, config['expected'])
    reference_ids = set(map(str, reference_ids))
    ecuador = gpd.read_file(data_paths['ecuador_boundary']).to_crs(4326)
    amazonia = gpd.read_file(data_paths['amazonia_boundary']).to_crs(4326)
    ocean, _ = source.natural_earth_layer('10m','physical','ocean',args.data_root)
    countries, _ = source.natural_earth_layer('110m','cultural','admin_0_countries',args.data_root)
    national_extent = source.national_extent(ecuador,config['national_panel']['margin_fraction'])
    requested = {'a': (national_extent,config['basemap']['national_zoom']),
                 'locator': (config['national_panel']['locator_extent'],config['basemap']['locator_zoom'])}
    requested.update({p:(config['detail_panels'][p]['extent'],config['basemap']['detail_zoom']) for p in 'bcd'})
    images = {}; basemap_hashes = {}
    for panel,(extent,zoom) in requested.items():
        images[panel], _, basemap_hashes[panel] = source.build_basemap_mosaic(
            extent, zoom, 'cache', source.resolve_path(args.data_root,config['outputs']['basemap_cache']),
            config['basemap'],args.data_root)
    figure, original_audit = source.build_figure(inventory,points,reference_ids,ecuador,
        amazonia,ocean,countries,images,config,'cache')
    national, b, c, d, legend_axis = figure.axes
    axes = {'a': national, 'b': b, 'c': c, 'd': d, 'legend': legend_axis}
    plt.rcParams.update({'font.family':'sans-serif',
                        'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                        'pdf.fonttype':42,'svg.fonttype':'none'})

    # Restore the South America locator at the original southeast position.
    # Area bubbles and burst coverage have separate legend boxes.
    for child in list(national.child_axes):
        child.remove()
    locator_bounds = config['national_panel']['locator_axes']
    key_bounds = [.578,.003,.185,.265]
    locator = national.inset_axes(locator_bounds,zorder=94)
    source.draw_locator_panel(locator,config['national_panel']['locator_extent'],images['locator'],
                              countries,config['figure']['colors'],config['national_panel']['locator_basemap_alpha'])
    settings = config['figure']['marker_area_pt2']
    low, high = points.area_ha.quantile([settings['lower_quantile'],settings['upper_quantile']])
    area_values = points.area_ha.quantile([.25,.5,.75]).to_numpy()
    geometry_path = ROOT.parents[1]/'data_pipeline/S1-GRiTS/src/s1grits/data/jpl_burst_geo.parquet'
    lookup_path = geometry_path.with_name('mgrs_burst_lookup_table.parquet')
    selected,selection_audit = select_bursts(geometry_path,lookup_path,ecuador)
    selected.to_parquet(out/'selected_burst_footprints.parquet',index=False)
    selected.drop(columns='geometry').to_csv(out/'selected_burst_ids.csv',index=False)
    layer_audit = {'a':draw_coverage(national,selected,national_extent,
                    fill_alpha=.09,line_alpha=.60,linewidth=.30)}
    for label in 'bcd':
        layer_audit[label] = draw_coverage(axes[label],selected,config['detail_panels'][label]['extent'],
                                         fill_alpha=.065,line_alpha=.48,linewidth=.45)
    visible_count = layer_audit['a']['footprints']
    marker_sizes = source.area_to_marker_size(points.area_ha,low,high,settings['minimum'],settings['maximum'])
    ref_mask = points.inventory_id.astype(str).isin(reference_ids)
    # Remove only the two national point collections from the original renderer;
    # retain geographic layers. Replace them with legible, single-pass symbols.
    from matplotlib.collections import PathCollection
    for collection in list(national.collections):
        if isinstance(collection,PathCollection) and collection.get_zorder() in (10,12):
            collection.remove()
    national.scatter(points.geometry.x,points.geometry.y,s=marker_sizes,
                     facecolors=GREEN,edgecolors=GREEN_EDGE,linewidths=.28,alpha=.85,zorder=12.5)
    ref_sizes = np.maximum(np.asarray(marker_sizes)[ref_mask]*settings['reference_scale'],settings['reference_minimum'])
    national.scatter(points.loc[ref_mask].geometry.x,points.loc[ref_mask].geometry.y,s=ref_sizes,
                     facecolors=YELLOW,edgecolors=GREEN_EDGE,linewidths=.30,alpha=.95,zorder=13)
    legend_axis.clear(); legend_axis.set_axis_off()
    legend_axis.legend(handles=[Patch(facecolor=GREEN,edgecolor=GREEN_EDGE,label='Official balsa inventory'),
        Patch(facecolor=YELLOW,edgecolor=GREEN_EDGE,label='Dated event subset')],
        loc='center left',bbox_to_anchor=(0,.5),ncol=2,frameon=False,borderpad=0,borderaxespad=0,
        handletextpad=.25,columnspacing=.55,handlelength=.72,handleheight=.8,fontsize=8)
    key = national.inset_axes(key_bounds,zorder=95)
    key.set_axis_off()
    sizes = source.area_to_marker_size(area_values,low,high,settings['minimum'],settings['maximum'])
    handles = [Line2D([],[],linestyle='none',marker='o',markersize=np.sqrt(size),
                      markerfacecolor='none',markeredgecolor='#555555',markeredgewidth=.6,
                      label=f'{value:.1f}') for value,size in zip(area_values,sizes)]
    area_legend = key.legend(handles=handles,title='Area (ha)',loc='center',ncol=1,
                          fontsize=6.5,title_fontsize=7.5,frameon=True,framealpha=.95,
                          facecolor='white',edgecolor='#777777',borderpad=.45,
                          handlelength=.9,handletextpad=.4,labelspacing=.5)
    area_legend.get_frame().set_linewidth(.4)
    burst_key = national.inset_axes([.540,.008,.212,.040],zorder=95)
    burst_key.set_axis_off()
    burst_legend = burst_key.legend(
        handles=[Patch(facecolor='none',edgecolor=BURST,linewidth=.75,label='Burst footprint')],
        loc='center',fontsize=6.5,frameon=True,framealpha=.95,
        facecolor='white',edgecolor='#777777',borderpad=.45,
        handlelength=.9,handletextpad=.4)
    burst_legend.get_frame().set_linewidth(.4)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    area_box = area_legend.get_window_extent(renderer)
    burst_box = burst_legend.get_window_extent(renderer)
    locator_box = locator.get_window_extent(renderer)
    # Shift both legend centres by the same rendered distance, aligning the
    # lower legend frame with the South America locator frame exactly.
    shift_px = locator_box.y0-burst_box.y0
    for legend,legend_axis in [(area_legend,key),(burst_legend,burst_key)]:
        relative_shift = shift_px/legend_axis.get_window_extent(renderer).height
        legend.set_bbox_to_anchor((.5,.5+relative_shift),transform=legend_axis.transAxes)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    area_box = area_legend.get_window_extent(renderer)
    burst_box = burst_legend.get_window_extent(renderer)
    locator_box = locator.get_window_extent(renderer)
    alignment_mm = (burst_box.y0-locator_box.y0)/figure.dpi*25.4
    if abs(alignment_mm) > .05:
        raise ValueError(f'Legend/locator bottom alignment failed: {alignment_mm:.4f} mm')
    layout_audit = {'legend_shift_up_mm':shift_px/figure.dpi*25.4,
                    'legend_locator_bottom_alignment_error_mm':alignment_mm}
    for label,display_box in [('locator',locator_box),('area_legend',area_box),('burst_legend',burst_box)]:
        axes_box = display_box.transformed(national.transAxes.inverted())
        bounds = [axes_box.x0,axes_box.y0,axes_box.width,axes_box.height]
        overlap = source.count_points_under_axes_bounds(points,reference_ids,national_extent,bounds)
        boundary = source.audit_boundary_under_axes_bounds(ecuador,national_extent,bounds,.003)
        layout_audit[label] = {'bounds':bounds,'stand_overlap':overlap,'boundary':boundary}
        if overlap['inventory_points_under_key'] or not boundary['boundary_clear']:
            raise ValueError(f'{label} overlaps stand markers or national boundary.')
        if axes_box.x0 < 0 or axes_box.y0 < 0 or axes_box.x1 > 1 or axes_box.y1 > 1:
            raise ValueError(f'{label} falls outside panel a.')
        if label!='locator':
            extent = source.axes_bounds_to_data_extent(national_extent,bounds)
            region = box(extent[0]-.02,extent[2]-.02,extent[1]+.02,extent[3]+.02)
            hits = int(selected.intersects(region).sum())
            layout_audit[label].update(burst_footprint_intersections=hits,burst_clearance_degrees=.02)
            if hits:
                raise ValueError(f'{label} overlaps burst coverage.')
    for left,right in [(area_box,burst_box),(area_box,locator_box),(burst_box,locator_box)]:
        if left.overlaps(right):
            raise ValueError('Legend/locator boxes overlap.')
    gap_mm = (area_box.y0-burst_box.y1)/figure.dpi*25.4
    layout_audit['legend_gap_mm'] = gap_mm
    if gap_mm < .6:
        raise ValueError(f'The separate legends need more space: {gap_mm:.3f} mm')
    # Enforce a 5 pt floor; retain readable source labels at final output size.
    from matplotlib.text import Text
    for text in figure.findobj(Text):
        if text.get_text() and text.get_fontsize() < 5:
            raise ValueError('Text below 5 pt glyph floor.')
    stem = out/'study_area_bursts'
    figure.savefig(stem.with_suffix('.pdf'),dpi=args.dpi,bbox_inches='tight',facecolor='white')
    figure.savefig(stem.with_suffix('.svg'),dpi=args.dpi,bbox_inches='tight',facecolor='white')
    figure.savefig(stem.with_suffix('.png'),dpi=args.dpi,bbox_inches='tight',facecolor='white')
    figure.savefig(stem.with_suffix('.tiff'),dpi=args.dpi,bbox_inches='tight',facecolor='white',
                   pil_kwargs={'compression':'tiff_lzw'})
    figure.savefig(out/'preview.png',dpi=300,bbox_inches='tight',facecolor='white')
    renderer = figure.canvas.get_renderer()
    for label,axis in axes.items():
        if label=='legend': continue
        bounds = axis.get_tightbbox(renderer).transformed(figure.dpi_scale_trans.inverted()).expanded(1.02,1.03)
        figure.savefig(out/f'panel_{label}.png',dpi=300,bbox_inches=bounds,facecolor='white')
    plt.close(figure)
    after = {p:digest(p) for p in protected}
    if before != after:
        raise RuntimeError('Protected manuscript files changed during rendering.')
    audit = {'selection':selection_audit,'visible_bursts':visible_count,'coverage_layers':layer_audit,'burst_detail_inset':False,
        'layout_audit':layout_audit,
        'inventory':inventory_audit,'references':reference_audit,
        'base_render_audit':original_audit,'protected_files_unchanged':True,
        'legend_label':'Burst footprint','basemap_hashes':basemap_hashes,
        'geometry_sha256':digest(geometry_path),'source_sha256':digest(Path(__file__)),
        'lookup_sha256':digest(lookup_path),'mainland_boundary_sha256':digest(data_paths['ecuador_boundary'])}
    (out/'QA.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    (out/'resolved_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'selected_bursts':len(selected),'visible_bursts':visible_count,
                      'mainland_covered_fraction':selection_audit['mainland_covered_fraction'],'protected_files_unchanged':True}))


if __name__=='__main__':
    main()
