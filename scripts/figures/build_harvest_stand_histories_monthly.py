"""Add monthly accepted-pixel counts to the compact Figure 6 histories."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib.dates as mdates
from matplotlib.patches import Patch
import pandas as pd

import build_harvest_stand_histories as figure

PREVIOUS = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v6_compact'
IMAGERY_SOURCE = PREVIOUS / 'imagery/input_imagery.json'
OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v7_monthly'
ORIGINAL_MAP = figure.draw_map
COUNT_MAX = 40
COUNT_HEIGHT = .56
MONTHLY = None
CASES = {716: [719,717,718,716,715], 728: [730,728,727,724,25621],
         22301: [22885,22883,22605,22302,22301]}


def main():
    global MONTHLY
    (OUT/'imagery').mkdir(parents=True, exist_ok=True)
    shutil.copy2(IMAGERY_SOURCE, OUT/'imagery/input_imagery.json')
    labels = pd.read_csv(PREVIOUS/'source_stands.csv')
    reports = pd.read_csv(PREVIOUS/'source_tda_reports.csv')
    activity_path = figure.PRODUCT/'octa/monthly_activity.parquet'
    monthly = pd.read_parquet(activity_path)
    monthly = monthly.loc[monthly.sample_id.isin(labels.sample_id)].copy()
    owners = pd.read_parquet(figure.PRODUCT/'cstr/node_ownership.parquet')
    owners = owners.loc[owners.sample_id.isin(labels.sample_id)].copy()
    assert owners.prediction_id.isin(reports.prediction_id).all()
    assert not owners.duplicated(['sample_id','unit_id','response_month']).any()
    key = ['sample_id','response_month']
    retained = owners.groupby(key).unit_id.nunique().rename('retained_pixels')
    latest = owners.loc[owners.prediction_id.isin(labels.latest_prediction_id)].groupby(key).unit_id.nunique().rename('latest_pixels')
    monthly = monthly.merge(retained, left_on=['sample_id','month'], right_index=True, how='left', validate='one_to_one')
    monthly = monthly.merge(latest, left_on=['sample_id','month'], right_index=True, how='left', validate='one_to_one')
    monthly[['retained_pixels','latest_pixels']] = monthly[['retained_pixels','latest_pixels']].fillna(0).astype(int)
    monthly['unassigned_pixels'] = monthly.active_pixels - monthly.retained_pixels
    monthly['earlier_pixels'] = monthly.retained_pixels - monthly.latest_pixels
    assert monthly[['unassigned_pixels','earlier_pixels']].ge(0).all().all()
    assert monthly.active_pixels.max() <= COUNT_MAX
    assert len(monthly) == len(labels)*108
    monthly = monthly.merge(labels[['case','letter','stand_id','sample_id']], on='sample_id', validate='many_to_one')
    monthly.to_csv(OUT/'source_monthly_detections.csv', index=False)
    MONTHLY = monthly
    figure.OUT = OUT
    figure.CASES = CASES
    figure.draw_history = draw_monthly_history
    figure.draw_map = draw_map
    figure.main()
    path = OUT/'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(monthly_count_definition='distinct accepted pixels per stand and calendar month, including detections outside retained TDA reports',
        monthly_count_scale=[0,COUNT_MAX], monthly_count_peak=int(monthly.active_pixels.max()),
        accepted_pixel_months=int(monthly.active_pixels.sum()), retained_pixel_months=int(monthly.retained_pixels.sum()),
        unassigned_pixel_months=int(monthly.unassigned_pixels.sum()),
        monthly_activity_sha256=hashlib.sha256(activity_path.read_bytes()).hexdigest(),
        monthly_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        right_map_stand_letters=False, peak_labels='earliest maximum month per stand; not event-level unique-pixel support')
    manifest['image_records'] = {k:v for k,v in manifest['image_records'].items() if int(k) in figure.CASES}
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def draw_map(ax, case, palette, panel, stp):
    ORIGINAL_MAP(ax, case, palette, panel, stp, show_labels=not stp)


def draw_monthly_history(ax, case, panel):
    ax.set_xlim(pd.Timestamp('2017-01-01'), pd.Timestamp('2026-01-01'))
    ax.set_ylim(4.65,-.85)
    ax.set_yticks(range(5), [f'{chr(65+i)}  {s}' for i,s in enumerate(case['ids'])])
    ax.set_xticks(pd.to_datetime([f'{y}-01-01' for y in [2017,2019,2021,2023,2025]]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=5.9, length=2, pad=2)
    ax.tick_params(axis='y', labelsize=6.3, length=0, pad=4)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#7d8983')
    ax.grid(axis='x', color='#e0e6e2', linewidth=.5, zorder=0)
    latest_ids = set(case['latest'].prediction_id)
    for index, stand in enumerate(case['ids']):
        m = MONTHLY.loc[MONTHLY.stand_id.eq(stand)].sort_values('month')
        sid = m.sample_id.iloc[0]
        dates = pd.to_datetime(m.month) + pd.Timedelta(days=14)
        baseline = index+.16
        scale = COUNT_HEIGHT/COUNT_MAX
        ax.axhspan(index-.45,index+.44, facecolor='#f2f5f3', edgecolor='none', zorder=-1)
        ax.axhline(baseline, color='#dce3df', lw=.35, zorder=0)
        bottom = pd.Series(baseline, index=m.index)
        for column, color in [('earlier_pixels','#76BCA7'),('latest_pixels',figure.TDA_COLOR),('unassigned_pixels','#939b97')]:
            heights = m[column]*scale
            ax.bar(dates, -heights, width=26, bottom=bottom, color=color, edgecolor='none', zorder=2)
            bottom = bottom-heights
        peak = m.loc[m.active_pixels.idxmax()]
        if peak.active_pixels:
            x = pd.Timestamp(peak.month)+pd.Timedelta(days=14)
            y = baseline-peak.active_pixels*scale-.025
            # A near-full-height peak has no room above it within its own lane.
            high_peak = peak.active_pixels >= 30
            if high_peak:
                x += pd.Timedelta(days=36)
            ax.text(x, y, str(int(peak.active_pixels)), ha='left' if high_peak else 'center',
                    va='top' if high_peak else 'bottom', fontsize=5.5, color='#263c31', zorder=5)
        for report in case['parents'].loc[case['parents'].sample_id.eq(sid)].itertuples():
            first = pd.Timestamp(report.first_response_month)
            end = pd.Timestamp(report.last_response_month)+pd.offsets.MonthBegin(1)
            last = report.prediction_id in latest_ids
            alpha = 1 if last else .6
            y = index+.34
            ax.plot([first,end],[y,y], color=figure.TDA_COLOR, lw=1, alpha=alpha, solid_capstyle='butt', zorder=3)
            ax.vlines([first,end], y-.055,y+.055, color=figure.TDA_COLOR, lw=.6, alpha=alpha, zorder=3)
            ax.scatter([pd.Timestamp(report.representative_month)],[y], marker='D', s=10,
                       facecolor=figure.TDA_COLOR, edgecolor='#142c23' if last else 'white',
                       linewidth=.4, alpha=alpha, zorder=4)
    ax.text(.01,.98,f'({panel})',transform=ax.transAxes,fontsize=8,weight='bold',va='top')
    return [Patch(facecolor=figure.TDA_COLOR, label='Monthly detections (0–40 pixels)')]


if __name__ == '__main__':
    main()
