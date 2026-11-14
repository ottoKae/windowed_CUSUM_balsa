"""Show Figure 6 histories as monthly bars within shaded TDA detection ranges."""
from pathlib import Path
import hashlib
import json

import matplotlib.dates as mdates
from matplotlib.patches import Patch
import pandas as pd

import build_harvest_stand_histories as figure
import build_harvest_stand_histories_monthly as monthly

OUT = figure.ROOT / 'runs/figures/harvest_stand_histories_20261112_v8_spans'
# Reuse the three temporal-event colours in the user's Figure 2(c).
BAR_COLORS = ['#3594BC', '#245A90', '#69429A']
SPAN_COLORS = ['#EAF4F8', '#E8EEF5', '#EAE0F3']
UNASSIGNED = '#626D7A'
COUNT_MAX = 40
LANE_HEIGHT = .75
LATEST_PURPLE = False
SHOW_COUNT_AXIS = True


def main():
    monthly.OUT = OUT
    monthly.draw_monthly_history = draw_span_bars
    figure.HISTORY_DATE_MARKERS = False
    monthly.main()
    path = OUT/'figure_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(spans_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        peak_labels=None, displayed_representative_month_markers=False,
        span_encoding='continuous half-open [first month start, month after last month start), confined to the stand lane',
        history_colors='chronological TDA report order within each stand; Figure 2(c) palette',
        count_axes='0 and 40 ticks at the right of every stand lane; identical linear scale')
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def draw_span_bars(ax, case, panel):
    start = pd.Timestamp('2017-01-01')
    end = pd.Timestamp('2026-01-01')
    ax.set_xlim(start,end)
    ax.set_ylim(4.65,-.85)
    ax.set_yticks(range(5), [f'{chr(65+i)}  {s}' for i,s in enumerate(case['ids'])])
    ax.set_xticks(pd.to_datetime([f'{y}-01-01' for y in [2017,2019,2021,2023,2025]]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=5.9, length=2, pad=2)
    ax.tick_params(axis='y', labelsize=6.3, length=0, pad=4)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#7d8983')
    ax.grid(axis='x', color='#e3e8eb', linewidth=.4, zorder=0)
    for index, stand in enumerate(case['ids']):
        m = monthly.MONTHLY.loc[monthly.MONTHLY.stand_id.eq(stand)].sort_values('month')
        dates = pd.to_datetime(m.month)
        reports = case['parents'].loc[case['parents'].sample_id.eq(m.sample_id.iloc[0])].sort_values('representative_month')
        assert len(reports) <= len(BAR_COLORS)
        baseline = index+.35
        top = baseline-LANE_HEIGHT
        colors = pd.Series(UNASSIGNED,index=m.index)
        for order, report in enumerate(reports.itertuples()):
            color_index = len(BAR_COLORS) - len(reports) + order if LATEST_PURPLE else order
            first = pd.Period(report.first_response_month,'M').start_time
            last_exclusive = (pd.Period(report.last_response_month,'M')+1).start_time
            within = (dates >= first) & (dates < last_exclusive)
            assert m.loc[within,'unassigned_pixels'].eq(0).all()
            colors.loc[within] = BAR_COLORS[color_index]
            ax.fill_betweenx([top,baseline],first,last_exclusive,color=SPAN_COLORS[color_index],lw=0,zorder=.5)
        ax.axhline(baseline,color='#a9b6bf',lw=.4,zorder=1)
        ax.bar(dates+pd.Timedelta(days=14), -m.active_pixels*LANE_HEIGHT/COUNT_MAX,
               width=27,bottom=baseline,color=colors.to_list(),edgecolor='none',zorder=2)
        if SHOW_COUNT_AXIS:
            ax.plot([end,end],[top,baseline],color='#a9b6bf',lw=.4,clip_on=False,zorder=1)
            for y, label in [(baseline,'0'),(top,'40')]:
                ax.plot([end,end+pd.Timedelta(days=12)],[y,y],color='#a9b6bf',lw=.4,clip_on=False)
                ax.text(end+pd.Timedelta(days=22),y,label,fontsize=5.1,color='#64747f',
                        va='center',ha='left',clip_on=False)
    ax.text(.01,.98,f'({panel})',transform=ax.transAxes,fontsize=8,weight='bold',va='top')
    return [Patch(facecolor=SPAN_COLORS[2],label='TDA detection range'),
            Patch(facecolor=UNASSIGNED,label='Unassigned detection')]


if __name__ == '__main__':
    main()
