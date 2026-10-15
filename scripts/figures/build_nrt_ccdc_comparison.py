#!/usr/bin/env python3
"""New figure only: full-horizon accuracy, paired effects, alert time and recurrence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'Windowed CUSUM': '#34745A', 'CuSum NRT monthly': '#838383',
          'CuSum NRT decrease': '#B49539', 'CCDC VV': '#4179A7', 'CCDC VV VH': '#203E61'}
LABELS = {'Windowed CUSUM': 'Windowed CUSUM', 'CuSum NRT monthly': 'NRT monthly',
          'CuSum NRT decrease': 'NRT + decrease', 'CCDC VV': 'CCDC VV', 'CCDC VV VH': 'CCDC VV + VH'}
MARKERS = ['o', 's', '^', 'D', 'v']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/cusum_nrt_ccdc_20260905_v1')
    args = parser.parse_args()
    run = args.run_dir
    if not (run / 'latency/COMPLETE.json').exists():
        raise ValueError('Complete monthly replay and latency reconciliation are required')
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    scores = pd.read_csv(run / 'diagnostics/method_intervals.csv')
    effects = pd.read_csv(run / 'paired_contrasts.csv')
    latency = pd.read_csv(run / 'latency/latency_metrics.csv')
    strata = pd.read_csv(run / 'diagnostics/stratified_reference_recall.csv')
    for name, frame in [('scores', scores), ('effects', effects), ('latency', latency), ('strata', strata)]:
        frame.to_csv(out / (name+'_source.csv'), index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
    width_in, height_in = 183/25.4, 138/25.4
    fig, axes = plt.subplots(2, 2, figsize=(width_in, height_in))
    fig.subplots_adjust(left=.19, right=.985, top=.93, bottom=.15, hspace=.66, wspace=.48)
    a, b, c, d = axes.ravel()
    titles = ['Catalogue-conditional F1', 'Paired F1 advantage', 'Active correct alerts', 'First and later references']
    for letter, ax, title in zip('abcd', axes.ravel(), titles):
        ax.text(-.03, 1.07, '('+letter+')', transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right')
        ax.set_title(title, fontsize=8, loc='left', pad=10)
        ax.set_axisbelow(True)
        ax.grid(axis='x' if ax in (a,b) else 'y', color='#dddddd', linewidth=.5)
    order = list(COLORS)
    for i, name in enumerate(order):
        r = scores.loc[scores.stage.eq(name) & scores.metric.eq('f1')].iloc[0]
        a.errorbar(r.estimate, i, xerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]],
                   fmt=MARKERS[i], color=COLORS[name], capsize=2, markersize=4, linewidth=1)
    a.set_yticks(range(len(order)), [LABELS[n] for n in order])
    a.invert_yaxis()
    a.set_xlim(.15, .95)
    a.set_xlabel('F1 (95% CI)')
    for i, name in enumerate(['W minus NRT', 'W minus CCDC']):
        r = effects.loc[effects.contrast.eq(name) & effects.metric.eq('f1')].iloc[0]
        b.errorbar(r.estimate, i, xerr=[[r.estimate-r.familywise_ci_low], [r.familywise_ci_high-r.estimate]],
                   fmt='o', color=COLORS['Windowed CUSUM'], capsize=2, markersize=4, linewidth=1)
    b.axvline(0, color='#888888', linestyle=':', linewidth=.7)
    b.set_yticks([0,1], ['W - NRT', 'W - CCDC'])
    b.set_ylim(1.5,-.5)
    b.set_xlim(-.025,.55)
    b.set_xlabel('F1 difference (97.5% CI)')
    handles = []
    for i, name in enumerate(order[:4]):
        frame = latency.loc[latency.method.eq(name)].sort_values('horizon_months')
        x = frame.horizon_months.to_numpy()
        assert np.all(np.diff(x) > 0)
        y = frame.active_correct_recall.to_numpy()
        handle, = c.plot(x, y, color=COLORS[name], marker=MARKERS[i], markersize=3,
                          linewidth=1.1, linestyle=['-', ':', '--', '-.'][i], label=LABELS[name])
        c.fill_between(x, frame.active_correct_recall_low, frame.active_correct_recall_high,
                       color=COLORS[name], alpha=.09, linewidth=0)
        handles.append(handle)
        rows = strata.loc[strata.method.eq(name) & strata.dimension.eq('sequence_position')].set_index('group')
        for j, group in enumerate(['First reference','Subsequent reference']):
            r = rows.loc[group]
            d.errorbar(j+(i-1.5)*.14, r.recall,
                yerr=[[r.recall-r.recall_low], [r.recall_high-r.recall]], fmt=MARKERS[i],
                color=COLORS[name], markersize=4, capsize=2, linewidth=1)
    c.set_xticks([0,1,2,3,6,12])
    c.set_xlabel('Months after reference')
    c.set_ylabel('Reference recall')
    c.set_ylim(0,1)
    d.set_xticks([0,1], ['First', 'Subsequent'])
    d.set_xlabel('Position in reference chronology')
    d.set_ylabel('Reference recall')
    d.set_xlim(-.5,1.5)
    d.set_ylim(0,1)
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.53,.01), handlelength=1.8, columnspacing=1.4)
    name = 'nrt_ccdc_accuracy_timeliness'
    fig.savefig(out / 'nrt_ccdc_accuracy_timeliness.pdf', facecolor='white')
    fig.savefig(out / 'nrt_ccdc_accuracy_timeliness.svg', facecolor='white')
    fig.savefig(out / 'nrt_ccdc_accuracy_timeliness.png', dpi=600, facecolor='white')
    fig.savefig(out / 'nrt_ccdc_accuracy_timeliness.tiff', dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    # Use the same Python rendering backend for individual QA previews.
    renderer = fig.canvas.get_renderer()
    for letter, ax in zip('abcd', axes.ravel()):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.08,1.12)
        fig.savefig(out / ('panel_'+letter+'.png'), dpi=300, bbox_inches=bbox, facecolor='white')
    plt.close(fig)
    print(out / (name+'.pdf'))


if __name__ == '__main__':
    main()
