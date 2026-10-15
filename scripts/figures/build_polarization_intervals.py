#!/usr/bin/env python3
"""Build manuscript Figure 8 from the frozen VV chronology and VH support."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import DATA_ROOT, GREEN, INK, RESULTS, YELLOW, configure_style, panel_label, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("runs/figures/fig08_polarization_intervals.pdf"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results", type=Path, default=RESULTS)
    return parser.parse_args()


def intervals(data_root: Path) -> pd.DataFrame:
    source = data_root / "frozen_results/polarization/vv_primary_adjacent_intervals_vh_pm2.parquet"
    frame = pd.read_parquet(source).rename(
        columns={"both_endpoints_vh_supported": "both_supported"}
    )
    if len(frame) != 3509 or int(frame.both_supported.sum()) != 1385:
        raise ValueError(
            "Expected 3,509 primary intervals and 1,385 intervals with both endpoints "
            f"VH-supported within +/-2 months; found {len(frame):,} and {int(frame.both_supported.sum()):,}"
        )
    return frame


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, float)); return x, np.arange(1, len(x) + 1) / len(x)


def main() -> None:
    args = parse_args(); configure_style(); frame = intervals(args.data_root)
    if len(frame) != 3509:
        raise ValueError(f"Expected 3,509 primary intervals, found {len(frame)}")
    bins = [(-np.inf, 12), (12, 24), (24, 48), (48, np.inf)]
    labels = ["<12", "12--23", "24--47", ">=48"]
    rows = []
    for label, (lower, upper) in zip(labels, bins):
        local = frame.loc[frame.interval_months.ge(lower) & frame.interval_months.lt(upper)]
        rows.append((label, len(local), int(local.both_supported.sum()), float(local.both_supported.mean())))

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": (1.0, 1.15, 0.95)})
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.96, wspace=0.32)
    x = np.arange(4); rates = np.asarray([row[3] for row in rows]) * 100
    axes[0].bar(x, rates, width=0.66, color=YELLOW, edgecolor=INK, linewidth=0.45)
    axes[0].axhline(frame.both_supported.mean() * 100, color="#777777", linestyle="--", linewidth=0.7)
    for position, (_, total, supported, rate) in enumerate(rows):
        axes[0].text(position, rate * 100 + 2, f"{rate * 100:.1f}%\n({supported}/{total})", ha="center", va="bottom", fontsize=5.8)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 70); axes[0].set_ylabel("VH-supported at both endpoints (%)")
    axes[0].set_xlabel("Primary VV interval (months)"); axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.4)
    panel_label(axes[0], "a")

    primary_x, primary_y = ecdf(frame.interval_months)
    supported_x, supported_y = ecdf(frame.loc[frame.both_supported, "interval_months"])
    axes[1].plot(primary_x, primary_y, color=GREEN, linewidth=1.35, label="Primary VV intervals")
    axes[1].plot(supported_x, supported_y, color=YELLOW, linewidth=1.35, label="VH-supported subset")
    axes[1].set_xlim(0, 108); axes[1].set_ylim(0, 1); axes[1].set_xlabel("Adjacent-event interval (months)")
    axes[1].set_ylabel("Empirical cumulative probability"); axes[1].grid(color="#DDDDDD", linewidth=0.4)
    axes[1].legend(loc="lower right", frameon=True); panel_label(axes[1], "b")

    summary = pd.read_csv(args.results / "polarization/spatial_block_bootstrap_vh_pm2.csv")
    medians = summary.loc[summary.metric.eq("median_months")].set_index("cohort")
    cohorts = ["primary_vv", "both_endpoints_vh_supported"]
    y = [1, 0]; colors = [GREEN, YELLOW]
    for position, cohort, color in zip(y, cohorts, colors):
        row = medians.loc[cohort]
        values = frame.interval_months if cohort == "primary_vv" else frame.loc[frame.both_supported, "interval_months"]
        q25, q75 = np.quantile(values, [0.25, 0.75])
        axes[2].plot([q25, q75], [position, position], color=color, linewidth=5.0,
                     solid_capstyle="butt", alpha=0.75, zorder=1)
        axes[2].errorbar(row.bootstrap_median, position,
                         xerr=[[row.bootstrap_median - row.ci95_low], [row.ci95_high - row.bootstrap_median]],
                         fmt="o", color=INK, markerfacecolor=color, markersize=4, capsize=2, linewidth=1, zorder=3)
        axes[2].text(
            row.bootstrap_median + 8, position,
            f"{row.bootstrap_median:.0f} mo; IQR {q25:.0f}--{q75:.0f}\n95% CI {row.ci95_low:.0f}--{row.ci95_high:.0f}",
            va="center", fontsize=6.0,
        )
    axes[2].set_yticks(y, ["Primary VV", "VH-supported"]); axes[2].set_xlim(0, 68)
    axes[2].set_xlabel("Interval (months)"); axes[2].grid(axis="x", color="#DDDDDD", linewidth=0.4)
    panel_label(axes[2], "c")
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
