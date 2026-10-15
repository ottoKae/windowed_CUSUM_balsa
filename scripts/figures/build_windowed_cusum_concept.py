#!/usr/bin/env python3
"""Build manuscript Figure 4 from one frozen polygon-first trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    DATA_ROOT, GREEN, INK, LIGHT_GREY, YELLOW, add_src_to_path,
    configure_style, load_case_series, panel_label, save_figure,
)

add_src_to_path()
from lr_cusum.core import cusum_amplitude  # noqa: E402
from lr_cusum.paper import load_paper_config, scan_series_n1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("runs/figures/fig04_windowed_cusum_concept.pdf"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_v1.json"))
    parser.add_argument("--historical-id", type=int, default=1499)
    return parser.parse_args()


def cumulative(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return np.cumsum(finite - finite.mean())


def main() -> None:
    args = parse_args()
    configure_style()
    case = load_case_series(args.data_root, args.historical_id)
    config = load_paper_config(args.config)
    vv = case.median[0]
    ledgers = scan_series_n1(
        vv, case.calendar, identifier=case.sample_id, band="VV_dB",
        config=config, all_component_candidates=True,
    )
    candidates = ledgers.sliding_candidates.loc[
        ledgers.sliding_candidates.candidate_valid
    ].copy()
    pairs = []
    for change_time, group in candidates.groupby("change_time"):
        rejected = group.loc[~group.passes_central.astype(bool)]
        retained = group.loc[
            group.passes_central.astype(bool)
            & group.passes_decrease.astype(bool)
            & group.passes_permutation.astype(bool)
        ]
        for rejected_row in rejected.itertuples(index=False):
            for retained_row in retained.itertuples(index=False):
                if abs(int(rejected_row.logical_window_id) - int(retained_row.logical_window_id)) == 1:
                    distance = abs(pd.Timestamp(change_time).to_period("M").ordinal - case.references[1].to_period("M").ordinal)
                    pairs.append((distance, rejected_row, retained_row))
    if not pairs:
        raise ValueError("No adjacent reject/retain window pair found for the representative trajectory")
    _, rejected_row, retained_row = min(pairs, key=lambda item: item[0])
    selected = [rejected_row, retained_row]
    if pd.Timestamp(rejected_row.change_time) != pd.Timestamp(retained_row.change_time):
        raise AssertionError("Illustrated windows must refer to the same candidate change month")
    if bool(rejected_row.passes_central) or not bool(retained_row.passes_central):
        raise AssertionError("Illustrated windows must show central rejection followed by admission")

    figure = plt.figure(figsize=(7.2, 4.5))
    grid = figure.add_gridspec(4, 2, height_ratios=(1.12, 1, 1, 0.82), hspace=0.12, wspace=0.05)
    top = figure.add_subplot(grid[0, :])
    top.plot(case.calendar, vv, color=GREEN, marker="o", markersize=1.8, linewidth=1.1, label="VV polygon median")
    for date in case.references:
        top.axvline(date, color=INK, linestyle="--", linewidth=0.75)
        top.text(date, -4.65, date.strftime("%Y-%m"), ha="center", va="bottom", fontsize=6.2)
    for row in selected:
        top.axvspan(row.window_start, row.window_end, color=LIGHT_GREY, alpha=0.42)
        top.axvspan(row.decision_start, row.decision_end, color=YELLOW, alpha=0.18)
    top.set_ylabel("VV (dB)"); top.set_xlim(case.calendar[0], case.calendar[-1]); top.set_ylim(-10.5, -4.3)
    top.xaxis.set_major_locator(mdates.YearLocator()); top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    top.grid(color="#D8D8D8", linewidth=0.35); top.legend(loc="lower left", frameon=True)
    panel_label(top, "a")

    window_axes = [figure.add_subplot(grid[1, i]) for i in range(2)]
    cusum_axes = [figure.add_subplot(grid[2, i]) for i in range(2)]
    for index, (row, axis, caxis) in enumerate(zip(selected, window_axes, cusum_axes)):
        start, end = int(row.window_start_index), int(row.window_end_index)
        values = vv[start:end]
        dates = case.calendar[start:end]
        axis.plot(dates, values, color=GREEN, marker="o", markersize=1.8, linewidth=1.0)
        axis.axvspan(row.decision_start, row.decision_end, color=YELLOW, alpha=0.18)
        axis.axvline(row.change_time, color="#405D80", linestyle="--", linewidth=1.0)
        axis.hlines([row.pre_mean, row.post_mean], dates[0], dates[-1], color="#56606A", linestyle=":", linewidth=0.8)
        axis.text(0.48, 0.50, f"{row.post_minus_pre:.1f} dB", transform=axis.transAxes, fontsize=6.5)
        axis.set_ylabel("VV (dB)" if index == 0 else "")
        axis.tick_params(labelbottom=False); axis.grid(color="#D8D8D8", linewidth=0.35)
        panel_label(axis, chr(ord("b") + index))

        finite = values[np.isfinite(values)]
        csum = cumulative(values)
        finite_dates = dates[np.isfinite(values)]
        caxis.plot(finite_dates, csum, color=INK, marker="o", markersize=1.7, linewidth=0.9)
        caxis.axvline(row.change_time, color="#405D80", linestyle="--", linewidth=1.0)
        caxis.text(0.50, 0.45, f"Q = {row.cusum_amplitude:.2f}", transform=caxis.transAxes, fontsize=6.5)
        caxis.set_ylabel("Cumulative deviation" if index == 0 else "")
        caxis.xaxis.set_major_locator(mdates.YearLocator()); caxis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        caxis.grid(color="#D8D8D8", linewidth=0.35)
        panel_label(caxis, chr(ord("d") + index))

    # Display the empirical null for the second illustrated window.
    row = selected[1]
    values = vv[int(row.window_start_index):int(row.window_end_index)]
    finite = values[np.isfinite(values)]
    rng = np.random.default_rng(int(row.permutation_seed))
    null = np.asarray([cusum_amplitude(rng.permutation(finite))[0] for _ in range(config.permutations)])
    null_axis = figure.add_subplot(grid[3, :])
    null_axis.hist(null, bins=28, density=True, color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.3)
    null_axis.axvline(row.cusum_amplitude, color="#405D80", linewidth=1.4, label="Observed CUSUM amplitude")
    null_axis.axvline(np.quantile(null, 0.975), color=INK, linestyle=":", linewidth=1.0, label="97.5th null percentile")
    null_axis.set_xlabel("CUSUM amplitude, Q"); null_axis.set_ylabel("Density")
    null_axis.legend(loc="upper left", frameon=True, ncol=2)
    panel_label(null_axis, "f")
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
