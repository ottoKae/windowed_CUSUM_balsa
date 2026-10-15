#!/usr/bin/env python3
"""Build the N1 window-length/step and polarization sensitivity figure.

The script is additive and reads only frozen N1 outputs. It verifies parity
between the reference-only and national VV event tables, evaluates four
polarization decision rules on the same 1,093-event reference cohort, performs
common-index spatial-block bootstrap resampling, and exports a three-panel
manuscript figure with auditable source tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lr_cusum.component_evaluation import (  # noqa: E402
    common_index_paired_block_bootstrap,
    evaluate_event_stages,
)
from lr_cusum.polygon_workflow import annotate_vv_with_vh  # noqa: E402
from lr_cusum.validation_closure import EVENT_MATCHER_OPTIMAL_V2  # noqa: E402


N1_ROOT = ROOT / "outputs/windowed_cusum_n1_full_40tile_v1"
REFERENCE_ROOT = ROOT / "outputs/balsa_validation_closure_40tiles_validated_v2_strict_decision"
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260831

DISPLAY_ORDER = ["VV only", "VH only", "VV or VH", "VV and VH"]
DISPLAY_LABELS = {
    "VV only": "VV only",
    "VH only": "VH only",
    "VV or VH": "VV or VH",
    "VV and VH": "VV–VH agreement",
}
COLORS = {
    "VV only": "#6F8F3D",
    "VH only": "#D4A72C",
    "VV or VH": "#4C78A8",
    "VV and VH": "#7A5AA6",
}
MARKERS = {
    "VV only": "o",
    "VH only": "o",
    "VV or VH": "^",
    "VV and VH": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n1-root", type=Path, default=N1_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_number(values: pd.Series) -> pd.Series:
    times = pd.to_datetime(values)
    return times.dt.year * 12 + times.dt.month


def collapse_any_polarization(events: pd.DataFrame, tolerance_months: int = 1) -> pd.DataFrame:
    """Fuse VV/VH events within a ground sample using the frozen union rule."""
    if events.empty:
        return pd.DataFrame(columns=list(events.columns))
    rows: list[dict[str, object]] = []
    for _, local in events.groupby("sample_id", sort=False):
        local = local.sort_values(["change_time", "Band"], kind="mergesort")
        local = local.assign(_month_number=month_number(local.change_time).to_numpy())
        clusters: list[list[int]] = []
        current: list[int] = []
        previous: int | None = None
        for index, number in zip(local.index, local._month_number):
            if current and previous is not None and int(number) - previous > tolerance_months:
                clusters.append(current)
                current = []
            current.append(index)
            previous = int(number)
        if current:
            clusters.append(current)
        for cluster in clusters:
            selected = local.loc[cluster]
            representative = selected.iloc[0].drop(labels="_month_number").to_dict()
            representative["Band"] = "VV_or_VH"
            representative["change_time"] = pd.to_datetime(selected.change_time).min()
            representative["polarization_support"] = int(selected.Band.nunique())
            rows.append(representative)
    return pd.DataFrame(rows).sort_values(
        ["sample_id", "change_time"], kind="mergesort"
    ).reset_index(drop=True)


def load_references(reference_root: Path) -> pd.DataFrame:
    references = pd.read_parquet(reference_root / "reference_events.parquet").copy()
    references["event_month"] = pd.to_datetime(references.event_month)
    folds = pd.read_csv(reference_root / "locked_outer_spatial_folds.csv")
    references = references.drop(columns=["outer_fold"], errors="ignore").merge(
        folds[["spatial_block_id", "outer_fold"]],
        on="spatial_block_id",
        how="left",
        validate="many_to_one",
    )
    if references.outer_fold.isna().any():
        raise ValueError("Reference events contain missing outer folds")
    return references


def normalized_event_keys(events: pd.DataFrame) -> set[tuple[str, str, str]]:
    local = events[["sample_id", "Band", "change_time"]].copy()
    local["change_time"] = pd.to_datetime(local.change_time).dt.strftime("%Y-%m")
    return set(local.astype(str).itertuples(index=False, name=None))


def build_polarization_strategies(
    n1_root: Path,
    references: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    national_path = n1_root / "national/ground_polygon_events.parquet"
    national = pd.read_parquet(national_path)
    national = national.loc[national.route.eq("polygon_first")].copy()
    reference_ids = set(references.sample_id.astype(str))
    national = national.loc[national.sample_id.astype(str).isin(reference_ids)].copy()
    national["change_time"] = pd.to_datetime(national.change_time)

    vv = national.loc[national.Band.eq("VV_dB")].copy()
    vh = national.loc[national.Band.eq("VH_dB")].copy()
    any_polarization = collapse_any_polarization(national, tolerance_months=1)
    _, agreement = annotate_vv_with_vh(national, tolerance_months=1)

    reference_only_vv_path = n1_root / "geometry/events/w48_s12.ground.parquet"
    reference_only_vv = pd.read_parquet(reference_only_vv_path)
    parity = normalized_event_keys(vv) == normalized_event_keys(reference_only_vv)
    if not parity:
        raise RuntimeError(
            "N1 VV parity failed between national and reference-only 48/12 events"
        )

    strategies = {
        "VV only": vv,
        "VH only": vh,
        "VV or VH": any_polarization,
        "VV and VH": agreement,
    }
    audit = {
        "vv_reference_parity": parity,
        "reference_sample_count": int(references.sample_id.astype(str).nunique()),
        "reference_event_count": int(len(references)),
        "strategy_event_counts": {name: int(len(frame)) for name, frame in strategies.items()},
    }
    return strategies, audit


def evaluate_polarization(
    strategies: dict[str, pd.DataFrame],
    references: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation = evaluate_event_stages(
        references,
        strategies,
        matcher_version=EVENT_MATCHER_OPTIMAL_V2,
        tolerances=(2,),
        fold_column="outer_fold",
    )
    contrasts = {
        f"{name} minus VV only": (name, "VV only")
        for name in DISPLAY_ORDER
        if name != "VV only"
    }
    bootstrap = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=2,
        replicates=replicates,
        seed=seed,
    )
    return (
        evaluation.stage_metrics,
        evaluation.fold_metrics,
        evaluation.block_metrics,
        bootstrap.stage_summary,
        bootstrap.contrast_summary,
    )


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=5)


def save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_figure(
    geometry: pd.DataFrame,
    geometry_contrasts: pd.DataFrame,
    polarization_metrics: pd.DataFrame,
    polarization_intervals: pd.DataFrame,
    output: Path,
) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.20, 2.55),
        gridspec_kw={"width_ratios": [1.55, 0.72, 1.18]},
    )

    # Panel a: all five structurally admissible window/step combinations.
    axis = axes[0]
    step_colors = {6: "#4C78A8", 12: "#E07B39"}
    metric_styles = {
        "f1": ("o", "-", 1.45),
        "precision": ("s", "--", 0.9),
        "recall": ("^", ":", 0.9),
    }
    for step, local in geometry.groupby("step_months"):
        local = local.sort_values("window_months")
        for metric, (marker, linestyle, linewidth) in metric_styles.items():
            axis.plot(
                local.window_months,
                local[metric],
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                markersize=4.2,
                color=step_colors[int(step)],
                alpha=1.0 if metric == "f1" else 0.80,
            )
    axis.set(
        xlabel="Window length (months)",
        ylabel="Event-level metric",
        xticks=[24, 36, 48],
        ylim=(0.62, 0.89),
    )
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.45, zorder=0)
    panel_title(axis, "a", "Window length and step")
    metric_handles = [
        Line2D([0], [0], color="#555555", marker="o", lw=1.45, label="F1"),
        Line2D([0], [0], color="#555555", marker="s", lw=0.9, ls="--", label="Precision"),
        Line2D([0], [0], color="#555555", marker="^", lw=0.9, ls=":", label="Recall"),
    ]
    metric_legend = axis.legend(
        handles=metric_handles,
        loc="upper left",
        frameon=False,
        handlelength=2.1,
        borderaxespad=0.2,
    )
    axis.add_artist(metric_legend)
    step_handles = [
        Line2D([0], [0], color=step_colors[6], lw=1.5, label="6-month step"),
        Line2D([0], [0], color=step_colors[12], lw=1.5, label="12-month step"),
    ]
    axis.legend(
        handles=step_handles,
        loc="lower right",
        frameon=False,
        borderaxespad=0.2,
        title="Window placement",
        title_fontsize=6.1,
    )

    # Panel b: the single inferentially decisive paired comparison.
    axis = axes[1]
    selected = geometry_contrasts.loc[
        geometry_contrasts.contrast.eq("w36_s12_minus_w48_s12")
        & geometry_contrasts.metric.eq("f1")
    ]
    if len(selected) != 1:
        raise ValueError("Expected one 36/12 minus 48/12 F1 contrast")
    row = selected.iloc[0]
    estimate = float(row.estimate)
    low = float(row.ci_low)
    high = float(row.ci_high)
    axis.axvline(0, color="#777777", linewidth=0.8, zorder=1)
    axis.errorbar(
        estimate,
        0,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="o",
        markersize=5,
        color="#2F6B4F",
        ecolor="#2F6B4F",
        elinewidth=1.2,
        capsize=2.5,
        zorder=3,
    )
    axis.set(
        xlabel=r"Paired $\Delta$F1",
        yticks=[0],
        yticklabels=["36/12 minus\n48/12"],
        ylim=(-0.62, 0.62),
        xlim=(-0.025, 0.027),
    )
    axis.grid(axis="x", color="#E3E3E3", linewidth=0.45, zorder=0)
    panel_title(axis, "b", "Paired difference")
    axis.text(
        0.5,
        0.10,
        f"{estimate:+.3f}\n95% CI {low:+.3f} to {high:+.3f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.7,
        color="#444444",
    )

    # Panel c: polarization operating points under the same frozen N1 protocol.
    axis = axes[2]
    point = polarization_metrics.set_index("stage")
    interval = polarization_intervals.pivot(
        index="stage", columns="metric", values=["ci_low", "ci_high"]
    )
    label_offsets = {
        "VV only": (5, -11, "left"),
        "VH only": (5, 5, "left"),
        "VV or VH": (5, -2, "left"),
        "VV and VH": (-5, -7, "right"),
    }
    for name in DISPLAY_ORDER:
        row = point.loc[name]
        recall = float(row.recall)
        precision = float(row.precision)
        x_low = float(interval.loc[name, ("ci_low", "recall")])
        x_high = float(interval.loc[name, ("ci_high", "recall")])
        y_low = float(interval.loc[name, ("ci_low", "precision")])
        y_high = float(interval.loc[name, ("ci_high", "precision")])
        axis.errorbar(
            recall,
            precision,
            xerr=np.array([[recall - x_low], [x_high - recall]]),
            yerr=np.array([[precision - y_low], [y_high - precision]]),
            fmt=MARKERS[name],
            markersize=5.2 if name == "VV only" else 4.6,
            markerfacecolor=COLORS[name],
            markeredgecolor="#303030",
            markeredgewidth=0.35,
            ecolor=COLORS[name],
            elinewidth=0.85,
            capsize=1.8,
            alpha=0.98,
            zorder=3,
        )
        dx, dy, alignment = label_offsets[name]
        axis.annotate(
            f"{DISPLAY_LABELS[name]}\nF1={row.f1:.3f}",
            (recall, precision),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=5.6,
            color="#303030",
        )
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.60, 0.89),
        ylim=(0.56, 0.92),
    )
    axis.grid(color="#E8E8E8", linewidth=0.4, zorder=0)
    panel_title(axis, "c", "Polarization decision rules")
    axis.text(
        0.02,
        0.02,
        "Error bars: 95% spatial-block CIs",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="#666666",
    )

    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.90, wspace=0.46)
    save_figure(figure, output)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    n1_root = args.n1_root.resolve()
    reference_root = args.reference_root.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    geometry_path = n1_root / "geometry/geometry_metrics.csv"
    geometry_contrasts_path = n1_root / "geometry/geometry_contrasts.csv"
    national_path = n1_root / "national/ground_polygon_events.parquet"
    reference_path = reference_root / "reference_events.parquet"
    fold_path = reference_root / "locked_outer_spatial_folds.csv"
    inputs = [geometry_path, geometry_contrasts_path, national_path, reference_path, fold_path]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required frozen inputs: " + ", ".join(missing))

    geometry = pd.read_csv(geometry_path)
    geometry_contrasts = pd.read_csv(geometry_contrasts_path)
    expected_geometries = {
        (24, 6), (36, 6), (48, 6), (36, 12), (48, 12),
    }
    observed_geometries = set(
        zip(geometry.window_months.astype(int), geometry.step_months.astype(int))
    )
    if observed_geometries != expected_geometries:
        raise ValueError(f"Unexpected N1 geometry grid: {sorted(observed_geometries)}")
    if (24, 12) in observed_geometries:
        raise ValueError("24/12 must remain excluded because W - 2S equals zero")

    references = load_references(reference_root)
    strategies, audit = build_polarization_strategies(n1_root, references)
    metrics, fold_metrics, block_metrics, intervals, contrasts = evaluate_polarization(
        strategies,
        references,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    event_dir = output / "polarization_events"
    for name, frame in strategies.items():
        slug = name.lower().replace(" ", "_")
        atomic_parquet(frame, event_dir / f"{slug}.parquet")
    atomic_csv(geometry, output / "source_data/geometry_metrics.csv")
    atomic_csv(geometry_contrasts, output / "source_data/geometry_contrasts.csv")
    atomic_csv(metrics, output / "source_data/polarization_metrics.csv")
    atomic_csv(fold_metrics, output / "source_data/polarization_fold_metrics.csv")
    atomic_parquet(block_metrics, output / "source_data/polarization_block_metrics.parquet")
    atomic_csv(intervals, output / "source_data/polarization_bootstrap_intervals.csv")
    atomic_csv(contrasts, output / "source_data/polarization_contrasts.csv")

    figure_base = output / "figure_06_window_length_step_polarization_n1"
    plot_figure(geometry, geometry_contrasts, metrics, intervals, figure_base)

    vv_metrics = metrics.loc[metrics.stage.eq("VV only")].iloc[0]
    geometry_vv = geometry.loc[
        geometry.window_months.eq(48) & geometry.step_months.eq(12)
    ].iloc[0]
    metric_parity = all(
        np.isclose(float(vv_metrics[column]), float(geometry_vv[column]), atol=1e-12)
        for column in ("precision", "recall", "f1")
    )
    if not metric_parity:
        raise RuntimeError("N1 VV metric parity failed after polarization evaluation")

    manifest = {
        "analysis": "N1 window-length/step and polarization sensitivity",
        "figure_claim": (
            "Window length and step jointly control the precision-recall balance, "
            "36/12 and frozen 48/12 have no detectable paired F1 difference, and "
            "polarization fusion provides distinct operating points without improving "
            "the overall F1 of VV-only detection."
        ),
        "protocol": {
            "detector": "Windowed CUSUM",
            "null_model": "N1 IID permutation",
            "permutation_threshold": 0.025,
            "polarization_window_months": 48,
            "polarization_step_months": 12,
            "polarization_agreement_tolerance_months": 1,
            "reference_match_tolerance_months": 2,
            "matcher": EVENT_MATCHER_OPTIMAL_V2,
            "bootstrap_unit": "complete spatial block",
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.seed),
        },
        "audit": {
            **audit,
            "vv_metric_parity": metric_parity,
            "spatial_blocks": int(references.spatial_block_id.nunique()),
            "valid_geometry_combinations": ["24/6", "36/6", "48/6", "36/12", "48/12"],
            "excluded_geometry": {
                "configuration": "24/12",
                "reason": "central decision interval W - 2S equals zero",
            },
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {
            "figure_pdf": str(figure_base.with_suffix(".pdf")),
            "figure_svg": str(figure_base.with_suffix(".svg")),
            "figure_png": str(figure_base.with_suffix(".png")),
            "figure_tiff": str(figure_base.with_suffix(".tiff")),
        },
    }
    atomic_json(manifest, output / "manifest.json")

    print(metrics[["stage", "precision", "recall", "f1", "detected_events"]].to_string(index=False))
    print(f"Figure written to {figure_base}")


if __name__ == "__main__":
    main()
