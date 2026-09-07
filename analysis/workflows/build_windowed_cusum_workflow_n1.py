"""Build a publication workflow figure for frozen N1 Windowed CUSUM.

The figure separates the production pipeline from historical accuracy
assessment. It reads frozen manifests, validates displayed values, and never
overwrites existing output or modifies manuscript sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


FIGURE_WIDTH_IN = 7.0079
FIGURE_HEIGHT_IN = 4.4094
INK = "#263238"
MUTED = "#65737E"
DATA = "#4472A6"
DATA_LIGHT = "#EDF4FA"
N1 = "#177E78"
N1_LIGHT = "#EAF6F4"
DECISION = "#D59633"
DECISION_LIGHT = "#FAF1DE"
EVAL = "#7566A0"
EVAL_LIGHT = "#F1EEF7"
APPLICATION = "#A46332"
APPLICATION_LIGHT = "#F7EFE8"
VV = "#72833D"
VH = "#D6A72C"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class FigureSpec:
    context_start: str
    context_end: str
    decision_start: str
    decision_end: str
    archive_tiles: int
    dynamic_tracks: int
    validated_polygons: int
    polygon_tracks: int
    pixel_memberships: int
    window_months: int
    step_months: int
    central_months: int
    minimum_observations: int
    permutation_alpha: float
    permutation_replicates: int
    near_threshold_replicates: int
    track_fusion_tolerance_months: int
    reference_events: int
    reference_polygons: int
    spatial_blocks: int
    outer_folds: int
    match_tolerance_months: int
    spatial_bootstrap_replicates: int
    primary_band: str
    support_band: str
    support_tolerance_months: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_spec(root: Path) -> tuple[FigureSpec, dict[str, Path]]:
    phase_b_path = root / "outputs/forestry_sample_rebuild/20260825_balsa_validated_40tiles_v1/phase_b_summary.json"
    cache_path = root / "outputs/forestry_sample_rebuild/20260825_balsa_validated_40tiles_v1_phase_d_full_cache_v1/phase_d_cache_report.json"
    n1_path = root / "outputs/windowed_cusum_n1_full_40tile_v1/run_manifest.json"
    polarization_path = root / "outputs/windowed_cusum_n1_polarization_interval_sensitivity_40tile_v1/run_manifest.json"
    protocol_path = root / "outputs/windowed_cusum_component_ablation_40tile_v2_prospective_lock/protocol.json"

    phase_b = _read_json(phase_b_path)
    cache = _read_json(cache_path)
    n1 = _read_json(n1_path)
    polarization = _read_json(polarization_path)
    protocol = _read_json(protocol_path)
    context_start, context_end = protocol["context_calendar"]
    decision_start = f"{protocol['reporting_years'][0]}-01"
    decision_end = f"{protocol['reporting_years'][1]}-12"
    window_months = int(protocol["window_length"])
    step_months = int(protocol["step"])

    spec = FigureSpec(
        context_start=context_start,
        context_end=context_end,
        decision_start=decision_start,
        decision_end=decision_end,
        archive_tiles=int(phase_b["catalog"]["tiles"]),
        dynamic_tracks=int(phase_b["catalog"]["dynamic_tracks"]),
        validated_polygons=int(phase_b["inventory"]["active_validated"]),
        polygon_tracks=int(cache["polygon_tracks"]),
        pixel_memberships=int(cache["pixel_associations"]),
        window_months=window_months,
        step_months=step_months,
        central_months=window_months - 2 * step_months,
        minimum_observations=int(protocol["minimum_observations"]),
        permutation_alpha=float(n1["alpha"]),
        permutation_replicates=int(n1["replicates"]),
        near_threshold_replicates=int(n1["near_threshold_replicates"]),
        track_fusion_tolerance_months=int(protocol["track_fusion_tolerance_months"]),
        reference_events=int(n1["reference_events"]),
        reference_polygons=int(n1["reference_polygons"]),
        spatial_blocks=int(n1["spatial_blocks"]),
        outer_folds=5,
        match_tolerance_months=int(protocol["primary_match_tolerance_months"]),
        spatial_bootstrap_replicates=int(n1["spatial_bootstrap_replicates"]),
        primary_band=str(polarization["primary_band"]),
        support_band="VH_dB",
        support_tolerance_months=int(polarization["vh_tolerance_months"]),
    )
    expected = {
        "context": ("2016-01", "2026-06"),
        "decision": ("2017-01", "2025-12"),
        "archive_tiles": 40,
        "dynamic_tracks": 56,
        "validated_polygons": 3498,
        "window_months": 48,
        "step_months": 12,
        "central_months": 24,
        "minimum_observations": 12,
        "permutation_alpha": 0.025,
        "permutation_replicates": 3999,
        "near_threshold_replicates": 19999,
        "track_fusion_tolerance_months": 1,
        "reference_events": 1093,
        "reference_polygons": 462,
        "spatial_blocks": 57,
        "outer_folds": 5,
        "match_tolerance_months": 2,
        "spatial_bootstrap_replicates": 5000,
        "primary_band": "VV_dB",
        "support_tolerance_months": 1,
    }
    observed = {
        "context": (spec.context_start, spec.context_end),
        "decision": (spec.decision_start, spec.decision_end),
        **{key: getattr(spec, key) for key in expected if key not in {"context", "decision"}},
    }
    mismatches = {key: {"expected": expected[key], "observed": observed[key]} for key in expected if observed[key] != expected[key]}
    if mismatches:
        raise ValueError(f"Frozen N1 semantic mismatch: {mismatches}")
    expected_missing = "source-absent month is flagged; pixel invalidity remains NaN with n_obs=0"
    if cache["source_missing_policy"] != expected_missing:
        raise ValueError("Unexpected missing-data policy")
    if int(cache["calendar_months"]) != 126:
        raise ValueError("The context calendar must contain 126 months")

    return spec, {
        "phase_b_summary": phase_b_path,
        "cache_report": cache_path,
        "n1_manifest": n1_path,
        "polarization_manifest": polarization_path,
        "component_protocol": protocol_path,
    }


def _box(ax, x, y, width, height, *, face, edge, linewidth=0.9, linestyle="-", radius=0.012, zorder=1):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, linestyle=linestyle,
        transform=ax.transAxes, clip_on=False, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color=MUTED, linewidth=1.0, linestyle="-", mutation_scale=8.0, connectionstyle="arc3,rad=0", zorder=4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=linewidth, linestyle=linestyle, color=color,
        connectionstyle=connectionstyle, transform=ax.transAxes,
        clip_on=False, zorder=zorder,
    ))


def _label(ax, x, y, text, *, size=7.0, color=INK, weight="normal", ha="center", va="center", zorder=6, linespacing=1.12):
    ax.text(
        x, y, text, fontsize=size, color=color, fontweight=weight,
        ha=ha, va=va, linespacing=linespacing, transform=ax.transAxes, zorder=zorder,
    )


def _panel_header(ax, x, y, letter, title):
    _label(ax, x, y, letter, size=9.0, weight="bold", ha="left")
    _label(ax, x + 0.028, y, title, size=7.8, weight="bold", ha="left")


def _draw_scene_stack(ax, x, y):
    for offset, alpha in ((0.012, 0.48), (0.006, 0.68), (0.0, 1.0)):
        ax.add_patch(Rectangle(
            (x + offset, y + offset), 0.032, 0.038,
            facecolor=DATA, edgecolor=WHITE, linewidth=0.45, alpha=alpha,
            transform=ax.transAxes, zorder=5,
        ))
    _label(ax, x + 0.019, y + 0.019, "S1", size=5.8, color=WHITE, weight="bold")


def _draw_cube(ax, x, y):
    for i, color in enumerate((DATA, "#78A7CE", "#B9D2E6")):
        ax.add_patch(Rectangle(
            (x + i * 0.010, y + i * 0.006), 0.032, 0.043,
            facecolor=color, edgecolor=WHITE, linewidth=0.45,
            transform=ax.transAxes, zorder=5 - i,
        ))
    for yy in (y + 0.014, y + 0.028):
        ax.plot([x, x + 0.032], [yy, yy], color=WHITE, linewidth=0.4, transform=ax.transAxes, zorder=7)


def _draw_polygon(ax, x, y):
    coords = [(x, y + 0.020), (x + 0.014, y + 0.050), (x + 0.043, y + 0.045), (x + 0.052, y + 0.014), (x + 0.026, y)]
    ax.add_patch(Polygon(coords, closed=True, facecolor=DATA_LIGHT, edgecolor=DATA, linewidth=0.9, transform=ax.transAxes, zorder=4))
    for dx, dy in ((0.012, 0.022), (0.022, 0.035), (0.031, 0.016), (0.040, 0.030)):
        ax.add_patch(Circle((x + dx, y + dy), 0.0027, facecolor=DATA, edgecolor="none", transform=ax.transAxes, zorder=6))


def _draw_trajectory(ax, x, y):
    xx = [x + i * 0.009 for i in range(7)]
    vv = [y + value for value in (0.030, 0.034, 0.029, 0.033, 0.018, 0.023, 0.027)]
    vh = [value - 0.012 for value in vv]
    ax.plot(xx, vv, color=VV, linewidth=1.15, transform=ax.transAxes, zorder=6)
    ax.plot(xx, vh, color=VH, linewidth=1.15, transform=ax.transAxes, zorder=6)
    ax.scatter(xx, vv, s=4, color=VV, transform=ax.transAxes, zorder=7)
    ax.scatter(xx, vh, s=4, color=VH, transform=ax.transAxes, zorder=7)


def _draw_data_panel(ax, spec):
    x0, width = 0.016, 0.239
    _box(ax, x0, 0.072, width, 0.864, face=DATA_LIGHT, edge="#B9CEDF", linewidth=0.8)
    _panel_header(ax, x0 + 0.012, 0.908, "a", "Data and stand trajectories")
    inner_x, inner_w = x0 + 0.025, width - 0.050
    stages = [
        (0.753, 0.105, "OPERA RTC-S1", "VV · VH · 30 m", "scene"),
        (0.574, 0.125, "S1-GRiTS monthly\ncubes", f"{spec.archive_tiles} tiles · {spec.dynamic_tracks} cubes\ntrack-specific; gaps retained", "cube"),
        (0.386, 0.125, "Complete polygon\nsupport", f"{spec.polygon_tracks:,} track-specific objects\nall interior cell centres", "polygon"),
        (0.174, 0.145, "Polygon-first\nmedian", f"monthly stand trajectory\n{spec.context_start}–{spec.context_end} context", "trajectory"),
    ]
    for index, (y, height, title, body, icon) in enumerate(stages):
        _box(ax, inner_x, y, inner_w, height, face=WHITE, edge=DATA, linewidth=0.85)
        icon_x, icon_y = inner_x + 0.012, y + height / 2 - 0.020
        if icon == "scene":
            _draw_scene_stack(ax, icon_x, icon_y)
        elif icon == "cube":
            _draw_cube(ax, icon_x, icon_y)
        elif icon == "polygon":
            _draw_polygon(ax, icon_x - 0.002, icon_y - 0.004)
        else:
            _draw_trajectory(ax, icon_x - 0.002, icon_y - 0.005)
        text_x = inner_x + 0.070
        _label(ax, text_x, y + height * 0.68, title, size=6.2, weight="bold", ha="left")
        _label(ax, text_x, y + height * 0.27, body, size=5.3, color=MUTED, ha="left")
        if index < len(stages) - 1:
            next_y = stages[index + 1][0] + stages[index + 1][1]
            _arrow(ax, (x0 + width / 2, y - 0.010), (x0 + width / 2, next_y + 0.008), color=DATA)
    _box(ax, inner_x + 0.016, 0.100, inner_w - 0.032, 0.046, face="#DCEAF4", edge="none", linewidth=0)
    _label(ax, x0 + width / 2, 0.123, f"Formal decisions: {spec.decision_start}–{spec.decision_end}", size=6.1, color=DATA, weight="bold")


def _draw_window_timeline(ax, x, y, width, height, spec):
    _box(ax, x, y, width, height, face=WHITE, edge="#B8D6D2", linewidth=0.8)
    _label(ax, x + 0.012, y + height - 0.020, f"{spec.window_months}-mo windows · {spec.step_months}-mo step", size=6.2, color=N1, weight="bold", ha="left")
    _label(ax, x + width - 0.012, y + height - 0.020, f"gold: central {spec.central_months} mo", size=5.2, color=DECISION, ha="right")
    axis_left, axis_right, axis_y = x + 0.020, x + width - 0.018, y + 0.040
    ax.plot([axis_left, axis_right], [axis_y, axis_y], color=MUTED, linewidth=0.65, transform=ax.transAxes, zorder=4)
    for frac, label, alignment in ((0.0, "2016", "left"), (0.86, "2025", "center"), (1.0, "2026", "right")):
        xx = axis_left + frac * (axis_right - axis_left)
        ax.plot([xx, xx], [axis_y - 0.004, axis_y + 0.004], color=MUTED, linewidth=0.55, transform=ax.transAxes, zorder=4)
        _label(ax, xx, axis_y - 0.014, label, size=5.3, color=MUTED, ha=alignment, va="top")
    total = axis_right - axis_left
    window_width = total * spec.window_months / 126.0
    step_width = total * spec.step_months / 126.0
    central_width = total * spec.central_months / 126.0
    bar_height = 0.018
    for index, start in enumerate((0.04, 0.04 + step_width, 0.04 + 2 * step_width), start=1):
        bar_y = y + height - 0.052 - (index - 1) * 0.030
        ax.add_patch(Rectangle((axis_left + start, bar_y), window_width, bar_height, facecolor="#CDE8E4", edgecolor=N1, linewidth=0.65, transform=ax.transAxes, zorder=4))
        ax.add_patch(Rectangle((axis_left + start + step_width, bar_y), central_width, bar_height, facecolor=DECISION_LIGHT, edgecolor=DECISION, linewidth=0.55, transform=ax.transAxes, zorder=5))
        _label(ax, axis_left + start - 0.006, bar_y + bar_height / 2, f"W{index}", size=5.5, color=N1, weight="bold", ha="right")


def _draw_n1_panel(ax, spec):
    x0, width = 0.270, 0.370
    _box(ax, x0, 0.072, width, 0.864, face=N1_LIGHT, edge="#A7CCC8", linewidth=0.9)
    _panel_header(ax, x0 + 0.012, 0.908, "b", "Frozen N1 Windowed CUSUM")
    _draw_window_timeline(ax, x0 + 0.022, 0.700, width - 0.044, 0.165, spec)
    center_x = x0 + width / 2
    pipeline = [
        (0.590, 0.068, "CUSUM amplitude Q and split month t*", N1),
        (0.472, 0.074, "Central-interval admission  |  negative-shift constraint", DECISION),
        (0.338, 0.086, f"Finite-sample permutation screening\np ≤ {spec.permutation_alpha:.3f} · {spec.permutation_replicates:,} permutations\n{spec.near_threshold_replicates:,} near the threshold", EVAL),
        (0.222, 0.070, f"Temporal–track consolidation\n±{spec.track_fusion_tolerance_months} month · greatest track support", N1),
    ]
    for index, (y, height, text, edge) in enumerate(pipeline):
        _box(ax, x0 + 0.052, y, width - 0.104, height, face=WHITE, edge=edge, linewidth=0.9)
        _label(ax, center_x, y + height / 2, text, size=6.2 if index != 2 else 5.8, weight="bold" if index == 0 else "normal")
        if index == 0:
            _arrow(ax, (center_x, 0.692), (center_x, y + height + 0.008), color=N1)
        if index < len(pipeline) - 1:
            next_y, next_h = pipeline[index + 1][0], pipeline[index + 1][1]
            _arrow(ax, (center_x, y - 0.007), (center_x, next_y + next_h + 0.007), color=N1)
    _arrow(ax, (center_x, 0.215), (center_x, 0.183), color=N1)
    _box(ax, x0 + 0.074, 0.113, width - 0.148, 0.061, face=N1, edge=N1, linewidth=0.9)
    _label(ax, center_x, 0.144, "Polygon-level candidate months", size=6.6, color=WHITE, weight="bold")
    _label(ax, center_x, 0.090, f"Minimum {spec.minimum_observations} finite observations per window", size=5.5, color=MUTED)


def _draw_eval_box(ax, x, y, width, height, spec):
    _box(ax, x, y, width, height, face=EVAL_LIGHT, edge=EVAL, linewidth=0.9, linestyle=(0, (3, 2)))
    _label(ax, x + 0.014, y + height - 0.022, "Historical spatial evaluation", size=6.7, color=EVAL, weight="bold", ha="left")
    cx = x + width / 2
    rows = [
        (y + height - 0.070, f"{spec.reference_events:,} reference-dated events · {spec.reference_polygons} polygons"),
        (y + height - 0.125, f"{spec.spatial_blocks} complete 6-km blocks · {spec.outer_folds} outer folds"),
        (y + height - 0.180, f"One-to-one matching · ±{spec.match_tolerance_months} months"),
        (y + 0.035, f"Precision · Recall · F1 · MAE\n{spec.spatial_bootstrap_replicates:,} paired block-bootstrap resamples"),
    ]
    for index, (yy, text) in enumerate(rows):
        _label(ax, cx, yy, text, size=5.65 if index < 3 else 5.55, weight="bold" if index == 0 else "normal")
        if index < len(rows) - 1:
            _arrow(ax, (cx, yy - 0.018), (cx, rows[index + 1][0] + 0.018), color=EVAL, linewidth=0.8, linestyle=(0, (3, 2)), mutation_scale=6.5)


def _draw_application_box(ax, x, y, width, height, spec):
    _box(ax, x, y, width, height, face=APPLICATION_LIGHT, edge=APPLICATION, linewidth=0.9)
    _label(ax, x + 0.014, y + height - 0.022, "National application", size=6.7, color=APPLICATION, weight="bold", ha="left")
    cx = x + width / 2
    rows = [
        (y + height - 0.074, f"Frozen N1 · {spec.validated_polygons:,} validated polygons"),
        (y + height - 0.137, "VV primary event histories\nVH support within ±1 month"),
        (y + height - 0.211, "Adjacent-event intervals\nannual candidate footprint"),
        (y + 0.035, "Prospective replay through June 2026\ncandidate-conditioned review"),
    ]
    for index, (yy, text) in enumerate(rows):
        _label(ax, cx, yy, text, size=5.7 if index != 3 else 5.35, weight="bold" if index == 0 else "normal")
        if index < len(rows) - 1:
            _arrow(ax, (cx, yy - 0.021), (cx, rows[index + 1][0] + 0.021), color=APPLICATION, linewidth=0.8, mutation_scale=6.5)
    _label(ax, x + 0.020, y + height - 0.137, "VV", size=5.6, color=VV, weight="bold", ha="left")
    _label(ax, x + width - 0.020, y + height - 0.137, "VH", size=5.6, color=VH, weight="bold", ha="right")


def _draw_branch_panel(ax, spec):
    x0, width = 0.655, 0.329
    _box(ax, x0, 0.072, width, 0.864, face="#F8F9FA", edge="#CFD5D8", linewidth=0.8)
    _panel_header(ax, x0 + 0.012, 0.908, "c", "Evaluation and application")
    _box(ax, x0 + 0.086, 0.820, width - 0.172, 0.045, face=N1, edge=N1, linewidth=0.8)
    _label(ax, x0 + width / 2, 0.843, "Frozen N1 outputs", size=6.2, color=WHITE, weight="bold")
    eval_x, eval_y, eval_w, eval_h = x0 + 0.020, 0.515, width - 0.040, 0.270
    app_x, app_y, app_w, app_h = x0 + 0.020, 0.155, width - 0.040, 0.315
    _arrow(ax, (x0 + width / 2, 0.815), (eval_x + eval_w / 2, eval_y + eval_h + 0.005), color=EVAL, linestyle=(0, (3, 2)))
    output_right = x0 + width - 0.086
    route_x = x0 + width - 0.009
    route_y = app_y + app_h + 0.015
    ax.plot([output_right, route_x], [0.843, 0.843], color=APPLICATION, linewidth=1.0, transform=ax.transAxes, zorder=4)
    ax.plot([route_x, route_x], [0.843, route_y], color=APPLICATION, linewidth=1.0, transform=ax.transAxes, zorder=4)
    _arrow(ax, (route_x, route_y), (app_x + app_w - 0.022, app_y + app_h + 0.005), color=APPLICATION, connectionstyle="arc3,rad=0.08")
    _draw_eval_box(ax, eval_x, eval_y, eval_w, eval_h, spec)
    _draw_application_box(ax, app_x, app_y, app_w, app_h, spec)
    _label(ax, x0 + width / 2, 0.105, "Mapped events are disturbance candidates;\ncausal harvest attribution is not assumed.", size=5.65, color=MUTED, weight="bold")


def build_figure(spec: FigureSpec):
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    _draw_data_panel(ax, spec)
    _draw_n1_panel(ax, spec)
    _draw_branch_panel(ax, spec)
    _arrow(ax, (0.255, 0.505), (0.269, 0.505), color=DATA, linewidth=1.2, mutation_scale=9)
    _arrow(ax, (0.640, 0.505), (0.654, 0.505), color=N1, linewidth=1.2, mutation_scale=9)
    return fig


def _write_manifest(output_dir, spec, inputs, output_paths):
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_contract": {
            "core_conclusion": "Orbit-specific monthly Sentinel-1 observations are converted to polygon-level trajectories; frozen N1 detects recurrent disturbance candidates; historical evaluation remains separate from national application.",
            "archetype": "asymmetric schematic-led workflow",
            "reuse_level": "style-only inheritance; scientific logic rebuilt from frozen N1 manifests",
            "final_size_mm": [178, 112],
            "backend": "Python/matplotlib only",
        },
        "spec": asdict(spec),
        "inputs": {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for key, path in inputs.items()},
        "outputs": {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for key, path in output_paths.items()},
        "scope": {"manuscript_modified": False, "archived_figures_overwritten": False},
        "software": {"python": sys.version, "platform": platform.platform(), "matplotlib": matplotlib.__version__},
    }
    manifest_path = output_dir / "workflow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec, inputs = load_and_validate_spec(root)
    fig = build_figure(spec)
    prefix = output_dir / "windowed_cusum_workflow_n1"
    svg_path = prefix.with_suffix(".svg")
    pdf_path = prefix.with_suffix(".pdf")
    png_path = prefix.with_suffix(".png")
    tiff_path = prefix.with_suffix(".tiff")
    output_paths = {"svg": svg_path, "pdf": pdf_path, "png": png_path, "tiff": tiff_path}
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    forbidden = (
        "bootstrap-based significance verification",
        "validated clear-cutting events",
        "pi >= tc",
        "national rotation map",
        "clear-cutting detection",
    )
    svg_text = output_paths["svg"].read_text(encoding="utf-8").lower()
    violations = [term for term in forbidden if term in svg_text]
    if violations:
        raise AssertionError(f"Forbidden legacy terms in SVG: {violations}")
    manifest_path = _write_manifest(output_dir, spec, inputs, output_paths)
    print(json.dumps({"outputs": {key: str(path) for key, path in output_paths.items()}, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
