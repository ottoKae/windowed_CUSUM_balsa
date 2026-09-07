#!/usr/bin/env python3
"""Freeze the nationwide N1 Pixel-first result at the validated support cutoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lr_cusum.component_evaluation import (  # noqa: E402
    common_index_paired_block_bootstrap,
    evaluate_event_stages,
)
from lr_cusum.polygon_workflow import (  # noqa: E402
    annotate_vv_with_vh,
    fuse_tracks,
)


ROUTE_LABELS = {
    "polygon_n1": "Polygon-first N1",
    "pixel_n1": "Pixel-first N1 connected r=0.20",
    "pixel_l0": "Pixel-first old L0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_references(data_root: Path) -> pd.DataFrame:
    references = pd.read_parquet(data_root / "references/reference_ground_events.parquet")
    references["spatial_block_id"] = references["frozen_spatial_block_id"]
    folds = pd.read_csv(data_root / "splits/outer_folds.csv")
    references = references.drop(columns="outer_fold", errors="ignore").merge(
        folds[["spatial_block_id", "outer_fold"]],
        on="spatial_block_id",
        how="left",
        validate="many_to_one",
    )
    if references.outer_fold.isna().any():
        raise ValueError("outer-fold map does not cover every reference event")
    return references


def one_to_one_matches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    tolerance_months: int = 1,
) -> pd.DataFrame:
    """Match two event routes by polygon and nearest month without reuse."""
    rows: list[dict[str, object]] = []
    sample_ids = sorted(set(left.sample_id.astype(str)) | set(right.sample_id.astype(str)))
    for sample_id in sample_ids:
        left_local = left.loc[left.sample_id.astype(str).eq(sample_id)].sort_values("change_time")
        right_local = right.loc[right.sample_id.astype(str).eq(sample_id)].sort_values("change_time")
        left_months = pd.to_datetime(left_local.change_time).dt.year * 12 + pd.to_datetime(
            left_local.change_time
        ).dt.month
        right_months = pd.to_datetime(right_local.change_time).dt.year * 12 + pd.to_datetime(
            right_local.change_time
        ).dt.month
        candidates = sorted(
            (abs(int(lm - rm)), li, ri)
            for li, lm in enumerate(left_months)
            for ri, rm in enumerate(right_months)
            if abs(int(lm - rm)) <= tolerance_months
        )
        used_left: set[int] = set()
        used_right: set[int] = set()
        for distance, left_index, right_index in candidates:
            if left_index in used_left or right_index in used_right:
                continue
            used_left.add(left_index)
            used_right.add(right_index)
            left_event = left_local.iloc[left_index]
            right_event = right_local.iloc[right_index]
            rows.append(
                {
                    "comparison": f"{left_name} vs {right_name}",
                    "sample_id": sample_id,
                    "polygon_id": left_event.get("polygon_id", right_event.get("polygon_id")),
                    "left_route": left_name,
                    "right_route": right_name,
                    "left_time": left_event.change_time,
                    "right_time": right_event.change_time,
                    "absolute_delta_months": int(distance),
                }
            )
    return pd.DataFrame(rows)


def route_summary(routes: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed_metrics = metrics.set_index("stage")
    for key, events in routes.items():
        label = ROUTE_LABELS[key]
        metric = indexed_metrics.loc[label]
        rows.append(
            {
                "route": label,
                "protocol": "N1" if key != "pixel_l0" else "L0",
                "vv_events_national": len(events),
                "polygons_with_event": events.sample_id.nunique(),
                "vh_supported_events": int(events.get("vh_supported", False).sum()),
                "reference_events": int(metric.reference_events),
                "reference_subset_detections": int(metric.detected_events),
                "tp": int(metric.tp),
                "fp": int(metric.fp),
                "fn": int(metric.fn),
                "precision": float(metric.precision),
                "recall": float(metric.recall),
                "f1": float(metric.f1),
                "temporal_mae_months": float(metric.temporal_mae_months),
            }
        )
    return pd.DataFrame(rows)


def report_markdown(
    threshold: float,
    raw_track_count: int,
    filtered_track_count: int,
    summary: pd.DataFrame,
    overlap: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> str:
    route_lines = [
        "| Route | National VV events | Polygons | Precision | Recall | F1 | MAE (months) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        route_lines.append(
            f"| {row.route} | {row.vv_events_national:,} | {row.polygons_with_event:,} | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | "
            f"{row.temporal_mae_months:.3f} |"
        )
    overlap_lines = [
        "| Comparison (+/-1 month) | Matches | Left matched | Right matched | Exact-month matches |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overlap.itertuples(index=False):
        overlap_lines.append(
            f"| {row.comparison} | {row.matches:,} | {row.left_match_fraction:.3f} | "
            f"{row.right_match_fraction:.3f} | {row.exact_month_fraction:.3f} |"
        )
    f1 = contrasts.loc[contrasts.metric.eq("f1")]
    contrast_lines = [
        "| Paired spatial-block contrast | Delta F1 | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in f1.itertuples(index=False):
        contrast_lines.append(
            f"| {row.contrast} | {row.estimate:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |"
        )
    return "\n".join(
        [
            "# Nationwide Pixel-first N1 result",
            "",
            f"The connected-support cutoff was frozen at `r={threshold:.2f}` because every outer fold selected it.",
            "The cutoff was applied to polygon-track candidates before cross-track fusion. Unscreened candidates remain unchanged in the parent Pixel-first directory.",
            "",
            f"This retained {filtered_track_count:,} of {raw_track_count:,} polygon-track candidates ({filtered_track_count / raw_track_count:.1%}).",
            "",
            "## Route comparison",
            "",
            *route_lines,
            "",
            "Validation metrics use the frozen 1,093 reference events, 462 polygons, 57 spatial blocks, five outer folds, optimal one-to-one matching, and +/-2 months.",
            "",
            "## Event overlap",
            "",
            *overlap_lines,
            "",
            "## Paired validation contrasts",
            "",
            *contrast_lines,
            "",
            "The formal Pixel-first N1 route is therefore a spatially validated alternative representation, not a replacement for Polygon-first N1. The old L0 result is retained only as a historical comparison.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")

    run_root = ROOT / "runs/windowed_cusum_balsa_paper_v1"
    pixel_root = run_root / "pixel-first"
    output = pixel_root / f"formal_n1_connected_r{args.threshold:.2f}"
    data_root = ROOT / "data/local/paper_v1"
    selection_path = ROOT / (
        "runs/numeric_refresh_20260905/representations_n1_48_12/"
        "connected.nested_selections.csv"
    )
    raw_track_path = pixel_root / "track_events.parquet"
    polygon_path = run_root / "polygon-first/vv_primary.parquet"
    legacy_path = data_root / "frozen_results/national/ground_polygon_vv_primary.parquet"
    detector_config_path = ROOT / "configs/paper_v1.json"
    raw_receipt_path = pixel_root / "COMPLETE.json"
    cache_receipt_path = run_root / "polygon_pixels/COMPLETE.json"

    selections = pd.read_csv(selection_path)
    expected = f"connected_r{args.threshold:g}"
    if len(selections) != 5 or selections.outer_fold.nunique() != 5:
        raise ValueError("connected-support selection must cover five outer folds")
    if set(selections.selected_name) != {expected}:
        raise ValueError(f"the frozen folds did not unanimously select {expected}")

    raw_track = pd.read_parquet(raw_track_path)
    required = {"connected_support_fraction", "track_key", "sample_id", "Band", "change_time"}
    if missing := required - set(raw_track.columns):
        raise ValueError(f"track candidate table is missing: {sorted(missing)}")
    filtered_track = raw_track.loc[
        raw_track.connected_support_fraction.ge(args.threshold)
    ].copy()
    ground = fuse_tracks(filtered_track, tolerance_months=1)
    pixel_n1, pixel_n1_supported = annotate_vv_with_vh(ground, tolerance_months=2)
    atomic_parquet(filtered_track, output / "track_events.parquet")
    atomic_parquet(ground, output / "ground_events.parquet")
    atomic_parquet(pixel_n1, output / "vv_primary.parquet")
    atomic_parquet(pixel_n1_supported, output / "vv_vh_supported.parquet")

    recomputed_polygon_n1 = pd.read_parquet(polygon_path)
    legacy = pd.read_parquet(legacy_path)
    polygon_n1 = legacy.loc[legacy.route.eq("polygon_first")].copy()
    pixel_l0 = legacy.loc[legacy.route.eq("pixel_first")].copy()
    event_key = ["sample_id", "change_time"]
    recomputed_keys = set(map(tuple, recomputed_polygon_n1[event_key].astype(str).to_numpy()))
    frozen_keys = set(map(tuple, polygon_n1[event_key].astype(str).to_numpy()))
    if recomputed_keys != frozen_keys:
        raise ValueError("recomputed and frozen Polygon-first N1 chronologies differ")
    routes = {"polygon_n1": polygon_n1, "pixel_n1": pixel_n1, "pixel_l0": pixel_l0}

    references = load_references(data_root)
    reference_ids = set(references.sample_id.astype(str))
    evaluation_routes = {
        ROUTE_LABELS[key]: events.loc[events.sample_id.astype(str).isin(reference_ids)]
        for key, events in routes.items()
    }
    evaluation = evaluate_event_stages(
        references,
        evaluation_routes,
        tolerances=(2,),
        fold_column="outer_fold",
    )
    contrasts = {
        "Pixel-first N1 minus Polygon-first N1": (
            ROUTE_LABELS["pixel_n1"], ROUTE_LABELS["polygon_n1"]
        ),
        "Pixel-first N1 minus Pixel-first old L0": (
            ROUTE_LABELS["pixel_n1"], ROUTE_LABELS["pixel_l0"]
        ),
    }
    bootstrap = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=2,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary = route_summary(routes, evaluation.stage_metrics)
    atomic_csv(summary, output / "route_summary.csv")
    atomic_csv(evaluation.stage_metrics, output / "validation_metrics.csv")
    atomic_csv(evaluation.fold_metrics, output / "validation_fold_metrics.csv")
    atomic_parquet(evaluation.block_metrics, output / "validation_block_metrics.parquet")
    atomic_csv(bootstrap.stage_summary, output / "spatial_bootstrap_stage_summary.csv")
    atomic_csv(bootstrap.contrast_summary, output / "paired_route_contrasts.csv")
    atomic_parquet(bootstrap.stage_replicates, output / "spatial_bootstrap_stage_replicates.parquet")
    atomic_parquet(bootstrap.contrast_replicates, output / "paired_route_contrast_replicates.parquet")

    annual = pd.concat(
        [
            events.assign(route=ROUTE_LABELS[key])
            for key, events in routes.items()
        ],
        ignore_index=True,
    )
    annual["year"] = pd.to_datetime(annual.change_time).dt.year
    annual = annual.groupby(["route", "year"]).size().rename("vv_events").reset_index()
    atomic_csv(annual, output / "annual_vv_event_counts.csv")

    match_frames = [
        one_to_one_matches(
            pixel_n1,
            polygon_n1,
            left_name=ROUTE_LABELS["pixel_n1"],
            right_name=ROUTE_LABELS["polygon_n1"],
        ),
        one_to_one_matches(
            pixel_n1,
            pixel_l0,
            left_name=ROUTE_LABELS["pixel_n1"],
            right_name=ROUTE_LABELS["pixel_l0"],
        ),
    ]
    matches = pd.concat(match_frames, ignore_index=True)
    atomic_parquet(matches, output / "pairwise_event_matches.parquet")
    overlap_rows = []
    pair_specs = [
        (ROUTE_LABELS["pixel_n1"], ROUTE_LABELS["polygon_n1"], pixel_n1, polygon_n1),
        (ROUTE_LABELS["pixel_n1"], ROUTE_LABELS["pixel_l0"], pixel_n1, pixel_l0),
    ]
    for left_name, right_name, left, right in pair_specs:
        label = f"{left_name} vs {right_name}"
        local = matches.loc[matches.comparison.eq(label)]
        overlap_rows.append(
            {
                "comparison": label,
                "tolerance_months": 1,
                "matches": len(local),
                "left_events": len(left),
                "right_events": len(right),
                "left_match_fraction": len(local) / max(1, len(left)),
                "right_match_fraction": len(local) / max(1, len(right)),
                "exact_month_fraction": float(local.absolute_delta_months.eq(0).mean()),
                "median_absolute_delta_months": float(local.absolute_delta_months.median()),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    atomic_csv(overlap, output / "pairwise_event_overlap.csv")

    report = report_markdown(
        args.threshold,
        len(raw_track),
        len(filtered_track),
        summary,
        overlap,
        bootstrap.contrast_summary,
    )
    atomic_text(report, output / "REPORT.md")

    output_files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "COMPLETE.json")
    receipt = {
        "status": "complete",
        "route": "pixel-first",
        "protocol": "N1",
        "connected_support_threshold": args.threshold,
        "threshold_application_level": "polygon-track candidates before cross-track fusion",
        "same_band_cross_track_tolerance_months": 1,
        "vv_vh_support_tolerance_months": 2,
        "validation_match_tolerance_months": 2,
        "selection": {
            "path": selection_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(selection_path),
            "folds": int(selections.outer_fold.nunique()),
            "selected_name": expected,
        },
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for path in (
                raw_track_path,
                raw_receipt_path,
                cache_receipt_path,
                polygon_path,
                legacy_path,
                detector_config_path,
            )
        ],
        "counts": {
            "raw_polygon_track_candidates": len(raw_track),
            "accepted_polygon_track_events": len(filtered_track),
            "ground_band_events": len(ground),
            "vv_primary_events": len(pixel_n1),
            "vv_vh_supported_events": len(pixel_n1_supported),
            "polygons_with_vv_event": int(pixel_n1.sample_id.nunique()),
        },
        "outputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        ],
    }
    atomic_text(json.dumps(receipt, indent=2), output / "COMPLETE.json")
    print(report)
    print(f"output={output}")


if __name__ == "__main__":
    main()
