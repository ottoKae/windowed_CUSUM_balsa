#!/usr/bin/env python3
"""Recompute the manuscript numeric refresh on one shared validation protocol.

This command performs the three deliberately narrow repairs requested for the
paper release:

1. verify the 1,093-event / 462-polygon reference inventory;
2. rerun all seven stand representations with the frozen N1 detector at 48/12;
3. recompute benchmark point estimates and paired intervals from the same OOF
   event tables and the same spatial-block bootstrap draws.

Large pixel-first work is cached one track at a time under ``--run-dir``.
Rerunning the command therefore resumes rather than restarting the N1 scan.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lr_cusum.component_evaluation import (  # noqa: E402
    common_index_paired_block_bootstrap,
    evaluate_event_stages,
)
from lr_cusum.paper import detect_series_n1, load_paper_config  # noqa: E402
from lr_cusum.polygon_workflow import (  # noqa: E402
    fuse_pixel_events,
    fuse_tracks,
    load_track_cache,
)
from lr_cusum.spatial_fusion import (  # noqa: E402
    space_time_dbscan_fusion,
    temporal_dbscan_fusion,
)


REPRESENTATIONS = ("median", "mean", "mode", "center")
DISPLAY_NAMES = {
    "median": "Polygon-first median",
    "mean": "Polygon-first mean",
    "mode": "Polygon-first mode",
    "center": "Single-cell spatial medoid",
    "connected": "Pixel-first connected support",
    "temporal_dbscan": "Pixel-first temporal DBSCAN",
    "space_time_dbscan": "Pixel-first space--time DBSCAN",
}
BENCHMARK_FILES = {
    "ReCuSum, decrease-filtered": "recusum_decrease.nested_oof_events.parquet",
    "ReCuSum, raw": "recusum_raw.nested_oof_events.parquet",
    "f-pol BOCD, decrease-filtered": "fpol_bocd_decrease.nested_oof_events.parquet",
    "BFAST-style, decrease-filtered": "bfast_decrease.nested_oof_events.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/local/paper_v1")
    parser.add_argument("--results", type=Path, default=ROOT / "results/formal_validation")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/numeric_refresh_20260905")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_v1.json")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--legacy-comparison-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--legacy-closure-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def histogram_mode(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    lower = max(np.floor(values.min()) - 1, -35)
    upper = min(np.ceil(values.max()) + 1, 10)
    step = max((upper - lower) / 500, 0.005)
    bins = np.arange(lower, upper + step, step)
    if len(bins) < 2:
        return float(values[0])
    counts, edges = np.histogram(values, bins=bins)
    index = int(np.argmax(counts))
    return float((edges[index] + edges[index + 1]) / 2)


def build_representations(
    associations: pd.DataFrame,
    polygon_tracks: pd.DataFrame,
    cache_dir: Path,
    run_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    metadata_path = run_dir / "representation_metadata.parquet"
    array_path = run_dir / "representations.npz"
    if metadata_path.exists() and array_path.exists():
        archive = np.load(array_path)
        return (
            pd.read_parquet(metadata_path),
            archive["values"],
            pd.DatetimeIndex(archive["calendar"]),
        )

    lookup = polygon_tracks.set_index("polygon_track_id")
    metadata: list[dict[str, object]] = []
    output: list[np.ndarray] = []
    calendar: pd.DatetimeIndex | None = None
    for track_key, local in associations.groupby("track_key", sort=True):
        pixels, arrays = load_track_cache(track_key, cache_dir)
        local_calendar = pd.DatetimeIndex(arrays["calendar"])
        if calendar is None:
            calendar = local_calendar
        elif not calendar.equals(local_calendar):
            raise ValueError("all reference caches must share the same explicit calendar")
        mapping = pixels.set_index("pixel_track_id").cache_index
        for polygon_track_id, group in local.groupby("polygon_track_id", sort=False):
            indexes = mapping.loc[group.pixel_track_id].to_numpy(int)
            data = arrays["values"][indexes].astype(float)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                median = np.nanmedian(data, axis=0)
                mean = np.nanmean(data, axis=0)
            mode = np.full_like(median, np.nan)
            for band_index in range(data.shape[1]):
                for month_index in range(data.shape[2]):
                    mode[band_index, month_index] = histogram_mode(
                        data[:, band_index, month_index]
                    )
            rows = group.native_row.to_numpy(float)
            columns = group.native_col.to_numpy(float)
            center_index = int(
                np.argmin((rows - rows.mean()) ** 2 + (columns - columns.mean()) ** 2)
            )
            output.append(np.stack((median, mean, mode, data[center_index])).astype(np.float32))
            row = lookup.loc[polygon_track_id]
            metadata.append(
                {
                    "series_index": len(output) - 1,
                    "polygon_track_id": polygon_track_id,
                    "sample_id": row.sample_id,
                    "polygon_id": row.polygon_id,
                    "role": row.role,
                    "species_nco": row.species_nco,
                    "tile_id": row.tile_id,
                    "track_key": row.track_key,
                    "interior_pixels": len(indexes),
                    "center_pixel_track_id": group.iloc[center_index].pixel_track_id,
                }
            )
    if calendar is None:
        raise ValueError("no reference trajectories were found")
    frame = pd.DataFrame(metadata)
    values = np.stack(output)
    atomic_parquet(frame, metadata_path)
    temporary = Path(str(array_path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            values=values,
            calendar=calendar.to_numpy(dtype="datetime64[ns]"),
            representation=np.asarray(REPRESENTATIONS),
        )
    os.replace(temporary, array_path)
    return frame, values, calendar


def _event_metadata(item, event: dict[str, object]) -> dict[str, object]:
    return {
        "polygon_track_id": item.polygon_track_id,
        "sample_id": item.sample_id,
        "polygon_id": item.polygon_id,
        "role": item.role,
        "species_nco": item.species_nco,
        "tile_id": item.tile_id,
        "track_key": item.track_key,
        **event,
    }


def detect_representation_worker(
    name: str,
    metadata_path: str,
    array_path: str,
    config_path: str,
    output_path: str,
) -> tuple[str, int, float]:
    started = time.perf_counter()
    output = Path(output_path)
    if output.exists():
        return name, len(pd.read_parquet(output)), time.perf_counter() - started
    metadata = pd.read_parquet(metadata_path)
    archive = np.load(array_path)
    values = archive["values"]
    dates = pd.DatetimeIndex(archive["calendar"])
    representation_index = REPRESENTATIONS.index(name)
    config = replace(load_paper_config(config_path), bands=("VV_dB",))
    rows: list[dict[str, object]] = []
    for item in metadata.itertuples(index=False):
        detected = detect_series_n1(
            values[int(item.series_index), representation_index, 0],
            dates,
            identifier=item.polygon_track_id,
            band="VV_dB",
            config=config,
        )
        rows.extend(_event_metadata(item, event) for event in detected.to_dict("records"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["sample_id", "Band", "change_time"])
    atomic_parquet(frame, output)
    return name, len(frame), time.perf_counter() - started


def detect_pixel_track_worker(
    track_key: str,
    used_pixel_ids: tuple[str, ...],
    cache_dir: str,
    config_path: str,
    output_path: str,
) -> tuple[str, int, int, float]:
    started = time.perf_counter()
    output = Path(output_path)
    if output.exists():
        cached = pd.read_parquet(output)
        return track_key, len(used_pixel_ids), len(cached), time.perf_counter() - started
    pixels, arrays = load_track_cache(track_key, cache_dir)
    selected = pixels.loc[pixels.pixel_track_id.isin(set(used_pixel_ids))]
    dates = pd.DatetimeIndex(arrays["calendar"])
    config = replace(load_paper_config(config_path), bands=("VV_dB",))
    rows: list[dict[str, object]] = []
    for pixel in selected.itertuples(index=False):
        detected = detect_series_n1(
            arrays["values"][int(pixel.cache_index), 0],
            dates,
            identifier=pixel.pixel_track_id,
            band="VV_dB",
            config=config,
        )
        for event in detected.to_dict("records"):
            rows.append(
                {
                    "pixel_track_id": pixel.pixel_track_id,
                    "track_key": track_key,
                    "native_row": int(pixel.native_row),
                    "native_col": int(pixel.native_col),
                    **event,
                    "selection_score": 1.0 - float(event["null_p_value"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=["pixel_track_id", "track_key", "native_row", "native_col", "Band", "change_time"]
        )
    atomic_parquet(frame, output)
    return track_key, len(selected), len(frame), time.perf_counter() - started


def ground_events(track_events: pd.DataFrame) -> pd.DataFrame:
    if track_events.empty:
        return pd.DataFrame(columns=["sample_id", "Band", "change_time"])
    return fuse_tracks(track_events, tolerance_months=1).loc[lambda x: x.Band.eq("VV_dB")]


def nested_support_selection(
    candidates: dict[str, pd.DataFrame],
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    for outer_fold in sorted(references.outer_fold.unique()):
        training = references.loc[references.outer_fold.ne(outer_fold)]
        held_out = references.loc[references.outer_fold.eq(outer_fold)]
        allowed_train = set(training.sample_id.astype(str))
        scored = []
        for name, events in candidates.items():
            evaluation = evaluate_event_stages(
                training,
                {name: events.loc[events.sample_id.astype(str).isin(allowed_train)]},
                tolerances=(2,),
            )
            metrics = evaluation.stage_metrics.iloc[0]
            scored.append((metrics.f1, metrics.precision, name))
        _, _, selected_name = max(scored, key=lambda item: (item[0], item[1], item[2]))
        allowed_test = set(held_out.sample_id.astype(str))
        selected = candidates[selected_name].loc[
            candidates[selected_name].sample_id.astype(str).isin(allowed_test)
        ].copy()
        selected["outer_fold"] = int(outer_fold)
        selected["selected_name"] = selected_name
        outputs.append(selected)
        selections.append({"outer_fold": int(outer_fold), "selected_name": selected_name})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(selections)


def add_stage_intervals(
    metrics: pd.DataFrame,
    stage_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = metrics.copy()
    for metric in ("precision", "recall", "f1"):
        intervals = stage_summary.loc[
            stage_summary.metric.eq(metric), ["stage", "ci_low", "ci_high"]
        ].rename(columns={"ci_low": f"{metric}_ci_low", "ci_high": f"{metric}_ci_high"})
        output = output.merge(intervals, on="stage", how="left", validate="one_to_one")
    return output


def run_representation_analysis(
    args: argparse.Namespace,
    references: pd.DataFrame,
    associations: pd.DataFrame,
    polygon_tracks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = args.run_dir / "representations_n1_48_12"
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = args.data_root / "polygon_pixels/cache"
    metadata, _, _ = build_representations(
        associations, polygon_tracks, cache_dir, destination
    )
    metadata_path = destination / "representation_metadata.parquet"
    array_path = destination / "representations.npz"

    tasks = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(REPRESENTATIONS))) as executor:
        for name in REPRESENTATIONS:
            output = destination / "track_events" / f"polygon_first_{name}.parquet"
            tasks.append(
                executor.submit(
                    detect_representation_worker,
                    name,
                    str(metadata_path),
                    str(array_path),
                    str(args.config),
                    str(output),
                )
            )
        for task in as_completed(tasks):
            name, events, elapsed = task.result()
            print(f"polygon representation {name}: {events:,} track events ({elapsed:.1f}s)", flush=True)

    representation_events: dict[str, pd.DataFrame] = {}
    for name in REPRESENTATIONS:
        track = pd.read_parquet(destination / "track_events" / f"polygon_first_{name}.parquet")
        ground = ground_events(track)
        atomic_parquet(ground, destination / "ground_events" / f"polygon_first_{name}.parquet")
        representation_events[DISPLAY_NAMES[name]] = ground

    pixel_dir = destination / "pixel_events_by_track"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    with ProcessPoolExecutor(max_workers=min(args.workers, associations.track_key.nunique())) as executor:
        for track_key, local in associations.groupby("track_key", sort=True):
            tasks.append(
                executor.submit(
                    detect_pixel_track_worker,
                    str(track_key),
                    tuple(sorted(set(local.pixel_track_id.astype(str)))),
                    str(cache_dir),
                    str(args.config),
                    str(pixel_dir / f"{track_key}.parquet"),
                )
            )
        for task in as_completed(tasks):
            track, pixels, events, elapsed = task.result()
            print(f"pixel N1 {track}: {pixels:,} pixels, {events:,} events ({elapsed:.1f}s)", flush=True)
    pixel_events = pd.concat(
        [pd.read_parquet(path) for path in sorted(pixel_dir.glob("*.parquet"))],
        ignore_index=True,
    )
    if "selection_score" not in pixel_events and "null_p_value" in pixel_events:
        pixel_events["selection_score"] = 1.0 - pixel_events["null_p_value"].astype(float)
    atomic_parquet(pixel_events, destination / "pixel_events.parquet")

    connected_track = fuse_pixel_events(
        pixel_events, associations, polygon_tracks, cache_dir, temporal_tolerance=1
    )
    atomic_parquet(connected_track, destination / "track_events/pixel_connected_candidates.parquet")
    connected_candidates = {}
    for ratio in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        name = f"connected_r{ratio:g}"
        connected_candidates[name] = ground_events(
            connected_track.loc[connected_track.connected_support_fraction.ge(ratio)]
        )

    family_specs = {
        "connected": connected_candidates,
        "temporal_dbscan": {},
        "space_time_dbscan": {},
    }
    for family, function in (
        ("temporal_dbscan", temporal_dbscan_fusion),
        ("space_time_dbscan", space_time_dbscan_fusion),
    ):
        for ratio in (0.20, 0.30, 0.40, 0.50):
            name = f"{family}_r{ratio:g}"
            path = destination / "track_events" / f"{name}.parquet"
            if path.exists():
                track_events = pd.read_parquet(path)
            else:
                track_events = function(
                    pixel_events,
                    associations,
                    polygon_tracks,
                    minimum_pixel_ratio=ratio,
                )
                atomic_parquet(track_events, path)
            family_specs[family][name] = ground_events(track_events)

    for family, candidates in family_specs.items():
        selected, selections = nested_support_selection(candidates, references)
        atomic_parquet(selected, destination / "ground_events" / f"{family}.nested_oof.parquet")
        atomic_csv(selections, destination / f"{family}.nested_selections.csv")
        representation_events[DISPLAY_NAMES[family]] = selected

    evaluation = evaluate_event_stages(
        references,
        representation_events,
        tolerances=(2,),
        fold_column="outer_fold",
    )
    contrasts = {
        f"{name} minus Polygon-first median": (name, "Polygon-first median")
        for name in representation_events
        if name != "Polygon-first median"
    }
    bootstrap = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=2,
        replicates=5000,
        seed=20260829,
    )
    metrics = add_stage_intervals(evaluation.stage_metrics, bootstrap.stage_summary)
    atomic_csv(metrics, destination / "stand_representation_metrics.csv")
    atomic_csv(evaluation.fold_metrics, destination / "stand_representation_fold_metrics.csv")
    atomic_parquet(evaluation.block_metrics, destination / "stand_representation_block_metrics.parquet")
    atomic_csv(bootstrap.stage_summary, destination / "stand_representation_stage_intervals.csv")
    atomic_csv(bootstrap.contrast_summary, destination / "stand_representation_contrasts.csv")
    atomic_parquet(bootstrap.stage_replicates, destination / "stand_representation_stage_replicates.parquet")
    atomic_parquet(bootstrap.contrast_replicates, destination / "stand_representation_contrast_replicates.parquet")

    export = metrics.rename(columns={"stage": "label"}).copy()
    reverse = {value: key for key, value in DISPLAY_NAMES.items()}
    export.insert(0, "name", export.label.map(reverse).map({
        "median": "frozen_median", "mean": "frozen_mean", "mode": "frozen_mode",
        "center": "frozen_center", "connected": "connected_nested_n1",
        "temporal_dbscan": "temporal_dbscan_nested_n1",
        "space_time_dbscan": "space_time_dbscan_nested_n1",
    }))
    export.insert(1, "family", "spatial_strategy")
    export.insert(2, "protocol", "fixed_n1_w48_s12")
    export.insert(3, "variant", export.label)
    export.insert(4, "role", "all")
    export = export.drop(columns="label")
    atomic_csv(export, args.results / "stand_representation_metrics.csv")
    atomic_csv(bootstrap.contrast_summary, args.results / "stand_representation_contrasts.csv")
    return metrics, bootstrap.contrast_summary


def run_benchmark_analysis(
    args: argparse.Namespace,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = args.run_dir / "benchmark_shared_oof"
    windowed = pd.read_parquet(
        args.data_root / "frozen_results/geometry/nested_oof_events.parquet"
    )
    methods: dict[str, pd.DataFrame] = {"Windowed CUSUM": windowed}
    for label, filename in BENCHMARK_FILES.items():
        methods[label] = pd.read_parquet(args.legacy_comparison_root / filename)
    methods["Maximum monthly drop"] = pd.read_parquet(
        args.legacy_closure_root / "maximum_drop.nested_oof_events.parquet"
    )
    methods["Classical lower CUSUM"] = pd.read_parquet(
        args.legacy_closure_root / "classical_cusum.nested_oof_events.parquet"
    )
    evaluation = evaluate_event_stages(
        references,
        methods,
        tolerances=(2,),
        fold_column="outer_fold",
    )
    contrasts = {
        f"Windowed CUSUM minus {name}": ("Windowed CUSUM", name)
        for name in methods
        if name != "Windowed CUSUM"
    }
    bootstrap = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=2,
        replicates=5000,
        seed=20260829,
    )
    atomic_csv(evaluation.stage_metrics, destination / "method_metrics.csv")
    atomic_csv(evaluation.fold_metrics, destination / "method_fold_metrics.csv")
    atomic_parquet(evaluation.block_metrics, destination / "method_block_metrics.parquet")
    atomic_parquet(evaluation.matches, destination / "method_matches.parquet")
    atomic_csv(bootstrap.stage_summary, destination / "method_stage_intervals.csv")
    atomic_csv(bootstrap.contrast_summary, destination / "method_contrasts.csv")
    atomic_parquet(bootstrap.stage_replicates, destination / "method_stage_replicates.parquet")
    atomic_parquet(bootstrap.contrast_replicates, destination / "method_contrast_replicates.parquet")

    atomic_csv(evaluation.stage_metrics, args.results / "method_comparison_metrics.csv")
    atomic_csv(bootstrap.contrast_summary, args.results / "method_comparison_contrasts.csv")
    legacy_nested = pd.read_csv(args.results / "nested_spatial_oof_metrics.csv")
    windowed_row = evaluation.stage_metrics.loc[
        evaluation.stage_metrics.stage.eq("Windowed CUSUM")
    ].iloc[0]
    mask = legacy_nested.family.eq("lr_cusum")
    for column in (
        "tp", "fp", "fn", "precision", "recall", "f1", "temporal_mae_months",
        "median_signed_error_months", "reference_events", "detected_events", "objects",
    ):
        legacy_nested.loc[mask, column] = windowed_row[column]
    atomic_csv(legacy_nested, args.results / "nested_spatial_oof_metrics.csv")
    return evaluation.stage_metrics, bootstrap.contrast_summary


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    references = pd.read_parquet(args.data_root / "references/reference_ground_events.parquet")
    if len(references) != 1093 or references.sample_id.nunique() != 462:
        raise ValueError(
            "reference freeze mismatch: expected 1,093 events in 462 polygons, found "
            f"{len(references):,} events in {references.sample_id.nunique():,} polygons"
        )
    fold_map = pd.read_csv(args.legacy_closure_root / "locked_outer_spatial_folds.csv")
    references = references.merge(
        fold_map, on="spatial_block_id", how="left", validate="many_to_one"
    )
    if references.outer_fold.isna().any() or references.spatial_block_id.nunique() != 57:
        raise ValueError("the frozen 57-block outer-fold map is incomplete")
    reference_ids = set(references.sample_id.astype(str))
    associations = pd.read_parquet(
        args.data_root / "polygon_pixels/pixel_associations.parquet"
    )
    associations = associations.loc[associations.sample_id.astype(str).isin(reference_ids)]
    polygon_tracks = pd.read_parquet(
        args.data_root / "polygon_pixels/polygon_tracks.parquet"
    )
    polygon_tracks = polygon_tracks.loc[
        polygon_tracks.sample_id.astype(str).isin(reference_ids)
    ]

    started = time.perf_counter()
    representation_metrics, representation_contrasts = run_representation_analysis(
        args, references, associations, polygon_tracks
    )
    benchmark_metrics, benchmark_contrasts = run_benchmark_analysis(args, references)
    receipt = {
        "protocol": "N1 fixed 48/12 stand representation plus shared-OOF benchmark",
        "reference_events": len(references),
        "reference_polygons": references.sample_id.nunique(),
        "spatial_blocks": references.spatial_block_id.nunique(),
        "reference_polygon_tracks": len(polygon_tracks),
        "reference_pixel_tracks": associations.pixel_track_id.nunique(),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {},
    }
    for path in sorted(args.run_dir.rglob("*")):
        if path.is_file():
            receipt["outputs"][str(path.relative_to(args.run_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    receipt["stand_representation_metrics"] = representation_metrics.to_dict("records")
    receipt["stand_representation_contrasts"] = representation_contrasts.to_dict("records")
    receipt["benchmark_metrics"] = benchmark_metrics.to_dict("records")
    receipt["benchmark_contrasts"] = benchmark_contrasts.to_dict("records")
    temporary = args.run_dir / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, args.run_dir / "receipt.json")
    print(json.dumps({
        "elapsed_seconds": receipt["elapsed_seconds"],
        "representation_metrics": representation_metrics[
            ["stage", "precision", "recall", "f1", "temporal_mae_months"]
        ].to_dict("records"),
        "benchmark_windowed": benchmark_metrics.loc[
            benchmark_metrics.stage.eq("Windowed CUSUM")
        ].iloc[0].to_dict(),
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
