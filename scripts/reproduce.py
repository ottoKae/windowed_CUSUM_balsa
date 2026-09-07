#!/usr/bin/env python3
"""Prepare, audit, and run the authorized Windowed CUSUM data package.

The Git checkout stays small. ``prepare`` creates an ignored local package from
the frozen research archive; ``audit`` verifies the study-defining counts;
``detect`` replays either polygon-first or pixel-first N1 detection by track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lr_cusum.paper import load_paper_config  # noqa: E402
from lr_cusum.component_evaluation import (  # noqa: E402
    common_index_paired_block_bootstrap,
    evaluate_event_stages,
)
from lr_cusum.component_ablation import derive_variant_events  # noqa: E402
from lr_cusum.polygon_workflow import (  # noqa: E402
    annotate_vv_with_vh,
    detect_pixels,
    detect_polygon_first,
    detect_polygon_first_components,
    extract_track_cache,
    fuse_pixel_events,
    fuse_tracks,
)
from lr_cusum.rotation import lifecycle_summary, rotation_periods  # noqa: E402
from lr_cusum.s1grits_shards import convert_store, discover_stores  # noqa: E402


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["destination"] = (ROOT / config["destination"]).resolve()
    config["sources"] = {key: Path(value).resolve() for key, value in config["sources"].items()}
    return config


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat, destination_stat = source.stat(), destination.stat()
        if (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def copy_tree(source: Path, destination: Path, *, suffixes: tuple[str, ...] | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes):
            copy_file(path, destination / path.relative_to(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def prepare(config: dict) -> None:
    src = config["sources"]
    out = config["destination"]
    sample = src["sample_release"]

    files = {
        sample / "inventory/balsa_validated.parquet": out / "inventory/balsa_validated.parquet",
        sample / "inventory/balsa_excluded_pending.parquet": out / "inventory/balsa_excluded_pending.parquet",
        sample / "roles/balsa_polygon_roles.parquet": out / "splits/polygon_roles.parquet",
        src["references"] / "reference_ground_events_exact.parquet": out / "references/reference_ground_events.parquet",
        src["references"] / "reference_track_events_exact.parquet": out / "references/reference_track_events.parquet",
        src["benchmark_closure"] / "locked_outer_spatial_folds.csv": out / "splits/outer_folds.csv",
        src["s1grits_root"] / "catalog.parquet": out / "s1grits/catalog.parquet",
    }
    for source, destination in files.items():
        copy_file(source, destination)

    # GeoPackage-to-Parquet conversion avoids shipping duplicate geometry formats.
    blocks_out = out / "splits/spatial_blocks.parquet"
    if not blocks_out.exists():
        import geopandas as gpd

        blocks_out.parent.mkdir(parents=True, exist_ok=True)
        blocks = gpd.read_file(sample / "roles/spatial_blocks.gpkg")
        blocks.to_parquet(blocks_out, index=False)

    # These are large exact-location products, so they remain under ignored data/local.
    copy_tree(src["pixel_cache"], out / "polygon_pixels", suffixes=(".npz", ".parquet"))
    copy_tree(src["frozen_results"], out / "frozen_results")
    for name in ("global_candidates.parquet", "sliding_candidates.parquet"):
        copy_file(
            src["component_candidates"] / "candidate_audit" / name,
            out / "frozen_results/component_ablation/candidate_audit" / name,
        )

    freeze_benchmark_protocol(src, out)

    # Public-domain administrative context is copied; cached Esri imagery is local only.
    natural_earth = src["natural_earth_root"]
    for relative in (
        Path("cultural/ne_110m_admin_0_countries"),
        Path("cultural/ne_10m_admin_1_states_provinces_lines"),
    ):
        parent, stem = relative.parent, relative.name
        for source in (natural_earth / parent).glob(stem + ".*"):
            copy_file(source, out / "admin" / parent / source.name)
    copy_tree(src["esri_cache"], out / "basemap")

    write_json(
        {
            "kind": "external_s1grits_zarr_archive",
            "path": str(src["s1grits_root"]),
            "note": "Catalog is copied locally; Zarr stores remain mounted in place to avoid duplicating the multi-terabyte archive.",
        },
        out / "s1grits/SOURCE.local.json",
    )
    build_manifest(config)
    print(f"prepared={out}")


def freeze_benchmark_protocol(src: dict[str, Path], out: Path) -> None:
    """Localize the exact fold selections and OOF event sources used by the paper."""

    destination = out / "frozen_protocol"
    comparison = src["benchmark_comparison"]
    closure = src["benchmark_closure"]
    sources = {
        "windowed_cusum": (
            out / "frozen_results/geometry/nested_selections.csv",
            out / "frozen_results/geometry/nested_oof_events.parquet",
        ),
        "recusum_decrease": (
            comparison / "recusum_decrease.nested_selections.csv",
            comparison / "recusum_decrease.nested_oof_events.parquet",
        ),
        "recusum_raw": (
            comparison / "recusum_raw.nested_selections.csv",
            comparison / "recusum_raw.nested_oof_events.parquet",
        ),
        "fpol_bocd_decrease": (
            comparison / "fpol_bocd_decrease.nested_selections.csv",
            comparison / "fpol_bocd_decrease.nested_oof_events.parquet",
        ),
        "bfast_decrease": (
            comparison / "bfast_decrease.nested_selections.csv",
            comparison / "bfast_decrease.nested_oof_events.parquet",
        ),
        "maximum_monthly_drop": (
            closure / "maximum_drop.nested_selections.csv",
            closure / "maximum_drop.nested_oof_events.parquet",
        ),
        "classical_lower_cusum": (
            closure / "classical_cusum.nested_selections.csv",
            closure / "classical_cusum.nested_oof_events.parquet",
        ),
    }
    rows = []
    for method, (selection, predictions) in sources.items():
        selection_out = destination / "selections" / f"{method}.csv"
        predictions_out = destination / "oof_predictions" / f"{method}.parquet"
        copy_file(selection, selection_out)
        copy_file(predictions, predictions_out)
        frame = pd.read_parquet(predictions_out)
        rows.append(
            {
                "method": method,
                "selection_path": selection_out.relative_to(out).as_posix(),
                "selection_sha256": sha256(selection_out),
                "oof_prediction_path": predictions_out.relative_to(out).as_posix(),
                "oof_prediction_sha256": sha256(predictions_out),
                "oof_rows": len(frame),
                "outer_folds": frame["outer_fold"].nunique(),
            }
        )
    atomic_csv(pd.DataFrame(rows), destination / "oof_source_ledger.csv")

    review = src.get("prospective_review")
    if review is not None and review.is_file():
        copy_file(review, out / "prospective_2026/prior_blinded_review_unblinded.csv")


def build_manifest(config: dict) -> None:
    out = config["destination"]
    records = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
        records.append(
            {
                "path": path.relative_to(out).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(
        {
            "dataset_id": config["dataset_id"],
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(row["bytes"] for row in records),
        },
        out / "MANIFEST.json",
    )


def check(name: str, actual: int, expected: int) -> dict:
    return {"name": name, "actual": int(actual), "expected": int(expected), "ok": int(actual) == int(expected)}


def audit(config: dict) -> None:
    root = config["destination"]
    expected = config["expected"]
    inventory = pd.read_parquet(root / "inventory/balsa_validated.parquet")
    roles = pd.read_parquet(root / "splits/polygon_roles.parquet")
    references = pd.read_parquet(root / "references/reference_ground_events.parquet")
    polygon_tracks = pd.read_parquet(root / "polygon_pixels/polygon_tracks.parquet")
    associations = pd.read_parquet(root / "polygon_pixels/pixel_associations.parquet")
    catalog = pd.read_parquet(root / "s1grits/catalog.parquet")
    outer_folds = pd.read_csv(root / "splits/outer_folds.csv")
    events = pd.read_parquet(root / "frozen_results/national/ground_polygon_vv_primary.parquet")
    events = events.loc[events.route.eq("polygon_first")].copy()
    intervals = events.assign(month=pd.to_datetime(events["change_time"]).dt.to_period("M"))
    intervals = intervals.sort_values(["sample_id", "month"])
    intervals["gap"] = intervals.groupby("sample_id")["month"].transform(
        lambda values: values.astype(int).diff()
    )

    report = [
        check("validated_polygons", len(inventory), expected["validated_polygons"]),
        check("polygon_roles", roles.sample_id.nunique(), expected["validated_polygons"]),
        check("reference_events", len(references), expected["reference_events"]),
        check("reference_polygons", references.sample_id.nunique(), expected["reference_polygons"]),
        check("reference_spatial_blocks", references.frozen_spatial_block_id.nunique(), expected["reference_spatial_blocks"]),
        check("outer_fold_spatial_blocks", outer_folds.spatial_block_id.nunique(), expected["reference_spatial_blocks"]),
        check("outer_folds", outer_folds.outer_fold.nunique(), 5),
        check("polygon_tracks", len(polygon_tracks), expected["polygon_tracks"]),
        check("pixel_associations", len(associations), expected["pixel_associations"]),
        check("unique_track_pixels", associations.pixel_track_id.nunique(), expected["unique_track_pixels"]),
        check("s1grits_tiles", catalog.tile_id.nunique(), expected["s1grits_tiles"]),
        check("s1grits_track_groups", catalog.loc[catalog.product_type.eq("smonthly"), "geometry_group_id"].nunique(), expected["s1grits_track_groups"]),
        check("national_vv_events", len(events), expected["national_vv_events"]),
        check("polygons_with_vv_events", events.sample_id.nunique(), expected["polygons_with_vv_events"]),
        check("polygons_with_two_or_more_events", (events.groupby("sample_id").size() >= 2).sum(), expected["polygons_with_two_or_more_events"]),
        check("adjacent_event_intervals", intervals.gap.notna().sum(), expected["adjacent_event_intervals"]),
    ]
    payload = {"dataset_id": config["dataset_id"], "checks": report, "ok": all(row["ok"] for row in report)}
    write_json(payload, root / "AUDIT.json")
    build_manifest(config)
    for row in report:
        print(f"{'OK' if row['ok'] else 'FAIL':4} {row['name']}: {row['actual']} (expected {row['expected']})")
    if not payload["ok"]:
        raise SystemExit("dataset audit failed")


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


def evaluation_references(root: Path) -> pd.DataFrame:
    references = pd.read_parquet(root / "references/reference_ground_events.parquet").copy()
    references["spatial_block_id"] = references["frozen_spatial_block_id"]
    folds = pd.read_csv(root / "splits/outer_folds.csv")
    if folds.duplicated("spatial_block_id").any():
        raise ValueError("outer-fold map contains duplicate spatial blocks")
    references = references.drop(columns=["outer_fold"], errors="ignore").merge(
        folds[["spatial_block_id", "outer_fold"]],
        on="spatial_block_id",
        how="left",
        validate="many_to_one",
    )
    if references["outer_fold"].isna().any():
        raise ValueError("outer-fold map does not cover every reference spatial block")
    if references["spatial_block_id"].nunique() != 57 or references["outer_fold"].nunique() != 5:
        raise ValueError("expected the frozen 57-block, five-fold evaluation protocol")
    return references


def detection_cache(root: Path, run_root: Path, tracks: list[str]) -> tuple[Path, str]:
    rebuilt = run_root / "polygon_pixels"
    required = [rebuilt / f"{key}.npz" for key in tracks]
    required += [rebuilt / f"{key}.pixels.parquet" for key in tracks]
    if all(path.is_file() for path in required):
        return rebuilt, "rebuilt_from_s1grits_zarr"
    return root / "polygon_pixels/cache", "prepared_frozen_pixel_cache"


def detect(
    config: dict,
    route: str,
    track: str | None,
    components: bool = False,
    workers: int = 1,
) -> None:
    root = config["destination"]
    index_root = root / "polygon_pixels"
    cache = index_root / "cache"
    associations = pd.read_parquet(index_root / "pixel_associations.parquet")
    polygon_tracks = pd.read_parquet(index_root / "polygon_tracks.parquet")
    all_tracks = sorted(associations.track_key.astype(str).unique())
    tracks = [track] if track else all_tracks
    run_root = ROOT / "runs" / config["dataset_id"]
    cache, cache_source = detection_cache(root, run_root, tracks)
    detector = load_paper_config(ROOT / "configs/paper_v1.json")
    output = run_root / route
    output.mkdir(parents=True, exist_ok=True)
    component_output = ROOT / "runs" / config["dataset_id"] / "component_ablation/candidates"
    if components and route != "polygon-first":
        raise ValueError("component ledgers are defined for polygon-first detection only")

    for key in tracks:
        path = output / "by_track" / f"{key}.parquet"
        global_path = component_output / "by_track" / f"{key}.global.parquet"
        sliding_path = component_output / "by_track" / f"{key}.sliding.parquet"
        complete = path.exists() and (
            not components or (global_path.exists() and sliding_path.exists())
        )
        if complete:
            print(f"skip={key}")
            continue
        local = associations.loc[associations.track_key.eq(key)].copy()
        if local.empty:
            raise ValueError(f"unknown track: {key}")
        local_tracks = polygon_tracks.loc[polygon_tracks.track_key.eq(key)].copy()
        if components:
            global_candidates, sliding_candidates = detect_polygon_first_components(
                local, local_tracks, cache, detector
            )
            atomic_parquet(global_candidates, global_path)
            atomic_parquet(sliding_candidates, sliding_path)
            events = derive_variant_events(
                global_candidates,
                sliding_candidates,
                "F4",
                reporting_only=True,
            )
            if len(events):
                events["selection_score"] = 1.0 - events["null_p_value"].astype(float)
                events = (
                    events.sort_values("selection_score", ascending=False)
                    .drop_duplicates(["polygon_track_id", "Band", "change_time"])
                    .sort_values(["polygon_track_id", "Band", "change_time"])
                    .reset_index(drop=True)
                )
        elif route == "polygon-first":
            events = detect_polygon_first(local, local_tracks, cache, detector)
        else:
            pixels = detect_pixels(local, cache, detector, workers=workers)
            events = fuse_pixel_events(pixels, local, local_tracks, cache)
        atomic_parquet(events, path)
        print(f"track={key} events={len(events)}")

    parts = [output / "by_track" / f"{key}.parquet" for key in all_tracks]
    completed = [path for path in parts if path.is_file()]
    if len(completed) != len(all_tracks):
        print(f"partial={len(completed)}/{len(all_tracks)}; rerun without --track to finish")
        return
    track_events = pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)
    atomic_parquet(track_events, output / "track_events.parquet")
    ground = fuse_tracks(track_events, tolerance_months=1)
    atomic_parquet(ground, output / "ground_events.parquet")
    primary, supported = annotate_vv_with_vh(ground, tolerance_months=2)
    atomic_parquet(primary, output / "vv_primary.parquet")
    atomic_parquet(supported, output / "vv_vh_supported.parquet")
    if components:
        global_parts = [
            component_output / "by_track" / f"{key}.global.parquet" for key in all_tracks
        ]
        sliding_parts = [
            component_output / "by_track" / f"{key}.sliding.parquet" for key in all_tracks
        ]
        if not all(path.is_file() for path in global_parts + sliding_parts):
            raise RuntimeError("component candidate ledgers are incomplete")
        atomic_parquet(
            pd.concat((pd.read_parquet(item) for item in global_parts), ignore_index=True),
            component_output / "global_candidates.parquet",
        )
        atomic_parquet(
            pd.concat((pd.read_parquet(item) for item in sliding_parts), ignore_index=True),
            component_output / "sliding_candidates.parquet",
        )
    write_json(
        {
            "route": route,
            "cache_source": cache_source,
            "cache_path": str(cache.resolve()),
            "tracks": len(all_tracks),
            "track_events": len(track_events),
            "ground_events": len(ground),
            "vv_primary_events": len(primary),
            "vv_vh_supported_events": len(supported),
            "components": bool(components),
        },
        output / "COMPLETE.json",
    )
    print(f"complete route={route} track_events={len(track_events)} ground_events={len(ground)}")


def extract(config: dict, track: str | None) -> None:
    root = config["destination"]
    associations = pd.read_parquet(root / "polygon_pixels/pixel_associations.parquet")
    tracks = [track] if track else sorted(associations.track_key.unique())
    output = ROOT / "runs" / config["dataset_id"] / "polygon_pixels"
    calendar = pd.period_range("2016-01", "2026-06", freq="M")
    for key in tracks:
        local = associations.loc[associations.track_key.eq(key)].copy()
        if local.empty:
            raise ValueError(f"unknown track: {key}")
        report = extract_track_cache(local, output, calendar=calendar)
        print(f"track={key} native_pixels={report['native_pixels']}")
    if track is None:
        reports = [output / f"{key}.json" for key in sorted(associations.track_key.unique())]
        if not all(path.is_file() for path in reports):
            raise RuntimeError("one or more track cache receipts are missing")
        write_json(
            {
                "status": "complete",
                "source": "S1-GRiTS catalog and Zarr",
                "calendar": "2016-01/2026-06",
                "tracks": len(reports),
                "associations": len(associations),
                "unique_track_pixels": associations.pixel_track_id.nunique(),
            },
            output / "COMPLETE.json",
        )


def convert_shards(cohort_root: Path, output: Path, cohort: str) -> None:
    records = []
    for store in discover_stores(cohort_root.resolve()):
        records.extend(convert_store(store, output.resolve(), cohort))
        print(f"store={store.name}")
    atomic_parquet(pd.DataFrame(records), output.resolve() / "conversion_manifest.parquet")
    print(f"converted_stores={len(records)} output={output.resolve()}")


def evaluate(config: dict, source: str = "auto") -> None:
    root = config["destination"]
    recomputed = ROOT / "runs" / config["dataset_id"] / "component_ablation/candidates"
    recomputed_files = (
        recomputed / "global_candidates.parquet",
        recomputed / "sliding_candidates.parquet",
    )
    if source == "auto":
        source = "recomputed" if all(path.is_file() for path in recomputed_files) else "frozen"
    if source == "recomputed" and not all(path.is_file() for path in recomputed_files):
        missing = [str(path) for path in recomputed_files if not path.is_file()]
        raise FileNotFoundError(
            "recomputed component ledgers are incomplete; run polygon-first "
            f"detection with --components first: {missing}"
        )
    component = (
        recomputed if source == "recomputed"
        else root / "frozen_results/component_ablation"
    )
    references = evaluation_references(root)
    reference_ids = set(references.sample_id.astype(str))
    if source == "recomputed":
        global_candidates = pd.read_parquet(component / "global_candidates.parquet")
        sliding_candidates = pd.read_parquet(component / "sliding_candidates.parquet")
    else:
        global_candidates = pd.read_parquet(component / "candidate_audit/global_candidates.parquet")
        sliding_candidates = pd.read_parquet(component / "candidate_audit/sliding_candidates.parquet")
        for name, candidates in (("global", global_candidates), ("sliding", sliding_candidates)):
            scores = pd.read_parquet(component / f"scores/{name}_n1.parquet")
            candidates["passes_permutation"] = candidates.candidate_id.astype(str).isin(
                set(scores.loc[scores.admitted, "candidate_id"].astype(str))
            )
    stages = {}
    for stage in ("F0", "F1", "F2", "F3", "F4", "R0", "R1", "R2", "R3"):
        track_events = derive_variant_events(
            global_candidates,
            sliding_candidates,
            stage,
            reporting_only=True,
        )
        ground_events = fuse_tracks(track_events, tolerance_months=1)
        stages[stage] = ground_events.loc[
            ground_events.sample_id.astype(str).isin(reference_ids)
        ].copy()
    result = evaluate_event_stages(
        references,
        stages,
        tolerances=(1, 2, 3),
        block_column="spatial_block_id",
        fold_column="outer_fold",
    )
    output = ROOT / "runs" / config["dataset_id"] / "component_ablation"
    atomic_parquet(result.stage_metrics, output / "stage_metrics.parquet")
    atomic_parquet(result.fold_metrics, output / "fold_metrics.parquet")
    atomic_parquet(result.block_metrics, output / "block_metrics.parquet")
    atomic_parquet(result.matches, output / "matches.parquet")
    contrasts = {
        "F1_minus_F0": ("F1", "F0"),
        "F2_minus_F1": ("F2", "F1"),
        "F3_minus_F2": ("F3", "F2"),
        "F4_minus_F3": ("F4", "F3"),
        "F4_minus_F0": ("F4", "F0"),
        "F4_minus_R0": ("F4", "R0"),
        "F4_minus_R1": ("F4", "R1"),
        "F4_minus_R2": ("F4", "R2"),
        "F4_minus_R3": ("F4", "R3"),
    }
    bootstrap = common_index_paired_block_bootstrap(
        result.block_metrics,
        contrasts,
        tolerance_months=2,
        replicates=5_000,
        seed=20260829,
    )
    atomic_parquet(bootstrap.stage_replicates, output / "bootstrap_stage_replicates.parquet")
    atomic_parquet(bootstrap.contrast_replicates, output / "bootstrap_contrast_replicates.parquet")
    atomic_parquet(bootstrap.stage_summary, output / "bootstrap_stage_summary.parquet")
    atomic_parquet(bootstrap.contrast_summary, output / "bootstrap_contrast_summary.parquet")
    write_json(
        {
            "candidate_source": source,
            "global_candidates": len(global_candidates),
            "sliding_candidates": len(sliding_candidates),
            "reference_events": len(references),
            "bootstrap_replicates": 5_000,
            "bootstrap_seed": 20260829,
        },
        output / "evaluation_receipt.json",
    )
    print(f"component_ablation={output} source={source}")


def summarize(config: dict) -> None:
    root = config["destination"]
    inventory = pd.read_parquet(root / "inventory/balsa_validated.parquet")
    run_root = ROOT / "runs" / config["dataset_id"]
    recomputed = run_root / "polygon-first/vv_primary.parquet"
    complete = run_root / "polygon-first/COMPLETE.json"
    event_source = recomputed if recomputed.is_file() and complete.is_file() else (
        root / "frozen_results/national/ground_polygon_vv_primary.parquet"
    )
    events = pd.read_parquet(event_source)
    if "route" in events:
        events = events.loc[events.route.eq("polygon_first")].copy()
    output = run_root / "national"
    lifecycle = lifecycle_summary(events, inventory.polygon_id, id_column="polygon_id")
    lifecycle = lifecycle.drop(columns="generation_count")
    intervals = rotation_periods(events, id_column="polygon_id")
    annual = (
        events.assign(year=pd.to_datetime(events.change_time).dt.year)
        .groupby("year").size().rename("detected_events").reset_index()
    )
    atomic_parquet(lifecycle, output / "polygon_event_counts.parquet")
    atomic_parquet(intervals, output / "adjacent_event_intervals.parquet")
    atomic_parquet(annual, output / "annual_event_counts.parquet")
    write_json({"event_source": str(event_source.resolve())}, output / "SOURCE.json")
    print(
        f"national events={len(events)} polygons={events.polygon_id.nunique()} "
        f"intervals={len(intervals)}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit", "summarize"):
        item = commands.add_parser(command)
        item.add_argument("--config", type=Path, default=ROOT / "configs/data.local.json")
    item = commands.add_parser("evaluate")
    item.add_argument("--config", type=Path, default=ROOT / "configs/data.local.json")
    item.add_argument(
        "--source", choices=("auto", "recomputed", "frozen"), default="auto",
        help="Use freshly generated component ledgers when available, or the frozen paper ledgers.",
    )
    item = commands.add_parser("detect")
    item.add_argument("--config", type=Path, default=ROOT / "configs/data.local.json")
    item.add_argument("--route", choices=("polygon-first", "pixel-first"), required=True)
    item.add_argument("--track", help="Optional single track key; omit for resumable all-track execution.")
    item.add_argument(
        "--workers", type=int, default=1,
        help="Independent pixel-sequence workers for Pixel-first detection.",
    )
    item.add_argument(
        "--components", action="store_true",
        help="Persist all N1-scored polygon-first candidates required by F0--F4 evaluation.",
    )
    item = commands.add_parser("extract")
    item.add_argument("--config", type=Path, default=ROOT / "configs/data.local.json")
    item.add_argument("--track", help="Optional single track key; omit for all tracks.")
    item = commands.add_parser("convert-shards")
    item.add_argument("cohort_root", type=Path)
    item.add_argument("output", type=Path)
    item.add_argument("--cohort", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "convert-shards":
        convert_shards(args.cohort_root, args.output, args.cohort)
        return
    config = load_config(args.config)
    if args.command == "prepare":
        prepare(config)
    elif args.command == "audit":
        audit(config)
    elif args.command == "evaluate":
        evaluate(config, args.source)
    elif args.command == "summarize":
        summarize(config)
    elif args.command == "detect":
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        detect(config, args.route, args.track, args.components, args.workers)
    else:
        extract(config, args.track)


if __name__ == "__main__":
    main()
