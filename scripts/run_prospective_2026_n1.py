#!/usr/bin/env python3
"""Replay the frozen N1 detector on right-censored 2026 observations.

The command rebuilds polygon-first monthly medians from the local full-interior
cache, evaluates January--June 2026 as-of snapshots, and audits overlap with the
previous detector-blind visual review. Reused review labels are evidence for
the same polygon/month only; unmatched N1 candidates remain unreviewed.
"""
from __future__ import annotations

import argparse
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

from lr_cusum.paper import detect_series_n1, load_paper_config  # noqa: E402
from lr_cusum.polygon_workflow import annotate_vv_with_vh, fuse_tracks, load_track_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/local/paper_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prospective_2026_n1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/prospective_2026_n1")
    return parser.parse_args()


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


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_number(value) -> int:
    period = pd.Period(value, freq="M")
    return int(period.year * 12 + period.month)


def build_polygon_series(data_root: Path, output: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    metadata_path = output / "polygon_series_metadata.parquet"
    values_path = output / "polygon_series.npz"
    if metadata_path.is_file() and values_path.is_file():
        archive = np.load(values_path)
        return pd.read_parquet(metadata_path), archive["values"], pd.DatetimeIndex(archive["calendar"])

    associations = pd.read_parquet(data_root / "polygon_pixels/pixel_associations.parquet")
    polygon_tracks = pd.read_parquet(data_root / "polygon_pixels/polygon_tracks.parquet")
    track_lookup = polygon_tracks.set_index("polygon_track_id")
    rows: list[dict[str, object]] = []
    series: list[np.ndarray] = []
    calendar: pd.DatetimeIndex | None = None
    for track_key, local in associations.groupby("track_key", sort=True):
        pixels, arrays = load_track_cache(track_key, data_root / "polygon_pixels/cache")
        local_calendar = pd.DatetimeIndex(arrays["calendar"])
        if calendar is None:
            calendar = local_calendar
        elif not calendar.equals(local_calendar):
            raise ValueError("all polygon caches must share one explicit calendar")
        cache_index = pixels.set_index("pixel_track_id").cache_index
        for polygon_track_id, group in local.groupby("polygon_track_id", sort=False):
            indexes = cache_index.loc[group.pixel_track_id].to_numpy(dtype=int)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                trajectory = np.nanmedian(arrays["values"][indexes], axis=0)
            item = track_lookup.loc[polygon_track_id]
            rows.append(
                {
                    "series_index": len(series),
                    "polygon_track_id": polygon_track_id,
                    "sample_id": item.sample_id,
                    "polygon_id": item.polygon_id,
                    "role": item.role,
                    "species_nco": item.species_nco,
                    "tile_id": item.tile_id,
                    "track_key": track_key,
                    "interior_pixels": len(indexes),
                }
            )
            series.append(trajectory.astype(np.float32))
    if calendar is None:
        raise ValueError("no polygon trajectories were found")
    metadata = pd.DataFrame(rows)
    values = np.stack(series)
    atomic_parquet(metadata, metadata_path)
    temporary = values_path.with_name(values_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, values=values, calendar=calendar.to_numpy(dtype="datetime64[ns]"))
    os.replace(temporary, values_path)
    return metadata, values, calendar


def support_counts(values: np.ndarray, calendar: pd.DatetimeIndex, event: dict) -> tuple[int, int]:
    months = calendar.to_period("M")
    start = int(months.get_loc(pd.Period(event["window_start"], freq="M")))
    end = int(months.get_loc(pd.Period(event["window_end"], freq="M"))) + 1
    change = int(event["change_index"])
    return int(np.isfinite(values[start:change]).sum()), int(np.isfinite(values[change:end]).sum())


def detect_snapshot(
    metadata: pd.DataFrame,
    values: np.ndarray,
    calendar: pd.DatetimeIndex,
    cutoff: pd.Period,
    config,
    track_tolerance: int,
    vh_tolerance: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_index = int(calendar.to_period("M").get_loc(cutoff)) + 1
    local_calendar = calendar[:cutoff_index]
    local_config = replace(config, start_year=cutoff.year, end_year=cutoff.year)
    rows: list[dict[str, object]] = []
    for item in metadata.itertuples(index=False):
        trajectory = values[int(item.series_index), :, :cutoff_index]
        for band_index, band in enumerate(local_config.bands):
            detected = detect_series_n1(
                trajectory[band_index],
                local_calendar,
                identifier=item.polygon_track_id,
                band=band,
                config=local_config,
            )
            for event in detected.to_dict("records"):
                pre_count, post_count = support_counts(trajectory[band_index], local_calendar, event)
                rows.append(
                    {
                        "route": "polygon_first",
                        "as_of_cutoff": cutoff.to_timestamp(),
                        "polygon_track_id": item.polygon_track_id,
                        "sample_id": item.sample_id,
                        "polygon_id": item.polygon_id,
                        "role": item.role,
                        "species_nco": item.species_nco,
                        "tile_id": item.tile_id,
                        "track_key": item.track_key,
                        "interior_pixels": item.interior_pixels,
                        "pre_observations": pre_count,
                        "post_observations": post_count,
                        "selection_score": 1.0 - float(event["null_p_value"]),
                        **event,
                    }
                )
    track = pd.DataFrame(rows)
    if track.empty:
        return track, pd.DataFrame(columns=["sample_id", "Band", "change_time"])
    ground = fuse_tracks(track, tolerance_months=track_tolerance)
    primary, _ = annotate_vv_with_vh(ground, tolerance_months=vh_tolerance)
    primary["as_of_cutoff"] = cutoff.to_timestamp()
    primary["available_followup_months"] = primary.change_time.map(lambda value: month_number(cutoff) - month_number(value))
    primary["right_censored"] = True
    return track, primary


def longest_consecutive(values: pd.Series) -> int:
    numbers = sorted({month_number(value) for value in values})
    best = current = int(bool(numbers))
    for previous, value in zip(numbers, numbers[1:], strict=False):
        current = current + 1 if value == previous + 1 else 1
        best = max(best, current)
    return best


def link_histories(snapshots: pd.DataFrame, tolerance: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    linked_rows: list[dict[str, object]] = []
    for sample_id, group in snapshots.groupby("sample_id", sort=True):
        clusters: list[dict[str, object]] = []
        for row in group.sort_values(["as_of_cutoff", "change_time"]).to_dict("records"):
            event_month = month_number(row["change_time"])
            cutoff = pd.Timestamp(row["as_of_cutoff"])
            eligible = [
                (abs(event_month - int(cluster["last_event_month"])), index)
                for index, cluster in enumerate(clusters)
                if cluster["last_cutoff"] != cutoff
                and abs(event_month - int(cluster["last_event_month"])) <= tolerance
            ]
            if eligible:
                _, index = min(eligible)
                cluster = clusters[index]
            else:
                cluster = {
                    "id": hashlib.sha1(f"N1:{sample_id}:{event_month}:{len(clusters)}".encode()).hexdigest()[:20]
                }
                clusters.append(cluster)
            cluster["last_event_month"] = event_month
            cluster["last_cutoff"] = cutoff
            row["candidate_cluster_id"] = cluster["id"]
            linked_rows.append(row)
    linked = pd.DataFrame(linked_rows)
    histories = []
    for cluster_id, group in linked.groupby("candidate_cluster_id", sort=True):
        ordered = group.sort_values("as_of_cutoff")
        histories.append(
            {
                "candidate_cluster_id": cluster_id,
                "sample_id": ordered.sample_id.iloc[0],
                "first_detected_cutoff": ordered.as_of_cutoff.min(),
                "latest_supported_cutoff": ordered.as_of_cutoff.max(),
                "first_estimated_change_time": ordered.change_time.iloc[0],
                "latest_estimated_change_time": ordered.change_time.iloc[-1],
                "event_month_span": max(month_number(value) for value in ordered.change_time) - min(month_number(value) for value in ordered.change_time),
                "snapshot_support": ordered.as_of_cutoff.nunique(),
                "longest_consecutive_snapshots": longest_consecutive(ordered.as_of_cutoff),
                "any_vh_supported": bool(ordered.vh_supported.any()),
                "maximum_track_support": int(ordered.track_support.max()),
            }
        )
    return linked, pd.DataFrame(histories)


def audit_prior_reviews(latest: pd.DataFrame, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_columns = [
        column for column in (
            "sample_id", "blind_review_id", "visual_label", "interpreted_event_month",
            "visual_confidence", "primary_evidence", "reviewer", "review_date",
        ) if column in reviews
    ]
    reviews = reviews[review_columns].drop_duplicates("sample_id")
    merged = latest.merge(reviews, on="sample_id", how="left", validate="many_to_one")
    reviewed = merged.visual_label.notna()
    merged["prior_review_event_delta_months"] = np.nan
    interpreted = reviewed & merged.interpreted_event_month.notna()
    merged.loc[interpreted, "prior_review_event_delta_months"] = merged.loc[interpreted].apply(
        lambda row: month_number(row["change_time"]) - month_number(row["interpreted_event_month"]), axis=1
    )
    merged["review_transfer_status"] = "no_prior_review"
    merged.loc[reviewed, "review_transfer_status"] = "same_polygon_prior_review"
    comparable = interpreted & merged.prior_review_event_delta_months.abs().le(2)
    merged.loc[comparable, "review_transfer_status"] = "same_polygon_event_within_2_months"
    exact = interpreted & merged.prior_review_event_delta_months.eq(0)
    merged.loc[exact, "review_transfer_status"] = "same_polygon_exact_event_month"
    summary = (
        merged.groupby(["review_transfer_status", "visual_label"], dropna=False)
        .size().rename("candidates").reset_index()
    )
    return merged, summary


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    paper_config = load_paper_config(ROOT / protocol["paper_config"])
    started = time.perf_counter()
    metadata, values, calendar = build_polygon_series(args.data_root, args.output / "series")
    snapshot_frames = []
    for cutoff_text in protocol["cutoffs"]:
        cutoff = pd.Period(cutoff_text, freq="M")
        track_path = args.output / "snapshots" / f"{cutoff}.track.parquet"
        ground_path = args.output / "snapshots" / f"{cutoff}.ground_vv.parquet"
        if track_path.is_file() and ground_path.is_file():
            ground = pd.read_parquet(ground_path)
            snapshot_frames.append(ground)
            print(f"reuse cutoff={cutoff} vv_candidates={len(ground)}", flush=True)
            continue
        track, ground = detect_snapshot(
            metadata, values, calendar, cutoff, paper_config,
            int(protocol["same_band_track_fusion_tolerance_months"]),
            int(protocol["vv_vh_support_tolerance_months"]),
        )
        atomic_parquet(track, track_path)
        atomic_parquet(ground, ground_path)
        snapshot_frames.append(ground)
        print(f"cutoff={cutoff} vv_candidates={len(ground)}", flush=True)
    snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    linked, histories = link_histories(snapshots, int(protocol["history_link_tolerance_months"]))
    latest_cutoff = pd.Period(protocol["cutoffs"][-1], freq="M").to_timestamp()
    latest = linked.loc[pd.to_datetime(linked.as_of_cutoff).eq(latest_cutoff)].copy()
    atomic_parquet(linked, args.output / "operational_candidates_all_cutoffs.parquet")
    atomic_parquet(histories, args.output / "candidate_histories.parquet")
    atomic_parquet(latest, args.output / "operational_candidates_latest.parquet")

    review_path = args.data_root / "prospective_2026/prior_blinded_review_unblinded.csv"
    review_summary = pd.DataFrame()
    if review_path.is_file():
        audited, review_summary = audit_prior_reviews(latest, pd.read_csv(review_path))
        atomic_csv(audited, args.output / "review_overlap_audit.csv")
        atomic_csv(review_summary, args.output / "review_overlap_summary.csv")
        atomic_csv(
            audited.loc[audited.review_transfer_status.eq("no_prior_review")],
            args.output / "n1_candidates_requiring_new_review.csv",
        )

    outputs = {}
    for path in sorted(item for item in args.output.rglob("*") if item.is_file()):
        outputs[path.relative_to(args.output).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    receipt = {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "paper_detector": paper_config.as_dict(),
        "operational_reporting_year": 2026,
        "operational_cutoffs": list(protocol["cutoffs"]),
        "operational_calendar_override": (
            "Each snapshot evaluates candidates in its 2026 reporting interval "
            "using observations available through that as-of cutoff."
        ),
        "calendar": [str(calendar[0].to_period("M")), str(calendar[-1].to_period("M"))],
        "polygon_tracks": len(metadata),
        "ground_polygons": metadata.sample_id.nunique(),
        "snapshot_rows": len(snapshots),
        "candidate_histories": len(histories),
        "latest_candidates": len(latest),
        "latest_vh_supported": int(latest.vh_supported.sum()) if len(latest) else 0,
        "prior_reviewed_polygon_overlap": int(latest.sample_id.isin(set(pd.read_csv(review_path).sample_id)).sum()) if review_path.is_file() else 0,
        "interpretation_boundary": (
            "Prior detector-blind labels are reused only as an overlap audit. "
            "N1 candidates without a matching reviewed polygon/month remain unreviewed; "
            "candidate-only review cannot estimate recall."
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": outputs,
    }
    atomic_json(receipt, args.output / "COMPLETE.json")
    print(json.dumps({key: receipt[key] for key in (
        "latest_candidates", "latest_vh_supported", "prior_reviewed_polygon_overlap", "elapsed_seconds"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
