"""Full-interior plantation polygon workflows for S1-GRiTS and LR-CuSum."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from .core import lr_cusum
from .detection import DetectionConfig, stable_seed
from .paper import PaperDetectionConfig, detect_series_n1, scan_series_n1


CALENDAR = pd.period_range("2017-01", "2025-12", freq="M")


def monthly_calendar(start: str = "2017-01", end: str = "2025-12") -> pd.PeriodIndex:
    """Build an explicit monthly input calendar without imputing missing months."""
    calendar = pd.period_range(start, end, freq="M")
    if not len(calendar):
        raise ValueError("calendar must contain at least one month")
    return calendar


def _track_id(sample_id: str, track_key: str) -> str:
    return hashlib.sha1(f"{sample_id}:{track_key}".encode()).hexdigest()[:20]


def _pixel_id(track_key: str, row: int, col: int) -> str:
    return f"{track_key}:{int(row)}:{int(col)}"


def _entry_map(resolver) -> dict[str, object]:
    output = {}
    for tile in resolver.tiles():
        for pair in resolver.tracks(
            tile, product_type="smonthly", require_static=True,
            strict_static_reference=True,
        ):
            output[pair.track_key] = pair.dynamic
    return output


def _geometry_pixels(geometry, transform, height: int, width: int):
    """Return native pixels whose centres fall inside one geometry."""
    from affine import Affine
    from rasterio.features import geometry_mask
    from rasterio.transform import rowcol
    from shapely.geometry import mapping

    affine = Affine(*transform)
    minx, miny, maxx, maxy = geometry.bounds
    corners = [rowcol(affine, x, y) for x, y in (
        (minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)
    )]
    row0 = max(0, min(value[0] for value in corners) - 1)
    row1 = min(height, max(value[0] for value in corners) + 2)
    col0 = max(0, min(value[1] for value in corners) - 1)
    col1 = min(width, max(value[1] for value in corners) + 2)
    if row0 >= row1 or col0 >= col1:
        return np.array([], np.int32), np.array([], np.int32), "outside"
    local_transform = affine * Affine.translation(col0, row0)
    inside = geometry_mask(
        [mapping(geometry)], out_shape=(row1 - row0, col1 - col0),
        transform=local_transform, invert=True, all_touched=False,
    )
    local_rows, local_cols = np.nonzero(inside)
    mode = "pixel_centres"
    if not len(local_rows):
        point = geometry.representative_point()
        row, col = rowcol(affine, point.x, point.y)
        if 0 <= row < height and 0 <= col < width:
            local_rows = np.asarray([row - row0]); local_cols = np.asarray([col - col0])
            mode = "representative_fallback"
    return (local_rows + row0).astype(np.int32), (local_cols + col0).astype(np.int32), mode


def build_polygon_pixel_index(
    polygons,
    object_index: pd.DataFrame,
    resolver,
    *,
    allowed_sample_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rasterize complete interiors for every frozen sample/track association."""
    import geopandas as gpd

    scope = object_index.loc[object_index.sample_type.eq("P")].copy()
    scope = scope.drop_duplicates(["sample_id", "track_key"])
    if allowed_sample_ids is not None:
        allowed = {str(value) for value in allowed_sample_ids}
        scope = scope.loc[scope.sample_id.astype(str).isin(allowed)]
    scope["source_row"] = scope.sample_id.astype(str).str.extract(r"plantation:(\d+)")[0]
    if scope.source_row.isna().any():
        raise ValueError("P sample_id must encode plantation source row")
    scope["source_row"] = scope.source_row.astype(int)
    entries = _entry_map(resolver)
    missing_tracks = sorted(set(scope.track_key.astype(str)) - set(entries))
    if missing_tracks:
        raise ValueError(f"sample tracks are absent from resolver: {missing_tracks}")

    rows, polygon_track_rows = [], []
    for track_key, local in scope.groupby("track_key", sort=True):
        entry = entries[str(track_key)]
        local = local.sort_values("sample_id").reset_index(drop=True)
        geo = gpd.GeoDataFrame(
            local.copy(), geometry=polygons.geometry.iloc[local.source_row].reset_index(drop=True),
            crs=polygons.crs,
        ).to_crs(entry.crs)
        for item in geo.itertuples(index=False):
            native_rows, native_cols, mode = _geometry_pixels(
                item.geometry, entry.transform, entry.height, entry.width
            )
            polygon_track_id = _track_id(str(item.sample_id), str(track_key))
            polygon_track_rows.append({
                "polygon_track_id": polygon_track_id,
                "sample_id": str(item.sample_id), "polygon_id": str(item.polygon_id),
                "role": str(item.role), "species_nco": str(item.species_nco),
                "tile_id": str(item.tile_id), "track_key": str(track_key),
                "source_row": int(item.source_row), "support_mode": mode,
                "interior_pixels": int(len(native_rows)),
                "polygon_area_m2": float(item.geometry.area),
                "dynamic_zarr_path": str(entry.zarr_path),
            })
            for native_row, native_col in zip(native_rows, native_cols, strict=True):
                rows.append({
                    "polygon_track_id": polygon_track_id,
                    "sample_id": str(item.sample_id), "polygon_id": str(item.polygon_id),
                    "role": str(item.role), "species_nco": str(item.species_nco),
                    "tile_id": str(item.tile_id), "track_key": str(track_key),
                    "native_row": int(native_row), "native_col": int(native_col),
                    "chunk_y": int(native_row) // 1024,
                    "chunk_x": int(native_col) // 1024,
                    "pixel_track_id": _pixel_id(str(track_key), native_row, native_col),
                    "dynamic_zarr_path": str(entry.zarr_path),
                })
    associations = pd.DataFrame(rows)
    polygon_tracks = pd.DataFrame(polygon_track_rows)
    report = {
        "ground_polygons": int(scope.sample_id.nunique()),
        "polygon_tracks": int(len(polygon_tracks)),
        "associations": int(len(associations)),
        "unique_native_pixels": int(associations.pixel_track_id.nunique()) if len(associations) else 0,
        "fallback_polygon_tracks": int(polygon_tracks.support_mode.eq("representative_fallback").sum()),
        "outside_polygon_tracks": int(polygon_tracks.support_mode.eq("outside").sum()),
        "pixel_rule": "native pixel centre within polygon; representative fallback when no centre exists",
        "track_rule": "reuse frozen P sample-track ownership; strict resolver static reference",
    }
    return associations, polygon_tracks, report


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def extract_track_cache(
    associations: pd.DataFrame,
    output: str | Path,
    *,
    calendar: pd.PeriodIndex = CALENDAR,
) -> dict:
    """Read every unique native pixel once, preserving monthly missingness."""
    import zarr

    if associations.track_key.nunique() != 1 or associations.dynamic_zarr_path.nunique() != 1:
        raise ValueError("one extraction cache must contain exactly one track store")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    track_key = str(associations.track_key.iloc[0])
    array_path = output / f"{track_key}.npz"
    pixel_path = output / f"{track_key}.pixels.parquet"
    report_path = output / f"{track_key}.json"
    if array_path.exists() and pixel_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = f"{calendar[0]}/{calendar[-1]}"
        if report.get("calendar") != expected:
            raise ValueError(
                f"cache calendar mismatch for {track_key}: "
                f"{report.get('calendar')} != {expected}"
            )
        return report

    pixels = associations[["pixel_track_id", "native_row", "native_col", "chunk_y", "chunk_x"]]
    pixels = pixels.drop_duplicates("pixel_track_id").sort_values(
        ["chunk_y", "chunk_x", "native_row", "native_col"]
    ).reset_index(drop=True)
    pixels["cache_index"] = np.arange(len(pixels), dtype=np.int32)
    group = zarr.open_group(str(associations.dynamic_zarr_path.iloc[0]), mode="r")
    source_periods = pd.DatetimeIndex(np.asarray(group["time"][:])).to_period("M")
    source_positions = np.flatnonzero(source_periods.isin(calendar))
    target_positions = calendar.get_indexer(source_periods[source_positions])
    values = np.full((len(pixels), 2, len(calendar)), np.nan, np.float32)
    counts = np.zeros((len(pixels), len(calendar)), np.uint16)
    decode_failures = []
    chunk_height, chunk_width = group["VV_dB"].chunks[-2:]
    for (chunk_y, chunk_x), local in pixels.groupby(["chunk_y", "chunk_x"], sort=True):
        y0, x0 = int(chunk_y) * chunk_height, int(chunk_x) * chunk_width
        y1 = min(y0 + chunk_height, group["VV_dB"].shape[1])
        x1 = min(x0 + chunk_width, group["VV_dB"].shape[2])
        rows = local.native_row.to_numpy(int) - y0
        cols = local.native_col.to_numpy(int) - x0
        indexes = local.cache_index.to_numpy(int)
        for source, target in zip(source_positions, target_positions, strict=True):
            obs_block = np.asarray(group["n_obs"][int(source), y0:y1, x0:x1])
            obs = obs_block[rows, cols].astype(np.uint16)
            replace = obs > counts[indexes, int(target)]
            if not replace.any():
                continue
            selected = indexes[replace]; rr, cc = rows[replace], cols[replace]
            counts[selected, int(target)] = obs[replace]
            for band_index, band in enumerate(("VV_dB", "VH_dB")):
                try:
                    block = np.asarray(group[band][int(source), y0:y1, x0:x1])
                except Exception as error:
                    decode_failures.append({
                        "band": band,
                        "source_index": int(source),
                        "source_month": str(source_periods[int(source)]),
                        "chunk_y": int(chunk_y),
                        "chunk_x": int(chunk_x),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    continue
                values[selected, band_index, int(target)] = block[rr, cc]
    valid = (counts > 0) & np.isfinite(values).all(axis=1)
    values[:, 0, :][~valid] = np.nan; values[:, 1, :][~valid] = np.nan
    _atomic_npz(
        array_path,
        values=values,
        valid=valid.astype(np.uint8),
        n_obs=counts,
        calendar=calendar.to_timestamp().to_numpy(dtype="datetime64[ns]"),
    )
    temporary = Path(str(pixel_path) + ".tmp")
    pixels.to_parquet(temporary, index=False); os.replace(temporary, pixel_path)
    report = {
        "track_key": track_key, "native_pixels": int(len(pixels)),
        "source_months": int(len(source_positions)), "target_months": int(len(calendar)),
        "calendar": f"{calendar[0]}/{calendar[-1]}",
        "observed_start": str(source_periods[source_positions].min()) if len(source_positions) else None,
        "observed_end": str(source_periods[source_positions].max()) if len(source_positions) else None,
        "valid_fraction": float(valid.mean()), "array_path": str(array_path.resolve()),
        "pixel_path": str(pixel_path.resolve()),
        "decode_failures": decode_failures,
    }
    temporary_report = Path(str(report_path) + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary_report, report_path)
    return report


def load_track_cache(track_key: str, cache_dir: str | Path):
    cache_dir = Path(cache_dir)
    arrays = np.load(cache_dir / f"{track_key}.npz")
    pixels = pd.read_parquet(cache_dir / f"{track_key}.pixels.parquet")
    return pixels, {name: arrays[name] for name in arrays.files}


def _cache_calendar(arrays: dict) -> pd.PeriodIndex:
    if "calendar" not in arrays:
        return CALENDAR
    return pd.DatetimeIndex(arrays["calendar"]).to_period("M")


def _series_events(
    identifier,
    values,
    config: DetectionConfig | PaperDetectionConfig,
    calendar: pd.PeriodIndex = CALENDAR,
    seeds: dict | None = None,
):
    if values.shape[-1] != len(calendar):
        raise ValueError("series length and cache calendar differ")
    dates = calendar.to_timestamp().to_numpy(); rows = []
    for band_index, band in enumerate(config.bands):
        if isinstance(config, PaperDetectionConfig):
            detected = detect_series_n1(
                values[band_index],
                dates,
                identifier=identifier,
                band=band,
                config=config,
                series_seed=None if seeds is None else seeds.get(band),
            )
            for record in detected.to_dict("records"):
                record["selection_score"] = 1.0 - float(record["null_p_value"])
                rows.append(record)
            continue
        for event in lr_cusum(
            values[band_index], dates, window_length=config.window_length,
            step=config.step, confidence_threshold=config.confidence_threshold,
            bootstrap_iterations=config.bootstrap_iterations,
            seed=stable_seed(f"{identifier}:{band}", config.seed),
            start_year=config.start_year, end_year=config.end_year,
            minimum_observations=config.minimum_observations,
        ):
            record = {"Band": band, **event.as_dict()}
            record["selection_score"] = float(record["bootstrap_confidence"])
            rows.append(record)
    return rows


def _score_column(frame: pd.DataFrame) -> str:
    """Return the common ranking field for N1 and legacy candidate tables."""
    if "selection_score" in frame:
        return "selection_score"
    if "bootstrap_confidence" in frame:
        return "bootstrap_confidence"
    raise KeyError("candidate table requires selection_score or bootstrap_confidence")


def detect_polygon_first(associations, polygon_tracks, cache_dir, config):
    """Aggregate all interior pixels monthly, then detect each polygon-track."""
    rows = []
    for track_key, local in associations.groupby("track_key", sort=True):
        pixels, arrays = load_track_cache(track_key, cache_dir)
        calendar = _cache_calendar(arrays)
        mapping = pixels.set_index("pixel_track_id").cache_index
        for polygon_track_id, group in local.groupby("polygon_track_id", sort=False):
            indexes = mapping.loc[group.pixel_track_id].to_numpy(int)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                series = np.nanmedian(arrays["values"][indexes], axis=0)
            metadata = polygon_tracks.loc[
                polygon_tracks.polygon_track_id.eq(polygon_track_id)
            ].iloc[0]
            seeds = {band: int(metadata[f"seed_{band}"]) for band in config.bands
                     if f"seed_{band}" in metadata}
            for event in _series_events(polygon_track_id, series, config, calendar, seeds):
                rows.append({
                    "route": "polygon_first", "polygon_track_id": polygon_track_id,
                    "sample_id": metadata.sample_id, "polygon_id": metadata.polygon_id,
                    "role": metadata.role, "species_nco": metadata.species_nco,
                    "tile_id": metadata.tile_id, "track_key": track_key,
                    "interior_pixels": len(indexes), **event,
                })
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(_score_column(result), ascending=False).drop_duplicates(
            ["polygon_track_id", "Band", "change_time"]
        ).sort_values(["polygon_track_id", "Band", "change_time"])
    return result.reset_index(drop=True)


def detect_polygon_first_components(
    associations,
    polygon_tracks,
    cache_dir,
    config: PaperDetectionConfig,
    bands: tuple[str, ...] = ("VV_dB",),
):
    """Build N1-scored global and sliding ledgers for polygon-first series.

    This is the auditable input to F0--F4 evaluation.  It is deliberately
    separate from normal inference because it scores candidates removed by
    intermediate component gates and is therefore more expensive.
    """

    unknown = set(bands) - set(config.bands)
    if unknown:
        raise ValueError(f"unknown component bands: {sorted(unknown)}")
    global_rows = []
    sliding_rows = []
    for track_key, local in associations.groupby("track_key", sort=True):
        pixels, arrays = load_track_cache(track_key, cache_dir)
        calendar = _cache_calendar(arrays)
        dates = calendar.to_timestamp().to_numpy()
        mapping = pixels.set_index("pixel_track_id").cache_index
        for polygon_track_id, group in local.groupby("polygon_track_id", sort=False):
            indexes = mapping.loc[group.pixel_track_id].to_numpy(int)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                series = np.nanmedian(arrays["values"][indexes], axis=0)
            row = polygon_tracks.loc[
                polygon_tracks.polygon_track_id.eq(polygon_track_id)
            ].iloc[0]
            metadata = {
                "route": "polygon_first",
                "polygon_track_id": polygon_track_id,
                "sample_id": row.sample_id,
                "polygon_id": row.polygon_id,
                "role": row.role,
                "species_nco": row.species_nco,
                "tile_id": row.tile_id,
                "track_key": track_key,
                "interior_pixels": len(indexes),
            }
            for band in bands:
                band_index = config.bands.index(band)
                ledgers = scan_series_n1(
                    series[band_index],
                    dates,
                    identifier=polygon_track_id,
                    band=band,
                    config=config,
                    all_component_candidates=True,
                    metadata=metadata,
                    series_seed=int(row[f"seed_{band}"]) if f"seed_{band}" in row else None,
                )
                global_rows.append(ledgers.global_candidates)
                sliding_rows.append(ledgers.sliding_candidates)
    return (
        pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame(),
        pd.concat(sliding_rows, ignore_index=True) if sliding_rows else pd.DataFrame(),
    )


def _detect_pixel_partition(track_key, pixel_ids, cache_dir, config):
    """Detect one independent subset of pixels from a single track cache."""
    pixels, arrays = load_track_cache(track_key, cache_dir)
    calendar = _cache_calendar(arrays)
    selected = pixels.loc[pixels.pixel_track_id.isin(set(pixel_ids))]
    rows = []
    for pixel in selected.itertuples(index=False):
        for event in _series_events(
            pixel.pixel_track_id, arrays["values"][int(pixel.cache_index)], config, calendar,
            {band: int(getattr(pixel, f"seed_{band}")) for band in config.bands
             if hasattr(pixel, f"seed_{band}")},
        ):
            rows.append({
                "pixel_track_id": pixel.pixel_track_id, "track_key": track_key,
                **({"native_row": int(pixel.native_row), "native_col": int(pixel.native_col)}
                   if hasattr(pixel, "native_row") else {}),
                **event,
            })
    return pd.DataFrame(rows)


def detect_pixels(associations, cache_dir, config, *, workers: int = 1):
    """Run LR-CuSum once per unique native pixel-track sequence."""
    jobs = []
    for track_key in sorted(associations.track_key.unique()):
        used = sorted(set(
            associations.loc[associations.track_key.eq(track_key), "pixel_track_id"]
        ))
        if not used:
            continue
        count = min(workers, len(used))
        partitions = [part.tolist() for part in np.array_split(used, count) if len(part)]
        jobs.extend((track_key, part, str(cache_dir), config) for part in partitions)
    if not jobs:
        return pd.DataFrame()
    if workers == 1:
        frames = [_detect_pixel_partition(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            frames = list(executor.map(_detect_pixel_partition, *zip(*jobs)))
    nonempty = [frame for frame in frames if len(frame)]
    result = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if len(result):
        result = result.sort_values(_score_column(result), ascending=False).drop_duplicates(
            ["pixel_track_id", "Band", "change_time"]
        ).sort_values(["pixel_track_id", "Band", "change_time"])
    return result.reset_index(drop=True)


def _largest_component(points: set[tuple[int, int]]) -> int:
    remaining = set(points); largest = 0
    while remaining:
        queue = deque([remaining.pop()]); size = 0
        while queue:
            row, col = queue.popleft(); size += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbour = (row + dr, col + dc)
                    if neighbour in remaining:
                        remaining.remove(neighbour); queue.append(neighbour)
        largest = max(largest, size)
    return largest


def _month_clusters(months: np.ndarray, tolerance: int):
    months = np.sort(np.unique(months.astype(int)))
    if not len(months):
        return []
    clusters, current = [], [int(months[0])]
    for value in months[1:]:
        if int(value) - current[-1] <= tolerance:
            current.append(int(value))
        else:
            clusters.append(current); current = [int(value)]
    clusters.append(current)
    return clusters


def fuse_pixel_events(
    pixel_events: pd.DataFrame,
    associations: pd.DataFrame,
    polygon_tracks: pd.DataFrame,
    cache_dir: str | Path,
    *,
    temporal_tolerance: int = 1,
) -> pd.DataFrame:
    """Create polygon-track candidates with temporal and connected support."""
    if pixel_events.empty:
        return pd.DataFrame()
    events = pixel_events.copy()
    times = pd.to_datetime(events.change_time)
    events["month_number"] = times.dt.year * 12 + times.dt.month
    # Coordinates in the association table define polygon membership. Avoid
    # duplicate coordinate columns from the reusable per-pixel event table.
    events = events.drop(columns=["native_row", "native_col"], errors="ignore")
    joined = associations.merge(events, on=["pixel_track_id", "track_key"], how="inner")
    rows = []
    for (polygon_track_id, band), local in joined.groupby(["polygon_track_id", "Band"]):
        metadata = polygon_tracks.loc[
            polygon_tracks.polygon_track_id.eq(polygon_track_id)
        ].iloc[0]
        polygon_pixels = associations.loc[
            associations.polygon_track_id.eq(polygon_track_id), "pixel_track_id"
        ].nunique()
        for cluster in _month_clusters(local.month_number.to_numpy(), temporal_tolerance):
            selected = local.loc[local.month_number.isin(cluster)].copy()
            score_column = _score_column(selected)
            support = selected.sort_values(score_column, ascending=False).drop_duplicates(
                "pixel_track_id"
            )
            points = set(zip(support.native_row.astype(int), support.native_col.astype(int)))
            largest = _largest_component(points)
            weights = support.groupby("month_number").pixel_track_id.nunique()
            representative = int(weights.idxmax())
            year, month0 = divmod(representative - 1, 12)
            rows.append({
                "route": "pixel_first", "polygon_track_id": polygon_track_id,
                "sample_id": metadata.sample_id, "polygon_id": metadata.polygon_id,
                "role": metadata.role, "species_nco": metadata.species_nco,
                "tile_id": metadata.tile_id, "track_key": metadata.track_key,
                "Band": band, "change_time": pd.Timestamp(year=year, month=month0 + 1, day=1),
                "support_pixels": int(len(points)), "largest_component_pixels": int(largest),
                "interior_pixels": int(polygon_pixels),
                "support_fraction": float(len(points) / max(1, polygon_pixels)),
                "connected_support_fraction": float(largest / max(1, polygon_pixels)),
                "median_confidence": float(support[score_column].median()),
                "median_amplitude": float(support.cusum_amplitude.median()),
            })
    return pd.DataFrame(rows).sort_values(
        ["polygon_track_id", "Band", "change_time"]
    ).reset_index(drop=True)


def fuse_tracks(events: pd.DataFrame, *, tolerance_months: int = 1) -> pd.DataFrame:
    """Fuse track-specific events to one ground sample while retaining support."""
    if events.empty:
        return pd.DataFrame(columns=["sample_id", "Band", "change_time"])
    local = events.copy(); times = pd.to_datetime(local.change_time)
    local["month_number"] = times.dt.year * 12 + times.dt.month
    rows = []
    for (sample_id, band), group in local.groupby(["sample_id", "Band"]):
        for cluster in _month_clusters(group.month_number.to_numpy(), tolerance_months):
            selected = group.loc[group.month_number.isin(cluster)]
            weights = selected.groupby("month_number").track_key.nunique()
            representative = int(weights.idxmax()); year, month0 = divmod(representative - 1, 12)
            first = selected.iloc[0]
            rows.append({
                "sample_id": sample_id, "polygon_id": first.polygon_id,
                "role": first.role, "species_nco": first.species_nco,
                "Band": band, "change_time": pd.Timestamp(year=year, month=month0 + 1, day=1),
                "track_support": int(selected.track_key.nunique()),
                "track_event_rows": int(len(selected)),
            })
    return pd.DataFrame(rows).sort_values(["sample_id", "Band", "change_time"]).reset_index(drop=True)


def annotate_vv_with_vh(events: pd.DataFrame, *, tolerance_months: int = 1):
    """One-to-one VH support annotation for ground-level VV-primary events."""
    if events.empty:
        return events.copy(), events.copy()
    group_columns = [column for column in ("route", "sample_id") if column in events]
    rows = []
    for keys, group in events.groupby(group_columns, sort=False):
        vv = group.loc[group.Band.eq("VV_dB")].sort_values("change_time").copy()
        vh = group.loc[group.Band.eq("VH_dB")].sort_values("change_time").copy()
        vv_month = pd.to_datetime(vv.change_time).dt.year * 12 + pd.to_datetime(vv.change_time).dt.month
        vh_month = pd.to_datetime(vh.change_time).dt.year * 12 + pd.to_datetime(vh.change_time).dt.month
        candidates = sorted(
            (abs(int(vm - hm)), vi, hi) for vi, vm in enumerate(vv_month)
            for hi, hm in enumerate(vh_month) if abs(int(vm - hm)) <= tolerance_months
        )
        matches = {}; used_vv, used_vh = set(), set()
        for distance, vi, hi in candidates:
            if vi not in used_vv and hi not in used_vh:
                used_vv.add(vi); used_vh.add(hi); matches[vi] = (hi, distance)
        for vi, (_, item) in enumerate(vv.iterrows()):
            output = item.to_dict(); match = matches.get(vi)
            output["vh_supported"] = match is not None
            output["vh_delta_months"] = match[1] if match else pd.NA
            output["vh_change_time"] = vh.iloc[match[0]].change_time if match else pd.NaT
            output["vh_track_support"] = int(vh.iloc[match[0]].track_support) if match else 0
            rows.append(output)
    primary = pd.DataFrame(rows)
    consensus = primary.loc[primary.vh_supported].copy()
    consensus["Band"] = "VV+VH"
    return primary.reset_index(drop=True), consensus.reset_index(drop=True)


def match_polygon_routes(vv_primary: pd.DataFrame, *, tolerance_months: int = 1) -> pd.DataFrame:
    """One-to-one Polygon-first/Pixel-first VV consensus at ground level."""
    rows = []
    for sample_id, group in vv_primary.groupby("sample_id", sort=False):
        polygon = group.loc[group.route.eq("polygon_first")].sort_values("change_time")
        pixel = group.loc[group.route.eq("pixel_first")].sort_values("change_time")
        pm = pd.to_datetime(polygon.change_time).dt.year * 12 + pd.to_datetime(polygon.change_time).dt.month
        xm = pd.to_datetime(pixel.change_time).dt.year * 12 + pd.to_datetime(pixel.change_time).dt.month
        candidates = sorted(
            (abs(int(pv - xv)), pi, xi) for pi, pv in enumerate(pm)
            for xi, xv in enumerate(xm) if abs(int(pv - xv)) <= tolerance_months
        )
        used_p, used_x = set(), set()
        for distance, pi, xi in candidates:
            if pi in used_p or xi in used_x:
                continue
            used_p.add(pi); used_x.add(xi)
            p = polygon.iloc[pi]; x = pixel.iloc[xi]
            rows.append({
                "sample_id": sample_id, "polygon_id": p.polygon_id,
                "role": p.role, "species_nco": p.species_nco,
                "change_time": p.change_time,
                "polygon_first_time": p.change_time, "pixel_first_time": x.change_time,
                "route_delta_months": int(distance),
                "polygon_track_support": int(p.track_support),
                "pixel_track_support": int(x.track_support),
                "polygon_vh_supported": bool(p.vh_supported),
                "pixel_vh_supported": bool(x.vh_supported),
                "dual_route_dual_pol": bool(p.vh_supported and x.vh_supported),
            })
    return pd.DataFrame(rows)


def match_sample_events(references, detections, tolerance_months: int) -> dict:
    """One-to-one sample-level event metrics."""
    references = references[["sample_id", "event_month"]].drop_duplicates()
    detections = detections[["sample_id", "change_time"]].drop_duplicates()
    true_positive = 0; errors = []
    for identifier in sorted(set(references.sample_id) | set(detections.sample_id)):
        r = pd.to_datetime(references.loc[references.sample_id.eq(identifier), "event_month"])
        d = pd.to_datetime(detections.loc[detections.sample_id.eq(identifier), "change_time"])
        rm = np.asarray([x.year * 12 + x.month for x in r]); dm = np.asarray([x.year * 12 + x.month for x in d])
        candidates = sorted(
            (abs(int(rv - dv)), ri, di) for ri, rv in enumerate(rm)
            for di, dv in enumerate(dm) if abs(int(rv - dv)) <= tolerance_months
        )
        used_r, used_d = set(), set()
        for distance, ri, di in candidates:
            if ri not in used_r and di not in used_d:
                used_r.add(ri); used_d.add(di); true_positive += 1; errors.append(distance)
    fp = len(detections) - true_positive; fn = len(references) - true_positive
    precision = true_positive / max(1, true_positive + fp)
    recall = true_positive / max(1, true_positive + fn)
    return {
        "tp": true_positive, "fp": fp, "fn": fn, "precision": precision,
        "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "temporal_mae_months": float(np.mean(errors)) if errors else np.nan,
        "reference_events": len(references), "detected_events": len(detections),
        "objects": references.sample_id.nunique(),
    }
