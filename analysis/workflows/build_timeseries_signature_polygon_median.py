"""Build a reproducible, simplified signature figure for reference polygon 1499.

This workflow is a side-effect-free replacement for the plotting cells preserved
in ``01_timeseries_figures.py`` and ``06_optical_sar_examples.py``.  It does not
modify either legacy workflow or the existing manuscript figure.  The temporal
panel contains only the frozen Polygon-first monthly medians for VV and VH.

The figure layout is:

* (a) 2017--2025 Polygon-first monthly median VV and VH time series;
* (b, c) PlanetScope and Sentinel-1 evidence before the March 2024 event;
* (d, e) PlanetScope and Sentinel-1 evidence after the March 2024 event.

The script refuses to overwrite any output.  Use a new output directory for each
run so that every rendered figure remains traceable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_ID = "plantation:8655"
DEFAULT_HISTORICAL_POLYGON_ID = "1499"
DEFAULT_TILE_ID = "17MPV"
DEFAULT_TRACK_ID = "TK18"
DEFAULT_CRS = "EPSG:32717"
DEFAULT_BEFORE_MONTH = "2024-02"
DEFAULT_AFTER_MONTH = "2024-03"
DEFAULT_BEFORE_DATE = "2024-02-17"
DEFAULT_AFTER_DATE = "2024-03-28"
DEFAULT_EXTENT_SCALE_FACTOR = 2.0
DEFAULT_SCALE_BAR_M = 1000.0
TIME_AXIS_FONT_SIZE = 7.2
PANEL_LABEL_FONT_SIZE = 9.0
# Match the Bisan Gardenia palette used by Fig. 3 while preserving the
# established semantic assignment: VV is green and VH is yellow.
TEMPORAL_PALETTE = {
    "VV": "#77964a",
    "VH": "#fbc040",
}
PUBLICATION_PAD_MM = 1.5
DEFAULT_MAP_HEIGHT_SCALE_FACTOR = 1.0
IQR_MAP_HEIGHT_SCALE_FACTOR = 0.8
DEFAULT_V9_IQR_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "jag_codex_20260821"
    / "signature_analysis"
    / "timeseries_signature_polygon_iqr_20260823_v9_publication_trimmed"
    / "timeseries_signature_polygon_iqr_manifest.json"
)
DEFAULT_SERIES_METADATA = (
    REPO_ROOT
    / "outputs"
    / "jag_codex_20260821"
    / "signature_analysis"
    / "national_balsa_series_metadata.parquet"
)
DEFAULT_SERIES_ARRAY = (
    REPO_ROOT
    / "outputs"
    / "jag_codex_20260821"
    / "signature_analysis"
    / "national_balsa_series.npz"
)
DEFAULT_REFERENCE_EVENTS = (
    REPO_ROOT
    / "outputs"
    / "plantation_reference_manifest_v1"
    / "reference_sample_events.csv"
)
DEFAULT_POLYGON_FULL_DIR = REPO_ROOT / "outputs" / "plantation_polygon_full_v1"
DEFAULT_POLYGON_TRACKS = DEFAULT_POLYGON_FULL_DIR / "index" / "polygon_tracks.parquet"
DEFAULT_PIXEL_ASSOCIATIONS = (
    DEFAULT_POLYGON_FULL_DIR / "index" / "pixel_associations.parquet"
)
DEFAULT_PIXEL_CACHE = DEFAULT_POLYGON_FULL_DIR / "cache"
LOCAL_FIGURE_INPUTS = REPO_ROOT / "data/local/paper_v1/figure_inputs/fig06"
DEFAULT_CASE_SHP = LOCAL_FIGURE_INPUTS / "case_stand.shp"
DEFAULT_PLANET_BEFORE = LOCAL_FIGURE_INPUTS / "planet_before.tif"
DEFAULT_PLANET_AFTER = LOCAL_FIGURE_INPUTS / "planet_after.tif"
DEFAULT_ZARR = LOCAL_FIGURE_INPUTS / "s1grits_reference.zarr"


@dataclass(frozen=True)
class PolygonSeries:
    """One frozen ground-polygon time series on a monthly calendar."""

    sample_id: str
    calendar: pd.DatetimeIndex
    vv: np.ndarray
    vh: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RasterPanel:
    """A georeferenced RGB panel ready for plotting."""

    rgb: np.ndarray
    extent: tuple[float, float, float, float]
    transform: rasterio.Affine
    crs: str


@dataclass(frozen=True)
class PolygonPixelIQR:
    """Within-polygon pixel quantiles on the frozen monthly calendar."""

    calendar: pd.DatetimeIndex
    vv_q25: np.ndarray
    vv_q50: np.ndarray
    vv_q75: np.ndarray
    vh_q25: np.ndarray
    vh_q50: np.ndarray
    vh_q75: np.ndarray
    valid_pixel_counts_vv: np.ndarray
    valid_pixel_counts_vh: np.ndarray
    metadata: dict[str, Any]


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n" + "\n".join(missing))


def _refuse_existing(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n" + "\n".join(existing)
        )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path, hash_limit_bytes: int = 10_000_000) -> dict[str, Any]:
    """Return a compact provenance record without hashing very large rasters."""

    stat = path.stat()
    signature: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    if stat.st_size <= hash_limit_bytes:
        signature["sha256"] = _sha256(path)
    else:
        signature["sha256"] = None
        signature["hash_note"] = "Not hashed because the file exceeds 10 MB."
    return signature


def zarr_signature(path: Path) -> dict[str, Any]:
    signature: dict[str, Any] = {"path": str(path.resolve())}
    for name in (".zmetadata", ".zattrs", "zarr.json"):
        candidate = path / name
        if candidate.exists() and candidate.is_file():
            signature[name] = file_signature(candidate)
    return signature


def load_polygon_first_series(
    metadata_path: Path,
    array_path: Path,
    sample_id: str,
    start_month: str = "2017-01",
    end_month: str = "2025-12",
) -> PolygonSeries:
    """Load one already-fused Polygon-first series from the frozen cache."""

    metadata = pd.read_parquet(metadata_path)
    required = {"series_index", "sample_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Series metadata lacks columns: {missing}")

    selected = metadata.loc[metadata["sample_id"].astype(str).eq(sample_id)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one metadata row for {sample_id!r}; found {len(selected)}"
        )
    row = selected.iloc[0]

    with np.load(array_path) as arrays:
        if not {"values", "calendar"}.issubset(arrays.files):
            raise ValueError("Series NPZ must contain 'values' and 'calendar'")
        values = np.asarray(arrays["values"])
        calendar = pd.DatetimeIndex(pd.to_datetime(arrays["calendar"]))

    index = int(row["series_index"])
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(
            "Expected frozen values with shape [polygon, band(VV,VH), month]; "
            f"received {values.shape}"
        )
    if not 0 <= index < values.shape[0]:
        raise IndexError(f"series_index {index} is outside the frozen array")
    if values.shape[2] != len(calendar):
        raise ValueError("Calendar length does not match the frozen series")

    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    periods = calendar.to_period("M")
    keep = (periods >= start) & (periods <= end)
    expected = pd.period_range(start, end, freq="M")
    selected_periods = periods[keep]
    if not selected_periods.equals(expected):
        raise ValueError(
            "Frozen series does not contain the complete requested monthly calendar"
        )

    record = {column: _json_value(row[column]) for column in metadata.columns}
    return PolygonSeries(
        sample_id=sample_id,
        calendar=selected_periods.to_timestamp(),
        vv=values[index, 0, keep].astype(np.float64),
        vh=values[index, 1, keep].astype(np.float64),
        metadata=record,
    )


def summarize_pixel_iqr(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize [pixel, band(VV,VH), month] values by Q25/Q50/Q75."""

    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("values must have shape [pixel, 2, month]")
    if valid.shape != (values.shape[0], values.shape[2]):
        raise ValueError("valid must have shape [pixel, month]")
    masked = np.where(valid[:, None, :] & np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        quantiles = np.nanquantile(masked, [0.25, 0.50, 0.75], axis=0)
    counts = np.isfinite(masked).sum(axis=0).astype(np.int32)
    return quantiles, counts


def load_polygon_pixel_iqr(
    polygon_tracks_path: Path,
    associations_path: Path,
    cache_dir: Path,
    series: PolygonSeries,
    median_tolerance_db: float = 1e-5,
) -> PolygonPixelIQR:
    """Load native polygon pixels and reproduce the frozen median plus IQR.

    Quantiles are calculated independently within each orbit track. For a
    multi-track polygon, corresponding track quantiles are fused by their
    median so no orbit receives extra weight. The present case has one track,
    hence the intervals are the exact Q25--Q75 distribution of its 439 native
    polygon-interior pixels.
    """

    tracks = pd.read_parquet(polygon_tracks_path)
    required_tracks = {"polygon_track_id", "sample_id", "track_key"}
    missing_tracks = sorted(required_tracks.difference(tracks.columns))
    if missing_tracks:
        raise ValueError(f"Polygon-track table lacks columns: {missing_tracks}")
    selected_tracks = tracks.loc[
        tracks["sample_id"].astype(str).eq(series.sample_id)
    ].copy()
    if selected_tracks.empty:
        raise ValueError(f"No polygon-track rows found for {series.sample_id}")

    associations = pd.read_parquet(associations_path)
    required_associations = {"polygon_track_id", "pixel_track_id", "track_key"}
    missing_associations = sorted(
        required_associations.difference(associations.columns)
    )
    if missing_associations:
        raise ValueError(
            f"Pixel-association table lacks columns: {missing_associations}"
        )
    wanted_track_ids = set(selected_tracks["polygon_track_id"].astype(str))
    associations = associations.loc[
        associations["polygon_track_id"].astype(str).isin(wanted_track_ids)
    ].copy()
    if associations.empty:
        raise ValueError(f"No native pixels found for {series.sample_id}")

    track_quantiles: list[np.ndarray] = []
    track_counts: list[np.ndarray] = []
    track_records: list[dict[str, Any]] = []
    for track_key, group in associations.groupby("track_key", sort=True):
        pixel_path = cache_dir / f"{track_key}.pixels.parquet"
        array_path = cache_dir / f"{track_key}.npz"
        _require_paths([pixel_path, array_path])
        pixels = pd.read_parquet(pixel_path)
        required_pixels = {"pixel_track_id", "cache_index"}
        missing_pixels = sorted(required_pixels.difference(pixels.columns))
        if missing_pixels:
            raise ValueError(f"Pixel cache index lacks columns: {missing_pixels}")
        mapping = pixels.set_index("pixel_track_id")["cache_index"]
        requested = group["pixel_track_id"].astype(str)
        missing_ids = sorted(set(requested).difference(mapping.index.astype(str)))
        if missing_ids:
            raise ValueError(
                f"Pixel cache {track_key} lacks {len(missing_ids)} associated pixels"
            )
        indexes = mapping.loc[requested].to_numpy(dtype=int)
        with np.load(array_path) as arrays:
            if not {"values", "valid"}.issubset(arrays.files):
                raise ValueError(f"Pixel cache {array_path} lacks values/valid arrays")
            values = np.asarray(arrays["values"][indexes], dtype=np.float64)
            valid = np.asarray(arrays["valid"][indexes], dtype=bool)
        if values.shape[2] != len(series.calendar):
            raise ValueError(
                f"Cache {track_key} has {values.shape[2]} months, expected "
                f"{len(series.calendar)}"
            )
        quantiles, counts = summarize_pixel_iqr(values, valid)
        track_quantiles.append(quantiles)
        track_counts.append(counts)
        track_records.append(
            {
                "track_key": str(track_key),
                "native_pixels": int(len(indexes)),
                "cache_values": file_signature(array_path),
                "cache_pixels": file_signature(pixel_path),
            }
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        fused = np.nanmedian(np.stack(track_quantiles), axis=0)
    counts = np.sum(np.stack(track_counts), axis=0)
    vv_difference = np.abs(fused[1, 0] - series.vv)
    vh_difference = np.abs(fused[1, 1] - series.vh)
    max_vv_difference = float(np.nanmax(vv_difference))
    max_vh_difference = float(np.nanmax(vh_difference))
    if max(max_vv_difference, max_vh_difference) > median_tolerance_db:
        raise ValueError(
            "Native-pixel Q50 does not reproduce the frozen Polygon-first median: "
            f"VV max difference={max_vv_difference:.6g} dB; "
            f"VH max difference={max_vh_difference:.6g} dB"
        )

    return PolygonPixelIQR(
        calendar=series.calendar.copy(),
        vv_q25=fused[0, 0],
        vv_q50=fused[1, 0],
        vv_q75=fused[2, 0],
        vh_q25=fused[0, 1],
        vh_q50=fused[1, 1],
        vh_q75=fused[2, 1],
        valid_pixel_counts_vv=counts[0],
        valid_pixel_counts_vh=counts[1],
        metadata={
            "sample_id": series.sample_id,
            "quantiles": [0.25, 0.50, 0.75],
            "track_fusion": "median of corresponding within-track quantiles",
            "track_count": int(len(track_records)),
            "native_pixel_track_sum": int(
                sum(record["native_pixels"] for record in track_records)
            ),
            "median_tolerance_db": float(median_tolerance_db),
            "max_abs_q50_minus_frozen_vv_db": max_vv_difference,
            "max_abs_q50_minus_frozen_vh_db": max_vh_difference,
            "tracks": track_records,
        },
    )


def load_reference_events(
    event_path: Path,
    sample_id: str,
    historical_polygon_id: str,
) -> pd.DataFrame:
    events = pd.read_csv(event_path)
    required = {"sample_id", "historical_id", "event_month", "event_order"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"Reference-event table lacks columns: {missing}")
    selected = events.loc[
        events["sample_id"].astype(str).eq(sample_id)
        & events["historical_id"].astype(str).eq(str(historical_polygon_id))
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No reference events link polygon {historical_polygon_id} to {sample_id}"
        )
    selected["event_month"] = pd.to_datetime(selected["event_month"])
    return selected.sort_values(["event_month", "event_order"]).reset_index(drop=True)


def target_buffered_bbox(
    bounds: Sequence[float],
    padding_m: float = 240.0,
    grid_resolution_m: float = 30.0,
) -> tuple[float, float, float, float]:
    """Expand target bounds and snap outward to a projected raster grid."""

    if len(bounds) != 4:
        raise ValueError("Bounds must contain xmin, ymin, xmax, ymax")
    if padding_m < 0:
        raise ValueError("padding_m must be non-negative")
    if grid_resolution_m <= 0:
        raise ValueError("grid_resolution_m must be positive")
    xmin, ymin, xmax, ymax = map(float, bounds)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Bounds must have positive width and height")
    return (
        float(np.floor((xmin - padding_m) / grid_resolution_m) * grid_resolution_m),
        float(np.floor((ymin - padding_m) / grid_resolution_m) * grid_resolution_m),
        float(np.ceil((xmax + padding_m) / grid_resolution_m) * grid_resolution_m),
        float(np.ceil((ymax + padding_m) / grid_resolution_m) * grid_resolution_m),
    )


def expand_aligned_bbox(
    bbox: Sequence[float],
    scale_factor: float = DEFAULT_EXTENT_SCALE_FACTOR,
    grid_resolution_m: float = 30.0,
) -> tuple[float, float, float, float]:
    """Scale a bbox about its centre while retaining exact grid dimensions.

    The requested width and height are rounded to whole raster cells. The new
    lower-left corner is snapped to the same grid. If the original extent has
    an odd number of cells, exact centring and grid alignment are incompatible;
    the deterministic downward snap then shifts the centre by at most half a
    grid cell.
    """

    if len(bbox) != 4:
        raise ValueError("BBox must contain xmin, ymin, xmax, ymax")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if grid_resolution_m <= 0:
        raise ValueError("grid_resolution_m must be positive")
    xmin, ymin, xmax, ymax = map(float, bbox)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("BBox must have positive width and height")

    width_cells = max(1, int(round(width * scale_factor / grid_resolution_m)))
    height_cells = max(1, int(round(height * scale_factor / grid_resolution_m)))
    new_width = width_cells * grid_resolution_m
    new_height = height_cells * grid_resolution_m
    centre_x = (xmin + xmax) / 2.0
    centre_y = (ymin + ymax) / 2.0
    new_xmin = float(
        np.floor((centre_x - new_width / 2.0) / grid_resolution_m)
        * grid_resolution_m
    )
    new_ymin = float(
        np.floor((centre_y - new_height / 2.0) / grid_resolution_m)
        * grid_resolution_m
    )
    return (
        new_xmin,
        new_ymin,
        float(new_xmin + new_width),
        float(new_ymin + new_height),
    )


def resize_aligned_bbox_height(
    bbox: Sequence[float],
    height_scale_factor: float = DEFAULT_MAP_HEIGHT_SCALE_FACTOR,
    grid_resolution_m: float = 30.0,
) -> tuple[float, float, float, float]:
    """Resize only the bbox height about its centre on a fixed raster grid."""

    if len(bbox) != 4:
        raise ValueError("BBox must contain xmin, ymin, xmax, ymax")
    if not 0 < height_scale_factor <= 1:
        raise ValueError("height_scale_factor must lie in (0, 1]")
    if grid_resolution_m <= 0:
        raise ValueError("grid_resolution_m must be positive")
    xmin, ymin, xmax, ymax = map(float, bbox)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("BBox must have positive width and height")
    height_cells = max(
        1,
        int(round((ymax - ymin) * height_scale_factor / grid_resolution_m)),
    )
    new_height = height_cells * grid_resolution_m
    centre_y = (ymin + ymax) / 2.0
    new_ymin = float(
        np.floor((centre_y - new_height / 2.0) / grid_resolution_m)
        * grid_resolution_m
    )
    return xmin, new_ymin, xmax, float(new_ymin + new_height)


def square_aligned_bbox(
    bbox: Sequence[float], grid_resolution_m: float = 30.0
) -> tuple[float, float, float, float]:
    """Expand the shorter side to produce a centred, grid-aligned square bbox.

    This changes only the geographic context shown around the target polygon;
    it never stretches raster pixels or changes their resolution.
    """

    if len(bbox) != 4:
        raise ValueError("BBox must contain xmin, ymin, xmax, ymax")
    if grid_resolution_m <= 0:
        raise ValueError("grid_resolution_m must be positive")
    xmin, ymin, xmax, ymax = map(float, bbox)
    width_cells = int(round((xmax - xmin) / grid_resolution_m))
    height_cells = int(round((ymax - ymin) / grid_resolution_m))
    if width_cells <= 0 or height_cells <= 0:
        raise ValueError("BBox must have positive grid-aligned dimensions")
    side_cells = max(width_cells, height_cells)
    side = side_cells * grid_resolution_m
    centre_x = (xmin + xmax) / 2.0
    centre_y = (ymin + ymax) / 2.0
    new_xmin = float(np.floor((centre_x - side / 2.0) / grid_resolution_m) * grid_resolution_m)
    new_ymin = float(np.floor((centre_y - side / 2.0) / grid_resolution_m) * grid_resolution_m)
    return new_xmin, new_ymin, new_xmin + side, new_ymin + side


def geometry_bbox_margins(
    bbox: Sequence[float], geometry_bounds: Sequence[float]
) -> dict[str, float]:
    """Return projected margins from a geometry envelope to its display bbox."""

    xmin, ymin, xmax, ymax = map(float, bbox)
    gxmin, gymin, gxmax, gymax = map(float, geometry_bounds)
    margins = {
        "left_m": gxmin - xmin,
        "bottom_m": gymin - ymin,
        "right_m": xmax - gxmax,
        "top_m": ymax - gymax,
    }
    if min(margins.values()) < 0:
        raise ValueError("Display bbox clips the target geometry")
    return margins


def load_case_geometry(
    shapefile: Path,
    historical_polygon_id: str,
    target_crs: str = DEFAULT_CRS,
    crop_padding_m: float = 240.0,
    grid_resolution_m: float = 30.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, tuple[float, float, float, float]]:
    gdf = gpd.read_file(shapefile).to_crs(target_crs)
    if "ID" not in gdf.columns:
        raise ValueError("Case shapefile must contain an 'ID' column")
    target = gdf.loc[gdf["ID"].astype(str).eq(str(historical_polygon_id))].copy()
    if len(target) != 1:
        raise ValueError(
            f"Expected one case geometry for ID {historical_polygon_id}; found {len(target)}"
        )

    bbox = target_buffered_bbox(
        target.total_bounds,
        padding_m=crop_padding_m,
        grid_resolution_m=grid_resolution_m,
    )
    return gdf, target, bbox


def read_planet_rgb(path: Path, bbox: Sequence[float]) -> RasterPanel:
    """Read the three display bands of a PlanetScope Visual product."""

    with rasterio.open(path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        bands = src.read(
            [1, 2, 3], window=window, boundless=True, fill_value=0
        ).astype(np.float32)
        transform = src.window_transform(window)
        crs = str(src.crs)
    rgb = np.moveaxis(bands, 0, -1)
    if np.nanmax(rgb) > 1.0:
        rgb /= 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    west, south, east, north = array_bounds(rgb.shape[0], rgb.shape[1], transform)
    extent = (float(west), float(east), float(south), float(north))
    return RasterPanel(rgb=rgb, extent=extent, transform=transform, crs=crs)


def _coordinate_slice(values: np.ndarray, low: float, high: float) -> slice:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Spatial coordinate must be a one-dimensional array")
    return slice(low, high) if values[0] < values[-1] else slice(high, low)


def read_sar_month(
    zarr_path: Path,
    bbox: Sequence[float],
    year_month: str,
) -> tuple[np.ndarray, np.ndarray, RasterPanel, dict[str, Any]]:
    """Read one cropped monthly VV/VH composite directly from S1-GRiTS Zarr."""

    dataset = xr.open_zarr(zarr_path, consolidated=False)
    try:
        required = {"VV_dB", "VH_dB", "n_obs"}
        missing = sorted(required.difference(dataset.data_vars))
        if missing:
            raise ValueError(f"S1-GRiTS Zarr lacks variables: {missing}")
        periods = pd.DatetimeIndex(dataset["time"].values).to_period("M")
        wanted = pd.Period(year_month, freq="M")
        indexes = np.flatnonzero(periods == wanted)
        if len(indexes) == 0:
            raise ValueError(f"Month {year_month} is absent from {zarr_path}")

        x_values = np.asarray(dataset["x"].values)
        y_values = np.asarray(dataset["y"].values)
        cropped = dataset[["VV_dB", "VH_dB", "n_obs"]].isel(time=indexes).sel(
            x=_coordinate_slice(x_values, float(bbox[0]), float(bbox[2])),
            y=_coordinate_slice(y_values, float(bbox[1]), float(bbox[3])),
        )
        if cropped.sizes.get("x", 0) == 0 or cropped.sizes.get("y", 0) == 0:
            raise ValueError(f"The requested bbox does not overlap {zarr_path}")

        monthly = cropped.median(dim="time", skipna=True).compute()
        vv = np.asarray(monthly["VV_dB"].values, dtype=np.float32)
        vh = np.asarray(monthly["VH_dB"].values, dtype=np.float32)
        n_obs = np.asarray(cropped["n_obs"].sum(dim="time").compute().values)
        valid = (n_obs > 0) & np.isfinite(vv) & np.isfinite(vh)
        vv[~valid] = np.nan
        vh[~valid] = np.nan

        x = np.asarray(monthly["x"].values, dtype=float)
        y = np.asarray(monthly["y"].values, dtype=float)
        if x[0] > x[-1]:
            x = x[::-1]
            vv = vv[:, ::-1]
            vh = vh[:, ::-1]
            n_obs = n_obs[:, ::-1]
        if y[0] < y[-1]:
            y = y[::-1]
            vv = vv[::-1, :]
            vh = vh[::-1, :]
            n_obs = n_obs[::-1, :]

        dx = float(np.median(np.abs(np.diff(x))))
        dy = float(np.median(np.abs(np.diff(y))))
        extent = (
            float(x.min() - dx / 2),
            float(x.max() + dx / 2),
            float(y.min() - dy / 2),
            float(y.max() + dy / 2),
        )
        transform = rasterio.Affine(dx, 0.0, extent[0], 0.0, -dy, extent[3])
        crs = str(dataset.attrs.get("crs", DEFAULT_CRS))
        metadata = {
            "year_month": year_month,
            "source_time_count": int(len(indexes)),
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
            "valid_fraction": float(valid.mean()),
            "n_obs_min_valid": int(n_obs[valid].min()) if valid.any() else None,
            "n_obs_max_valid": int(n_obs[valid].max()) if valid.any() else None,
        }
        placeholder = RasterPanel(
            rgb=np.empty((*vv.shape, 3), dtype=np.float32),
            extent=extent,
            transform=transform,
            crs=crs,
        )
        return vv, vh, placeholder, metadata
    finally:
        dataset.close()


def joint_stretch_pair(
    first: np.ndarray,
    second: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Apply one percentile stretch to a matched pair of arrays."""

    finite = np.concatenate(
        [first[np.isfinite(first)], second[np.isfinite(second)]]
    )
    if finite.size == 0:
        raise ValueError("Cannot stretch a pair with no finite values")
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low:
        high = low + 1e-6

    def scale(array: np.ndarray) -> np.ndarray:
        result = np.clip((array - low) / (high - low), 0.0, 1.0)
        result[~np.isfinite(result)] = 0.0
        return result.astype(np.float32)

    return scale(first), scale(second), (low, high)


def build_sar_false_colour_pair(
    before_vv: np.ndarray,
    before_vh: np.ndarray,
    after_vv: np.ndarray,
    after_vh: np.ndarray,
    fixed_ranges: dict[str, Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    before_ratio = before_vv - before_vh
    after_ratio = after_vv - after_vh

    def apply_range(
        first: np.ndarray, second: np.ndarray, limits: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        if len(limits) != 2:
            raise ValueError("Each fixed stretch range must contain low and high")
        low, high = map(float, limits)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid fixed stretch range: {limits}")

        def scale(array: np.ndarray) -> np.ndarray:
            result = np.clip((array - low) / (high - low), 0.0, 1.0)
            result[~np.isfinite(result)] = 0.0
            return result.astype(np.float32)

        return scale(first), scale(second), (low, high)

    if fixed_ranges is None:
        vv_before, vv_after, vv_range = joint_stretch_pair(before_vv, after_vv)
        vh_before, vh_after, vh_range = joint_stretch_pair(before_vh, after_vh)
        ratio_before, ratio_after, ratio_range = joint_stretch_pair(
            before_ratio, after_ratio
        )
    else:
        required = {"VV_dB", "VH_dB", "VV_minus_VH_dB"}
        missing = sorted(required.difference(fixed_ranges))
        if missing:
            raise ValueError(f"Fixed SAR stretch lacks ranges: {missing}")
        vv_before, vv_after, vv_range = apply_range(
            before_vv, after_vv, fixed_ranges["VV_dB"]
        )
        vh_before, vh_after, vh_range = apply_range(
            before_vh, after_vh, fixed_ranges["VH_dB"]
        )
        ratio_before, ratio_after, ratio_range = apply_range(
            before_ratio, after_ratio, fixed_ranges["VV_minus_VH_dB"]
        )
    rgb_before = np.stack([vv_before, vh_before, ratio_before], axis=-1)
    rgb_after = np.stack([vv_after, vh_after, ratio_after], axis=-1)
    ranges = {
        "VV_dB": [float(vv_range[0]), float(vv_range[1])],
        "VH_dB": [float(vh_range[0]), float(vh_range[1])],
        "VV_minus_VH_dB": [float(ratio_range[0]), float(ratio_range[1])],
    }
    return rgb_before, rgb_after, ranges


def load_frozen_sar_stretch_ranges(path: Path) -> dict[str, list[float]]:
    """Load the recorded false-colour stretch from a previous figure manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        ranges = payload["image_panels"]["sar_false_colour"]["stretch_ranges"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Manifest does not contain SAR stretch ranges: {path}") from error
    required = ("VV_dB", "VH_dB", "VV_minus_VH_dB")
    result: dict[str, list[float]] = {}
    for name in required:
        if name not in ranges or len(ranges[name]) != 2:
            raise ValueError(f"Manifest has no valid {name} stretch range")
        low, high = map(float, ranges[name])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Manifest has invalid {name} stretch range")
        result[name] = [low, high]
    return result


def rasterized_inside_mask(
    geometry: gpd.GeoDataFrame,
    shape: tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    return rasterize(
        [(item, 1) for item in geometry.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def sparse_degree_ticks(
    bbox: Sequence[float],
    crs: str,
    decimals: int = 2,
    latitude_count: int = 2,
    longitude_count: int = 1,
) -> tuple[list[float], list[str], list[float], list[str]]:
    """Return sparse geographic ticks that lie inside the displayed extent."""

    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    if latitude_count < 1:
        raise ValueError("latitude_count must be positive")
    if longitude_count < 1:
        raise ValueError("longitude_count must be positive")
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_projected = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xmin, ymin, xmax, ymax = map(float, bbox)
    lon_min, lat_min = to_lonlat.transform(xmin, ymin)
    lon_max, lat_max = to_lonlat.transform(xmax, ymax)
    lon_low, lon_high = sorted((lon_min, lon_max))
    lat_low, lat_high = sorted((lat_min, lat_max))
    mid_lon = (lon_min + lon_max) / 2
    mid_lat = (lat_min + lat_max) / 2

    step = 10.0 ** (-decimals)
    if longitude_count == 1:
        central_lon = round(mid_lon, decimals)
        if not lon_low <= central_lon <= lon_high:
            central_lon = min(max(central_lon, lon_low), lon_high)
        lon_values = [central_lon]
    else:
        # Prefer rounded interior longitude ticks (rather than the bbox edges),
        # giving the present Ecuador crop a symmetric pair around the
        # rounded central meridian when two ticks are requested.
        central = round(mid_lon, decimals)
        if longitude_count == 2:
            candidates = [central - step, central + step]
        else:
            candidates = [
                central + offset * step
                for offset in range(-longitude_count, longitude_count + 1)
            ]
        valid = [value for value in candidates if lon_low <= value <= lon_high]
        if len(valid) >= longitude_count:
            valid.sort(key=lambda value: (abs(value - central), value))
            lon_values = sorted(valid[:longitude_count])
        else:
            lon_values = np.linspace(lon_low, lon_high, longitude_count).tolist()

    first_lat = np.ceil((lat_low - 1e-12) / step) * step
    last_lat = np.floor((lat_high + 1e-12) / step) * step
    lat_values = np.arange(first_lat, last_lat + step / 2.0, step)
    # At southern latitudes the southernmost valid ticks appear first. This
    # keeps 0.54 S and 0.53 S for the present extent without crowding 0.52 S.
    lat_values = lat_values[:latitude_count]
    x_ticks = [to_projected.transform(lon, mid_lat)[0] for lon in lon_values]
    y_ticks = [to_projected.transform(mid_lon, lat)[1] for lat in lat_values]
    x_labels = [
        f"{abs(lon):.{decimals}f}\N{DEGREE SIGN}{'W' if lon < 0 else 'E'}"
        for lon in lon_values
    ]
    y_labels = [
        f"{abs(lat):.{decimals}f}\N{DEGREE SIGN}{'S' if lat < 0 else 'N'}"
        for lat in lat_values
    ]
    return x_ticks, x_labels, y_ticks, y_labels


def validate_metric_crs(crs_like: str) -> CRS:
    """Require a projected CRS whose horizontal axes are measured in metres."""

    crs = CRS.from_user_input(crs_like)
    if not crs.is_projected:
        raise ValueError(f"Scale bars require a projected CRS; received {crs_like}")
    horizontal_axes = crs.axis_info[:2]
    units = {str(axis.unit_name).lower() for axis in horizontal_axes}
    if not units or not units.issubset({"metre", "meter"}):
        raise ValueError(
            f"Scale bars require metre-based horizontal axes; received units {units}"
        )
    return crs


def scale_bar_fraction(bbox: Sequence[float], scale_bar_m: float) -> float:
    """Return the scale-bar length as a fraction of the map width."""

    map_width_m = float(bbox[2]) - float(bbox[0])
    if map_width_m <= 0:
        raise ValueError("Map bbox must have positive width")
    if scale_bar_m <= 0 or scale_bar_m > map_width_m:
        raise ValueError("Scale bar must be positive and no longer than the map width")
    return float(scale_bar_m / map_width_m)


def plot_series_panel(
    ax: plt.Axes,
    series: PolygonSeries,
    events: pd.DataFrame,
    pixel_iqr: PolygonPixelIQR | None = None,
    draw_legend: bool = True,
    draw_panel_label: bool = True,
    fixed_ylim: tuple[float, float] | None = None,
    y_tick_interval: float = 2.0,
) -> None:
    ax.axvspan(
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-04-30"),
        color="#d9d9d9",
        alpha=0.45,
        linewidth=0,
        zorder=0,
    )
    if pixel_iqr is not None:
        if not pixel_iqr.calendar.equals(series.calendar):
            raise ValueError("Pixel-IQR and Polygon-first calendars do not match")
        ax.fill_between(
            pixel_iqr.calendar,
            pixel_iqr.vv_q25,
            pixel_iqr.vv_q75,
            color=TEMPORAL_PALETTE["VV"],
            alpha=0.16,
            linewidth=0,
            label="_nolegend_",
            zorder=1,
        )
        ax.fill_between(
            pixel_iqr.calendar,
            pixel_iqr.vh_q25,
            pixel_iqr.vh_q75,
            color=TEMPORAL_PALETTE["VH"],
            alpha=0.16,
            linewidth=0,
            label="_nolegend_",
            zorder=1,
        )
    ax.plot(
        series.calendar,
        series.vv,
        color=TEMPORAL_PALETTE["VV"],
        linewidth=1.5,
        label="Polygon-first VV median",
        zorder=3,
    )
    ax.plot(
        series.calendar,
        series.vh,
        color=TEMPORAL_PALETTE["VH"],
        linewidth=1.5,
        label="Polygon-first VH median",
        zorder=3,
    )

    for values, colour in (
        (series.vv, TEMPORAL_PALETTE["VV"]),
        (series.vh, TEMPORAL_PALETTE["VH"]),
    ):
        valid = np.isfinite(values)
        ax.scatter(
            series.calendar[valid],
            values[valid],
            s=7.0,
            color=colour,
            edgecolor="white",
            linewidth=0.22,
            alpha=0.88,
            label="_nolegend_",
            zorder=3.5,
        )

    for row in events.itertuples(index=False):
        event_month = pd.Timestamp(row.event_month)
        focal = event_month.to_period("M") == pd.Period("2024-03", freq="M")
        ax.axvline(
            event_month,
            color="#111111" if focal else "#777777",
            linestyle="--" if focal else ":",
            linewidth=1.25 if focal else 0.8,
            alpha=0.9 if focal else 0.7,
            zorder=2,
        )
        ax.text(
            event_month,
            1.012,
            event_month.strftime("%Y-%m"),
            transform=ax.get_xaxis_transform(),
            rotation=0,
            rotation_mode="anchor",
            ha="center",
            va="bottom",
            color="#111111" if focal else "#666666",
            fontsize=TIME_AXIS_FONT_SIZE,
            clip_on=False,
        )

    focal_months = pd.to_datetime(["2024-02-01", "2024-03-01", "2024-04-01"])
    lookup = pd.DataFrame(
        {"VV": series.vv, "VH": series.vh}, index=series.calendar
    )
    for band, colour in (
        ("VV", TEMPORAL_PALETTE["VV"]),
        ("VH", TEMPORAL_PALETTE["VH"]),
    ):
        values = lookup.reindex(focal_months)[band].to_numpy(dtype=float)
        ax.scatter(
            focal_months,
            values,
            s=18,
            color=colour,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

    finite = np.concatenate(
        [series.vv[np.isfinite(series.vv)], series.vh[np.isfinite(series.vh)]]
    )
    ymin = np.floor((finite.min() - 0.8) / 2.0) * 2.0
    ymax = np.ceil((finite.max() + 0.8) / 2.0) * 2.0
    if fixed_ylim is None:
        ax.set_ylim(ymin, ymax)
    else:
        fixed_ymin, fixed_ymax = map(float, fixed_ylim)
        if fixed_ymax <= fixed_ymin:
            raise ValueError("fixed_ylim must be ordered as (ymin, ymax)")
        ax.set_ylim(fixed_ymin, fixed_ymax)
    # Use the actual frozen calendar so context months (e.g. 2016 and 2026H1)
    # remain visible when the updated archive is rendered.  The previous
    # 2017--2025 behaviour is retained automatically for the archived series.
    finite_mask = np.isfinite(series.vv) | np.isfinite(series.vh)
    if not finite_mask.any():
        raise ValueError("The series contains no finite VV/VH observations")
    valid_positions = np.flatnonzero(finite_mask)
    first_valid = series.calendar[valid_positions[0]]
    last_valid = series.calendar[valid_positions[-1]]
    ax.set_xlim(first_valid, last_valid)
    if y_tick_interval <= 0:
        raise ValueError("y_tick_interval must be positive")
    if fixed_ylim is None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    else:
        first_tick = np.ceil(fixed_ymin / y_tick_interval) * y_tick_interval
        ax.set_yticks(
            np.arange(first_tick, fixed_ymax + y_tick_interval * 0.01, y_tick_interval)
        )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(
        axis="both",
        labelsize=TIME_AXIS_FONT_SIZE,
        length=2.2,
        width=0.65,
        # Bring the year labels slightly closer to the x-axis ticks while
        # leaving the complete time-series geometry and annotations unchanged.
        pad=1.5,
    )
    ax.set_ylabel("Backscatter (dB)")
    ax.grid(axis="both", color="#d0d0d0", linewidth=0.5, alpha=0.65)
    if draw_legend:
        ax.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            fontsize=6.5,
            ncol=1,
            handlelength=2.0,
            labelspacing=0.25,
            borderpad=0.28,
            handletextpad=0.55,
        )
    if draw_panel_label:
        ax.text(
            0.0,
            1.04,
            "(a)",
            transform=ax.transAxes,
            va="bottom",
            ha="center",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
        )


def _iqr_legend_content() -> tuple[list[tuple[Patch, Line2D]], list[str]]:
    """Build paired patch/line handles for median-plus-IQR legends."""

    handles = []
    labels = []
    for band, colour in (
        ("VV", TEMPORAL_PALETTE["VV"]),
        ("VH", TEMPORAL_PALETTE["VH"]),
    ):
        handles.append(
            (
                Patch(facecolor=colour, edgecolor="none", alpha=0.16),
                Line2D(
                    [0],
                    [0],
                    color=colour,
                    linewidth=1.5,
                    marker="o",
                    markersize=3.2,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                ),
            )
        )
        labels.append(f"Polygon-first {band} median")
    return handles, labels


def plot_iqr_legend_band(ax: plt.Axes) -> None:
    """Draw a one-row legend whose handles jointly encode median and IQR."""

    ax.set_axis_off()
    handles, labels = _iqr_legend_content()
    legend = ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        loc="center",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        fontsize=TIME_AXIS_FONT_SIZE,
        handlelength=2.2,
        columnspacing=2.0,
        handletextpad=0.65,
        borderpad=0.3,
    )
    legend.get_frame().set_edgecolor("#bdbdbd")
    legend.get_frame().set_linewidth(0.55)


def plot_iqr_legend_inside(ax: plt.Axes) -> plt.Legend:
    """Place the two-row IQR legend in the reserved lower-left data area."""

    handles, labels = _iqr_legend_content()
    legend = ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        loc="lower left",
        ncol=1,
        frameon=True,
        framealpha=0.94,
        fontsize=TIME_AXIS_FONT_SIZE,
        handlelength=2.2,
        labelspacing=0.25,
        handletextpad=0.6,
        borderpad=0.3,
    )
    legend.get_frame().set_edgecolor("#bdbdbd")
    legend.get_frame().set_linewidth(0.55)
    return legend


def _format_panel_date(value: str, *, include_day: bool) -> str:
    """Format image-panel dates consistently as abbreviated English dates."""

    if include_day:
        return pd.Timestamp(value).strftime("%d %b. %Y")
    return pd.Period(value, freq="M").strftime("%b. %Y")


def plot_map_panel(
    ax: plt.Axes,
    panel: RasterPanel,
    target_polygon: gpd.GeoDataFrame,
    bbox: Sequence[float],
    title: str,
    panel_label: str,
    panel_note: str,
    show_x_labels: bool,
    show_y_labels: bool,
    shade_outside_polygons: bool,
    scale_bar_m: float = DEFAULT_SCALE_BAR_M,
    longitude_tick_count: int = 1,
) -> None:
    validate_metric_crs(panel.crs)
    scale_bar_fraction(bbox, scale_bar_m)
    ax.imshow(panel.rgb, extent=panel.extent, origin="upper", interpolation="nearest")
    if shade_outside_polygons:
        inside = rasterized_inside_mask(
            target_polygon, panel.rgb.shape[:2], panel.transform
        )
        overlay = np.zeros((*inside.shape, 4), dtype=np.float32)
        overlay[~inside, :3] = 0.88
        overlay[~inside, 3] = 0.18
        ax.imshow(
            overlay,
            extent=panel.extent,
            origin="upper",
            interpolation="nearest",
        )

    # A dark casing plus a white inner line remains visible on vegetation,
    # clouds, exposed soil, and all three channels of the SAR false colour.
    target_polygon.boundary.plot(
        ax=ax, color="#111111", linewidth=2.25, alpha=0.98, zorder=4
    )
    target_polygon.boundary.plot(
        ax=ax, color="#ffffff", linewidth=1.05, alpha=1.0, zorder=5
    )
    xmin, ymin, xmax, ymax = map(float, bbox)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    map_aspect = (ymax - ymin) / (xmax - xmin)
    ax.set_aspect("equal", adjustable="box", anchor="N")
    ax.set_box_aspect(map_aspect)
    if title:
        ax.set_title(title, fontsize=7.2, pad=2.0, fontweight="semibold")

    x_ticks, x_labels, y_ticks, y_labels = sparse_degree_ticks(
        bbox, panel.crs, longitude_count=longitude_tick_count
    )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        x_labels if show_x_labels else [], fontsize=TIME_AXIS_FONT_SIZE
    )
    if show_x_labels and len(x_labels) >= 2:
        x_tick_labels = ax.get_xticklabels()
        # Center each geographic label directly below its corresponding tick.
        # The symmetric 79.41/79.39 pair leaves enough separation in the
        # compact map panels for centered labels without overlap.
        for tick_label in x_tick_labels:
            tick_label.set_ha("center")
    # Keep the same geographic y-grid in every map panel, while suppressing
    # duplicate labels on panels c--e.
    ax.set_yticks(y_ticks)
    if show_y_labels:
        ax.set_yticklabels(
            y_labels,
            fontsize=TIME_AXIS_FONT_SIZE,
            rotation=90,
            rotation_mode="anchor",
            va="center",
        )
    else:
        # Keep the axis free of duplicate labels (and preserve the historical
        # layout contract) while drawing the same geographic latitude grid as
        # the labelled panel.
        ax.set_yticks([])
        for y_tick in y_ticks:
            ax.axhline(
                float(y_tick), color="black", linewidth=0.42, alpha=0.52,
                linestyle="-", zorder=2.5,
            )
    ax.set_axisbelow(True)
    ax.grid(
        axis="both",
        color="black",
        linewidth=0.42,
        alpha=0.52,
        linestyle="-",
        zorder=2.5,
    )
    ax.yaxis.tick_left()
    ax.tick_params(axis="both", length=2.2, width=0.65, pad=1.5)

    # Scaling these projected offsets with the bar length preserves the old
    # bar's physical placement when both the view and bar length are doubled.
    display_scale = scale_bar_m / 500.0
    # Anchor the scale bar at the lower-right while preserving its physical
    # length and a small projected margin from the map frame.
    scale_x = xmax - 45.0 * display_scale - scale_bar_m
    scale_y = ymin + 55.0 * display_scale
    ax.add_patch(
        Rectangle(
            (scale_x, scale_y),
            scale_bar_m,
            13.0 * display_scale,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
    )
    ax.text(
        scale_x + scale_bar_m / 2.0,
        scale_y + 23.0 * display_scale,
        f"{int(scale_bar_m)} m",
        color="white",
        fontsize=5.6,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=6,
        path_effects=[
            path_effects.withStroke(linewidth=1.2, foreground="black")
        ],
    )
    ax.text(
        0.018,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        color="white",
        zorder=6,
        path_effects=[path_effects.withStroke(linewidth=1.4, foreground="black")],
    )
    ax.text(
        0.5,
        0.94,
        panel_note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
        color="white",
        zorder=6,
        path_effects=[path_effects.withStroke(linewidth=1.3, foreground="black")],
    )


def render_figure(
    series: PolygonSeries,
    events: pd.DataFrame,
    planet_before: RasterPanel,
    planet_after: RasterPanel,
    sar_before: RasterPanel,
    sar_after: RasterPanel,
    target_polygon: gpd.GeoDataFrame,
    bbox: Sequence[float],
    before_date: str,
    after_date: str,
    before_month: str,
    after_month: str,
    pixel_iqr: PolygonPixelIQR | None = None,
    dedicated_iqr_legend: bool = False,
    inside_iqr_legend: bool = False,
    time_ylim: tuple[float, float] | None = None,
    time_y_tick_interval: float = 2.0,
    figure_width_inches: float = 7.15,
    map_wspace: float = 0.06,
    map_left_margin: float = 0.075,
    map_right_margin: float = 0.985,
    map_row_height_scale: float = 1.0,
    longitude_tick_count: int = 1,
    scale_bar_m: float = DEFAULT_SCALE_BAR_M,
    map_gap_pt: float = 0.0,
) -> plt.Figure:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if dedicated_iqr_legend and inside_iqr_legend:
        raise ValueError("Choose either a dedicated or an inside IQR legend")
    if map_gap_pt < 0:
        raise ValueError("map_gap_pt must be non-negative")
    # Convert the requested physical gap to the GridSpec wspace convention.
    # This keeps the gap at one point even when the overall figure width changes.
    available_fraction = map_right_margin - map_left_margin
    gap_fraction = map_gap_pt / (72.0 * figure_width_inches)
    map_wspace_effective = (
        0.0
        if map_gap_pt == 0
        else (gap_fraction * 4.0) / (available_fraction - gap_fraction * 3.0)
    )
    if map_gap_pt > 0:
        map_wspace = map_wspace_effective
    if inside_iqr_legend:
        if pixel_iqr is None:
            raise ValueError("The inside IQR legend requires pixel-IQR data")
        # Removing the separate legend band allows a taller time-series panel
        # while retaining square map panels in a compact journal layout.
        figure = plt.figure(figsize=(7.0, 4.05), constrained_layout=False)
        if not np.isclose(figure_width_inches, 7.0):
            figure.set_size_inches(figure_width_inches, 4.05, forward=True)
        grid = GridSpec(
            2,
            4,
            figure=figure,
            height_ratios=[0.90, float(map_row_height_scale)],
            # Compact the vertical separation between the time-series panel
            # and the four-panel map row for the near-final layout.
            hspace=0.14,
            wspace=map_wspace,
            left=map_left_margin,
            right=map_right_margin,
            top=0.92,
            bottom=0.08,
        )
        time_ax = figure.add_subplot(grid[0, :])
        planet_before_ax = figure.add_subplot(grid[1, 0])
        sar_before_ax = figure.add_subplot(grid[1, 1])
        planet_after_ax = figure.add_subplot(grid[1, 2])
        sar_after_ax = figure.add_subplot(grid[1, 3])
        plot_series_panel(
            time_ax,
            series,
            events,
            pixel_iqr=pixel_iqr,
            draw_legend=False,
            draw_panel_label=False,
            fixed_ylim=time_ylim,
            y_tick_interval=time_y_tick_interval,
        )
        plot_iqr_legend_inside(time_ax)
    elif dedicated_iqr_legend:
        if pixel_iqr is None:
            raise ValueError("The dedicated IQR legend requires pixel-IQR data")
        # The square map row releases vertical space for a 16--20% taller time
        # panel plus a non-overlapping legend band.
        figure = plt.figure(figsize=(7.0, 3.90), constrained_layout=False)
        if not np.isclose(figure_width_inches, 7.0):
            figure.set_size_inches(figure_width_inches, 3.90, forward=True)
        grid = GridSpec(
            3,
            4,
            figure=figure,
            height_ratios=[0.80, 0.12, 1.0],
            hspace=0.12,
            wspace=map_wspace,
            left=map_left_margin,
            right=map_right_margin,
            top=0.92,
            bottom=0.08,
        )
        time_ax = figure.add_subplot(grid[0, :])
        planet_before_ax = figure.add_subplot(grid[2, 0])
        sar_before_ax = figure.add_subplot(grid[2, 1])
        planet_after_ax = figure.add_subplot(grid[2, 2])
        sar_after_ax = figure.add_subplot(grid[2, 3])
        # Add this last so figure.axes[1:5] remains the four-map row used by
        # the shared layout diagnostics.
        legend_ax = figure.add_subplot(grid[1, :])
        plot_series_panel(
            time_ax,
            series,
            events,
            pixel_iqr=pixel_iqr,
            draw_legend=False,
            draw_panel_label=False,
            fixed_ylim=time_ylim,
            y_tick_interval=time_y_tick_interval,
        )
        plot_iqr_legend_band(legend_ax)
        legend_position = legend_ax.get_position()
        legend_ax.set_position(
            [
                legend_position.x0,
                legend_position.y0 - 0.018,
                legend_position.width,
                legend_position.height,
            ]
        )
        for map_ax in (
            planet_before_ax,
            sar_before_ax,
            planet_after_ax,
            sar_after_ax,
        ):
            map_position = map_ax.get_position()
            map_ax.set_position(
                [
                    map_position.x0,
                    map_position.y0 - 0.026,
                    map_position.width,
                    map_position.height,
                ]
            )
    else:
        # Approximately 18.2 x 10.7 cm. The lower row groups optical and SAR
        # evidence by disturbance state: before (b, c), then after (d, e).
        figure = plt.figure(figsize=(7.0, 4.20), constrained_layout=False)
        if not np.isclose(figure_width_inches, 7.0):
            figure.set_size_inches(figure_width_inches, 4.20, forward=True)
        grid = GridSpec(
            2,
            4,
            figure=figure,
            height_ratios=[0.55, 1.0],
            hspace=0.22,
            wspace=map_wspace,
            left=map_left_margin,
            right=map_right_margin,
            top=0.92,
            bottom=0.105,
        )
        time_ax = figure.add_subplot(grid[0, :])
        planet_before_ax = figure.add_subplot(grid[1, 0])
        sar_before_ax = figure.add_subplot(grid[1, 1])
        planet_after_ax = figure.add_subplot(grid[1, 2])
        sar_after_ax = figure.add_subplot(grid[1, 3])
        plot_series_panel(
            time_ax,
            series,
            events,
            pixel_iqr=pixel_iqr,
            fixed_ylim=time_ylim,
            y_tick_interval=time_y_tick_interval,
        )

    plot_map_panel(
        planet_before_ax,
        planet_before,
        target_polygon,
        bbox,
        "",
        "(b)",
        _format_panel_date(before_date, include_day=True),
        show_x_labels=True,
        show_y_labels=True,
        shade_outside_polygons=True,
        scale_bar_m=scale_bar_m,
        longitude_tick_count=longitude_tick_count,
    )
    plot_map_panel(
        sar_before_ax,
        sar_before,
        target_polygon,
        bbox,
        "",
        "(c)",
        _format_panel_date(before_month, include_day=False),
        show_x_labels=True,
        show_y_labels=False,
        shade_outside_polygons=False,
        scale_bar_m=scale_bar_m,
        longitude_tick_count=longitude_tick_count,
    )
    plot_map_panel(
        planet_after_ax,
        planet_after,
        target_polygon,
        bbox,
        "",
        "(d)",
        _format_panel_date(after_date, include_day=True),
        show_x_labels=True,
        show_y_labels=False,
        shade_outside_polygons=True,
        scale_bar_m=scale_bar_m,
        longitude_tick_count=longitude_tick_count,
    )
    plot_map_panel(
        sar_after_ax,
        sar_after,
        target_polygon,
        bbox,
        "",
        "(e)",
        _format_panel_date(after_month, include_day=False),
        show_x_labels=True,
        show_y_labels=False,
        shade_outside_polygons=False,
        scale_bar_m=scale_bar_m,
        longitude_tick_count=longitude_tick_count,
    )
    if inside_iqr_legend:
        # Preserve every component of panel (a) and create title clearance by
        # moving only the b--e image row.  The shift is small enough to retain
        # the compact page layout while preventing the group headings from
        # entering the time-axis label zone.
        # Move the image row down by an additional 7 pt relative to v37.  The
        # group-heading baseline is then placed 2 pt above its bbox, giving a
        # net 5 pt downward movement of the heading relative to v37 (2 pt
        # relative to v39) while retaining a measurable clearance from the map
        # frame.
        map_row_shift = -0.010 - 7.0 / (72.0 * figure.get_figheight())
        for map_ax in (
            planet_before_ax,
            sar_before_ax,
            planet_after_ax,
            sar_after_ax,
        ):
            map_position = map_ax.get_position()
            map_ax.set_position(
                [
                    map_position.x0,
                    map_position.y0 + map_row_shift,
                    map_position.width,
                    map_position.height,
                ]
            )
    calibrate_time_panel_outer_width(
        figure,
        time_ax,
        (planet_before_ax, sar_before_ax, planet_after_ax, sar_after_ax),
    )
    if dedicated_iqr_legend or inside_iqr_legend:
        time_outer = _tight_bbox_figure_fraction(figure, time_ax)
        if inside_iqr_legend:
            figure.text(
                time_outer[0],
                time_outer[3] - 0.004,
                "(a)",
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONT_SIZE,
                fontweight="bold",
            )
        else:
            figure.text(
                time_outer[0],
                time_outer[3] + 0.005,
                "(a)",
                ha="left",
                va="bottom",
                fontsize=PANEL_LABEL_FONT_SIZE,
                fontweight="bold",
            )
    before_left = planet_before_ax.get_position().x0
    left_group_right = sar_before_ax.get_position().x1
    right_group_left = planet_after_ax.get_position().x0
    after_right = sar_after_ax.get_position().x1
    divider_x = (left_group_right + right_group_left) / 2
    image_bottom = min(ax.get_position().y0 for ax in (
        planet_before_ax, sar_before_ax, planet_after_ax, sar_after_ax
    ))
    image_top = max(ax.get_position().y1 for ax in (
        planet_before_ax, sar_before_ax, planet_after_ax, sar_after_ax
    ))
    group_header_gap_above_map_pt = 2.0
    header_y = (
        image_top
        + group_header_gap_above_map_pt / (72.0 * figure.get_figheight())
    )
    figure.text(
        (before_left + left_group_right) / 2,
        header_y,
        "Before stand-disturbance",
        ha="center",
        va="bottom",
        fontsize=TIME_AXIS_FONT_SIZE,
        fontweight="bold",
        color="#222222",
    )
    figure.text(
        (right_group_left + after_right) / 2,
        header_y,
        "After stand-disturbance",
        ha="center",
        va="bottom",
        fontsize=TIME_AXIS_FONT_SIZE,
        fontweight="bold",
        color="#222222",
    )
    figure.add_artist(
        Line2D(
            [divider_x, divider_x],
            [image_bottom, header_y + 0.009],
            transform=figure.transFigure,
            color="#b5b5b5",
            linewidth=0.65,
            zorder=10,
        )
    )
    return figure


def _tight_bbox_figure_fraction(
    figure: plt.Figure, ax: plt.Axes
) -> tuple[float, float, float, float]:
    renderer = figure.canvas.get_renderer()
    tight = ax.get_tightbbox(renderer)
    return (
        float(tight.x0 / figure.bbox.width),
        float(tight.y0 / figure.bbox.height),
        float(tight.x1 / figure.bbox.width),
        float(tight.y1 / figure.bbox.height),
    )


def calibrate_time_panel_outer_width(
    figure: plt.Figure,
    time_ax: plt.Axes,
    map_axes: Sequence[plt.Axes],
    iterations: int = 3,
) -> None:
    """Match the decorated time-panel width to the decorated four-map row."""

    if len(map_axes) != 4:
        raise ValueError("Exactly four map axes are required")
    for _ in range(iterations):
        figure.canvas.draw()
        map_boxes = [_tight_bbox_figure_fraction(figure, ax) for ax in map_axes]
        target_left = min(item[0] for item in map_boxes)
        target_right = max(item[2] for item in map_boxes)
        time_box = _tight_bbox_figure_fraction(figure, time_ax)
        position = time_ax.get_position()
        left_decoration = position.x0 - time_box[0]
        right_decoration = time_box[2] - position.x1
        new_x0 = target_left + left_decoration
        new_x1 = target_right - right_decoration
        if new_x1 <= new_x0:
            raise ValueError("Decorations leave no positive width for the time panel")
        time_ax.set_position([new_x0, position.y0, new_x1 - new_x0, position.height])


def figure_layout_diagnostics(figure: plt.Figure) -> dict[str, Any]:
    """Measure final panel alignment in both figure and physical coordinates."""

    figure.canvas.draw()
    time_position = figure.axes[0].get_position()
    map_positions = [axis.get_position() for axis in figure.axes[1:5]]
    time_tight = _tight_bbox_figure_fraction(figure, figure.axes[0])
    map_tight = [
        _tight_bbox_figure_fraction(figure, axis) for axis in figure.axes[1:5]
    ]
    map_outer_left = min(item[0] for item in map_tight)
    map_outer_right = max(item[2] for item in map_tight)
    figure_width, figure_height = figure.get_size_inches()
    map_aspects = [
        float((position.height * figure_height) / (position.width * figure_width))
        for position in map_positions
    ]
    left_delta = float(time_tight[0] - map_outer_left)
    right_delta = float(time_tight[2] - map_outer_right)
    tolerance = 0.002
    legend_diagnostics: dict[str, Any] = {"enabled": False}
    if len(figure.axes) > 5 and figure.axes[5].get_legend() is not None:
        renderer = figure.canvas.get_renderer()
        legend = figure.axes[5].get_legend()
        legend_bbox = legend.get_window_extent(renderer)
        time_bbox = figure.axes[0].get_tightbbox(renderer)
        map_bbox_top = max(
            axis.get_tightbbox(renderer).y1 for axis in figure.axes[1:5]
        )
        header_boxes = [
            text.get_window_extent(renderer)
            for text in figure.texts
            if text.get_text()
            in {"Before stand-disturbance", "After stand-disturbance"}
        ]
        overlaps_headers = any(
            legend_bbox.overlaps(header_bbox) for header_bbox in header_boxes
        )
        legend_diagnostics = {
            "enabled": True,
            "columns": int(legend._ncols),
            "bbox_figure_fraction": [
                float(legend_bbox.x0 / figure.bbox.width),
                float(legend_bbox.y0 / figure.bbox.height),
                float(legend_bbox.x1 / figure.bbox.width),
                float(legend_bbox.y1 / figure.bbox.height),
            ],
            "time_panel_outer_bottom_figure_fraction": float(
                time_bbox.y0 / figure.bbox.height
            ),
            "map_panel_outer_top_figure_fraction": float(
                map_bbox_top / figure.bbox.height
            ),
            "overlaps_time_panel": bool(legend_bbox.overlaps(time_bbox)),
            "overlaps_map_panels": bool(legend_bbox.y0 < map_bbox_top),
            "overlaps_group_headers": bool(overlaps_headers),
            "non_overlap_pass": bool(
                not legend_bbox.overlaps(time_bbox)
                and legend_bbox.y0 >= map_bbox_top
                and not overlaps_headers
            ),
        }
    elif figure.axes[0].get_legend() is not None:
        renderer = figure.canvas.get_renderer()
        legend = figure.axes[0].get_legend()
        legend_bbox = legend.get_window_extent(renderer)
        axes_bbox = figure.axes[0].bbox
        legend_top_data = float(
            figure.axes[0].transData.inverted().transform(
                (legend_bbox.x0, legend_bbox.y1)
            )[1]
        )
        data_floor = min(
            float(np.nanmin(np.asarray(line.get_ydata(), dtype=float)))
            for line in figure.axes[0].lines[:2]
        )
        for collection in figure.axes[0].collections[:2]:
            vertices = np.concatenate(
                [path.vertices for path in collection.get_paths()], axis=0
            )
            data_floor = min(data_floor, float(np.nanmin(vertices[:, 1])))
        clearance_db = float(data_floor - legend_top_data)
        within_axes = bool(
            legend_bbox.x0 >= axes_bbox.x0
            and legend_bbox.x1 <= axes_bbox.x1
            and legend_bbox.y0 >= axes_bbox.y0
            and legend_bbox.y1 <= axes_bbox.y1
        )
        legend_diagnostics = {
            "enabled": True,
            "placement": "inside_time_panel_lower_left",
            "columns": int(legend._ncols),
            "bbox_figure_fraction": [
                float(legend_bbox.x0 / figure.bbox.width),
                float(legend_bbox.y0 / figure.bbox.height),
                float(legend_bbox.x1 / figure.bbox.width),
                float(legend_bbox.y1 / figure.bbox.height),
            ],
            "within_time_axes": within_axes,
            "legend_top_data_db": legend_top_data,
            "lowest_iqr_or_median_db": data_floor,
            "vertical_data_clearance_db": clearance_db,
            "non_overlap_pass": bool(within_axes and clearance_db > 0.0),
        }
    return {
        "figure_size_in": [float(figure_width), float(figure_height)],
        "time_panel_x0": float(time_position.x0),
        "time_panel_x1": float(time_position.x1),
        "map_row_x0": float(map_positions[0].x0),
        "map_row_x1": float(map_positions[-1].x1),
        "time_panel_outer_x0": float(time_tight[0]),
        "time_panel_outer_x1": float(time_tight[2]),
        "map_row_outer_x0": float(map_outer_left),
        "map_row_outer_x1": float(map_outer_right),
        "outer_left_alignment_delta_figure_fraction": left_delta,
        "outer_right_alignment_delta_figure_fraction": right_delta,
        "horizontal_alignment_tolerance": tolerance,
        "horizontal_alignment_pass": bool(
            abs(left_delta) <= tolerance and abs(right_delta) <= tolerance
        ),
        "map_panel_physical_height_to_width": map_aspects,
        "time_panel_physical_height_in": float(
            time_position.height * figure_height
        ),
        "legend": legend_diagnostics,
        "time_to_map_vertical_gap_figure_fraction": float(
            time_position.y0 - max(position.y1 for position in map_positions)
        ),
    }


def publication_crop_bbox(
    figure: plt.Figure, padding_mm: float = PUBLICATION_PAD_MM
) -> Bbox:
    """Return a content-aware output bbox with a small print-safe margin."""

    if padding_mm < 0:
        raise ValueError("padding_mm must be non-negative")
    figure.canvas.draw()
    tight = figure.get_tightbbox(figure.canvas.get_renderer())
    padding_in = padding_mm / 25.4
    return Bbox.from_extents(
        tight.x0 - padding_in,
        tight.y0 - padding_in,
        tight.x1 + padding_in,
        tight.y1 + padding_in,
    )


def _series_event_values(series: PolygonSeries) -> dict[str, dict[str, float]]:
    lookup = pd.DataFrame(
        {"VV_dB": series.vv, "VH_dB": series.vh}, index=series.calendar
    )
    values: dict[str, dict[str, float]] = {}
    for month in ("2024-02", "2024-03", "2024-04"):
        timestamp = pd.Period(month, freq="M").to_timestamp()
        row = lookup.loc[timestamp]
        values[month] = {
            "VV_dB": float(row["VV_dB"]),
            "VH_dB": float(row["VH_dB"]),
        }
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument(
        "--historical-polygon-id", default=DEFAULT_HISTORICAL_POLYGON_ID
    )
    parser.add_argument("--tile-id", default=DEFAULT_TILE_ID)
    parser.add_argument("--track-id", default=DEFAULT_TRACK_ID)
    parser.add_argument("--series-metadata", type=Path, default=DEFAULT_SERIES_METADATA)
    parser.add_argument("--series-array", type=Path, default=DEFAULT_SERIES_ARRAY)
    parser.add_argument("--reference-events", type=Path, default=DEFAULT_REFERENCE_EVENTS)
    parser.add_argument("--polygon-tracks", type=Path, default=DEFAULT_POLYGON_TRACKS)
    parser.add_argument(
        "--pixel-associations", type=Path, default=DEFAULT_PIXEL_ASSOCIATIONS
    )
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXEL_CACHE)
    parser.add_argument(
        "--include-pixel-iqr",
        action="store_true",
        help="Add Q25--Q75 shading from native polygon-interior pixels.",
    )
    parser.add_argument("--case-shapefile", type=Path, default=DEFAULT_CASE_SHP)
    parser.add_argument("--planet-before", type=Path, default=DEFAULT_PLANET_BEFORE)
    parser.add_argument("--planet-after", type=Path, default=DEFAULT_PLANET_AFTER)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--before-month", default=DEFAULT_BEFORE_MONTH)
    parser.add_argument("--after-month", default=DEFAULT_AFTER_MONTH)
    parser.add_argument("--before-date", default=DEFAULT_BEFORE_DATE)
    parser.add_argument("--after-date", default=DEFAULT_AFTER_DATE)
    parser.add_argument("--target-crs", default=DEFAULT_CRS)
    parser.add_argument("--crop-padding-m", type=float, default=240.0)
    parser.add_argument("--grid-resolution-m", type=float, default=30.0)
    parser.add_argument(
        "--extent-scale-factor",
        type=float,
        default=DEFAULT_EXTENT_SCALE_FACTOR,
        help=(
            "Scale the original target-buffered crop about its centre; the default "
            "2.0 doubles both map dimensions while preserving 30 m alignment."
        ),
    )
    parser.add_argument(
        "--map-height-scale-factor",
        type=float,
        default=DEFAULT_MAP_HEIGHT_SCALE_FACTOR,
        help="Crop the expanded map height about its centre; 0.8 makes v10 square.",
    )
    parser.add_argument(
        "--dedicated-iqr-legend",
        action="store_true",
        help="Use the taller time panel and separate one-row IQR legend band.",
    )
    parser.add_argument(
        "--inside-iqr-legend",
        action="store_true",
        help="Place the two-row IQR legend inside the lower-left time panel.",
    )
    parser.add_argument("--time-ymin", type=float, default=None)
    parser.add_argument("--time-ymax", type=float, default=None)
    parser.add_argument("--time-ytick-interval", type=float, default=2.0)
    parser.add_argument(
        "--figure-width-inches",
        type=float,
        default=7.15,
        help="Overall figure width; map panels grow without changing geographic aspect.",
    )
    parser.add_argument(
        "--map-wspace",
        type=float,
        default=0.06,
        help="Horizontal spacing between map columns as a GridSpec fraction.",
    )
    parser.add_argument("--map-left-margin", type=float, default=0.075)
    parser.add_argument("--map-right-margin", type=float, default=0.985)
    parser.add_argument("--map-row-height-scale", type=float, default=1.0)
    parser.add_argument("--longitude-tick-count", type=int, default=1)
    parser.add_argument("--scale-bar-m", type=float, default=DEFAULT_SCALE_BAR_M)
    parser.add_argument("--map-gap-pt", type=float, default=0.0)
    parser.add_argument(
        "--square-map-panels",
        action="store_true",
        help="Expand the geographic crop to a centred square while keeping the 1x4 layout.",
    )
    parser.add_argument(
        "--series-start-month",
        default="2017-01",
        help="First month to display from the frozen series (YYYY-MM).",
    )
    parser.add_argument(
        "--series-end-month",
        default="2025-12",
        help="Last month to display from the frozen series (YYYY-MM).",
    )
    parser.add_argument(
        "--sar-stretch-manifest",
        type=Path,
        default=None,
        help="Freeze SAR false-colour ranges from a previous figure manifest.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--publication-padding-mm",
        type=float,
        default=PUBLICATION_PAD_MM,
        help="Print-safe margin added around the union of all visible artists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="A new directory; existing output files are never overwritten.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.time_ymin is None) != (args.time_ymax is None):
        raise ValueError("--time-ymin and --time-ymax must be provided together")
    if args.time_ytick_interval <= 0:
        raise ValueError("--time-ytick-interval must be positive")
    if args.figure_width_inches <= 0:
        raise ValueError("--figure-width-inches must be positive")
    if args.map_wspace < 0:
        raise ValueError("--map-wspace must be non-negative")
    if not (0.0 <= args.map_left_margin < args.map_right_margin <= 1.0):
        raise ValueError("Map margins must satisfy 0 <= left < right <= 1")
    if args.map_row_height_scale <= 0:
        raise ValueError("--map-row-height-scale must be positive")
    if args.longitude_tick_count < 1:
        raise ValueError("--longitude-tick-count must be positive")
    if args.scale_bar_m <= 0:
        raise ValueError("--scale-bar-m must be positive")
    if args.map_gap_pt < 0:
        raise ValueError("--map-gap-pt must be non-negative")
    time_ylim = (
        (float(args.time_ymin), float(args.time_ymax))
        if args.time_ymin is not None
        else None
    )
    inputs = [
        args.series_metadata,
        args.series_array,
        args.reference_events,
        args.case_shapefile,
        args.planet_before,
        args.planet_after,
        args.zarr,
    ]
    if args.include_pixel_iqr:
        inputs.extend(
            [args.polygon_tracks, args.pixel_associations, args.pixel_cache]
        )
    if args.sar_stretch_manifest is not None:
        inputs.append(args.sar_stretch_manifest)
    _require_paths(inputs)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        "timeseries_signature_polygon_iqr"
        if args.include_pixel_iqr
        else "timeseries_signature_polygon_median"
    )
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"
    tiff_path = output_dir / f"{output_stem}.tiff"
    manifest_path = output_dir / f"{output_stem}_manifest.json"
    _refuse_existing([pdf_path, png_path, svg_path, tiff_path, manifest_path])

    series = load_polygon_first_series(
        args.series_metadata,
        args.series_array,
        args.sample_id,
        start_month=args.series_start_month,
        end_month=args.series_end_month,
    )
    if series.metadata.get("primary_tile_id") != args.tile_id:
        raise ValueError(
            f"Frozen series belongs to {series.metadata.get('primary_tile_id')}, "
            f"not requested tile {args.tile_id}"
        )
    pixel_iqr = None
    if args.include_pixel_iqr:
        pixel_iqr = load_polygon_pixel_iqr(
            args.polygon_tracks,
            args.pixel_associations,
            args.pixel_cache,
            series,
        )
    events = load_reference_events(
        args.reference_events, args.sample_id, args.historical_polygon_id
    )
    _, target_polygon, base_bbox = load_case_geometry(
        args.case_shapefile,
        args.historical_polygon_id,
        target_crs=args.target_crs,
        crop_padding_m=args.crop_padding_m,
        grid_resolution_m=args.grid_resolution_m,
    )
    expanded_bbox = expand_aligned_bbox(
        base_bbox,
        scale_factor=args.extent_scale_factor,
        grid_resolution_m=args.grid_resolution_m,
    )
    bbox = resize_aligned_bbox_height(
        expanded_bbox,
        height_scale_factor=args.map_height_scale_factor,
        grid_resolution_m=args.grid_resolution_m,
    )
    rectangular_bbox = bbox
    if args.square_map_panels:
        bbox = square_aligned_bbox(bbox, args.grid_resolution_m)
    target_margins = geometry_bbox_margins(bbox, target_polygon.total_bounds)
    planet_before = read_planet_rgb(args.planet_before, bbox)
    planet_after = read_planet_rgb(args.planet_after, bbox)

    before_vv, before_vh, sar_template_before, sar_before_metadata = read_sar_month(
        args.zarr, bbox, args.before_month
    )
    after_vv, after_vh, sar_template_after, sar_after_metadata = read_sar_month(
        args.zarr, bbox, args.after_month
    )
    if sar_template_before.extent != sar_template_after.extent:
        raise ValueError("Before/after SAR crops do not share one spatial extent")
    fixed_stretch_ranges = None
    if args.sar_stretch_manifest is not None:
        fixed_stretch_ranges = load_frozen_sar_stretch_ranges(
            args.sar_stretch_manifest
        )
    sar_rgb_before, sar_rgb_after, stretch_ranges = build_sar_false_colour_pair(
        before_vv,
        before_vh,
        after_vv,
        after_vh,
        fixed_ranges=fixed_stretch_ranges,
    )
    sar_before = RasterPanel(
        sar_rgb_before,
        sar_template_before.extent,
        sar_template_before.transform,
        sar_template_before.crs,
    )
    sar_after = RasterPanel(
        sar_rgb_after,
        sar_template_after.extent,
        sar_template_after.transform,
        sar_template_after.crs,
    )

    figure = render_figure(
        series=series,
        events=events,
        planet_before=planet_before,
        planet_after=planet_after,
        sar_before=sar_before,
        sar_after=sar_after,
        target_polygon=target_polygon,
        bbox=bbox,
        before_date=args.before_date,
        after_date=args.after_date,
        before_month=args.before_month,
        after_month=args.after_month,
        pixel_iqr=pixel_iqr,
        dedicated_iqr_legend=args.dedicated_iqr_legend,
        inside_iqr_legend=args.inside_iqr_legend,
        time_ylim=time_ylim,
        time_y_tick_interval=args.time_ytick_interval,
        figure_width_inches=args.figure_width_inches,
        map_wspace=args.map_wspace,
        map_left_margin=args.map_left_margin,
        map_right_margin=args.map_right_margin,
        map_row_height_scale=args.map_row_height_scale,
        longitude_tick_count=args.longitude_tick_count,
        scale_bar_m=args.scale_bar_m,
        map_gap_pt=args.map_gap_pt,
    )
    layout_diagnostics = figure_layout_diagnostics(figure)
    crop_bbox = publication_crop_bbox(figure, args.publication_padding_mm)
    crop_diagnostics = {
        "mode": "content_aware_union_bbox",
        "padding_mm": float(args.publication_padding_mm),
        "native_width_mm": float(crop_bbox.width * 25.4),
        "native_height_mm": float(crop_bbox.height * 25.4),
        "recommended_latex_width": "\\textwidth",
        "nominal_two_column_target_width_mm": 180.0,
        "latex_scale_to_180mm": float(180.0 / (crop_bbox.width * 25.4)),
    }
    try:
        figure.savefig(
            pdf_path,
            format="pdf",
            bbox_inches=crop_bbox,
            metadata={
                "Title": "Polygon-first Sentinel-1 signature for reference polygon 1499",
                "Author": "LR-CuSum reproducible figure workflow",
                "Subject": "Sentinel-1 stand-disturbance signature",
                "Creator": "Matplotlib",
            },
        )
        figure.savefig(
            png_path, format="png", dpi=args.dpi, bbox_inches=crop_bbox
        )
        figure.savefig(
            tiff_path, format="tiff", dpi=600, bbox_inches=crop_bbox
        )
        figure.savefig(
            svg_path,
            format="svg",
            bbox_inches=crop_bbox,
            metadata={
                "Title": "Polygon-first Sentinel-1 signature for reference polygon 1499",
                "Creator": "Matplotlib",
                "Description": "Before/after optical and Sentinel-1 evidence",
            },
        )
    finally:
        plt.close(figure)

    temporal_iqr_manifest: dict[str, Any] = {"enabled": False}
    if pixel_iqr is not None:
        temporal_iqr_manifest = {
            "enabled": True,
            "definition": (
                "Q25--Q75 across valid native polygon-interior pixels within each "
                "track and month; corresponding track quantiles are median-fused."
            ),
            "display_alpha": 0.16,
            "valid_pixel_count_vv": {
                "minimum": int(pixel_iqr.valid_pixel_counts_vv.min()),
                "median": float(np.median(pixel_iqr.valid_pixel_counts_vv)),
                "maximum": int(pixel_iqr.valid_pixel_counts_vv.max()),
            },
            "valid_pixel_count_vh": {
                "minimum": int(pixel_iqr.valid_pixel_counts_vh.min()),
                "median": float(np.median(pixel_iqr.valid_pixel_counts_vh)),
                "maximum": int(pixel_iqr.valid_pixel_counts_vh.max()),
            },
            "metadata": pixel_iqr.metadata,
        }
    input_manifest = {
        "series_metadata": file_signature(args.series_metadata),
        "series_array": file_signature(args.series_array),
        "reference_events": file_signature(args.reference_events),
        "case_shapefile": file_signature(args.case_shapefile),
        "planet_before": file_signature(args.planet_before),
        "planet_after": file_signature(args.planet_after),
        "s1grits_zarr": zarr_signature(args.zarr),
    }
    if pixel_iqr is not None:
        input_manifest.update(
            {
                "polygon_tracks": file_signature(args.polygon_tracks),
                "pixel_associations": file_signature(args.pixel_associations),
                "pixel_cache_directory": str(args.pixel_cache.resolve()),
            }
        )
    iqr_caption = (
        " Shading denotes the within-polygon native-pixel interquartile range "
        "(Q25--Q75) for each polarization."
        if pixel_iqr is not None
        else ""
    )

    manifest = {
        "schema_version": (
            9
            if args.inside_iqr_legend
            else (
                8
                if args.dedicated_iqr_legend
                else (7 if pixel_iqr is not None else 6)
            )
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": str(Path(__file__).resolve()),
        "outputs": {
            "pdf": file_signature(pdf_path),
            "png": file_signature(png_path),
            "svg": file_signature(svg_path),
            "tiff": file_signature(tiff_path),
        },
        "case": {
            "sample_id": args.sample_id,
            "historical_polygon_id": str(args.historical_polygon_id),
            "tile_id": args.tile_id,
            "track_id": args.track_id,
            "target_crs": args.target_crs,
            "bbox_projected": [float(value) for value in bbox],
            "crop": {
                "mode": "scaled_target_polygon_buffer",
                "padding_m": float(args.crop_padding_m),
                "base_bbox_projected": [float(value) for value in base_bbox],
                "base_width_m": float(base_bbox[2] - base_bbox[0]),
                "base_height_m": float(base_bbox[3] - base_bbox[1]),
                "extent_scale_factor": float(args.extent_scale_factor),
                "expanded_bbox_projected": [
                    float(value) for value in expanded_bbox
                ],
                "map_height_scale_factor": float(args.map_height_scale_factor),
                "grid_alignment_m": float(args.grid_resolution_m),
                "width_m": float(bbox[2] - bbox[0]),
                "height_m": float(bbox[3] - bbox[1]),
                "centre_shift_m": {
                    "x": float(
                        (bbox[0] + bbox[2] - base_bbox[0] - base_bbox[2]) / 2.0
                    ),
                    "y": float(
                        (bbox[1] + bbox[3] - base_bbox[1] - base_bbox[3]) / 2.0
                    ),
                },
                "target_geometry_margins": target_margins,
            },
            "reference_events": [
                pd.Timestamp(value).strftime("%Y-%m")
                for value in events["event_month"]
            ],
        },
        "temporal_panel": {
            "aggregation": (
                "Within each track and month, median over all valid polygon-interior "
                "pixels; then median across track-level polygon series."
            ),
            "calendar_start": series.calendar.min().strftime("%Y-%m"),
            "calendar_end": series.calendar.max().strftime("%Y-%m"),
            "month_count": int(len(series.calendar)),
            "display_start_month": series.calendar[np.flatnonzero(
                np.isfinite(series.vv) | np.isfinite(series.vh)
            )[0]].strftime("%Y-%m"),
            "display_end_month": series.calendar[np.flatnonzero(
                np.isfinite(series.vv) | np.isfinite(series.vh)
            )[-1]].strftime("%Y-%m"),
            "valid_months_vv": int(np.isfinite(series.vv).sum()),
            "valid_months_vh": int(np.isfinite(series.vh).sum()),
            "missing_policy": "NaN values remain gaps; no zero fill or interpolation.",
            "legend_layout": (
                "inside time panel lower-left, one column, two rows"
                if args.inside_iqr_legend
                else (
                    "dedicated band, one row, two columns"
                    if args.dedicated_iqr_legend
                    else "lower-left, one column, two rows"
                )
            ),
            "dedicated_iqr_legend": bool(args.dedicated_iqr_legend),
            "inside_iqr_legend": bool(args.inside_iqr_legend),
            "fixed_y_limits_db": (
                [float(time_ylim[0]), float(time_ylim[1])]
                if time_ylim is not None
                else None
            ),
            "y_tick_interval_db": float(args.time_ytick_interval),
            "event_label_orientation": "horizontal above the plotting frame",
            "time_axis_font_size_pt": TIME_AXIS_FONT_SIZE,
            "event_label_font_size_pt": TIME_AXIS_FONT_SIZE,
            "axis_tick_width_pt": 0.65,
            "axis_tick_length_pt": 2.2,
            "panel_label_font_size_pt": PANEL_LABEL_FONT_SIZE,
            "temporal_palette": "Bisan Gardenia",
            "temporal_palette_hex": dict(TEMPORAL_PALETTE),
            "figure_width_inches": float(args.figure_width_inches),
            "series_line_width_pt": 1.5,
            "map_wspace": float(args.map_wspace),
            "map_left_margin": float(args.map_left_margin),
            "map_right_margin": float(args.map_right_margin),
            "map_row_height_scale": float(args.map_row_height_scale),
            "time_map_hspace": 0.14 if args.inside_iqr_legend else None,
            "map_gap_pt": float(args.map_gap_pt),
            "map_wspace_effective": float(
                args.map_wspace if args.map_gap_pt == 0 else (
                    (args.map_gap_pt / (72.0 * args.figure_width_inches)) * 4.0
                    / ((args.map_right_margin - args.map_left_margin)
                       - (args.map_gap_pt / (72.0 * args.figure_width_inches)) * 3.0)
                )
            ),
            "square_map_panels": bool(args.square_map_panels),
            "rectangular_bbox_projected": [float(value) for value in rectangular_bbox],
            "square_bbox_projected": [float(value) for value in bbox],
            "monthly_scatter": {
                "enabled": True,
                "marker_area_pt2": 7.0,
                "valid_points_vv": int(np.isfinite(series.vv).sum()),
                "valid_points_vh": int(np.isfinite(series.vh).sum()),
            },
            "pixel_iqr": temporal_iqr_manifest,
            "frozen_metadata": series.metadata,
            "illustrative_values": _series_event_values(series),
        },
        "image_panels": {
            "layout": {
                "reading_order": ["b", "c", "d", "e"],
                "semantic_order": [
                    "PlanetScope_before",
                    "Sentinel1_before",
                    "PlanetScope_after",
                    "Sentinel1_after",
                ],
                "before_disturbance": ["b", "c"],
                "after_disturbance": ["d", "e"],
                "diagnostics": layout_diagnostics,
                "publication_crop": crop_diagnostics,
            },
                "display": {
                    "scale_bar_m": float(args.scale_bar_m),
                    "scale_bar_anchor": "lower-right",
                "geographic_grid": {
                    "enabled": True,
                    "color": "black",
                    "linewidth_pt": 0.42,
                    "alpha": 0.52,
                    "source": "sparse geographic axis ticks",
                },
                "map_width_m": float(bbox[2] - bbox[0]),
                "map_height_m": float(bbox[3] - bbox[1]),
                "scale_bar_fraction_of_map_width": scale_bar_fraction(
                    bbox, args.scale_bar_m
                ),
                "sentinel1_pixel_spacing_m": 30.0,
                "context_polygon_boundaries_visible": False,
                "planet_outside_mask": "target polygon only",
                "target_boundary": "black casing with white inner line",
                "panel_notes": {
                    "b": _format_panel_date(args.before_date, include_day=True),
                    "c": _format_panel_date(args.before_month, include_day=False),
                    "d": _format_panel_date(args.after_date, include_day=True),
                    "e": _format_panel_date(args.after_month, include_day=False),
                },
                "panel_note_alignment": "centered at top of each map panel",
                "group_header_anchor": "2 pt above the shifted map-row bbox",
                "group_header_labels": [
                    "Before stand-disturbance",
                    "After stand-disturbance",
                ],
                "group_header_font_size_pt": TIME_AXIS_FONT_SIZE,
                "group_header_downshift_pt": 5.0,
                "map_row_shift_figure_fraction": -0.03401,
                "map_row_shift_relative_to_v37_pt": 7.0,
                "map_row_shift_relative_to_v39_pt": 2.0,
                "group_header_downshift_relative_to_v37_pt": 5.0,
                "group_header_downshift_relative_to_v39_pt": 2.0,
                "group_header_gap_above_map_pt": 2.0,
                "coordinate_labels": {
                    "longitude_per_panel": (
                        sparse_degree_ticks(
                            bbox, args.target_crs,
                            longitude_count=args.longitude_tick_count,
                        )[1]
                    ),
                    "latitude_on_panel_b": [
                        "0.54\N{DEGREE SIGN}S",
                        "0.53\N{DEGREE SIGN}S",
                    ],
                    "latitude_on_panels_c_to_e": [],
                    "selection_rule": (
                        f"{args.longitude_tick_count} longitude tick(s) inside the "
                        "shared extent (a symmetric rounded pair when count=2), "
                        "plus the two southernmost valid 0.01-degree latitude ticks."
                    ),
                },
            },
            "planet_before_date": args.before_date,
            "planet_after_date": args.after_date,
            "sar_before": sar_before_metadata,
            "sar_after": sar_after_metadata,
            "sar_false_colour": {
                "red": "VV_dB",
                "green": "VH_dB",
                "blue": "VV_dB - VH_dB",
                "joint_percentile_stretch": [2.0, 98.0],
                "stretch_ranges": stretch_ranges,
                "stretch_mode": (
                    "frozen_from_manifest"
                    if args.sar_stretch_manifest is not None
                    else "joint_percentile_from_current_extent"
                ),
                "stretch_manifest": (
                    file_signature(args.sar_stretch_manifest)
                    if args.sar_stretch_manifest is not None
                    else None
                ),
            },
        },
        "caption_guidance": (
            "The black-cased white boundary identifies historical reference polygon "
            "1499 (frozen sample ID plantation:8655). Panel (a) shows the 2017-2025 "
            "Polygon-first monthly median VV and VH trajectories of the same stand. "
            "Panels (b) and (d) are PlanetScope observations before and after the "
            "March 2024 disturbance; panels (c) and (e) are the corresponding "
            "February and March Sentinel-1 monthly median composites. All image "
            f"panels share a {(bbox[2] - bbox[0]) / 1000:.2f} km by "
            f"{(bbox[3] - bbox[1]) / 1000:.2f} km extent and a true "
            f"{args.scale_bar_m:.0f} m scale bar in EPSG:32717; Sentinel-1 "
            "composites have 30 m pixel spacing."
            + iqr_caption
        ),
        "inputs": input_manifest,
        "non_overwrite_guarantee": (
            "The workflow aborts when any target PDF, PNG, SVG, or manifest already exists."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")
    print(f"TIFF: {tiff_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
