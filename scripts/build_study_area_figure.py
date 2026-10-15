#!/usr/bin/env python3
"""Build the reproducible four-panel Ecuador BALSA study-area figure.

The workflow is intentionally independent of the historical notebook. It
loads repository-relative inputs, validates the scientific sample semantics,
assembles panels (a)--(d), writes publication outputs and provenance, and can
atomically publish the validated PDF to the manuscript directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Conda on Windows may not expose GDAL_DATA even though the packaged resource
# directory exists. Resolve that standard location before importing GeoPandas
# so vector reads do not emit misleading missing-resource warnings.
if "GDAL_DATA" not in os.environ:
    conda_gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
    if conda_gdal_data.is_dir():
        os.environ["GDAL_DATA"] = str(conda_gdal_data)

import cartopy
from cartopy.io import shapereader
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pyproj import Geod
from shapely.geometry import box


DEFAULT_CONFIG = "configs/study_area_figure_v1.json"
PANEL_ORDER = ("b", "c", "d")
TILE_SIZE = 256
WEB_MERCATOR_LAT_LIMIT = 85.05112878
GEOD = Geod(ellps="WGS84")


class FigureBuildError(RuntimeError):
    """Raised when a reproducibility or data-contract assertion fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the publication-ready Ecuador BALSA study-area figure."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help="Repository-relative or absolute JSON configuration path.",
    )
    parser.add_argument(
        "--layout-variant",
        choices=("v8", "v9a", "v9b", "v9c", "v10"),
        default="v8",
        help=(
            "Apply a frozen study-area layout variant: V9-A moves only the "
            "panel-(c) national callout, V9-B adds twofold detail enlargement, "
            "V9-C adds moderate official-polygon contrast, and V10 repositions "
            "panels (c)/(d) with uniform lower-right scale bars."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the canonical output directory.",
    )
    parser.add_argument(
        "--basemap-mode",
        choices=("cache", "download", "none"),
        default=None,
        help="Override the configured basemap mode.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Override raster output DPI, primarily for smoke testing.",
    )
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip the optional SVG output.",
    )
    parser.add_argument(
        "--export-panels",
        action="store_true",
        help="Also export panel-level PNG diagnostics.",
    )
    parser.add_argument(
        "--export-style-comparison",
        action="store_true",
        help="Export a controlled three-palette polygon-style comparison.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Atomically replace the manuscript study_area.pdf after validation.",
    )
    parser.add_argument(
        "--allow-vector-only-publish",
        action="store_true",
        help="Permit publication when --basemap-mode=none. Disabled by default.",
    )
    return parser.parse_args(argv)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FigureBuildError(f"Configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FigureBuildError(f"Invalid JSON configuration: {path}: {exc}") from exc
    if config.get("schema_version") != "study-area-figure-config/1.0":
        raise FigureBuildError(
            "Unsupported configuration schema: "
            f"{config.get('schema_version')!r}"
        )
    return config


def apply_layout_variant(
    config: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Return a self-contained frozen V8/V9 rendering configuration."""

    resolved = copy.deepcopy(dict(config))
    if variant == "v8":
        return resolved

    # V9-A changes only the national callout. It provides the single-factor
    # control required before changing any detail-map extent or class style.
    resolved["detail_panels"]["c"]["national_callout"] = {
        "longitude": -79.132,
        "latitude": -0.493,
        "horizontal_alignment": "left",
    }
    resolved["national_panel"]["required_collision_free_callouts"] = ["c", "d"]
    if variant == "v9a":
        return resolved

    enlarged_panels = {
        "b": {
            "name": "northern_western_zoom2x",
            # Preserve the common 2.55:1 geographic aspect while adding a
            # three-percent geographic margin around every displayed
            # inventory polygon. This prevents polygon 2640 from being
            # clipped by the upper frame in the original V9-B candidate.
            "extent": [
                -80.005886900603,
                -79.869985875450,
                0.628161137526,
                0.681455650748,
            ],
            "longitude_ticks": [-79.98, -79.90],
            "latitude_ticks": [0.65],
        },
        "c": {
            "name": "central_western_zoom2x",
            "extent": [
                -79.455387121875,
                -79.258161571875,
                -0.5000001875,
                -0.4226567875,
            ],
            "national_callout": {
                "longitude": -79.243,
                "latitude": -0.470,
                "horizontal_alignment": "left",
            },
            "longitude_ticks": [-79.40, -79.30],
            "latitude_ticks": [-0.46],
        },
        "d": {
            "name": "eastern_inventory_only_zoom2x",
            "extent": [-77.13244010, -76.90195110, -0.29731040, -0.20692240],
            "longitude_ticks": [-77.10, -76.95],
            "latitude_ticks": [-0.25],
        },
    }
    for panel, settings in enlarged_panels.items():
        resolved["detail_panels"][panel].update(settings)
    resolved["expected"]["panel_intersection_counts"] = {
        "b": 125,
        "c": 150,
        "d": 71,
    }
    resolved["expected"]["panel_reference_counts"] = {
        "b": 2,
        "c": 23,
        "d": 0,
    }
    resolved["expected"]["panel_complete_counts"] = {
        "b": 125,
        "c": 150,
        "d": 71,
    }
    resolved["v8_layout_targets"]["eastern_panel_minimum_complete_polygons"] = 65

    if variant == "v10":
        # Expand only the western and southern sides of panels (c) and (d).
        # Keeping their eastern and northern limits fixed moves all mapped
        # content right and upward on the page. The southern margin makes the
        # common lower-right scale-bar footprint genuinely data-free.
        resolved["detail_panels"]["c"].update(
            {
                "name": "central_western_repositioned",
                "extent": [
                    -79.489901593125,
                    -79.258161571875,
                    -0.513535282500,
                    -0.422656787500,
                ],
            }
        )
        resolved["detail_panels"]["d"].update(
            {
                "name": "eastern_inventory_only_repositioned",
                "extent": [
                    -77.155489000000,
                    -76.901951100000,
                    -0.306349200000,
                    -0.206922400000,
                ],
            }
        )
        resolved["detail_map_decorations"]["scale_bar_y_candidates"] = [0.055]
        resolved["detail_map_decorations"]["require_uniform_scale_anchor"] = True
        resolved["expected"]["panel_intersection_counts"] = {
            "b": 125,
            "c": 151,
            "d": 71,
        }
        resolved["expected"]["panel_reference_counts"] = {
            "b": 2,
            "c": 23,
            "d": 0,
        }
        resolved["expected"]["panel_complete_counts"] = {
            "b": 125,
            "c": 151,
            "d": 71,
        }
        resolved["selected_palette"] = (
            "V10 repositioned detail maps with uniform scale anchors"
        )

    if variant == "v9c":
        polygon_style = resolved["figure"]["detail_polygon_style"]
        polygon_style["inventory_alpha"] = 0.90
        polygon_style["inventory_edge"] = "#4B55B8"
        polygon_style["inventory_edge_width"] = 0.16
        resolved["selected_palette"] = "V9-C moderate official-polygon contrast"
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        display = path.relative_to(root).as_posix()
    except ValueError:
        display = str(path)
    return {
        "path": display,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def shapefile_records(path: Path, root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sidecar in sorted(path.parent.glob(f"{path.stem}.*")):
        if sidecar.is_file() and not sidecar.name.endswith(".shp.xml"):
            records.append(path_record(sidecar, root))
    if not records:
        raise FigureBuildError(f"No shapefile components found for {path}")
    return records


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    names = (
        "numpy",
        "pandas",
        "geopandas",
        "matplotlib",
        "Pillow",
        "pyproj",
        "shapely",
        "cartopy",
        "pyarrow",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def normalize_inventory_ids(values: pd.Series) -> pd.Series:
    def normalize(value: Any) -> str:
        if pd.isna(value):
            raise FigureBuildError("Inventory ID contains a missing value.")
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    return values.map(normalize)


def load_inventory(
    path: Path,
    expected: Mapping[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    if not path.exists():
        raise FigureBuildError(f"Inventory shapefile not found: {path}")
    inventory = gpd.read_file(path)
    required = {"ID", "nco", "geometry"}
    missing = required.difference(inventory.columns)
    if missing:
        raise FigureBuildError(f"Inventory is missing columns: {sorted(missing)}")
    if inventory.crs is None:
        raise FigureBuildError("Inventory CRS is missing.")

    expected_count = int(expected["inventory_polygons"])
    if len(inventory) != expected_count:
        raise FigureBuildError(
            f"Expected {expected_count} inventory polygons, found {len(inventory)}."
        )
    inventory = inventory.copy()
    inventory["inventory_id"] = normalize_inventory_ids(inventory["ID"])
    if inventory["inventory_id"].nunique() != expected_count:
        raise FigureBuildError("Inventory IDs are not unique.")
    if inventory.geometry.is_empty.any() or inventory.geometry.isna().any():
        raise FigureBuildError("Inventory contains empty or missing geometries.")
    invalid_count = int((~inventory.geometry.is_valid).sum())
    if invalid_count:
        raise FigureBuildError(f"Inventory contains {invalid_count} invalid geometries.")
    species = sorted(
        inventory["nco"].dropna().astype(str).str.strip().str.upper().unique()
    )
    if species != ["BALSA"]:
        raise FigureBuildError(f"Unexpected inventory species values: {species}")

    metric_epsg = int(expected["metric_epsg"])
    metric = inventory.to_crs(epsg=metric_epsg)
    metric["area_ha"] = metric.geometry.area / 10_000.0
    if (metric["area_ha"] <= 0).any():
        raise FigureBuildError("Inventory contains non-positive polygon area.")

    area_field = str(expected.get("area_field", "")).strip()
    area_check: dict[str, Any] = {"field": area_field or None}
    if area_field and area_field in metric.columns:
        source_area = pd.to_numeric(metric[area_field], errors="coerce")
        if source_area.isna().any():
            raise FigureBuildError(f"Area field {area_field!r} contains missing values.")
        difference = (metric["area_ha"] - source_area).abs()
        tolerance = float(expected["area_absolute_tolerance_ha"])
        area_check.update(
            {
                "maximum_absolute_difference_ha": float(difference.max()),
                "tolerance_ha": tolerance,
                "passed": bool(difference.max() <= tolerance),
            }
        )
        if not area_check["passed"]:
            raise FigureBuildError(
                f"Geometry-derived area disagrees with {area_field!r}: "
                f"max difference {difference.max():.6f} ha exceeds {tolerance:.6f} ha."
            )

    points_metric = metric[["inventory_id", "area_ha", "geometry"]].copy()
    points_metric.geometry = metric.geometry.representative_point()
    points_wgs84 = points_metric.to_crs(epsg=4326)
    inventory_wgs84 = metric.to_crs(epsg=4326)

    quantiles = metric["area_ha"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    audit = {
        "polygon_count": int(len(metric)),
        "unique_id_count": int(metric["inventory_id"].nunique()),
        "source_crs": str(inventory.crs),
        "metric_crs": f"EPSG:{metric_epsg}",
        "species_values": species,
        "invalid_geometry_count": invalid_count,
        "area_quantiles_ha": {
            f"q{int(q * 100):02d}": float(value) for q, value in quantiles.items()
        },
        "area_field_check": area_check,
    }
    return inventory_wgs84, points_wgs84, audit


def load_reference_matches(
    path: Path,
    inventory_wgs84: gpd.GeoDataFrame,
    inventory_points_wgs84: gpd.GeoDataFrame,
    expected: Mapping[str, Any],
) -> tuple[pd.DataFrame, set[str], dict[str, Any]]:
    if not path.exists():
        raise FigureBuildError(f"Reference-event table not found: {path}")
    events = pd.read_parquet(path)
    required = {"sample_id", "longitude", "latitude", "event_id"}
    missing = required.difference(events.columns)
    if missing:
        raise FigureBuildError(f"Reference events are missing columns: {sorted(missing)}")

    coordinate_counts = events.groupby("sample_id")[["longitude", "latitude"]].nunique()
    if (coordinate_counts > 1).any(axis=None):
        bad = coordinate_counts[(coordinate_counts > 1).any(axis=1)].index.tolist()
        raise FigureBuildError(
            f"Reference samples have inconsistent coordinates: {bad[:10]}"
        )
    references = events.sort_values(["sample_id", "event_id"]).drop_duplicates("sample_id")
    expected_count = int(expected["reference_polygons"])
    if len(references) != expected_count:
        raise FigureBuildError(
            f"Expected {expected_count} reference polygons, found {len(references)}."
        )

    reference_points = gpd.GeoDataFrame(
        references[["sample_id", "longitude", "latitude"]].copy(),
        geometry=gpd.points_from_xy(references["longitude"], references["latitude"]),
        crs="EPSG:4326",
    )
    metric_epsg = int(expected["metric_epsg"])
    reference_metric = reference_points.to_crs(epsg=metric_epsg)
    inventory_metric = inventory_wgs84.to_crs(epsg=metric_epsg)[
        ["inventory_id", "area_ha", "geometry"]
    ]
    joined = gpd.sjoin(
        reference_metric,
        inventory_metric,
        how="left",
        predicate="within",
    )
    matched = joined.loc[joined["inventory_id"].notna()].copy()
    matched_counts = matched.groupby("sample_id").size()
    ambiguous = matched_counts.loc[matched_counts.ne(1)]
    matched_ids = set(matched["sample_id"].astype(str))
    missing_samples = sorted(set(references["sample_id"].astype(str)) - matched_ids)
    if missing_samples or len(ambiguous):
        raise FigureBuildError(
            "Reference-to-inventory spatial join is not one-to-one: "
            f"unmatched={len(missing_samples)}, ambiguous={len(ambiguous)}."
        )

    matches = matched[
        ["sample_id", "longitude", "latitude", "inventory_id", "area_ha"]
    ].copy()
    matches["sample_id"] = matches["sample_id"].astype(str)
    matches["inventory_id"] = matches["inventory_id"].astype(str)
    matches = matches.sort_values("sample_id").reset_index(drop=True)
    reference_inventory_ids = set(matches["inventory_id"])
    if len(reference_inventory_ids) != expected_count:
        raise FigureBuildError(
            "More than one reference sample maps to the same official inventory polygon."
        )

    available_ids = set(inventory_points_wgs84["inventory_id"])
    if not reference_inventory_ids.issubset(available_ids):
        raise FigureBuildError("Reference matches contain unknown inventory IDs.")

    audit = {
        "event_count": int(events["event_id"].nunique()),
        "reference_polygon_count": int(len(references)),
        "matched_inventory_polygon_count": int(len(reference_inventory_ids)),
        "unmatched_count": 0,
        "ambiguous_count": 0,
        "predicate": "within",
        "metric_crs": f"EPSG:{metric_epsg}",
    }
    return matches, reference_inventory_ids, audit


def area_to_marker_size(
    areas: Iterable[float],
    lower_area: float,
    upper_area: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    values = np.asarray(list(areas), dtype=float)
    if not np.isfinite(values).all():
        raise FigureBuildError("Marker-size input contains non-finite areas.")
    if upper_area <= lower_area:
        raise FigureBuildError("Marker-size area range must be positive.")
    scaled = (np.clip(values, lower_area, upper_area) - lower_area) / (
        upper_area - lower_area
    )
    return minimum + scaled * (maximum - minimum)


def format_longitude(value: float, _position: int | None = None, decimals: int = 0) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-10):
        return "0°"
    suffix = "W" if value < 0 else "E"
    return f"{abs(value):.{decimals}f}°{suffix}"


def format_latitude(value: float, _position: int | None = None, decimals: int = 0) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-10):
        return "0°"
    suffix = "S" if value < 0 else "N"
    return f"{abs(value):.{decimals}f}°{suffix}"


def validate_extent(extent: Sequence[float], label: str) -> tuple[float, float, float, float]:
    if len(extent) != 4:
        raise FigureBuildError(f"{label} extent must contain four values.")
    xmin, xmax, ymin, ymax = map(float, extent)
    if not (xmin < xmax and ymin < ymax):
        raise FigureBuildError(f"Invalid {label} extent: {extent}")
    return xmin, xmax, ymin, ymax


def world_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    clipped_lat = float(np.clip(lat, -WEB_MERCATOR_LAT_LIMIT, WEB_MERCATOR_LAT_LIMIT))
    scale = TILE_SIZE * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(clipped_lat))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return x, y


def required_tiles(extent: Sequence[float], zoom: int) -> list[tuple[int, int, int]]:
    xmin, xmax, ymin, ymax = validate_extent(extent, "basemap")
    left, top = world_pixel(xmin, ymax, zoom)
    right, bottom = world_pixel(xmax, ymin, zoom)
    x0 = int(math.floor(left / TILE_SIZE))
    x1 = int(math.floor(np.nextafter(right, -np.inf) / TILE_SIZE))
    y0 = int(math.floor(top / TILE_SIZE))
    y1 = int(math.floor(np.nextafter(bottom, -np.inf) / TILE_SIZE))
    limit = 2**zoom
    tiles = []
    for x in range(max(0, x0), min(limit - 1, x1) + 1):
        for y in range(max(0, y0), min(limit - 1, y1) + 1):
            tiles.append((zoom, x, y))
    if not tiles:
        raise FigureBuildError(f"No basemap tiles selected for extent {extent} at zoom {zoom}.")
    return tiles


def download_tile(
    url: str,
    destination: Path,
    timeout_seconds: int,
    retries: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    last_error: Exception | None = None

    def attempt_with_curl(curl_executable: str) -> bool:
        nonlocal last_error
        command = [
            curl_executable,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--connect-timeout",
            str(timeout_seconds),
            "--max-time",
            str(max(60, timeout_seconds * 3)),
            "--output",
            str(temporary),
            url,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, destination)
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            return False

    curl = shutil.which("curl")
    # The system TLS stack is more reliable than Python/OpenSSL behind some
    # managed Windows proxies. Prefer it there and retain urllib elsewhere.
    if os.name == "nt" and curl is not None and attempt_with_curl(curl):
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LR-CuSum-study-area-figure/1.0"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            temporary.write_bytes(payload)
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, destination)
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    # Some managed Windows networks terminate Python/OpenSSL connections while
    # the system curl client succeeds through the same proxy. Keep this as a
    # transparent, HTTPS-only fallback; the downloaded payload is still
    # decoded and verified before the atomic cache replacement.
    if os.name != "nt" and curl is not None and attempt_with_curl(curl):
        return
    raise FigureBuildError(f"Failed to download basemap tile {url}: {last_error}")


def build_basemap_mosaic(
    extent: Sequence[float],
    zoom: int,
    mode: str,
    cache_dir: Path,
    settings: Mapping[str, Any],
    root: Path,
) -> tuple[np.ndarray | None, list[dict[str, Any]], str | None]:
    if mode == "none":
        return None, [], None
    tiles = required_tiles(extent, zoom)
    x_values = sorted({x for _, x, _ in tiles})
    y_values = sorted({y for _, _, y in tiles})
    mosaic = Image.new(
        "RGB",
        (len(x_values) * TILE_SIZE, len(y_values) * TILE_SIZE),
    )
    records: list[dict[str, Any]] = []
    url_template = str(settings["url_template"])
    service = str(settings["service_name"])
    for z, x, y in tiles:
        url = url_template.format(z=z, x=x, y=y)
        tile_path = cache_dir / service / str(z) / str(x) / f"{y}.png"
        if not tile_path.exists():
            if mode == "cache":
                raise FigureBuildError(
                    f"Basemap tile is absent in cache mode: {tile_path}. "
                    "Run once with --basemap-mode download."
                )
            download_tile(
                url,
                tile_path,
                int(settings["timeout_seconds"]),
                int(settings["retries"]),
            )
        try:
            with Image.open(tile_path) as tile:
                tile_rgb = tile.convert("RGB")
                if tile_rgb.size != (TILE_SIZE, TILE_SIZE):
                    raise FigureBuildError(
                        f"Unexpected tile dimensions for {tile_path}: {tile_rgb.size}"
                    )
                mosaic.paste(
                    tile_rgb,
                    ((x - x_values[0]) * TILE_SIZE, (y - y_values[0]) * TILE_SIZE),
                )
        except OSError as exc:
            raise FigureBuildError(f"Cannot read basemap tile {tile_path}: {exc}") from exc
        record = path_record(tile_path, root)
        record.update({"z": z, "x": x, "y": y, "url": url})
        records.append(record)

    xmin, xmax, ymin, ymax = validate_extent(extent, "basemap")
    left, top = world_pixel(xmin, ymax, zoom)
    right, bottom = world_pixel(xmax, ymin, zoom)
    origin_x = x_values[0] * TILE_SIZE
    origin_y = y_values[0] * TILE_SIZE
    crop = (
        int(round(left - origin_x)),
        int(round(top - origin_y)),
        int(round(right - origin_x)),
        int(round(bottom - origin_y)),
    )
    cropped = mosaic.crop(crop)
    array = np.asarray(cropped)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    return array, records, digest


def natural_earth_layer(
    resolution: str,
    category: str,
    name: str,
    root: Path,
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    try:
        shp = Path(shapereader.natural_earth(resolution, category, name)).resolve()
    except Exception as exc:
        raise FigureBuildError(
            f"Natural Earth layer is unavailable: {resolution}/{category}/{name}: {exc}"
        ) from exc
    layer = gpd.read_file(shp).to_crs(epsg=4326)
    return layer, shapefile_records(shp, root)


def clipped(layer: gpd.GeoDataFrame, extent: Sequence[float]) -> gpd.GeoDataFrame:
    xmin, xmax, ymin, ymax = validate_extent(extent, "clip")
    selector = layer.geometry.intersects(box(xmin, ymin, xmax, ymax))
    local = layer.loc[selector].copy()
    if local.empty:
        return local
    return gpd.clip(local, box(xmin, ymin, xmax, ymax))


def build_panel_membership(
    inventory: gpd.GeoDataFrame,
    reference_ids: set[str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Record the exact inventory membership and clipping status of each inset."""

    records: list[dict[str, Any]] = []
    for panel in PANEL_ORDER:
        extent = validate_extent(config["detail_panels"][panel]["extent"], panel)
        frame = box(extent[0], extent[2], extent[1], extent[3])
        selected = inventory.loc[inventory.geometry.intersects(frame)]
        for row in selected.itertuples():
            inventory_id = str(row.inventory_id)
            records.append(
                {
                    "panel": panel,
                    "panel_name": str(config["detail_panels"][panel]["name"]),
                    "inventory_id": inventory_id,
                    "area_ha": float(row.area_ha),
                    "is_complete_within_panel": bool(row.geometry.within(frame)),
                    "is_visually_interpreted": inventory_id in reference_ids,
                }
            )
    return pd.DataFrame.from_records(records)


def configure_matplotlib(settings: Mapping[str, Any]) -> None:
    fonts = list(settings["fonts"])
    font_size = float(settings["body_font_size_pt"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
        }
    )


def draw_panel_label(
    axis: plt.Axes,
    label: str,
    figure_settings: Mapping[str, Any],
) -> None:
    position = figure_settings["panel_label_positions"][label]
    axis.text(
        float(position[0]),
        float(position[1]),
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=float(figure_settings["panel_label_size_pt"]),
        fontweight="bold",
        zorder=100,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )


def apply_text_halo(text: Any, width: float = 1.4) -> None:
    text.set_path_effects(
        [
            path_effects.Stroke(linewidth=width, foreground="white", alpha=0.88),
            path_effects.Normal(),
        ]
    )


def draw_north_arrow(axis: plt.Axes, font_size: float) -> None:
    x = 0.055
    base_y = 0.19
    width = 0.025
    height = 0.13
    black = MplPolygon(
        [(x, base_y + height), (x - width, base_y), (x, base_y + height * 0.35)],
        transform=axis.transAxes,
        facecolor="black",
        edgecolor="black",
        linewidth=0.4,
        zorder=80,
    )
    white = MplPolygon(
        [(x, base_y + height), (x + width, base_y), (x, base_y + height * 0.35)],
        transform=axis.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=0.4,
        zorder=80,
    )
    axis.add_patch(black)
    axis.add_patch(white)
    axis.text(
        x,
        base_y + height + 0.025,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=font_size,
        zorder=80,
    )


def draw_metric_scale_bar(
    axis: plt.Axes,
    extent: Sequence[float],
    font_size: float,
    axes_anchor: Sequence[float],
    length_km: float,
) -> None:
    xmin, xmax, ymin, ymax = validate_extent(extent, "scale bar")
    x_span = xmax - xmin
    y_span = ymax - ymin
    anchor_x, anchor_y = map(float, axes_anchor)
    start_lon = xmin + anchor_x * x_span
    latitude = ymin + anchor_y * y_span
    end_lon, _, _ = GEOD.fwd(start_lon, latitude, 90.0, length_km * 1_000.0)
    tick_height = 0.018 * y_span
    line_width = 1.0 if length_km <= 2.0 else 1.25
    tick_width = 0.55 if length_km <= 2.0 else 0.65
    axis.plot(
        [start_lon, end_lon],
        [latitude, latitude],
        color="black",
        linewidth=line_width,
        solid_capstyle="butt",
        zorder=70,
    )
    axis.vlines(
        [start_lon, end_lon],
        latitude - tick_height / 2.0,
        latitude + tick_height / 2.0,
        colors="black",
        linewidth=tick_width,
        zorder=70,
    )
    axis.text(
        (start_lon + end_lon) / 2.0,
        latitude + tick_height * 0.85,
        f"{length_km:g} km",
        fontsize=font_size,
        ha="center",
        va="bottom",
        color="black",
        zorder=71,
        bbox={
            "boxstyle": "round,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.70,
        },
    )


def choose_scale_bar_anchor(
    extent: Sequence[float],
    polygons: gpd.GeoDataFrame,
    decoration_settings: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """Choose the least-obstructed lower-right scale-bar anchor."""

    xmin, xmax, ymin, ymax = validate_extent(extent, "scale-bar placement")
    x_span = xmax - xmin
    y_span = ymax - ymin
    length_km = float(decoration_settings["scale_bar_length_km"])
    x_margin = float(decoration_settings["scale_bar_x_margin_fraction"])
    padding = float(decoration_settings["scale_bar_collision_padding_fraction"])
    end_lon = xmax - x_margin * x_span
    candidates: list[dict[str, Any]] = []
    for y_fraction in decoration_settings["scale_bar_y_candidates"]:
        latitude = ymin + float(y_fraction) * y_span
        start_lon, _, _ = GEOD.fwd(end_lon, latitude, 270.0, length_km * 1_000.0)
        _, _, measured_length_m = GEOD.inv(
            start_lon,
            latitude,
            end_lon,
            latitude,
        )
        footprint = box(
            start_lon - padding * x_span,
            latitude - padding * y_span,
            end_lon + padding * x_span,
            latitude + 0.085 * y_span,
        )
        intersections = int(polygons.geometry.intersects(footprint).sum())
        candidates.append(
            {
                "axes_anchor": [
                    float((start_lon - xmin) / x_span),
                    float(y_fraction),
                ],
                "intersecting_polygons": intersections,
                "right_endpoint_axes_fraction": float(1.0 - x_margin),
                "measured_length_m": float(measured_length_m),
                "length_error_m": float(measured_length_m - length_km * 1_000.0),
                "footprint_extent": list(map(float, footprint.bounds)),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            int(item["intersecting_polygons"]),
            float(item["axes_anchor"][1]),
        ),
    )
    return list(selected["axes_anchor"]), {
        "length_km": length_km,
        "corner": str(decoration_settings["scale_bar_corner"]),
        "selected": selected,
        "candidates": candidates,
        "collision_free": bool(selected["intersecting_polygons"] == 0),
    }


def draw_basemap(
    axis: plt.Axes,
    image: np.ndarray | None,
    extent: Sequence[float],
    colors: Mapping[str, str],
    alpha: float,
) -> None:
    axis.set_facecolor(colors["land"])
    if image is not None:
        axis.imshow(
            image,
            extent=extent,
            origin="upper",
            interpolation="bilinear",
            alpha=alpha,
            zorder=0,
            aspect="auto",
        )


def select_locator_ecuador(countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for column in ("ADM0_A3", "SOV_A3", "ISO_A3"):
        if column in countries.columns:
            selected = countries.loc[countries[column].astype(str).str.upper() == "ECU"]
            if len(selected) == 1:
                return selected.copy()
    for column in ("ADMIN", "NAME", "SOVEREIGNT"):
        if column in countries.columns:
            selected = countries.loc[
                countries[column].astype(str).str.strip().str.casefold() == "ecuador"
            ]
            if len(selected) == 1:
                return selected.copy()
    raise FigureBuildError(
        "Natural Earth countries must contain exactly one identifiable Ecuador polygon."
    )


def set_geographic_aspect(
    axis: plt.Axes,
    extent: Sequence[float],
) -> float:
    _, _, ymin, ymax = validate_extent(extent, "geographic aspect")
    middle_latitude = (ymin + ymax) / 2.0
    aspect = 1.0 / math.cos(math.radians(middle_latitude))
    axis.set_aspect(aspect, adjustable="box", anchor="C")
    return aspect


def draw_locator_panel(
    axis: plt.Axes,
    extent: Sequence[float],
    image: np.ndarray | None,
    countries: gpd.GeoDataFrame,
    colors: Mapping[str, str],
    alpha: float,
) -> None:
    axis.set_facecolor(colors["ocean"])
    if image is not None:
        axis.imshow(
            image,
            extent=extent,
            origin="upper",
            interpolation="bilinear",
            alpha=alpha,
            zorder=0,
            aspect="auto",
        )
    local_countries = clipped(countries, extent)
    if local_countries.empty:
        raise FigureBuildError("No Natural Earth countries intersect the locator extent.")
    local_countries.plot(
        ax=axis,
        facecolor="#F7F6F1",
        edgecolor="#777777",
        linewidth=0.28,
        alpha=0.78,
        zorder=1,
    )
    locator_ecuador = clipped(select_locator_ecuador(countries), extent)
    if locator_ecuador.empty:
        raise FigureBuildError("Ecuador does not intersect the configured locator extent.")
    locator_ecuador.plot(
        ax=axis,
        facecolor=colors["reference"],
        edgecolor="#B33A18",
        linewidth=0.35,
        zorder=3,
    )
    xmin, xmax, ymin, ymax = validate_extent(extent, "locator")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    set_geographic_aspect(axis, extent)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#555555")


def build_class_legend(
    axis: plt.Axes,
    colors: Mapping[str, str],
    polygon_settings: Mapping[str, Any],
    font_size: float,
) -> None:
    handles: list[Patch] = [
        Patch(
            facecolor=str(polygon_settings["inventory_fill"]),
            edgecolor=str(polygon_settings["inventory_edge"]),
            linewidth=float(polygon_settings["inventory_edge_width"]),
            label="Official BALSA inventory",
        ),
        Patch(
            facecolor=str(polygon_settings["inventory_fill"]),
            edgecolor=colors["reference"],
            linewidth=float(polygon_settings["reference_edge_width"]),
            label="Visually interpreted subset",
        ),
    ]
    axis.set_axis_off()
    axis.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=2,
        frameon=False,
        borderpad=0.0,
        borderaxespad=0.0,
        labelspacing=0.2,
        handletextpad=0.25,
        columnspacing=0.40,
        handlelength=0.72,
        handleheight=0.80,
        fontsize=font_size,
    )


def draw_area_key(
    parent: plt.Axes,
    axes_bounds: Sequence[float],
    area_values: Sequence[float],
    marker_settings: Mapping[str, Any],
    lower_area: float,
    upper_area: float,
    font_size: float,
) -> plt.Axes:
    inset = parent.inset_axes(axes_bounds, zorder=95)
    inset.set_axis_off()
    marker_sizes = area_to_marker_size(
        area_values,
        lower_area,
        upper_area,
        float(marker_settings["minimum"]),
        float(marker_settings["maximum"]),
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=0.65,
            markersize=math.sqrt(float(size)),
            label=f"{area:.1f}",
        )
        for area, size in zip(area_values, marker_sizes)
    ]
    legend = inset.legend(
        handles=handles,
        title="Area (ha)",
        loc="center",
        ncol=1,
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="#777777",
        borderpad=0.30,
        labelspacing=0.22,
        handletextpad=0.35,
        fontsize=font_size,
        title_fontsize=font_size,
    )
    legend.get_frame().set_linewidth(0.4)
    return inset


def count_points_under_axes_bounds(
    points: gpd.GeoDataFrame,
    reference_ids: set[str],
    extent: Sequence[float],
    axes_bounds: Sequence[float],
) -> dict[str, Any]:
    x0, y0, width, height = map(float, axes_bounds)
    data_extent = axes_bounds_to_data_extent(extent, axes_bounds)
    x = points.geometry.x
    y = points.geometry.y
    inside = (
        (x >= data_extent[0])
        & (x <= data_extent[1])
        & (y >= data_extent[2])
        & (y <= data_extent[3])
    )
    reference = points["inventory_id"].astype(str).isin(reference_ids)
    return {
        "axes_bounds": [x0, y0, width, height],
        "data_extent": data_extent,
        "inventory_points_under_key": int(inside.sum()),
        "reference_points_under_key": int((inside & reference).sum()),
    }


def axes_bounds_to_data_extent(
    extent: Sequence[float],
    axes_bounds: Sequence[float],
) -> list[float]:
    xmin, xmax, ymin, ymax = validate_extent(extent, "axes-bounds conversion")
    x0, y0, width, height = map(float, axes_bounds)
    return [
        xmin + x0 * (xmax - xmin),
        xmin + (x0 + width) * (xmax - xmin),
        ymin + y0 * (ymax - ymin),
        ymin + (y0 + height) * (ymax - ymin),
    ]


def audit_boundary_under_axes_bounds(
    boundary: gpd.GeoDataFrame,
    extent: Sequence[float],
    axes_bounds: Sequence[float],
    clearance_fraction: float,
) -> dict[str, Any]:
    xmin, xmax, ymin, ymax = validate_extent(extent, "inset boundary overlap")
    data_extent = axes_bounds_to_data_extent(extent, axes_bounds)
    x_padding = float(clearance_fraction) * (xmax - xmin)
    y_padding = float(clearance_fraction) * (ymax - ymin)
    buffered_key = box(
        data_extent[0] - x_padding,
        data_extent[2] - y_padding,
        data_extent[1] + x_padding,
        data_extent[3] + y_padding,
    )
    hits = boundary.boundary.intersects(buffered_key)
    return {
        "data_extent": data_extent,
        "clearance_fraction": float(clearance_fraction),
        "boundary_segments_under_clearance": int(hits.sum()),
        "boundary_clear": bool(not hits.any()),
    }


def audit_text_artist_overlap(
    text_artist: Any,
    axis: plt.Axes,
    points: gpd.GeoDataFrame,
    reference_ids: set[str],
    boundary: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Conservatively audit an axis-aligned rendered text bounding box."""

    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    display_box = text_artist.get_window_extent(renderer=renderer)
    inverse = axis.transData.inverted()
    lower_left = inverse.transform((display_box.x0, display_box.y0))
    upper_right = inverse.transform((display_box.x1, display_box.y1))
    data_extent = [
        float(lower_left[0]),
        float(upper_right[0]),
        float(lower_left[1]),
        float(upper_right[1]),
    ]
    x = points.geometry.x
    y = points.geometry.y
    inside = (
        (x >= data_extent[0])
        & (x <= data_extent[1])
        & (y >= data_extent[2])
        & (y <= data_extent[3])
    )
    references = points["inventory_id"].astype(str).isin(reference_ids)
    text_box = box(
        data_extent[0],
        data_extent[2],
        data_extent[1],
        data_extent[3],
    )
    boundary_hits = boundary.boundary.intersects(text_box)
    return {
        "text": str(text_artist.get_text()),
        "data_extent": data_extent,
        "inventory_points_under_text": int(inside.sum()),
        "reference_points_under_text": int((inside & references).sum()),
        "boundary_segments_under_text": int(boundary_hits.sum()),
        "collision_free": bool(
            not inside.any() and not boundary_hits.any()
        ),
    }


def draw_national_panel(
    axis: plt.Axes,
    inventory: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    reference_ids: set[str],
    ecuador: gpd.GeoDataFrame,
    amazonia: gpd.GeoDataFrame,
    ocean: gpd.GeoDataFrame,
    extent: Sequence[float],
    image: np.ndarray | None,
    locator_image: np.ndarray | None,
    locator_countries: gpd.GeoDataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    figure_settings = config["figure"]
    colors = figure_settings["colors"]
    basemap_settings = config["basemap"]
    national_settings = config["national_panel"]
    marker_settings = figure_settings["marker_area_pt2"]
    body_font_size = float(figure_settings["body_font_size_pt"])
    draw_basemap(axis, image, extent, colors, float(basemap_settings["alpha"]))

    local_ocean = clipped(ocean, extent)
    if not local_ocean.empty:
        local_ocean.plot(
            ax=axis,
            facecolor=colors["ocean"],
            edgecolor="none",
            alpha=0.72,
            zorder=1,
        )
    local_amazonia = clipped(amazonia, extent)
    if not local_amazonia.empty:
        local_amazonia.plot(
            ax=axis,
            facecolor=colors["amazonia"],
            edgecolor="none",
            alpha=0.30,
            zorder=2,
        )
    ecuador.boundary.plot(
        ax=axis,
        color=colors["boundary"],
        linewidth=0.85,
        zorder=5,
    )

    q_low = float(points["area_ha"].quantile(float(marker_settings["lower_quantile"])))
    q_high = float(points["area_ha"].quantile(float(marker_settings["upper_quantile"])))
    marker_sizes = area_to_marker_size(
        points["area_ha"],
        q_low,
        q_high,
        float(marker_settings["minimum"]),
        float(marker_settings["maximum"]),
    )
    point_sizes = pd.Series(marker_sizes, index=points.index)
    axis.scatter(
        points.geometry.x,
        points.geometry.y,
        s=marker_sizes,
        c=colors["inventory"],
        edgecolors="#3F47A6",
        linewidths=float(marker_settings["inventory_edge_width"]),
        alpha=float(marker_settings["inventory_alpha"]),
        zorder=10,
        rasterized=False,
    )
    reference_mask = points["inventory_id"].isin(reference_ids)
    reference_points = points.loc[reference_mask]
    reference_sizes = np.maximum(
        point_sizes.loc[reference_mask].to_numpy()
        * float(marker_settings["reference_scale"]),
        float(marker_settings["reference_minimum"]),
    )
    axis.scatter(
        reference_points.geometry.x,
        reference_points.geometry.y,
        s=reference_sizes,
        facecolors="none",
        edgecolors=colors["reference"],
        linewidths=float(marker_settings["reference_edge_width"]),
        alpha=0.97,
        zorder=12,
        rasterized=False,
    )

    detail_settings = config["detail_panels"]
    callout_text_artists: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        detail_panel_settings = detail_settings[panel]
        panel_extent = validate_extent(detail_panel_settings["extent"], panel)
        xmin, xmax, ymin, ymax = panel_extent
        rectangle = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="none",
                edgecolor="#262626",
                linewidth=0.80,
                zorder=30,
        )
        rectangle.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.0, foreground="white", alpha=0.92),
                path_effects.Normal(),
            ]
        )
        axis.add_patch(rectangle)
        callout = detail_panel_settings["national_callout"]
        callout_text = axis.text(
            float(callout["longitude"]),
            float(callout["latitude"]),
            f"({panel})",
            ha=str(callout["horizontal_alignment"]),
            va="center",
            fontsize=body_font_size,
            fontweight="bold",
            color="black",
            zorder=31,
        )
        apply_text_halo(callout_text, width=1.8)
        callout_text_artists[panel] = callout_text

    regional_text_artists: dict[str, Any] = {}
    for label in national_settings["regional_labels"]:
        region_text = axis.text(
            float(label["longitude"]),
            float(label["latitude"]),
            str(label["text"]),
            ha="center",
            va="center",
            rotation=float(label.get("rotation", 0.0)),
            rotation_mode="anchor",
            fontsize=body_font_size,
            fontstyle="italic",
            color=str(label.get("color", colors["context_text"])),
            alpha=0.92,
            zorder=20,
        )
        apply_text_halo(region_text, width=1.6)
        regional_text_artists[str(label["text"])] = region_text

    xmin, xmax, ymin, ymax = validate_extent(extent, "national")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.xaxis.set_major_locator(
        MultipleLocator(float(national_settings["longitude_tick_step"]))
    )
    axis.yaxis.set_major_locator(
        MultipleLocator(float(national_settings["latitude_tick_step"]))
    )
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format_longitude(x, p, 0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, p: format_latitude(y, p, 0)))
    axis.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        direction="out",
        length=2.6,
        width=0.5,
        pad=1.5,
    )
    axis.tick_params(
        axis="y",
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        direction="out",
        length=2.6,
        width=0.5,
        pad=1.5,
    )
    for tick in axis.get_yticklabels():
        tick.set_rotation(90)
        tick.set_va("center")
    axis.grid(
        color=colors["grid"],
        linestyle="--",
        linewidth=0.35,
        alpha=0.28,
        zorder=3,
    )
    geographic_aspect = set_geographic_aspect(axis, extent)
    draw_panel_label(axis, "a", figure_settings)

    area_values = points["area_ha"].quantile([0.25, 0.50, 0.75]).to_numpy()
    locator_bounds = national_settings["locator_axes"]
    area_key_bounds = national_settings["area_key_axes"]
    locator_overlap = count_points_under_axes_bounds(
        points,
        reference_ids,
        extent,
        locator_bounds,
    )
    area_key_overlap = count_points_under_axes_bounds(
        points,
        reference_ids,
        extent,
        area_key_bounds,
    )
    boundary_clearance = float(
        national_settings["inset_boundary_clearance_fraction"]
    )
    locator_boundary = audit_boundary_under_axes_bounds(
        ecuador,
        extent,
        locator_bounds,
        boundary_clearance,
    )
    area_key_boundary = audit_boundary_under_axes_bounds(
        ecuador,
        extent,
        area_key_bounds,
        boundary_clearance,
    )
    if locator_overlap["inventory_points_under_key"] != 0:
        raise FigureBuildError("The locator inset obscures inventory points.")
    if area_key_overlap["inventory_points_under_key"] != 0:
        raise FigureBuildError("The area key obscures inventory points.")
    if not locator_boundary["boundary_clear"]:
        raise FigureBuildError("The locator inset obscures the Ecuador boundary.")
    if not area_key_boundary["boundary_clear"]:
        raise FigureBuildError("The area key obscures the Ecuador boundary.")
    locator_axis = axis.inset_axes(locator_bounds, zorder=94)
    draw_locator_panel(
        locator_axis,
        national_settings["locator_extent"],
        locator_image,
        locator_countries,
        colors,
        float(national_settings["locator_basemap_alpha"]),
    )
    area_key_axis = draw_area_key(
        axis,
        area_key_bounds,
        area_values,
        marker_settings,
        q_low,
        q_high,
        body_font_size,
    )
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    locator_box = locator_axis.get_tightbbox(renderer)
    area_key_box = area_key_axis.get_tightbbox(renderer)
    millimetres_per_pixel = 25.4 / float(axis.figure.dpi)
    locator_area_separation_mm = float(
        (locator_box.x0 - area_key_box.x1) * millimetres_per_pixel
    )
    minimum_separation = float(
        config["v7_layout_targets"]["locator_area_separation_mm_min"]
    )
    if locator_area_separation_mm < minimum_separation:
        raise FigureBuildError(
            "Locator/area-key separation is too small: "
            f"{locator_area_separation_mm:.3f} mm is below "
            f"{minimum_separation:.3f} mm."
        )
    pacific_label_audit = audit_text_artist_overlap(
        regional_text_artists["PACIFIC\nOCEAN"],
        axis,
        points,
        reference_ids,
        ecuador,
    )
    if not pacific_label_audit["collision_free"]:
        raise FigureBuildError(
            "The PACIFIC OCEAN label overlaps a sample point or Ecuador boundary."
        )
    callout_label_audits = {
        panel: audit_text_artist_overlap(
            artist,
            axis,
            points,
            reference_ids,
            ecuador,
        )
        for panel, artist in callout_text_artists.items()
    }
    required_callouts = list(
        national_settings.get("required_collision_free_callouts", ["d"])
    )
    for panel in required_callouts:
        if not callout_label_audits[panel]["collision_free"]:
            raise FigureBuildError(
                f"Panel ({panel}) national callout overlaps a sample point or "
                "the Ecuador boundary."
            )
    return {
        "marker_lower_area_ha": q_low,
        "marker_upper_area_ha": q_high,
        "area_legend_values_ha": [float(value) for value in area_values],
        "reference_points_drawn": int(reference_mask.sum()),
        "geographic_aspect": geographic_aspect,
        "locator_overlap_audit": locator_overlap,
        "area_key_overlap_audit": area_key_overlap,
        "locator_boundary_audit": locator_boundary,
        "area_key_boundary_audit": area_key_boundary,
        "pacific_ocean_label_audit": pacific_label_audit,
        "national_callout_label_audits": callout_label_audits,
        "locator_area_key_separation_mm": locator_area_separation_mm,
        "locator_area_key_minimum_separation_mm": minimum_separation,
    }


def draw_detail_panel(
    axis: plt.Axes,
    panel: str,
    inventory: gpd.GeoDataFrame,
    reference_ids: set[str],
    image: np.ndarray | None,
    config: Mapping[str, Any],
) -> tuple[int, int, float, dict[str, Any]]:
    settings = config["detail_panels"][panel]
    extent = validate_extent(settings["extent"], panel)
    figure_settings = config["figure"]
    colors = figure_settings["colors"]
    polygon_settings = figure_settings["detail_polygon_style"]
    decoration_settings = config["detail_map_decorations"]
    basemap = config["basemap"]
    body_font_size = float(figure_settings["body_font_size_pt"])
    draw_basemap(axis, image, extent, colors, float(basemap["alpha"]))
    local = clipped(inventory, extent)
    if local.empty:
        raise FigureBuildError(f"No BALSA polygons intersect detail panel ({panel}).")
    local.plot(
        ax=axis,
        facecolor=str(polygon_settings["inventory_fill"]),
        edgecolor=str(polygon_settings["inventory_edge"]),
        linewidth=float(polygon_settings["inventory_edge_width"]),
        alpha=float(polygon_settings["inventory_alpha"]),
        zorder=10,
    )
    local_reference = local.loc[local["inventory_id"].astype(str).isin(reference_ids)]
    if not local_reference.empty:
        collection_start = len(axis.collections)
        local_reference.boundary.plot(
            ax=axis,
            color=colors["reference"],
            linewidth=float(polygon_settings["reference_edge_width"]),
            alpha=float(polygon_settings["reference_alpha"]),
            zorder=12,
        )
        for collection in axis.collections[collection_start:]:
            if hasattr(collection, "set_joinstyle"):
                collection.set_joinstyle("round")
            if hasattr(collection, "set_capstyle"):
                collection.set_capstyle("round")

    xmin, xmax, ymin, ymax = extent
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_xticks([float(value) for value in settings["longitude_ticks"]])
    axis.set_yticks([float(value) for value in settings["latitude_ticks"]])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format_longitude(x, p, 2)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, p: format_latitude(y, p, 2)))
    axis.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        direction="out",
        length=2.2,
        width=0.45,
        pad=1.1,
        labelsize=body_font_size,
    )
    axis.tick_params(
        axis="y",
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        direction="out",
        length=2.2,
        width=0.45,
        pad=1.6,
        labelsize=body_font_size,
    )
    for tick in axis.get_yticklabels():
        tick.set_rotation(90)
        tick.set_va("center")
    axis.grid(
        color=colors["grid"],
        linestyle="--",
        linewidth=0.28,
        alpha=0.20,
        zorder=2,
    )
    geographic_aspect = set_geographic_aspect(axis, extent)
    draw_panel_label(axis, panel, figure_settings)
    if bool(settings["show_north_arrow"]):
        draw_north_arrow(axis, body_font_size)
    if bool(settings["show_scale_bar"]):
        scale_anchor, scale_bar_audit = choose_scale_bar_anchor(
            extent,
            local,
            decoration_settings,
        )
        draw_metric_scale_bar(
            axis,
            extent,
            body_font_size,
            scale_anchor,
            float(decoration_settings["scale_bar_length_km"]),
        )
    else:
        scale_bar_audit = {"drawn": False}
    return (
        int(len(local)),
        int(len(local_reference)),
        geographic_aspect,
        scale_bar_audit,
    )


def measure_detail_label_overflow(
    figure: plt.Figure,
    detail_axes: Mapping[str, plt.Axes],
    tolerance_mm: float = 0.15,
) -> dict[str, Any]:
    """Check that inset longitude/latitude labels remain within map edges."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    millimetres_per_pixel = 25.4 / float(figure.dpi)
    result: dict[str, Any] = {}
    for panel, axis in detail_axes.items():
        axis_box = axis.get_window_extent(renderer)
        x_labels = [label for label in axis.get_xticklabels() if label.get_visible()]
        y_labels = [label for label in axis.get_yticklabels() if label.get_visible()]
        x_overflows: list[float] = []
        for label in x_labels:
            label_box = label.get_window_extent(renderer)
            x_overflows.append(
                max(
                    0.0,
                    (axis_box.x0 - label_box.x0) * millimetres_per_pixel,
                    (label_box.x1 - axis_box.x1) * millimetres_per_pixel,
                )
            )
        y_overflows: list[float] = []
        for label in y_labels:
            label_box = label.get_window_extent(renderer)
            y_overflows.append(
                max(
                    0.0,
                    (axis_box.y0 - label_box.y0) * millimetres_per_pixel,
                    (label_box.y1 - axis_box.y1) * millimetres_per_pixel,
                )
            )
        maximum = max(x_overflows + y_overflows, default=0.0)
        result[panel] = {
            "longitude_label_count": len(x_labels),
            "latitude_label_count": len(y_labels),
            "maximum_overflow_mm": float(maximum),
            "tolerance_mm": float(tolerance_mm),
            "within_panel_edges": bool(maximum <= tolerance_mm),
        }
    return result


def measure_detail_frame_heights(
    figure: plt.Figure,
    detail_axes: Mapping[str, plt.Axes],
    tolerance_mm: float = 0.2,
) -> dict[str, Any]:
    figure.canvas.draw()
    millimetres_per_pixel = 25.4 / float(figure.dpi)
    heights = {
        panel: float(axis.get_window_extent().height * millimetres_per_pixel)
        for panel, axis in detail_axes.items()
    }
    spread = max(heights.values()) - min(heights.values())
    return {
        "panel_heights_mm": heights,
        "height_spread_mm": float(spread),
        "tolerance_mm": float(tolerance_mm),
        "within_tolerance": bool(spread <= tolerance_mm),
    }


def _polygon_parts(geometry: Any) -> Iterable[Any]:
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms


def _pixel_ring(
    coordinates: Iterable[Sequence[float]],
    extent: Sequence[float],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    xmin, xmax, ymin, ymax = validate_extent(extent, "readability raster")
    return [
        (
            int(round((float(x) - xmin) / (xmax - xmin) * (width - 1))),
            int(round((ymax - float(y)) / (ymax - ymin) * (height - 1))),
        )
        for x, y, *_ in coordinates
    ]


def audit_reference_boundary_readability(
    local_reference: gpd.GeoDataFrame,
    extent: Sequence[float],
    width_px: int,
    height_px: int,
    line_width_pt: float,
    dpi: int,
    minimum_fill_pixels: int,
    minimum_visible_interior_fraction: float,
) -> dict[str, Any]:
    """Rasterize each interpreted polygon to verify that indigo interiors survive."""

    line_width_px = max(1, int(round(line_width_pt * dpi / 72.0)))
    records: list[dict[str, Any]] = []
    for row in local_reference.itertuples():
        fill_image = Image.new("L", (width_px, height_px), 0)
        boundary_image = Image.new("L", (width_px, height_px), 0)
        fill_draw = ImageDraw.Draw(fill_image)
        boundary_draw = ImageDraw.Draw(boundary_image)
        for polygon in _polygon_parts(row.geometry):
            exterior = _pixel_ring(polygon.exterior.coords, extent, width_px, height_px)
            if len(exterior) < 3:
                continue
            fill_draw.polygon(exterior, fill=255)
            boundary_draw.line(
                exterior,
                fill=255,
                width=line_width_px,
                joint="curve",
            )
            for interior in polygon.interiors:
                ring = _pixel_ring(interior.coords, extent, width_px, height_px)
                if len(ring) < 3:
                    continue
                fill_draw.polygon(ring, fill=0)
                boundary_draw.line(
                    ring,
                    fill=255,
                    width=line_width_px,
                    joint="curve",
                )
        fill_mask = np.asarray(fill_image, dtype=np.uint8) > 0
        boundary_mask = np.asarray(boundary_image, dtype=np.uint8) > 0
        fill_pixels = int(fill_mask.sum())
        boundary_pixels = int((fill_mask & boundary_mask).sum())
        visible_fraction = (
            float((fill_pixels - boundary_pixels) / fill_pixels)
            if fill_pixels
            else 0.0
        )
        records.append(
            {
                "inventory_id": str(row.inventory_id),
                "fill_pixels": fill_pixels,
                "boundary_pixels_inside_fill": boundary_pixels,
                "visible_interior_fraction": visible_fraction,
                "eligible": bool(fill_pixels >= minimum_fill_pixels),
            }
        )
    eligible = [record for record in records if record["eligible"]]
    visible = [float(record["visible_interior_fraction"]) for record in eligible]
    passing = [value >= minimum_visible_interior_fraction for value in visible]
    return {
        "interpreted_polygon_count": int(len(records)),
        "eligible_polygon_count": int(len(eligible)),
        "line_width_pt": float(line_width_pt),
        "line_width_px": int(line_width_px),
        "minimum_fill_pixels": int(minimum_fill_pixels),
        "minimum_visible_interior_fraction": float(
            minimum_visible_interior_fraction
        ),
        "median_visible_interior_fraction": (
            float(np.median(visible)) if visible else None
        ),
        "lower_quartile_visible_interior_fraction": (
            float(np.quantile(visible, 0.25)) if visible else None
        ),
        "passing_polygon_fraction": (
            float(np.mean(passing)) if passing else None
        ),
        "records": records,
    }


def measure_right_column_gaps(
    figure: plt.Figure,
    detail_axes: Mapping[str, plt.Axes],
    legend_axis: plt.Axes,
) -> dict[str, Any]:
    """Measure the three intentional vertical spacers in physical units."""

    figure.canvas.draw()
    millimetres_per_pixel = 25.4 / float(figure.dpi)
    boxes = {
        panel: detail_axes[panel].get_window_extent()
        for panel in PANEL_ORDER
    }
    legend_box = legend_axis.get_window_extent()
    gaps = {
        "b_to_c_mm": float((boxes["b"].y0 - boxes["c"].y1) * millimetres_per_pixel),
        "c_to_d_mm": float((boxes["c"].y0 - boxes["d"].y1) * millimetres_per_pixel),
        "d_to_legend_mm": float(
            (boxes["d"].y0 - legend_box.y1) * millimetres_per_pixel
        ),
    }
    map_gaps = [gaps["b_to_c_mm"], gaps["c_to_d_mm"]]
    gaps["map_gap_spread_mm"] = float(max(map_gaps) - min(map_gaps))
    return gaps


def measure_v7_layout(
    figure: plt.Figure,
    national_axis: plt.Axes,
    detail_axes: Mapping[str, plt.Axes],
    legend_axis: plt.Axes,
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure final canvas margins, detail widths, and left-edge alignment."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    millimetres_per_pixel = 25.4 / float(figure.dpi)
    visible_axes = [national_axis, *[detail_axes[p] for p in PANEL_ORDER], legend_axis]
    content_box = Bbox.union([axis.get_tightbbox(renderer) for axis in visible_axes])
    canvas_box = figure.bbox
    canvas_clearance = {
        "top_mm": float((canvas_box.y1 - content_box.y1) * millimetres_per_pixel),
        "bottom_mm": float((content_box.y0 - canvas_box.y0) * millimetres_per_pixel),
        "left_mm": float((content_box.x0 - canvas_box.x0) * millimetres_per_pixel),
        "right_mm": float((canvas_box.x1 - content_box.x1) * millimetres_per_pixel),
    }

    detail_widths = {
        panel: float(
            detail_axes[panel].get_window_extent(renderer).width
            * millimetres_per_pixel
        )
        for panel in PANEL_ORDER
    }
    baseline_width = float(targets["v6_detail_width_mm"])
    width_increases = {
        panel: float(width / baseline_width - 1.0)
        for panel, width in detail_widths.items()
    }
    width_spread = max(detail_widths.values()) - min(detail_widths.values())

    label_offsets: dict[str, float] = {}
    for panel, axis in detail_axes.items():
        label_artist = next(
            (artist for artist in axis.texts if artist.get_text() == f"({panel})"),
            None,
        )
        if label_artist is None:
            raise FigureBuildError(f"Panel ({panel}) label artist was not found.")
        label_box = label_artist.get_window_extent(renderer)
        axis_box = axis.get_window_extent(renderer)
        label_offsets[panel] = float(
            (label_box.x0 - axis_box.x0) * millimetres_per_pixel
        )
    label_spread = max(label_offsets.values()) - min(label_offsets.values())

    legend = legend_axis.get_legend()
    if legend is None:
        raise FigureBuildError("The detail-column class legend was not found.")
    legend_box = legend.get_window_extent(renderer)
    detail_box = detail_axes["d"].get_window_extent(renderer)
    legend_left_error = float(
        (legend_box.x0 - detail_box.x0) * millimetres_per_pixel
    )
    legend_right_overflow = float(
        max(0.0, legend_box.x1 - detail_box.x1) * millimetres_per_pixel
    )
    return {
        "canvas_clearance_mm": canvas_clearance,
        "detail_widths_mm": detail_widths,
        "detail_width_increase_fraction": width_increases,
        "detail_width_spread_mm": float(width_spread),
        "panel_label_left_offsets_mm": label_offsets,
        "panel_label_left_spread_mm": float(label_spread),
        "legend_left_alignment_error_mm": legend_left_error,
        "legend_right_overflow_mm": legend_right_overflow,
    }


def measure_composition_alignment(
    figure: plt.Figure,
    national_axis: plt.Axes,
    detail_axes: Mapping[str, plt.Axes],
    legend_axis: plt.Axes,
    tolerance_mm: float,
) -> dict[str, Any]:
    """Measure visible left/right composition alignment after rendering."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    left_box = national_axis.get_tightbbox(renderer)
    right_box = Bbox.union(
        [detail_axes[panel].get_tightbbox(renderer) for panel in PANEL_ORDER]
        + [legend_axis.get_tightbbox(renderer)]
    )
    millimetres_per_pixel = 25.4 / float(figure.dpi)
    top_signed = (left_box.y1 - right_box.y1) * millimetres_per_pixel
    bottom_signed = (left_box.y0 - right_box.y0) * millimetres_per_pixel
    height_signed = (left_box.height - right_box.height) * millimetres_per_pixel
    maximum_absolute = max(abs(top_signed), abs(bottom_signed))
    return {
        "left_visible_height_mm": float(left_box.height * millimetres_per_pixel),
        "right_visible_height_mm": float(right_box.height * millimetres_per_pixel),
        "top_signed_difference_mm": float(top_signed),
        "bottom_signed_difference_mm": float(bottom_signed),
        "height_signed_difference_mm": float(height_signed),
        "maximum_edge_mismatch_mm": float(maximum_absolute),
        "tolerance_mm": float(tolerance_mm),
        "within_tolerance": bool(maximum_absolute <= tolerance_mm),
    }


def national_extent(
    ecuador: gpd.GeoDataFrame,
    margin_fraction: float,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = map(float, ecuador.total_bounds)
    x_margin = (xmax - xmin) * margin_fraction
    y_margin = (ymax - ymin) * margin_fraction
    return xmin - x_margin, xmax + x_margin, ymin - y_margin, ymax + y_margin


def build_figure(
    inventory: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    reference_ids: set[str],
    ecuador: gpd.GeoDataFrame,
    amazonia: gpd.GeoDataFrame,
    ocean: gpd.GeoDataFrame,
    locator_countries: gpd.GeoDataFrame,
    basemap_images: Mapping[str, np.ndarray | None],
    config: Mapping[str, Any],
    basemap_mode: str,
) -> tuple[plt.Figure, dict[str, Any]]:
    settings = config["figure"]
    configure_matplotlib(settings)
    width_inches = float(settings["width_mm"]) / 25.4
    height_inches = float(settings["height_mm"]) / 25.4
    figure = plt.figure(figsize=(width_inches, height_inches), facecolor="white")
    margins = settings["margins"]
    layout = settings["layout"]
    main_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=list(layout["main_width_ratios"]),
        left=float(margins["left"]),
        right=float(margins["right"]),
        bottom=float(margins["bottom"]),
        top=float(margins["top"]),
        wspace=float(layout["main_wspace"]),
    )
    right_height_ratios = list(layout["right_height_ratios"])
    right_grid = main_grid[0, 1].subgridspec(
        len(right_height_ratios),
        1,
        height_ratios=right_height_ratios,
        hspace=float(layout["right_hspace"]),
    )
    national_axis = figure.add_subplot(main_grid[0, 0])
    axes_rows = layout["right_axes_rows"]
    detail_axes = {
        panel: figure.add_subplot(right_grid[int(axes_rows[panel]), 0])
        for panel in PANEL_ORDER
    }
    legend_axis = figure.add_subplot(right_grid[int(axes_rows["legend"]), 0])

    national_bounds = national_extent(
        ecuador,
        float(config["national_panel"]["margin_fraction"]),
    )
    national_audit = draw_national_panel(
        national_axis,
        inventory,
        points,
        reference_ids,
        ecuador,
        amazonia,
        ocean,
        national_bounds,
        basemap_images["a"],
        basemap_images["locator"],
        locator_countries,
        config,
    )
    panel_counts: dict[str, int] = {}
    panel_reference_counts: dict[str, int] = {}
    panel_aspects: dict[str, float] = {}
    scale_bar_audits: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        count, reference_count, panel_aspect, scale_bar_audit = draw_detail_panel(
            detail_axes[panel],
            panel,
            inventory,
            reference_ids,
            basemap_images[panel],
            config,
        )
        panel_counts[panel] = count
        panel_reference_counts[panel] = reference_count
        panel_aspects[panel] = panel_aspect
        scale_bar_audits[panel] = scale_bar_audit

    figure.canvas.draw()
    detail_position = detail_axes["d"].get_position()
    legend_position = legend_axis.get_position()
    legend_axis.set_position(
        [
            detail_position.x0,
            legend_position.y0,
            detail_position.width,
            legend_position.height,
        ]
    )
    build_class_legend(
        legend_axis,
        settings["colors"],
        settings["detail_polygon_style"],
        float(settings["body_font_size_pt"]),
    )
    figure.canvas.draw()
    readability_settings = config["v8_layout_targets"]
    target_dpi = int(config["figure"]["raster_dpi"])
    reference_readability: dict[str, Any] = {}
    for panel, axis in detail_axes.items():
        axis_box = axis.get_window_extent()
        width_px = max(1, int(round(axis_box.width / figure.dpi * target_dpi)))
        height_px = max(1, int(round(axis_box.height / figure.dpi * target_dpi)))
        local = clipped(inventory, config["detail_panels"][panel]["extent"])
        local_reference = local.loc[
            local["inventory_id"].astype(str).isin(reference_ids)
        ]
        reference_readability[panel] = audit_reference_boundary_readability(
            local_reference,
            config["detail_panels"][panel]["extent"],
            width_px,
            height_px,
            float(settings["detail_polygon_style"]["reference_edge_width"]),
            target_dpi,
            int(readability_settings["readability_minimum_fill_pixels"]),
            float(
                readability_settings["readability_minimum_visible_interior_fraction"]
            ),
        )
    alignment_audit = measure_composition_alignment(
        figure,
        national_axis,
        detail_axes,
        legend_axis,
        float(layout["alignment_tolerance_mm"]),
    )
    label_overflow_audit = measure_detail_label_overflow(figure, detail_axes)
    detail_height_audit = measure_detail_frame_heights(figure, detail_axes)
    right_column_gap_audit = measure_right_column_gaps(
        figure,
        detail_axes,
        legend_axis,
    )
    v7_layout_audit = measure_v7_layout(
        figure,
        national_axis,
        detail_axes,
        legend_axis,
        config["v7_layout_targets"],
    )
    return figure, {
        "national_extent": list(map(float, national_bounds)),
        "detail_panel_counts": panel_counts,
        "detail_panel_reference_counts": panel_reference_counts,
        "detail_panel_geographic_aspects": panel_aspects,
        "detail_scale_bars": scale_bar_audits,
        "detail_coordinate_labels": label_overflow_audit,
        "detail_frame_heights": detail_height_audit,
        "right_column_gaps": right_column_gap_audit,
        "reference_boundary_readability": reference_readability,
        "v7_layout": v7_layout_audit,
        "national_panel": national_audit,
        "composition_alignment": alignment_audit,
        "attribution_drawn_in_artwork": False,
    }


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        raise FigureBuildError(f"Unsupported tabular output: {path}")
    os.replace(temporary, path)


def save_figure_atomic(
    figure: plt.Figure,
    path: Path,
    dpi: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    save_kwargs: dict[str, Any] = {
        "format": path.suffix.lstrip("."),
        "facecolor": "white",
    }
    if path.suffix.lower() == ".png":
        save_kwargs["dpi"] = dpi
        save_kwargs["metadata"] = {"Software": "LR-CuSum study-area builder"}
    elif path.suffix.lower() == ".pdf":
        save_kwargs["metadata"] = metadata
    elif path.suffix.lower() == ".svg":
        save_kwargs["metadata"] = {"Date": None, "Creator": "LR-CuSum study-area builder"}
    figure.savefig(temporary, **save_kwargs)
    os.replace(temporary, path)


def measure_raster_canvas_clearance(
    path: Path,
    dpi: int,
    threshold: int = 250,
) -> dict[str, Any]:
    """Measure white canvas clearance from the rendered PNG itself."""

    image = np.asarray(Image.open(path).convert("RGB"))
    foreground = np.any(image < int(threshold), axis=2)
    rows, columns = np.where(foreground)
    if len(rows) == 0:
        raise FigureBuildError("Rendered PNG contains no detectable foreground.")
    millimetres_per_pixel = 25.4 / float(dpi)
    return {
        "threshold": int(threshold),
        "top_mm": float(rows.min() * millimetres_per_pixel),
        "bottom_mm": float(
            (image.shape[0] - 1 - rows.max()) * millimetres_per_pixel
        ),
        "left_mm": float(columns.min() * millimetres_per_pixel),
        "right_mm": float(
            (image.shape[1] - 1 - columns.max()) * millimetres_per_pixel
        ),
    }


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def publish_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    shutil.copyfile(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise FigureBuildError("Manuscript publication copy failed checksum verification.")
    os.replace(temporary, destination)


def export_panel_diagnostics(
    figure: plt.Figure,
    output_dir: Path,
    dpi: int,
) -> None:
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    renderer = figure.canvas.get_renderer()
    names = (
        "panel_a_national",
        "panel_b_northern_western",
        "panel_c_central_western",
        "panel_d_eastern_inventory_only",
    )
    for axis, name in zip(figure.axes[:4], names):
        bbox = axis.get_tightbbox(renderer).transformed(figure.dpi_scale_trans.inverted())
        target = panel_dir / f"{name}.png"
        temporary = target.with_name(target.stem + ".tmp" + target.suffix)
        figure.savefig(temporary, dpi=dpi, bbox_inches=bbox, facecolor="white")
        os.replace(temporary, target)


def export_palette_comparison(
    inventory: gpd.GeoDataFrame,
    reference_ids: set[str],
    basemap_image: np.ndarray | None,
    config: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Export three controlled styles over the same central detail extent."""

    panel = "c"
    extent = validate_extent(config["detail_panels"][panel]["extent"], panel)
    local = clipped(inventory, extent)
    local_reference = local.loc[
        local["inventory_id"].astype(str).isin(reference_ids)
    ]
    candidates = list(config["palette_candidates"])
    if len(candidates) != 3:
        raise FigureBuildError("Exactly three palette candidates are required.")
    settings = config["figure"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(float(settings["width_mm"]) / 25.4, 48.0 / 25.4),
        facecolor="white",
    )
    for index, (axis, candidate) in enumerate(zip(axes, candidates), start=1):
        draw_basemap(
            axis,
            basemap_image,
            extent,
            settings["colors"],
            float(config["basemap"]["alpha"]),
        )
        local.plot(
            ax=axis,
            facecolor=str(candidate["inventory_fill"]),
            edgecolor="#F1F2FF",
            linewidth=0.14,
            alpha=float(candidate["inventory_alpha"]),
            zorder=10,
        )
        before = len(axis.collections)
        local_reference.boundary.plot(
            ax=axis,
            color=str(candidate["reference_edge"]),
            linewidth=float(candidate["reference_edge_width"]),
            alpha=float(candidate["reference_alpha"]),
            zorder=12,
        )
        for collection in axis.collections[before:]:
            if hasattr(collection, "set_joinstyle"):
                collection.set_joinstyle("round")
            if hasattr(collection, "set_capstyle"):
                collection.set_capstyle("round")
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        set_geographic_aspect(axis, extent)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{index}. {candidate['name']}",
            fontsize=float(settings["body_font_size_pt"]),
            pad=3.0,
        )
    figure.subplots_adjust(left=0.012, right=0.994, bottom=0.045, top=0.86, wspace=0.035)
    png_path = output_dir / "study_area_palette_comparison.png"
    pdf_path = output_dir / "study_area_palette_comparison.pdf"
    save_figure_atomic(figure, png_path, dpi)
    save_figure_atomic(
        figure,
        pdf_path,
        dpi,
        metadata={
            "Title": "Controlled BALSA polygon palette comparison",
            "Creator": "scripts/build_study_area_figure.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return [png_path, pdf_path]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve_repo_root()
    config_path = resolve_path(root, args.config)
    config = apply_layout_variant(load_config(config_path), args.layout_variant)
    data_paths = {key: resolve_path(root, value) for key, value in config["data"].items()}
    output_dir = (
        resolve_path(root, args.output_dir)
        if args.output_dir is not None
        else resolve_path(root, config["outputs"]["canonical_dir"])
    )
    cache_dir = resolve_path(root, config["outputs"]["basemap_cache"])
    manuscript_pdf = resolve_path(root, config["outputs"]["manuscript_pdf"])
    basemap_mode = args.basemap_mode or str(config["basemap"]["mode"])
    dpi = int(args.dpi or config["figure"]["raster_dpi"])
    if dpi <= 0:
        raise FigureBuildError("DPI must be positive.")
    if args.publish and basemap_mode == "none" and not args.allow_vector_only_publish:
        raise FigureBuildError(
            "Refusing to publish a vector-only diagnostic. Use a cached/downloaded "
            "basemap or pass --allow-vector-only-publish explicitly."
        )

    print("Loading and validating the official BALSA inventory ...")
    inventory, points, inventory_audit = load_inventory(data_paths["inventory_shp"], config["expected"])
    print("Resolving the visually interpreted disturbance-timing subset ...")
    matches, reference_ids, reference_audit = load_reference_matches(
        data_paths["reference_events"],
        inventory,
        points,
        config["expected"],
    )
    print(
        f"Validated {inventory_audit['polygon_count']:,} inventory polygons and "
        f"{reference_audit['reference_polygon_count']:,} one-to-one reference matches."
    )
    inventory_ids = set(points["inventory_id"].astype(str))
    subset_is_inventory_member = reference_ids.issubset(inventory_ids)
    if not subset_is_inventory_member:
        raise FigureBuildError(
            "The visually interpreted disturbance-timing IDs are not a subset "
            "of the official BALSA inventory."
        )
    reference_audit["display_role"] = "visually_interpreted_disturbance_timing_subset"
    reference_audit["is_subset_of_official_inventory"] = True

    ecuador = gpd.read_file(data_paths["ecuador_boundary"]).to_crs(epsg=4326)
    amazonia = gpd.read_file(data_paths["amazonia_boundary"]).to_crs(epsg=4326)
    if ecuador.empty or ecuador.geometry.is_empty.any():
        raise FigureBuildError("Ecuador boundary is empty or invalid.")
    if amazonia.empty:
        raise FigureBuildError("Amazonia context layer is empty.")

    print("Loading Natural Earth context layers ...")
    ocean, ocean_records = natural_earth_layer("10m", "physical", "ocean", root)
    locator_countries, country_records = natural_earth_layer(
        "110m", "cultural", "admin_0_countries", root
    )
    locator_ecuador = select_locator_ecuador(locator_countries)
    if len(locator_ecuador) != 1:
        raise FigureBuildError("The locator must resolve exactly one Ecuador polygon.")

    national_bounds = national_extent(
        ecuador,
        float(config["national_panel"]["margin_fraction"]),
    )
    requested = {
        "a": (national_bounds, int(config["basemap"]["national_zoom"])),
        "locator": (
            config["national_panel"]["locator_extent"],
            int(config["basemap"]["locator_zoom"]),
        ),
    }
    for panel in PANEL_ORDER:
        requested[panel] = (
            config["detail_panels"][panel]["extent"],
            int(config["basemap"]["detail_zoom"]),
        )

    basemap_images: dict[str, np.ndarray | None] = {}
    basemap_manifest: dict[str, Any] = {}
    for panel, (extent, zoom) in requested.items():
        print(f"Preparing basemap for {panel} at zoom {zoom} ({basemap_mode}) ...")
        image, tile_records, mosaic_hash = build_basemap_mosaic(
            extent,
            zoom,
            basemap_mode,
            cache_dir,
            config["basemap"],
            root,
        )
        basemap_images[panel] = image
        basemap_manifest[panel] = {
            "extent": list(map(float, extent)),
            "zoom": zoom,
            "mosaic_sha256": mosaic_hash,
            "tiles": tile_records,
        }

    print("Rendering the A4-width four-panel composition ...")
    figure, render_audit = build_figure(
        inventory,
        points,
        reference_ids,
        ecuador,
        amazonia,
        ocean,
        locator_countries,
        basemap_images,
        config,
        basemap_mode,
    )
    alignment_audit = render_audit["composition_alignment"]
    if not bool(alignment_audit["within_tolerance"]):
        plt.close(figure)
        raise FigureBuildError(
            "Left/right visible composition alignment failed: maximum edge "
            f"mismatch {alignment_audit['maximum_edge_mismatch_mm']:.3f} mm "
            f"exceeds {alignment_audit['tolerance_mm']:.3f} mm."
        )
    height_audit = render_audit["detail_frame_heights"]
    if not bool(height_audit["within_tolerance"]):
        plt.close(figure)
        raise FigureBuildError(
            "Detail-panel frame heights are inconsistent: spread "
            f"{height_audit['height_spread_mm']:.3f} mm exceeds "
            f"{height_audit['tolerance_mm']:.3f} mm."
        )
    v8_targets = config["v8_layout_targets"]
    gap_audit = render_audit["right_column_gaps"]
    map_gap_min, map_gap_max = map(
        float,
        v8_targets["map_gap_mm_target"],
    )
    for key in ("b_to_c_mm", "c_to_d_mm"):
        observed_gap = float(gap_audit[key])
        if not map_gap_min <= observed_gap <= map_gap_max:
            plt.close(figure)
            raise FigureBuildError(
                f"Right-column gap {key}={observed_gap:.3f} mm is outside "
                f"[{map_gap_min:.3f}, {map_gap_max:.3f}] mm."
            )
    if float(gap_audit["map_gap_spread_mm"]) > float(
        v8_targets["map_gap_spread_mm_max"]
    ):
        plt.close(figure)
        raise FigureBuildError("The b--c and c--d map gaps are not mutually equal.")
    d_legend_min, d_legend_max = map(
        float,
        v8_targets["d_legend_gap_mm_target"],
    )
    d_legend_gap = float(gap_audit["d_to_legend_mm"])
    if not d_legend_min <= d_legend_gap <= d_legend_max:
        plt.close(figure)
        raise FigureBuildError(
            f"Panel-(d)-to-legend gap {d_legend_gap:.3f} mm is outside "
            f"[{d_legend_min:.3f}, {d_legend_max:.3f}] mm."
        )
    for panel, label_audit in render_audit["detail_coordinate_labels"].items():
        if not bool(label_audit["within_panel_edges"]):
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) coordinate labels overflow the map frame by "
                f"{label_audit['maximum_overflow_mm']:.3f} mm."
            )
        if int(label_audit["longitude_label_count"]) != 2:
            plt.close(figure)
            raise FigureBuildError(f"Panel ({panel}) must show two longitude labels.")
        if int(label_audit["latitude_label_count"]) != 1:
            plt.close(figure)
            raise FigureBuildError(f"Panel ({panel}) must show one latitude label.")
    for panel, scale_audit in render_audit["detail_scale_bars"].items():
        if not bool(scale_audit["collision_free"]):
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) has no collision-free lower-right scale-bar position."
            )
        if abs(float(scale_audit["selected"]["length_error_m"])) > 1.0e-4:
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) scale-bar geodesic error exceeds 0.1 mm."
            )
    if bool(config["detail_map_decorations"].get("require_uniform_scale_anchor", False)):
        anchors = [
            (
                float(audit["selected"]["right_endpoint_axes_fraction"]),
                float(audit["selected"]["axes_anchor"][1]),
            )
            for audit in render_audit["detail_scale_bars"].values()
        ]
        if any(
            abs(x - anchors[0][0]) > 1.0e-12 or abs(y - anchors[0][1]) > 1.0e-12
            for x, y in anchors[1:]
        ):
            plt.close(figure)
            raise FigureBuildError("Detail scale bars do not share one lower-right anchor.")
    v7_targets = config["v7_layout_targets"]
    v7_audit = render_audit["v7_layout"]
    clearances = v7_audit["canvas_clearance_mm"]
    if float(clearances["top_mm"]) < float(
        v7_targets["canvas_top_clearance_mm_min"]
    ):
        plt.close(figure)
        raise FigureBuildError(
            "Top canvas clearance is too small: "
            f"{clearances['top_mm']:.3f} mm."
        )
    if min(float(value) for value in clearances.values()) < 0.0:
        plt.close(figure)
        raise FigureBuildError("Visible figure content crosses the canvas boundary.")
    minimum_increase = float(
        v7_targets["detail_width_increase_fraction_min"]
    )
    maximum_increase = float(
        v7_targets["detail_width_increase_fraction_max"]
    )
    for panel, increase in v7_audit["detail_width_increase_fraction"].items():
        if not minimum_increase <= float(increase) <= maximum_increase:
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) width increase {float(increase):.4f} is outside "
                f"the [{minimum_increase:.4f}, {maximum_increase:.4f}] target."
            )
    if float(v7_audit["detail_width_spread_mm"]) > 0.05:
        plt.close(figure)
        raise FigureBuildError("Detail-panel widths are not mutually aligned.")
    if float(v7_audit["panel_label_left_spread_mm"]) > float(
        v7_targets["panel_label_left_spread_mm_max"]
    ):
        plt.close(figure)
        raise FigureBuildError("Panel-label left edges are not aligned.")
    if abs(float(v7_audit["legend_left_alignment_error_mm"])) > float(
        v7_targets["legend_left_alignment_error_mm_max"]
    ):
        plt.close(figure)
        raise FigureBuildError(
            "Class legend does not align with the detail-map left edge: "
            f"error {float(v7_audit['legend_left_alignment_error_mm']):.3f} mm."
        )
    if float(v7_audit["legend_right_overflow_mm"]) > float(
        v7_targets["legend_right_overflow_mm_max"]
    ):
        plt.close(figure)
        raise FigureBuildError(
            "Class legend exceeds the detail-column right edge by "
            f"{float(v7_audit['legend_right_overflow_mm']):.3f} mm."
        )
    expected_panel_counts = config["expected"]["panel_intersection_counts"]
    expected_reference_counts = config["expected"]["panel_reference_counts"]
    for panel in PANEL_ORDER:
        observed = int(render_audit["detail_panel_counts"][panel])
        expected_count = int(expected_panel_counts[panel])
        if observed != expected_count:
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) polygon count changed: expected {expected_count}, "
                f"observed {observed}."
            )
        observed_reference = int(
            render_audit["detail_panel_reference_counts"][panel]
        )
        expected_reference = int(expected_reference_counts[panel])
        if observed_reference != expected_reference:
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) interpreted-subset count changed: expected "
                f"{expected_reference}, observed {observed_reference}."
            )

    panel_membership = build_panel_membership(inventory, reference_ids, config)
    expected_complete_counts = config["expected"].get("panel_complete_counts")
    if expected_complete_counts is not None:
        for panel in PANEL_ORDER:
            observed_complete = int(
                panel_membership.loc[
                    panel_membership["panel"] == panel,
                    "is_complete_within_panel",
                ].sum()
            )
            expected_complete = int(expected_complete_counts[panel])
            if observed_complete != expected_complete:
                plt.close(figure)
                raise FigureBuildError(
                    f"Panel ({panel}) complete-polygon count changed: expected "
                    f"{expected_complete}, observed {observed_complete}."
                )
    eastern_membership = panel_membership.loc[panel_membership["panel"] == "d"]
    complete_eastern = int(eastern_membership["is_complete_within_panel"].sum())
    minimum_complete = int(v8_targets["eastern_panel_minimum_complete_polygons"])
    if complete_eastern < minimum_complete:
        plt.close(figure)
        raise FigureBuildError(
            f"Eastern panel contains only {complete_eastern} complete polygons; "
            f"at least {minimum_complete} are required."
        )
    eastern_references = int(eastern_membership["is_visually_interpreted"].sum())
    expected_eastern_references = int(
        v8_targets["eastern_panel_reference_polygons"]
    )
    if eastern_references != expected_eastern_references:
        plt.close(figure)
        raise FigureBuildError(
            "Eastern inventory-only semantics changed: expected "
            f"{expected_eastern_references} interpreted polygons, observed "
            f"{eastern_references}."
        )
    reference_width = float(
        config["figure"]["detail_polygon_style"]["reference_edge_width"]
    )
    if reference_width > float(v8_targets["reference_boundary_width_pt_max"]):
        plt.close(figure)
        raise FigureBuildError(
            f"Interpreted boundary width {reference_width:.3f} pt exceeds the "
            "V8 legibility limit."
        )
    for panel, readability in render_audit[
        "reference_boundary_readability"
    ].items():
        passing_fraction = readability["passing_polygon_fraction"]
        if passing_fraction is None:
            continue
        if float(passing_fraction) < float(
            v8_targets["readability_minimum_passing_polygon_fraction"]
        ):
            plt.close(figure)
            raise FigureBuildError(
                f"Panel ({panel}) preserves visible indigo interiors in only "
                f"{float(passing_fraction):.1%} of eligible interpreted polygons."
            )
    render_audit["eastern_panel"] = {
        "role": "official_inventory_only_spatial_contrast",
        "intersection_count": int(len(eastern_membership)),
        "complete_polygon_count": complete_eastern,
        "interpreted_subset_count": eastern_references,
        "inventory_area_ha": float(eastern_membership["area_ha"].sum()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bounds_frame = pd.DataFrame(
        [
            {
                "panel": panel,
                "name": config["detail_panels"][panel]["name"],
                "xmin": config["detail_panels"][panel]["extent"][0],
                "xmax": config["detail_panels"][panel]["extent"][1],
                "ymin": config["detail_panels"][panel]["extent"][2],
                "ymax": config["detail_panels"][panel]["extent"][3],
                "inventory_polygons": render_audit["detail_panel_counts"][panel],
                "interpreted_subset_polygons": render_audit[
                    "detail_panel_reference_counts"
                ][panel],
            }
            for panel in PANEL_ORDER
        ]
    )
    bounds_path = output_dir / "study_area_panel_bounds.csv"
    membership_path = output_dir / "study_area_panel_inventory_membership.parquet"
    matches_path = output_dir / "study_area_reference_matches.parquet"
    write_dataframe_atomic(bounds_frame, bounds_path)
    write_dataframe_atomic(panel_membership, membership_path)
    write_dataframe_atomic(matches, matches_path)

    pdf_path = output_dir / "study_area.pdf"
    png_path = output_dir / "study_area.png"
    svg_path = output_dir / "study_area.svg"
    pdf_metadata = {
        "Title": "Study region, official BALSA inventory, and interpreted subset",
        "Author": "LR-CuSum project",
        "Subject": "Reproducible Ecuador BALSA study-area figure",
        "Keywords": "Ecuador, BALSA, Sentinel-1, plantation inventory",
        "Creator": "scripts/build_study_area_figure.py",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    save_figure_atomic(figure, pdf_path, dpi, metadata=pdf_metadata)
    save_figure_atomic(figure, png_path, dpi)
    if not args.no_svg:
        save_figure_atomic(figure, svg_path, dpi)
    if args.export_panels:
        figure.canvas.draw()
        export_panel_diagnostics(figure, output_dir, dpi)
    plt.close(figure)

    raster_clearance = measure_raster_canvas_clearance(png_path, dpi)
    v7_targets = config["v7_layout_targets"]
    v6_bottom = float(v7_targets["v6_raster_bottom_clearance_mm"])
    bottom_ratio = float(raster_clearance["bottom_mm"] / v6_bottom)
    ratio_min, ratio_max = map(
        float,
        v7_targets["bottom_clearance_ratio_target"],
    )
    raster_clearance["v6_bottom_mm"] = v6_bottom
    raster_clearance["bottom_ratio_to_v6"] = bottom_ratio
    raster_clearance["ratio_target"] = [ratio_min, ratio_max]
    if not ratio_min <= bottom_ratio <= ratio_max:
        raise FigureBuildError(
            "Rendered bottom clearance was not reduced by approximately half: "
            f"ratio {bottom_ratio:.3f} is outside [{ratio_min:.3f}, {ratio_max:.3f}]."
        )
    render_audit["raster_canvas_clearance"] = raster_clearance

    output_paths = [pdf_path, png_path, bounds_path, membership_path, matches_path]
    if not args.no_svg:
        output_paths.append(svg_path)
    if args.export_style_comparison:
        output_paths.extend(
            export_palette_comparison(
                inventory,
                reference_ids,
                basemap_images["c"],
                config,
                output_dir,
                dpi,
            )
        )
    if args.publish:
        publish_pdf(pdf_path, manuscript_pdf)
        print(f"Published manuscript asset: {manuscript_pdf}")

    input_manifest = {
        "inventory_shapefile": shapefile_records(data_paths["inventory_shp"], root),
        "reference_events": path_record(data_paths["reference_events"], root),
        "ecuador_boundary": shapefile_records(data_paths["ecuador_boundary"], root),
        "amazonia_boundary": shapefile_records(data_paths["amazonia_boundary"], root),
        "configuration": path_record(config_path, root),
        "natural_earth": {
            "ocean_10m": ocean_records,
            "admin_0_countries_110m": country_records,
            "cartopy_data_dir": str(Path(cartopy.config["data_dir"]).resolve()),
        },
    }
    output_manifest = {path.name: path_record(path, root) for path in output_paths}
    manifest = {
        "schema_version": "study-area-figure-manifest/1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "python": sys.version,
        "packages": package_versions(),
        "inputs": input_manifest,
        "inventory_audit": inventory_audit,
        "reference_audit": reference_audit,
        "render_audit": render_audit,
        "figure": {
            "layout_variant": str(args.layout_variant),
            "width_mm": float(config["figure"]["width_mm"]),
            "height_mm": float(config["figure"]["height_mm"]),
            "raster_dpi": dpi,
            "basemap_mode": basemap_mode,
            "svg_written": not args.no_svg,
            "body_font_size_pt": float(config["figure"]["body_font_size_pt"]),
            "panel_label_size_pt": float(config["figure"]["panel_label_size_pt"]),
            "selected_palette": str(config["selected_palette"]),
        },
        "basemap": basemap_manifest,
        "outputs": output_manifest,
        "publication": {
            "requested": bool(args.publish),
            "manuscript_pdf": str(manuscript_pdf),
            "published_sha256": sha256_file(manuscript_pdf) if args.publish else None,
        },
    }
    manifest_path = output_dir / "study_area_manifest.json"
    write_json_atomic(manifest, manifest_path)
    print(f"Canonical PDF: {pdf_path}")
    print(f"Preview PNG: {png_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FigureBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
