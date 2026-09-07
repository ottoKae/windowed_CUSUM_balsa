"""Build a green/yellow study-area figure without changing the canonical figure.

The renderer reuses the validated study-area workflow and cached basemap, but
overrides only the official-inventory and visually interpreted reference
colors.  It writes a new output directory and never publishes over
``manuscript/JAG_Codex_20260821/figures/study_area.pdf``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = Path(__file__).with_name("build_study_area_figure.py")
CONFIG_PATH = REPO_ROOT / "configs" / "study_area_figure_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "manuscript_figures" / "study_area_green_yellow_v1"
INVENTORY_GREEN = "#2F8F5B"
INVENTORY_EDGE = "#246B45"
REFERENCE_YELLOW = "#F2C744"
FILL_ALPHA = 0.50
RASTER_DPI = 600
FIGURE_WIDTH_MM = 180.0


def load_source():
    spec = importlib.util.spec_from_file_location("study_area_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load study-area source: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = load_source()
    config = source.apply_layout_variant(source.load_config(CONFIG_PATH), "v8")

    # The national panel uses colors.inventory/reference; detail panels use the
    # corresponding detail_polygon_style values.  Keep all other layout,
    # basemap, extent, geometry and audit settings identical to study_area.pdf.
    colors = config["figure"]["colors"]
    colors["inventory"] = INVENTORY_GREEN
    colors["reference"] = REFERENCE_YELLOW
    polygon_style = config["figure"]["detail_polygon_style"]
    polygon_style["inventory_fill"] = INVENTORY_GREEN
    polygon_style["inventory_edge"] = INVENTORY_EDGE
    config["figure"]["marker_area_pt2"]["inventory_alpha"] = FILL_ALPHA
    polygon_style["inventory_alpha"] = FILL_ALPHA
    polygon_style["reference_alpha"] = FILL_ALPHA

    data_paths = {
        key: source.resolve_path(REPO_ROOT, value)
        for key, value in config["data"].items()
    }
    inventory, points, inventory_audit = source.load_inventory(
        data_paths["inventory_shp"], config["expected"]
    )
    matches, reference_ids, reference_audit = source.load_reference_matches(
        data_paths["reference_events"],
        inventory,
        points,
        config["expected"],
    )
    ecuador = gpd.read_file(data_paths["ecuador_boundary"]).to_crs(epsg=4326)
    amazonia = gpd.read_file(data_paths["amazonia_boundary"]).to_crs(epsg=4326)
    ocean, _ = source.natural_earth_layer("10m", "physical", "ocean", REPO_ROOT)
    locator_countries, _ = source.natural_earth_layer(
        "110m", "cultural", "admin_0_countries", REPO_ROOT
    )

    national_bounds = source.national_extent(
        ecuador, float(config["national_panel"]["margin_fraction"])
    )
    requested = {
        "a": (national_bounds, int(config["basemap"]["national_zoom"])),
        "locator": (
            config["national_panel"]["locator_extent"],
            int(config["basemap"]["locator_zoom"]),
        ),
    }
    for panel in source.PANEL_ORDER:
        requested[panel] = (
            config["detail_panels"][panel]["extent"],
            int(config["basemap"]["detail_zoom"]),
        )
    cache_dir = source.resolve_path(REPO_ROOT, config["outputs"]["basemap_cache"])
    basemap_images = {}
    basemap_manifest = {}
    for panel, (extent, zoom) in requested.items():
        image, records, mosaic_hash = source.build_basemap_mosaic(
            extent,
            zoom,
            "cache",
            cache_dir,
            config["basemap"],
            REPO_ROOT,
        )
        basemap_images[panel] = image
        basemap_manifest[panel] = {
            "extent": list(map(float, extent)),
            "zoom": int(zoom),
            "mosaic_sha256": mosaic_hash,
            "tile_count": len(records),
        }

    # Keep publication settings explicit in this wrapper as well as in the
    # delegated renderer, so static audits can verify the output contract.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, render_audit = source.build_figure(
        inventory,
        points,
        reference_ids,
        ecuador,
        amazonia,
        ocean,
        locator_countries,
        basemap_images,
        config,
        "cache",
    )
    # Make the national-panel encoding mutually exclusive: marker area is the
    # polygon area, while fill color is dated-event membership.  The delegated
    # renderer first draws all inventory points and yellow reference rings; the
    # opaque yellow overlay below converts the reference subset into an
    # immediately legible second color class without changing its area size.
    national_axis = figure.axes[0]
    marker_settings = config["figure"]["marker_area_pt2"]
    q_low = float(points["area_ha"].quantile(float(marker_settings["lower_quantile"])))
    q_high = float(points["area_ha"].quantile(float(marker_settings["upper_quantile"])))
    marker_sizes = source.area_to_marker_size(
        points["area_ha"],
        q_low,
        q_high,
        float(marker_settings["minimum"]),
        float(marker_settings["maximum"]),
    )
    reference_mask = points["inventory_id"].astype(str).isin(reference_ids)
    # The historical renderer used a fixed indigo outline for national
    # inventory markers.  Redraw the complete inventory layer with the
    # requested green outline before adding the yellow dated-event subset.
    national_axis.scatter(
        points.geometry.x,
        points.geometry.y,
        s=marker_sizes,
        facecolors=INVENTORY_GREEN,
        edgecolors=INVENTORY_EDGE,
        linewidths=0.30,
        alpha=FILL_ALPHA,
        zorder=12.5,
        rasterized=False,
    )
    reference_sizes = np.maximum(
        np.asarray(marker_sizes)[reference_mask.to_numpy()]
        * float(marker_settings["reference_scale"]),
        float(marker_settings["reference_minimum"]),
    )
    reference_points = points.loc[reference_mask]
    national_axis.scatter(
        reference_points.geometry.x,
        reference_points.geometry.y,
        s=reference_sizes,
        facecolors=REFERENCE_YELLOW,
        edgecolors=INVENTORY_EDGE,
        linewidths=0.28,
        alpha=FILL_ALPHA,
        zorder=13,
        rasterized=False,
    )

    # Replace the old outline-style global legend with a mutually exclusive
    # color legend.  The national area key remains separate and continues to
    # encode polygon area through bubble size.
    legend_axis = figure.axes[-1]
    legend_axis.clear()
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles=[
            Patch(
                facecolor=INVENTORY_GREEN,
                edgecolor=INVENTORY_EDGE,
                linewidth=0.45,
                label="Official balsa inventory",
            ),
            Patch(
                facecolor=REFERENCE_YELLOW,
                edgecolor=INVENTORY_EDGE,
                linewidth=0.45,
                label="Dated event subset",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=2,
        frameon=False,
        borderpad=0.0,
        borderaxespad=0.0,
        labelspacing=0.2,
        handletextpad=0.25,
        columnspacing=0.55,
        handlelength=0.72,
        handleheight=0.80,
        fontsize=float(config["figure"]["body_font_size_pt"]),
    )
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "study_area_green_yellow"
    outputs = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    metadata = {
        "Title": "Ecuador BALSA study area with green inventory and yellow interpreted subset",
        "Creator": "scripts/build_study_area_green_yellow.py",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(outputs["pdf"], dpi=RASTER_DPI, bbox_inches="tight", facecolor="white", metadata=metadata)
    # The SVG backend accepts only the standard Title/Creator fields.
    figure.savefig(
        outputs["svg"],
        dpi=RASTER_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": metadata["Title"], "Creator": metadata["Creator"]},
    )
    figure.savefig(outputs["png"], dpi=RASTER_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["tiff"], dpi=RASTER_DPI, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source_renderer": str(SOURCE_SCRIPT.resolve()),
        "config": str(CONFIG_PATH.resolve()),
        "palette": {
            "official_inventory": INVENTORY_GREEN,
            "visually_interpreted_subset": REFERENCE_YELLOW,
            "inventory_edge": INVENTORY_EDGE,
            "description": "green/yellow colorblind-conscious replacement; no red",
            "fill_alpha": FILL_ALPHA,
        },
        "national_marker_encoding": {
            "size": "polygon area in hectares",
            "color": "dated-event membership",
            "green": "inventory polygon without dated event",
            "yellow": "inventory polygon containing a visually dated event",
        },
        "data": {
            "inventory": str(data_paths["inventory_shp"].resolve()),
            "reference_events": str(data_paths["reference_events"].resolve()),
            "ecuador_boundary": str(data_paths["ecuador_boundary"].resolve()),
            "amazonia_boundary": str(data_paths["amazonia_boundary"].resolve()),
        },
        "input_sha256": {
            key: sha256(path)
            for key, path in data_paths.items()
            if path.is_file()
        },
        "counts": {
            "inventory_polygons": int(inventory_audit["polygon_count"]),
            "reference_polygons": int(reference_audit["reference_polygon_count"]),
            "reference_inventory_matches": int(reference_audit["matched_inventory_polygon_count"]),
        },
        "figure": {
            "width_mm": FIGURE_WIDTH_MM,
            "raster_dpi": RASTER_DPI,
        },
        "basemap": basemap_manifest,
        "render_audit": render_audit,
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    (output_dir / "study_area_green_yellow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "outputs": manifest["outputs"], "counts": manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
