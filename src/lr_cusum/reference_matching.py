"""Match historical clear-cut references to frozen plantation sample identities."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


CUT_TOKEN = "\u780d\u4f10\u65f6\u95f4"


def normalize_month(value):
    if pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed.year, parsed.month, 1)


def read_polygon_events(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    id_column = "ID"
    cut_columns = [column for column in frame.columns if CUT_TOKEN in str(column)]
    if not cut_columns:
        raise ValueError(f"no clear-cut columns found in {path}")
    rows = []
    for _, record in frame.iterrows():
        for order, column in enumerate(cut_columns, 1):
            month = normalize_month(record[column])
            if pd.notna(month):
                rows.append({
                    "reference_source": "polygon", "historical_id": int(record[id_column]),
                    "event_order": order, "event_month": month,
                    "source_column": str(column),
                })
    return pd.DataFrame(rows)


def read_point_events(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    value_columns = [column for column in frame.columns if str(column) != "ID"]
    rows = []
    for _, record in frame.iterrows():
        order = 0
        for column in value_columns:
            month = normalize_month(record[column])
            if pd.notna(month):
                order += 1
                rows.append({
                    "reference_source": "point", "historical_id": int(record["ID"]),
                    "event_order": order, "event_month": month,
                    "source_column": str(column),
                })
    return pd.DataFrame(rows)


def polygon_sample_map(roles_path: Path, polygon_path: Path):
    roles = pd.read_parquet(roles_path)
    roles = roles.loc[roles.species_nco.astype(str).str.upper().eq("BALSA")].copy()
    points = gpd.GeoDataFrame(
        roles, geometry=gpd.points_from_xy(roles.model_center_x, roles.model_center_y), crs="EPSG:32717"
    )
    polygons = gpd.read_file(polygon_path).to_crs(points.crs)
    polygons = polygons.rename(columns={"ID": "reference_polygon_id"})
    polygons = polygons[["reference_polygon_id", "OBJECTID", "geometry"]].copy()
    polygons["reference_polygon_area_m2"] = polygons.geometry.area
    joined = gpd.sjoin(points, polygons, how="left", predicate="within")
    counts = joined.dropna(subset=["reference_polygon_id"]).groupby("sample_id").reference_polygon_id.nunique()
    if (counts > 1).any():
        raise ValueError("a frozen centre falls in multiple reference polygons")
    matched = joined.dropna(subset=["reference_polygon_id"]).copy()
    matched["reference_polygon_id"] = matched.reference_polygon_id.astype(int)
    matched["match_method"] = "frozen_center_within_reference_polygon"
    matched["match_status"] = "exact"
    columns = [
        "reference_polygon_id", "sample_id", "polygon_id", "source_row", "OBJECTID_left",
        "species_nco", "role", "spatial_block_id", "longitude", "latitude",
        "reference_polygon_area_m2", "match_method", "match_status",
    ]
    matched = matched.rename(columns={"OBJECTID_left": "sample_OBJECTID"})
    columns = ["sample_OBJECTID" if column == "OBJECTID_left" else column for column in columns]
    return matched[columns].sort_values("reference_polygon_id").reset_index(drop=True), polygons


def point_sample_map(point_path: Path, polygons: gpd.GeoDataFrame,
                     polygon_map: pd.DataFrame, roles_path: Path):
    historical = gpd.read_file(point_path).rename(columns={"ID": "historical_point_id"})
    historical = historical.to_crs(polygons.crs)[["historical_point_id", "geometry"]]
    within = gpd.sjoin(
        historical, polygons[["reference_polygon_id", "geometry"]], how="left", predicate="within"
    )
    exact = within.dropna(subset=["reference_polygon_id"]).copy()
    exact["reference_polygon_id"] = exact.reference_polygon_id.astype(int)
    exact = exact.merge(polygon_map, on="reference_polygon_id", how="left", validate="many_to_one")
    exact["distance_m"] = 0.0
    exact["match_method"] = "historical_point_within_reference_polygon"
    exact["match_status"] = "exact"

    unmatched_ids = set(historical.historical_point_id) - set(exact.historical_point_id)
    roles = pd.read_parquet(roles_path)
    roles = roles.loc[roles.species_nco.astype(str).str.upper().eq("BALSA")].copy()
    centers = gpd.GeoDataFrame(
        roles[["sample_id", "polygon_id", "role", "spatial_block_id", "longitude", "latitude"]],
        geometry=gpd.points_from_xy(roles.model_center_x, roles.model_center_y), crs=polygons.crs,
    )
    remaining = historical[historical.historical_point_id.isin(unmatched_ids)]
    nearest = gpd.sjoin_nearest(remaining, centers, how="left", distance_col="distance_m")
    nearest = nearest.sort_values(["historical_point_id", "distance_m", "sample_id"]).drop_duplicates(
        "historical_point_id"
    )
    nearest["reference_polygon_id"] = pd.NA
    nearest["match_method"] = "nearest_frozen_balsa_center"
    nearest["match_status"] = np.select(
        [nearest.distance_m <= 45, nearest.distance_m <= 90, nearest.distance_m <= 300],
        ["candidate_le_45m", "candidate_le_90m", "candidate_le_300m"], default="unmatched_gt_300m",
    )
    columns = [
        "historical_point_id", "reference_polygon_id", "sample_id", "polygon_id", "role",
        "spatial_block_id", "longitude", "latitude", "distance_m", "match_method", "match_status",
    ]
    result = pd.concat([exact[columns], nearest[columns]], ignore_index=True)
    return result.sort_values("historical_point_id").reset_index(drop=True)


def attach_events(polygon_events, point_events, polygon_map, point_map):
    polygon = polygon_events.merge(
        polygon_map, left_on="historical_id", right_on="reference_polygon_id",
        how="left", validate="many_to_one",
    )
    polygon["match_status"] = polygon.match_status.fillna("unmatched_polygon_id")
    point = point_events.merge(
        point_map, left_on="historical_id", right_on="historical_point_id",
        how="left", validate="many_to_one",
    )
    events = pd.concat([polygon, point], ignore_index=True, sort=False)
    events["event_id"] = (
        events.reference_source.astype(str) + ":" + events.historical_id.astype(str)
        + ":" + events.event_order.astype(str)
    )
    first = ["event_id", "reference_source", "historical_id", "event_order", "event_month",
             "sample_id", "polygon_id", "match_status", "match_method", "distance_m"]
    remaining = [column for column in events.columns if column not in first]
    return events[first + remaining].sort_values(
        ["reference_source", "historical_id", "event_month"]
    ).reset_index(drop=True)

