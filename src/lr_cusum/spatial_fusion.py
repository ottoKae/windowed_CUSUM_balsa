"""Notebook polygon fusion strategies adapted to native S1-GRiTS pixel indexes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def _joined_pixel_events(pixel_events, associations):
    events = pixel_events.drop(columns=["native_row", "native_col"], errors="ignore").copy()
    joined = associations.merge(events, on=["pixel_track_id", "track_key"], how="inner")
    joined["time_days"] = (
        pd.to_datetime(joined.change_time) - pd.Timestamp("2000-01-01")
    ).dt.days.astype(float)
    return joined


def _metadata(polygon_tracks):
    return polygon_tracks.set_index("polygon_track_id")


def temporal_dbscan_fusion(
    pixel_events,
    associations,
    polygon_tracks,
    *,
    eps_days: float = 60,
    minimum_pixel_ratio: float = .40,
):
    """Notebook Strategy 2a: temporal DBSCAN on pixel event rows."""
    joined = _joined_pixel_events(pixel_events, associations)
    metadata = _metadata(polygon_tracks); rows = []
    for (polygon_track_id, band), local in joined.groupby(["polygon_track_id", "Band"], sort=False):
        total = associations.loc[
            associations.polygon_track_id.eq(polygon_track_id), "pixel_track_id"
        ].nunique()
        minimum = max(3, int(total * minimum_pixel_ratio))
        if len(local) < minimum:
            continue
        labels = DBSCAN(eps=eps_days, min_samples=minimum).fit_predict(
            local[["time_days"]].to_numpy(float)
        )
        for label in np.unique(labels):
            if label < 0:
                continue
            cluster = local.loc[labels == label]
            if len(cluster) < minimum:
                continue
            first = metadata.loc[polygon_track_id]
            rows.append({
                "route": "notebook_temporal_dbscan", "polygon_track_id": polygon_track_id,
                "sample_id": first.sample_id, "polygon_id": first.polygon_id,
                "role": first.role, "species_nco": first.species_nco,
                "tile_id": first.tile_id, "track_key": first.track_key, "Band": band,
                "change_time": pd.to_datetime(cluster.change_time).median(),
                "support_pixels": int(cluster.pixel_track_id.nunique()),
                "support_fraction": float(cluster.pixel_track_id.nunique() / max(1, total)),
                "cluster_rows": int(len(cluster)), "interior_pixels": int(total),
            })
    return pd.DataFrame(rows)


def space_time_dbscan_fusion(
    pixel_events,
    associations,
    polygon_tracks,
    *,
    space_scale_factor: float = 3.0,
    time_threshold_days: float = 60,
    minimum_pixel_ratio: float = .40,
):
    """Notebook Strategy 2b: adaptive space-time DBSCAN event fusion."""
    joined = _joined_pixel_events(pixel_events, associations)
    metadata = _metadata(polygon_tracks); rows = []
    for (polygon_track_id, band), local in joined.groupby(["polygon_track_id", "Band"], sort=False):
        polygon_pixels = associations.loc[associations.polygon_track_id.eq(polygon_track_id)]
        coordinates = polygon_pixels[["native_row", "native_col"]].drop_duplicates().to_numpy(float)
        total = len(coordinates); minimum = max(3, int(total * minimum_pixel_ratio))
        if total < 2 or len(local) < minimum:
            continue
        neighbours = NearestNeighbors(n_neighbors=2).fit(coordinates)
        distances, _ = neighbours.kneighbors(coordinates)
        spatial_threshold = float(np.median(distances[:, 1]) * space_scale_factor)
        if not np.isfinite(spatial_threshold) or spatial_threshold <= 0:
            continue
        features = np.column_stack((
            local.native_row.to_numpy(float) / spatial_threshold,
            local.native_col.to_numpy(float) / spatial_threshold,
            local.time_days.to_numpy(float) / time_threshold_days,
        ))
        labels = DBSCAN(eps=1.0, min_samples=minimum).fit_predict(features)
        for label in np.unique(labels):
            if label < 0:
                continue
            cluster = local.loc[labels == label]
            if len(cluster) < minimum:
                continue
            first = metadata.loc[polygon_track_id]
            rows.append({
                "route": "notebook_space_time_dbscan", "polygon_track_id": polygon_track_id,
                "sample_id": first.sample_id, "polygon_id": first.polygon_id,
                "role": first.role, "species_nco": first.species_nco,
                "tile_id": first.tile_id, "track_key": first.track_key, "Band": band,
                "change_time": pd.to_datetime(cluster.change_time).median(),
                "support_pixels": int(cluster.pixel_track_id.nunique()),
                "support_fraction": float(cluster.pixel_track_id.nunique() / max(1, total)),
                "cluster_rows": int(len(cluster)), "interior_pixels": int(total),
                "space_threshold_pixels": spatial_threshold,
            })
    return pd.DataFrame(rows)

