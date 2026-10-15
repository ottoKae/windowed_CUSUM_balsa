"""Frozen B20 v1.1 aggregation and descriptive native-geometry components.

No reference dates or judgements enter either public function. Execution
receipts must be checked by the caller before publishing complete results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .harvest_quarter_aggregation import (
    _identifier, _json, _prepare_events, _require, quarter_label, quarter_number,
)

PROTOCOL_VERSION = "stand_event_b20_execution_v1_1"
WINDOW_STARTS = pd.period_range("2017-01", "2025-10", freq="M")
EDGE_MONTHS = {"2017-01", "2017-02", "2025-11", "2025-12"}
RANKING = ["remaining_unique_pixels_desc", "remaining_unique_pixel_month_nodes_desc", "window_start_month_asc"]

CANDIDATE_COLUMNS = [
    "method", "sample_id", "candidate_id", "prediction_id", "allocation_order",
    "window_start_month", "window_end_month", "window_start", "window_end",
    "initial_support_pixels", "accepted_support_pixels", "initial_response_nodes", "accepted_response_nodes",
    "response_pixels", "total_pixels", "support_fraction", "source_event_count", "source_alias_count",
    "source_response_key", "source_unit_ids", "source_pixel_track_ids", "source_event_ids", "source_node_ids",
    "monthly_counts", "response_months", "representative_month", "predicted_month",
    "representative_month_has_response", "predicted_quarter", "quarter_index", "quarter_start", "quarter_end",
    "first_response_month", "last_response_month", "span_months", "cross_year_repeated_pixels",
    "aggregation_edge_exposed", "short_context_2025", "support_scope",
]
MEMBER_COLUMNS = [
    "method", "sample_id", "unit_id", "pixel_track_id", "track_key", "native_row", "native_col",
    "response_month", "change_time", "decision_year", "node_id", "source_event_id", "source_event_ids",
    "source_alias_count", "total_pixels", "candidate_id", "prediction_id", "part_id", "allocation_order",
    "destination_type", "residual_reason", "initially_covered_by_qualified_window",
    "aggregation_edge_exposed", "short_context_2025",
]


def _aliases(prepared, duplicate_rows):
    aliases = {(r.unit_id, r.response_month): {str(r.source_event_id)} for r in prepared.itertuples(index=False)}
    for row in duplicate_rows:
        aliases[row["unit_id"], row["response_month"]].add(str(row["source_event_id"]))
    return aliases


def build_bounded_events(units, membership, events, *, allow_multiple_months_per_year=False,
                         source_method="windowed_w36_d12_s12",
                         validate_frozen_gates=True,
                         require_annual_decision=True,
                         output_method="B20",
                         protocol_version=PROTOCOL_VERSION):
    """Return (candidates, node members including residuals, diagnostics).

    Denominators use all frozen membership cells. This three-argument API
    does not infer observation opportunity or execution success from events.
    Source aliases are retained but never affect ranking, dates or identities.
    """
    _require(units, ["unit_id", "pixel_track_id"], "units")
    if units[["unit_id", "pixel_track_id"]].isna().any().any() or units.unit_id.duplicated().any() or units.pixel_track_id.duplicated().any():
        raise ValueError("Master/native identities must be present and unique")
    _require(membership, ["sample_id", "unit_id"], "membership")
    ownership = membership[["sample_id", "unit_id"]].copy()
    if ownership.isna().any().any() or ownership.duplicated().any():
        raise ValueError("Duplicate or missing polygon membership")
    if not set(ownership.unit_id).issubset(units.unit_id):
        raise ValueError("Unknown master cell in polygon membership")
    # A no-central-gate control can legitimately emit several months in a year.
    # Keep the default strict for the frozen baseline; aggregate control nodes
    # by distinct cell/month without silently restoring the removed gate.
    ev, duplicate_rows = _prepare_events(
        events, units,
        allow_multiple_months_per_year=allow_multiple_months_per_year,
        detector_method=source_method,
        validate_frozen_gates=validate_frozen_gates,
        require_annual_decision=require_annual_decision,
    )
    if not set(ev.unit_id).issubset(ownership.unit_id):
        raise ValueError("An accepted pixel event has no frozen polygon membership")
    alias_map = _aliases(ev, duplicate_rows)
    ev = ev.drop(columns="sample_id", errors="ignore")
    native = units.set_index("unit_id")
    for name in ("track_key", "native_row", "native_col"):
        if name not in ev:
            ev[name] = ev.unit_id.map(native[name]) if name in native else None
    joined = ownership.merge(ev, on="unit_id", how="inner", validate="many_to_many")
    denominators = ownership.groupby("sample_id").unit_id.nunique()
    candidate_rows, member_rows = [], []
    start_ordinals = WINDOW_STARTS.asi8
    for sample_id, group in joined.groupby("sample_id", sort=True):
        group = group.sort_values(["unit_id", "response_month"], kind="stable").reset_index(drop=True)
        n_total = int(denominators[sample_id])
        months = pd.PeriodIndex(group.response_month, freq="M").asi8
        pixels = group.unit_id.to_numpy()
        windows = [(months >= start) & (months <= start + 2) for start in start_ordinals]
        initial_pixels = np.array([len(np.unique(pixels[mask])) for mask in windows], dtype=int)
        initial_nodes = np.array([int(mask.sum()) for mask in windows], dtype=int)
        initially_qualified = np.zeros(len(group), dtype=bool)
        for mask, count in zip(windows, initial_pixels):
            if 5 * count >= n_total:
                initially_qualified |= mask
        records = []
        for index, row in enumerate(group.itertuples(index=False)):
            aliases = sorted(alias_map[row.unit_id, row.response_month])
            records.append({
                "method": output_method, "sample_id": sample_id, "unit_id": row.unit_id,
                "pixel_track_id": row.pixel_track_id, "track_key": row.track_key,
                "native_row": row.native_row, "native_col": row.native_col,
                "response_month": row.response_month, "change_time": pd.Timestamp(row.response_month + "-01"),
                "decision_year": int(row.response_month[:4]),
                "node_id": _identifier("node_", sample_id, row.unit_id, row.response_month),
                "source_event_id": aliases[0], "source_event_ids": _json(aliases), "source_alias_count": len(aliases),
                "total_pixels": n_total, "candidate_id": "", "prediction_id": "", "part_id": "",
                "allocation_order": None, "destination_type": "residual", "residual_reason": "remaining_support_below_threshold",
                "initially_covered_by_qualified_window": bool(initially_qualified[index]),
                "aggregation_edge_exposed": row.response_month in EDGE_MONTHS,
                "short_context_2025": row.response_month.startswith("2025-"),
            })
        remaining = np.ones(len(group), dtype=bool)
        allocation_order = 0
        while remaining.any():
            masks = [mask & remaining for mask in windows]
            counts = [len(np.unique(pixels[mask])) for mask in masks]
            node_counts = [int(mask.sum()) for mask in masks]
            selected = min(range(len(windows)), key=lambda i: (-counts[i], -node_counts[i], int(start_ordinals[i])))
            if 5 * counts[selected] < n_total:
                break
            chosen = np.flatnonzero(masks[selected])
            allocation_order += 1
            start, end = WINDOW_STARTS[selected], WINDOW_STARTS[selected] + 2
            local = group.iloc[chosen]
            node_keys = sorted((sample_id, row.unit_id, row.response_month) for row in local.itertuples(index=False))
            identity = ((protocol_version, sample_id, str(start), node_keys)
                        if protocol_version == PROTOCOL_VERSION and output_method == "B20"
                        else (protocol_version, output_method, sample_id, str(start), node_keys))
            prediction_id = _identifier("B20_", *identity)
            node_ids = sorted(records[i]["node_id"] for i in chosen)
            alias_ids = sorted({alias for i in chosen for alias in json.loads(records[i]["source_event_ids"])})
            unique_units = sorted(local.unit_id.unique().tolist())
            response_months = sorted(local.response_month.unique().tolist())
            median = int(np.floor(np.median(months[chosen]) + .5))
            representative = str(pd.Period(ordinal=median, freq="M"))
            qindex = quarter_number(representative)
            qstart = pd.Timestamp(year=qindex // 4, month=(qindex % 4) * 3 + 1, day=1)
            candidate_rows.append({
                "method": output_method, "sample_id": sample_id, "candidate_id": prediction_id, "prediction_id": prediction_id,
                "allocation_order": allocation_order, "window_start_month": str(start), "window_end_month": str(end),
                "window_start": start.start_time, "window_end": end.end_time.normalize(),
                "initial_support_pixels": int(initial_pixels[selected]), "accepted_support_pixels": counts[selected],
                "initial_response_nodes": int(initial_nodes[selected]), "accepted_response_nodes": len(chosen),
                "response_pixels": counts[selected], "total_pixels": n_total, "support_fraction": counts[selected] / n_total,
                "source_event_count": len(chosen), "source_alias_count": len(alias_ids),
                "source_response_key": _identifier("responses_", node_ids),
                "source_unit_ids": _json(unique_units), "source_pixel_track_ids": _json(sorted(local.pixel_track_id.unique().tolist())),
                "source_event_ids": _json(alias_ids), "source_node_ids": _json(node_ids),
                "monthly_counts": _json({str(month): int(count) for month, count in local.groupby("response_month").unit_id.nunique().items()}),
                "response_months": _json(response_months), "representative_month": representative, "predicted_month": representative,
                "representative_month_has_response": representative in response_months,
                "predicted_quarter": quarter_label(qindex), "quarter_index": qindex,
                "quarter_start": qstart, "quarter_end": qstart + pd.offsets.QuarterEnd(),
                "first_response_month": response_months[0], "last_response_month": response_months[-1],
                "span_months": int(months[chosen].max() - months[chosen].min() + 1),
                "cross_year_repeated_pixels": int(local.groupby("unit_id").size().gt(1).sum()),
                "aggregation_edge_exposed": any(records[i]["aggregation_edge_exposed"] for i in chosen),
                "short_context_2025": any(records[i]["short_context_2025"] for i in chosen), "support_scope": "bounded_episode",
            })
            for i in chosen:
                records[i].update(candidate_id=prediction_id, prediction_id=prediction_id, allocation_order=allocation_order,
                                  destination_type="candidate", residual_reason="")
            remaining[chosen] = False
        member_rows.extend(records)
    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    members = pd.DataFrame(member_rows, columns=MEMBER_COLUMNS)
    members["allocation_order"] = members.allocation_order.astype("Int64")
    if not members.empty and (members.node_id.duplicated().any() or members.duplicated(["sample_id", "unit_id", "response_month"]).any()):
        raise AssertionError("One node must occur exactly once in the membership ledger")
    diagnostics = {
        "protocol_version": protocol_version, "output_method": output_method,
        "source_method": source_method, "window_count": len(WINDOW_STARTS), "window_ranking": RANKING,
        "input_pixel_event_rows": int(len(events)), "deduplicated_pixel_month_events": int(len(ev)),
        "duplicate_pixel_month_rows": len(duplicate_rows), "spatially_supported_stands": int(len(denominators)),
        "membership_nodes": len(members), "accepted_candidates": len(candidates),
        "assigned_nodes": int(members.destination_type.eq("candidate").sum()),
        "residual_nodes": int(members.destination_type.eq("residual").sum()),
        "residual_initially_qualified_nodes": int((members.destination_type.eq("residual") & members.initially_covered_by_qualified_window.eq(True)).sum()),
    }
    if diagnostics["membership_nodes"] != diagnostics["assigned_nodes"] + diagnostics["residual_nodes"]:
        raise AssertionError("Node allocation failed conservation")
    return candidates, members, diagnostics


def _native_geometry(units):
    """Restore actual native centers and footprints, then project to 32717."""
    os.environ.setdefault("PROJ_DATA", str(Path(sys.prefix) / "Library/share/proj"))
    os.environ.setdefault("GDAL_DATA", str(Path(sys.prefix) / "Library/share/gdal"))
    import geopandas as gpd
    from pyproj import Transformer
    import shapely

    _require(units, ["unit_id", "transform", "native_row", "native_col", "track_crs", "native_geometry_wkb"], "geometry units")
    if units.unit_id.duplicated().any():
        raise ValueError("Duplicate geometry unit identity")
    work = units.reset_index(drop=True).copy()
    if work[["track_crs", "native_row", "native_col", "native_geometry_wkb"]].isna().any().any():
        raise ValueError("Missing native coordinate system, coordinates or footprint")
    affine = np.asarray([json.loads(value) if isinstance(value, str) else value for value in work["transform"]], dtype=float)
    if affine.ndim != 2 or affine.shape[1] not in (6, 9) or not np.isfinite(affine).all():
        raise ValueError("Native affine transforms must have six or nine finite coefficients")
    col, row = work.native_col.to_numpy(float) + .5, work.native_row.to_numpy(float) + .5
    x = affine[:, 0] * col + affine[:, 1] * row + affine[:, 2]
    y = affine[:, 3] * col + affine[:, 4] * row + affine[:, 5]
    native = shapely.from_wkb(work.native_geometry_wkb.to_numpy())
    projected = np.empty(len(work), dtype=object)
    center_x, center_y = np.empty(len(work)), np.empty(len(work))
    for crs, indices in work.groupby("track_crs", sort=True).groups.items():
        ix = np.asarray(indices)
        tx = Transformer.from_crs(crs, "EPSG:32717", always_xy=True)
        center_x[ix], center_y[ix] = tx.transform(x[ix], y[ix])
        projected[ix] = gpd.GeoSeries(native[ix], crs=crs).to_crs("EPSG:32717").to_numpy()
    if not np.isfinite(center_x).all() or not np.isfinite(center_y).all():
        raise ValueError("Non-finite reconstructed native center")
    return {unit: (float(px), float(py), geom) for unit, px, py, geom in zip(work.unit_id, center_x, center_y, projected)}


def add_spatiotemporal_parts(candidates, members, units, polygons):
    """Add 45 m / same-or-adjacent-month components without filtering candidates.

    Polygon membership uses strict center containment. Component footprints
    are clipped native-cell unions; neither buffers nor filled holes are used.
    """
    os.environ.setdefault("PROJ_DATA", str(Path(sys.prefix) / "Library/share/proj"))
    os.environ.setdefault("GDAL_DATA", str(Path(sys.prefix) / "Library/share/gdal"))
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Point
    import shapely

    _require(polygons, ["sample_id", "geometry"], "polygons")
    if polygons.crs is None or polygons.crs.to_epsg() != 32717:
        raise ValueError("Polygons must use frozen EPSG:32717")
    if polygons.sample_id.duplicated().any():
        raise ValueError("Polygon sample identities must be unique")
    enriched_candidates, enriched_members = candidates.copy(), members.copy()
    part_columns = ["part_id", "candidate_id", "prediction_id", "sample_id", "source_unit_ids", "unit_ids",
                    "source_node_ids", "response_months", "response_pixels", "response_nodes", "monthly_counts",
                    "first_response_month", "last_response_month", "edge_pixel_fraction", "support_area_m2", "geometry"]
    geometry = _native_geometry(units) if len(members) else {}
    polygon_by_id = polygons.set_index("sample_id").geometry.to_dict()
    clipped, centers, edge_flags = {}, {}, {}
    for sample, unit in members[["sample_id", "unit_id"]].drop_duplicates().itertuples(index=False, name=None):
        if sample not in polygon_by_id or unit not in geometry:
            raise ValueError("A node has no frozen native geometry or polygon")
        px, py, footprint = geometry[unit]
        polygon = polygon_by_id[sample]
        point = Point(px, py)
        if not point.within(polygon):
            raise ValueError(f"Native center is not strictly inside its polygon: {sample}, {unit}")
        centers[sample, unit] = (px, py)
        edge_flags[sample, unit] = not polygon.covers(footprint)
        clipped[sample, unit] = footprint.intersection(polygon)
    enriched_members["center_x_32717"] = [centers[r.sample_id, r.unit_id][0] for r in members.itertuples(index=False)]
    enriched_members["center_y_32717"] = [centers[r.sample_id, r.unit_id][1] for r in members.itertuples(index=False)]
    enriched_members["edge_pixel"] = [edge_flags[r.sample_id, r.unit_id] for r in members.itertuples(index=False)]
    enriched_members["part_id"] = ""
    parts, summaries, edge_count = [], [], 0
    accepted = enriched_members.loc[enriched_members.destination_type.eq("candidate")]
    if set(accepted.prediction_id) != set(candidates.prediction_id):
        raise ValueError("Candidate/member prediction identities disagree")
    for prediction_id, local in accepted.groupby("prediction_id", sort=True):
        local = local.sort_values("node_id", kind="stable")
        sample = local.sample_id.iloc[0]
        if local.sample_id.nunique() != 1:
            raise ValueError("A candidate cannot span different stands")
        coordinates = local[["center_x_32717", "center_y_32717"]].to_numpy(float)
        month_numbers = pd.PeriodIndex(local.response_month, freq="M").asi8
        parent = list(range(len(local)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left, right in sorted(cKDTree(coordinates).query_pairs(r=45.0)):
            if abs(int(month_numbers[left]) - int(month_numbers[right])) > 1:
                continue
            a, b = coordinates[left], coordinates[right]
            connection = Point(*a) if np.array_equal(a, b) else LineString([a, b])
            if not polygon_by_id[sample].covers(connection):
                continue
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
            edge_count += 1
        components = {}
        for index in range(len(local)):
            components.setdefault(find(index), []).append(index)
        component_counts = []
        for indices in components.values():
            component = local.iloc[indices]
            node_keys = sorted((r.sample_id, r.unit_id, r.response_month) for r in component.itertuples(index=False))
            part_id = _identifier("part_", prediction_id, node_keys)
            enriched_members.loc[component.index, "part_id"] = part_id
            ids = sorted(component.unit_id.unique().tolist())
            months = sorted(component.response_month.unique().tolist())
            part_geometry = shapely.union_all([clipped[sample, unit] for unit in ids])
            component_counts.append(len(ids))
            parts.append({
                "part_id": part_id, "candidate_id": prediction_id, "prediction_id": prediction_id, "sample_id": sample,
                "source_unit_ids": _json(ids), "unit_ids": _json(ids), "source_node_ids": _json(sorted(component.node_id)),
                "response_months": _json(months), "response_pixels": len(ids), "response_nodes": len(component),
                "monthly_counts": _json({str(month): int(count) for month, count in component.groupby("response_month").unit_id.nunique().items()}),
                "first_response_month": months[0], "last_response_month": months[-1],
                "edge_pixel_fraction": sum(edge_flags[sample, unit] for unit in ids) / len(ids),
                "support_area_m2": float(part_geometry.area), "geometry": part_geometry,
            })
        all_ids = sorted(local.unit_id.unique())
        summaries.append({"prediction_id": prediction_id, "part_count": len(component_counts),
                          "largest_part_pixels": max(component_counts), "single_pixel_part_count": component_counts.count(1),
                          "connected_concentration_fraction": max(component_counts) / len(all_ids),
                          "edge_pixel_fraction": sum(edge_flags[sample, unit] for unit in all_ids) / len(all_ids),
                          "support_area_m2": float(shapely.union_all([clipped[sample, unit] for unit in all_ids]).area)})
    summary_columns = ["prediction_id", "part_count", "largest_part_pixels", "single_pixel_part_count",
                       "connected_concentration_fraction", "edge_pixel_fraction", "support_area_m2"]
    enriched_candidates = enriched_candidates.merge(pd.DataFrame(summaries, columns=summary_columns), on="prediction_id", how="left", validate="one_to_one")
    parts_gdf = gpd.GeoDataFrame(parts, columns=part_columns, geometry="geometry", crs="EPSG:32717")
    if len(parts_gdf):
        parts_gdf = parts_gdf.sort_values(["prediction_id", "part_id"], kind="stable").reset_index(drop=True)
    diagnostic = {"crs": "EPSG:32717", "spatial_radius_m": 45.0, "temporal_radius_calendar_months": 1,
                  "accepted_graph_edges": edge_count, "parts": len(parts_gdf), "candidates": len(enriched_candidates),
                  "strict_center_membership_pairs_checked": len(centers), "candidate_filtering": False}
    return enriched_candidates, enriched_members, parts_gdf, diagnostic
