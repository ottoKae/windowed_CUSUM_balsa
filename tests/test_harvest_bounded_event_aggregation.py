"""B20 v1.1 source allocation, dating, and descriptive geometry contracts."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PROJ_DATA", str(Path(sys.prefix) / "Library/share/proj"))
os.environ.setdefault("GDAL_DATA", str(Path(sys.prefix) / "Library/share/gdal"))
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import harvest_bounded_event_aggregation as agg


def fixture(n, records=()):
    units = pd.DataFrame([{"unit_id": f"u{i:03}", "pixel_track_id": f"track:0:{i}",
                           "track_key": "track", "native_row": 0, "native_col": i} for i in range(n)],
                         columns=["unit_id", "pixel_track_id", "track_key", "native_row", "native_col"])
    membership = pd.DataFrame({"unit_id": units.unit_id, "sample_id": "plot"})
    events = pd.DataFrame([{"unit_id": f"u{i:03}", "pixel_track_id": f"track:0:{i}",
                            "change_time": month + "-01", "candidate_id": f"event-{i}-{month}"}
                           for i, month in records],
                          columns=["unit_id", "pixel_track_id", "change_time", "candidate_id"])
    return units, membership, events


def geometry_fixture(coordinates, records, polygon=None, crs=None):
    units, membership, events = fixture(len(coordinates), records)
    systems = crs or ["EPSG:32717"] * len(coordinates)
    transforms, footprints = [], []
    for i, ((x, y), system) in enumerate(zip(coordinates, systems)):
        if system == "EPSG:32617":
            y -= 10000000
        # Each pixel keeps its own genuine transform and native column.
        transforms.append(json.dumps([30, 0, x - 15 - 30 * i, 0, -30, y + 15]))
        footprints.append(box(x - 15, y - 15, x + 15, y + 15).wkb)
    units["transform"], units["native_geometry_wkb"], units["track_crs"] = transforms, footprints, systems
    if polygon is None:
        xy = np.asarray(coordinates)
        polygon = box(xy[:, 0].min() - 20, xy[:, 1].min() - 20,
                      xy[:, 0].max() + 20, xy[:, 1].max() + 20)
    polygons = gpd.GeoDataFrame({"sample_id": ["plot"]}, geometry=[polygon], crs="EPSG:32717")
    return units, membership, events, polygons


def test_example_a_cross_year_same_pixels_choose_more_nodes_then_earlier_window():
    args = fixture(100, [(i, month) for month in ("2020-12", "2021-01") for i in range(20)])
    candidates, members, diagnostic = agg.build_bounded_events(*args)
    assert len(candidates) == 1 and len(members) == 40
    row = candidates.iloc[0]
    assert row.window_start_month == "2020-11" and row.window_end_month == "2021-01"
    assert row.response_pixels == 20 and row.accepted_response_nodes == 40
    assert row.support_fraction == .2 and row.cross_year_repeated_pixels == 20
    assert row.representative_month == "2021-01" and row.predicted_quarter == "2021Q1"
    assert row.initial_support_pixels == row.accepted_support_pixels == 20
    assert row.initial_response_nodes == row.accepted_response_nodes == 40
    assert members.initially_covered_by_qualified_window.all()
    assert members.destination_type.eq("candidate").all()
    assert diagnostic["residual_nodes"] == 0 and diagnostic["window_count"] == 106


def test_example_b_greedy_31_then_nine_initially_qualified_residuals():
    records = [(i, "2020-02") for i in range(10)] + [(i, "2020-03") for i in range(10, 20)]
    records += [(i, "2020-04") for i in range(20, 31)] + [(i, "2020-05") for i in range(31, 40)]
    candidates, members, diagnostic = agg.build_bounded_events(*fixture(100, records))
    assert len(candidates) == 1 and candidates.iloc[0].window_start_month == "2020-02"
    assert candidates.iloc[0].response_pixels == 31
    residual = members.loc[members.destination_type.eq("residual")]
    assert len(residual) == 9 and residual.response_month.eq("2020-05").all()
    assert residual.initially_covered_by_qualified_window.all()
    assert residual.prediction_id.eq("").all() and residual.total_pixels.eq(100).all()
    assert diagnostic["assigned_nodes"] == 31 and diagnostic["residual_initially_qualified_nodes"] == 9


@pytest.mark.parametrize("total,count,passes", [(100, 19, False), (100, 20, True), (6, 1, False),
                                                 (6, 2, True), (1, 1, True), (5, 1, True)])
def test_integer_gate_keeps_all_members_and_fixed_denominator(total, count, passes):
    candidates, members, _ = agg.build_bounded_events(*fixture(total, [(i, "2020-04") for i in range(count)]))
    assert len(candidates) == int(passes)
    assert members.total_pixels.eq(total).all()
    assert members.initially_covered_by_qualified_window.eq(passes).all()


def test_aliases_never_change_ranking_dates_or_candidate_and_node_ids():
    units, membership, events = fixture(10, [(0, "2020-04"), (1, "2020-06")])
    expected, original, _ = agg.build_bounded_events(units, membership, events)
    extra = events.iloc[[0]].copy()
    extra["candidate_id"] = "another-alias"
    changed = pd.concat([events, extra, extra], ignore_index=True).sample(frac=1, random_state=8)
    actual, members, diagnostic = agg.build_bounded_events(units.sample(frac=1), membership.sample(frac=1), changed)
    assert actual.prediction_id.tolist() == expected.prediction_id.tolist()
    assert actual.source_node_ids.tolist() == expected.source_node_ids.tolist()
    assert members.node_id.tolist() == original.node_id.tolist()
    assert actual.iloc[0].representative_month == "2020-05"
    assert not actual.iloc[0].representative_month_has_response
    assert actual.iloc[0].source_event_count == 2 and actual.iloc[0].source_alias_count == 3
    assert set(json.loads(members.iloc[0].source_event_ids)) == {"event-0-2020-04", "another-alias"}
    assert diagnostic["duplicate_pixel_month_rows"] == 2


def test_one_pixel_other_years_are_not_removed_and_initial_counts_stay_fixed():
    records = [(i, "2019-07") for i in range(20)] + [(i, "2020-07") for i in range(20)]
    candidates, members, _ = agg.build_bounded_events(*fixture(100, records))
    assert len(candidates) == 2 and len(members) == 40
    assert members.node_id.is_unique
    assert candidates.initial_support_pixels.eq(20).all()
    assert set(candidates.representative_month) == {"2019-07", "2020-07"}


def test_no_four_month_chain_and_input_permutation_does_not_change_memberships():
    args = fixture(20, [(i, f"2020-{i // 5 + 1:02}") for i in range(20)])
    expected, members, _ = agg.build_bounded_events(*args)
    actual, shuffled, _ = agg.build_bounded_events(args[0], args[1], args[2].sample(frac=1, random_state=2))
    assert len(expected) == 2 and expected.span_months.le(3).all()
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(members, shuffled)
    assert expected.iloc[1].initial_support_pixels > expected.iloc[1].accepted_support_pixels


def test_edges_use_actual_months_and_short_context_uses_any_2025_member():
    records = [(0, "2017-03"), (0, "2024-12"), (1, "2024-12"), (2, "2025-01")]
    candidates, members, _ = agg.build_bounded_events(*fixture(5, records))
    march = candidates.loc[candidates.representative_month.eq("2017-03")].iloc[0]
    assert march.window_start_month == "2017-01" and not march.aggregation_edge_exposed
    december = candidates.loc[candidates.representative_month.eq("2024-12")].iloc[0]
    assert december.short_context_2025 and not december.aggregation_edge_exposed
    _, edges, _ = agg.build_bounded_events(*fixture(100, [(0, "2017-01"), (1, "2017-02"), (2, "2025-11"), (3, "2025-12")]))
    assert edges.aggregation_edge_exposed.all() and edges.destination_type.eq("residual").all()


def test_empty_inputs_and_reject_conflicting_annual_native_events():
    candidates, members, diagnostic = agg.build_bounded_events(*fixture(0))
    assert candidates.empty and members.empty and diagnostic["accepted_candidates"] == 0
    candidates, members, _ = agg.build_bounded_events(*fixture(10))
    assert candidates.empty and members.empty
    with pytest.raises(ValueError, match="Conflicting accepted months"):
        agg.build_bounded_events(*fixture(5, [(0, "2020-01"), (0, "2020-02")]))
    units, membership, events = fixture(5, [(0, "2020-01")])
    events.loc[0, "pixel_track_id"] = "wrong-track"
    with pytest.raises(ValueError, match="native identity mismatch"):
        agg.build_bounded_events(units, membership, events)


def test_common_adapter_keeps_other_method_and_multiple_months_without_w36_gates():
    units, membership, events = fixture(5, [(0, "2020-01"), (0, "2020-02")])
    events["method"] = "recusum_pixel"
    events["Band"] = "VV+VH"
    events["null_p_value"] = .9
    events["post_minus_pre"] = 1.0
    events["decision_start"] = pd.to_datetime(["2019-12-01", "2019-12-01"])
    candidates, members, diagnostic = agg.build_bounded_events(
        units, membership, events,
        allow_multiple_months_per_year=True,
        source_method=None,
        validate_frozen_gates=False,
        require_annual_decision=False,
        output_method="B20:recusum_pixel",
        protocol_version="common_pixel_b20_v1",
    )
    assert len(candidates) == 1
    assert candidates.iloc[0].method == "B20:recusum_pixel"
    assert candidates.iloc[0].accepted_response_nodes == 2
    assert set(members.response_month) == {"2020-01", "2020-02"}
    assert diagnostic["source_method"] is None


def test_historical_defaults_preserve_candidate_identity():
    args = fixture(5, [(0, "2020-01")])
    historical, _, _ = agg.build_bounded_events(*args)
    explicit, _, _ = agg.build_bounded_events(
        *args,
        source_method="windowed_w36_d12_s12",
        validate_frozen_gates=True,
        require_annual_decision=True,
        output_method="B20",
        protocol_version=agg.PROTOCOL_VERSION,
    )
    pd.testing.assert_frame_equal(historical, explicit)


def test_source_without_membership_cannot_silently_disappear():
    units, membership, events = fixture(5, [(0, "2020-01")])
    with pytest.raises(ValueError, match="no frozen polygon membership"):
        agg.build_bounded_events(units, membership.iloc[1:], events)


def test_spatial_parts_same_adjacent_month_45m_and_do_not_filter():
    xy = [(500000, 9900000), (500030, 9900000), (500060, 9900000), (500200, 9900000)]
    args = geometry_fixture(xy, [(0, "2020-01"), (1, "2020-02"), (2, "2020-03"), (3, "2020-01")])
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, nodes, parts, diagnostic = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert len(enriched) == len(candidates) == 1 and len(parts) == 2
    assert enriched.iloc[0].part_count == 2 and enriched.iloc[0].largest_part_pixels == 3
    assert enriched.iloc[0].single_pixel_part_count == 1
    assert enriched.iloc[0].connected_concentration_fraction == .75
    assert nodes.part_id.ne("").all() and diagnostic["candidate_filtering"] is False
    assert enriched.response_pixels.tolist() == candidates.response_pixels.tolist()
    assert set(parts.prediction_id) == set(candidates.prediction_id)
    rerun = agg.add_spatiotemporal_parts(candidates, members.sample(frac=1, random_state=7), args[0], args[3])
    assert set(rerun[2].part_id) == set(parts.part_id)


def test_same_position_cross_year_uses_safe_point_and_unique_footprint_area():
    args = geometry_fixture([(500000, 9900000)], [(0, "2020-12"), (0, "2021-01")])
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, nodes, parts, _ = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert len(parts) == 1 and parts.iloc[0].response_pixels == 1 and parts.iloc[0].response_nodes == 2
    assert enriched.iloc[0].support_area_m2 == pytest.approx(900)
    assert nodes.part_id.nunique() == 1 and enriched.iloc[0].connected_concentration_fraction == 1


def test_hole_blocks_connection_and_support_geometry_is_clipped():
    outer = [(499975, 9899975), (500055, 9899975), (500055, 9900025), (499975, 9900025)]
    hole = [(500010, 9899990), (500020, 9899990), (500020, 9900010), (500010, 9900010)]
    polygon = Polygon(outer, [hole])
    args = geometry_fixture([(500000, 9900000), (500030, 9900000)], [(0, "2020-01"), (1, "2020-01")], polygon)
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, _, parts, _ = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert len(parts) == 2 and enriched.iloc[0].edge_pixel_fraction == 1
    assert enriched.iloc[0].support_area_m2 == pytest.approx(1600)
    assert all(polygon.covers(part) for part in parts.geometry)


def test_real_geometry_connects_across_crs_and_ignores_native_rowcol_spacing():
    args = geometry_fixture([(500000, 9900000), (500030, 9900000)], [(0, "2020-01"), (1, "2020-01")],
                            crs=["EPSG:32717", "EPSG:32617"])
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, nodes, parts, _ = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert len(parts) == 1 and enriched.iloc[0].largest_part_pixels == 2
    assert np.abs(nodes.center_y_32717 - 9900000).max() < 1e-6


def test_center_outside_or_on_boundary_is_not_accepted_even_if_footprint_intersects():
    args = geometry_fixture([(500000, 9900000)], [(0, "2020-01")], box(500000, 9899990, 500010, 9900010))
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    with pytest.raises(ValueError, match="strictly inside"):
        agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])


def test_temporally_separated_same_area_parts_preserve_candidate_pixel_union():
    args = geometry_fixture([(500000, 9900000)], [(0, "2020-11"), (0, "2021-01")])
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, _, parts, _ = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert len(parts) == 2 and parts.response_pixels.sum() == 2
    assert enriched.iloc[0].response_pixels == 1
    assert enriched.iloc[0].support_area_m2 == pytest.approx(900)
    assert enriched.iloc[0].connected_concentration_fraction == 1


def test_spatial_distance_boundary_is_inclusive_and_above_45m_is_not_connected():
    args = geometry_fixture([(500000, 9900000), (500045, 9900000), (500090.001, 9900000)],
                            [(0, "2020-01"), (1, "2020-01"), (2, "2020-01")])
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, _, parts, diagnostic = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert enriched.iloc[0].part_count == 2 and sorted(parts.response_pixels) == [1, 2]
    assert diagnostic["accepted_graph_edges"] == 1


def test_all_residual_geometry_keeps_nodes_without_fabricating_parts():
    args = geometry_fixture([(500000 + 30 * i, 9900000) for i in range(6)], [(0, "2020-01")])
    candidates, members, _ = agg.build_bounded_events(*args[:3])
    enriched, nodes, parts, diagnostic = agg.add_spatiotemporal_parts(candidates, members, args[0], args[3])
    assert enriched.empty and parts.empty and len(nodes) == 1
    assert nodes.part_id.eq("").all() and nodes.destination_type.eq("residual").all()
    assert diagnostic["strict_center_membership_pairs_checked"] == 1
