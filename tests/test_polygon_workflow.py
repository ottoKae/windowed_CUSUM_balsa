import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import box

from lr_cusum.paper import PaperDetectionConfig
from lr_cusum.polygon_workflow import (
    _geometry_pixels, _largest_component, _series_events, annotate_vv_with_vh, fuse_tracks,
    match_sample_events,
)


def test_geometry_pixels_uses_pixel_centres_and_falls_back_for_tiny_polygon():
    transform = Affine(30, 0, 0, 0, -30, 90)
    rows, cols, mode = _geometry_pixels(box(0, 30, 60, 90), transform, 3, 3)
    assert mode == "pixel_centres"
    assert set(zip(rows, cols)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    rows, cols, mode = _geometry_pixels(box(1, 61, 2, 62), transform, 3, 3)
    assert mode == "representative_fallback"
    assert len(rows) == len(cols) == 1


def test_largest_component_uses_eight_neighbour_connectivity():
    assert _largest_component({(0, 0), (1, 1), (5, 5)}) == 2


def test_polygon_routes_accept_the_paper_n1_detector():
    calendar = pd.period_range("2017-01", "2021-12", freq="M")
    signal = np.r_[np.full(30, -8.0), np.full(30, -14.0)]
    values = np.vstack([signal, signal - 6.0])
    config = PaperDetectionConfig(
        permutations=19,
        refined_permutations=39,
        refine_low=0.0,
        refine_high=0.0,
        alpha=0.5,
        start_year=2017,
        end_year=2021,
    )
    events = _series_events("stand", values, config, calendar)
    assert events
    assert all("null_p_value" in event for event in events)
    assert all(event["selection_score"] == 1.0 - event["null_p_value"] for event in events)


def test_track_fusion_and_sample_event_matching_are_one_to_one():
    events = pd.DataFrame({
        "sample_id": ["a", "a", "a"], "polygon_id": ["a"] * 3,
        "role": ["test"] * 3, "species_nco": ["BALSA"] * 3,
        "Band": ["VV_dB"] * 3,
        "track_key": ["t1", "t2", "t1"],
        "change_time": pd.to_datetime(["2020-05-01", "2020-06-01", "2023-01-01"]),
    })
    fused = fuse_tracks(events, tolerance_months=1)
    assert len(fused) == 2
    assert fused.track_support.max() == 2
    references = pd.DataFrame({"sample_id": ["a"], "event_month": pd.to_datetime(["2020-06-01"])})
    metrics = match_sample_events(references, fused, 1)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1


def test_vv_vh_support_is_one_to_one():
    events = pd.DataFrame({
        "route": ["r"] * 4, "sample_id": ["a"] * 4,
        "Band": ["VV_dB", "VV_dB", "VH_dB", "VH_dB"],
        "change_time": pd.to_datetime(["2020-05-01", "2020-06-01", "2020-05-01", "2023-01-01"]),
        "track_support": [1, 1, 1, 1],
    })
    primary, dual = annotate_vv_with_vh(events, tolerance_months=1)
    assert len(primary) == 2
    assert primary.vh_supported.sum() == 1
    assert len(dual) == 1
