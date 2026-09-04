import numpy as np
import pandas as pd

from lr_cusum.baselines import (
    bfast_detect_seasonal,
    fpol_bocd_detect,
    is_decrease,
    recusum_detect,
)
from lr_cusum.spatial_fusion import temporal_dbscan_fusion
from lr_cusum.notebook_baselines import is_decrease as legacy_is_decrease
from lr_cusum.notebook_spatial import temporal_dbscan_fusion as legacy_temporal_dbscan


def test_direction_filter_recognizes_downward_state_change():
    assert legacy_is_decrease is is_decrease
    assert legacy_temporal_dbscan is temporal_dbscan_fusion
    values = np.r_[np.zeros(24), np.full(24, -4.0)]
    assert is_decrease(values, 24)
    assert not is_decrease(-values, 24)


def test_notebook_baselines_return_valid_indexes():
    rng = np.random.default_rng(4)
    vv = np.r_[rng.normal(-8, .1, 36), rng.normal(-13, .1, 36)]
    vh = np.r_[rng.normal(-14, .1, 36), rng.normal(-19, .1, 36)]
    fpol = fpol_bocd_detect(vv, vh, delta_m=6)
    bfast = bfast_detect_seasonal(vv, bic_improvement=0)
    recusum = recusum_detect(vv, confidence=.90, bootstrap=100, minimum_segment=12)
    assert all(0 <= value < len(vv) for value in fpol)
    assert bfast is None or 0 <= bfast < len(vv)
    assert all(0 <= value < len(vv) for value in recusum)


def test_temporal_dbscan_requires_polygon_support():
    events = pd.DataFrame({
        "pixel_track_id": ["p0", "p1", "p2"], "track_key": ["t"] * 3,
        "Band": ["VV_dB"] * 3, "change_time": pd.to_datetime(["2020-01-01"] * 3),
    })
    associations = pd.DataFrame({
        "pixel_track_id": ["p0", "p1", "p2", "p3"], "track_key": ["t"] * 4,
        "polygon_track_id": ["pt"] * 4, "native_row": [0, 0, 1, 1],
        "native_col": [0, 1, 0, 1],
    })
    tracks = pd.DataFrame({
        "polygon_track_id": ["pt"], "sample_id": ["s"], "polygon_id": ["g"],
        "role": ["test"], "species_nco": ["BALSA"], "tile_id": ["17MPU"],
        "track_key": ["t"],
    })
    result = temporal_dbscan_fusion(
        events, associations, tracks, minimum_pixel_ratio=.5
    )
    assert len(result) == 1
    assert result.iloc[0].support_pixels == 3
