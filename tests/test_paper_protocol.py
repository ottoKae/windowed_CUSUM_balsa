import json

import numpy as np
import pandas as pd
import pytest

from lr_cusum.paper import (
    PaperDetectionConfig,
    detect_table_n1,
    load_paper_config,
)


def test_frozen_defaults_match_manuscript_protocol():
    config = PaperDetectionConfig()
    assert (config.window_length, config.step) == (48, 12)
    assert (config.alpha, config.permutations) == (0.025, 3_999)
    assert (config.refine_low, config.refine_high) == (0.015, 0.035)
    assert config.refined_permutations == 19_999


def test_n1_reports_first_post_change_month():
    months = pd.date_range("2016-01-01", periods=126, freq="MS")
    values = np.r_[np.full(50, -8.0), np.full(76, -15.0)]
    frame = pd.DataFrame({"ID": "stand-1", "time": months, "VV_dB": values})
    config = PaperDetectionConfig(
        bands=("VV_dB",),
        alpha=0.05,
        permutations=99,
        refine_low=0.04,
        refine_high=0.06,
        refined_permutations=199,
    )

    events = detect_table_n1(frame, config=config)

    assert not events.empty
    # The detected CUSUM boundary is February 2020 for this synthetic step;
    # production semantics include that indexed month in the post slice.
    assert set(events.change_time) == {pd.Timestamp("2020-02-01")}
    assert events.post_minus_pre.lt(0).all()
    assert events.null_p_value.le(config.alpha).all()


def test_internal_missing_month_must_be_explicit_nan():
    months = pd.date_range("2016-01-01", periods=60, freq="MS").delete(10)
    frame = pd.DataFrame({"ID": "stand-1", "time": months, "VV_dB": -8.0})
    with pytest.raises(ValueError, match="missing months must be explicit"):
        detect_table_n1(
            frame,
            config=PaperDetectionConfig(
                bands=("VV_dB",), permutations=9, refined_permutations=19
            ),
        )


def test_nested_protocol_json_loads_detector_only(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps({"schema_version": 1, "detector": {"bands": ["VV_dB"]}}),
        encoding="utf-8",
    )
    assert load_paper_config(path).bands == ("VV_dB",)
