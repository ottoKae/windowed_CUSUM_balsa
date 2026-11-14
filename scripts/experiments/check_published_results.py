"""Verify published aggregate metrics without accessing individual study records."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT / "results/summary"
    for name, source in json.loads((folder / "SOURCES.json").read_text()).items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == source["sha256"], name
        assert len(pd.read_csv(folder / name)) == source["rows"], name
    for name, expected in [("table03_components.csv", 5), ("table04_configuration.csv", 17),
                           ("table05_benchmarks.csv", 14), ("temporal_blocks.csv", 5)]:
        frame = pd.read_csv(folder / name)
        assert len(frame) == expected and frame.variant_id.is_unique, name
        assert frame.n_stands.eq(459).all() and frame.n_references.eq(1049).all(), name
        assert frame.matching_rule.eq("reference_pm1_interval_overlap").all(), name
        for col, value in [("catalogue_precision", frame.n_matches / frame.n_reports),
                           ("recall", frame.n_matches / frame.n_references),
                           ("catalogue_f1", 2 * frame.n_matches / (frame.n_references + frame.n_reports))]:
            np.testing.assert_allclose(frame[col], value, rtol=0, atol=1e-14)
    components = pd.read_csv(folder / "table03_components.csv").set_index("variant_id")
    effects = pd.read_csv(folder / "table03_adjacent_effects.csv")
    for row in effects.itertuples():
        expected = components.loc[row.current_variant, "catalogue_f1"] - components.loc[row.previous_variant, "catalogue_f1"]
        assert abs(row.difference - expected) < 1e-14
        assert row.n_bootstrap == 5000
    baselines = [components.loc["F4_permutation"]]
    for filename, key in [("table04_configuration.csv", "full"), ("table05_benchmarks.csv", "windowed_cusum_w36_firstpost")]:
        baselines.append(pd.read_csv(folder / filename).set_index("variant_id").loc[key])
    for row in baselines:
        assert (row.n_reports, row.n_matches, row.n_original_pairs_retained) == (1208, 945, 495)
    annual = pd.read_csv(folder / "national_canton_annual_reports.csv").set_index("year")
    quarterly = pd.read_csv(folder / "national_canton_quarterly_reports.csv").set_index("quarter")
    for frame in [annual, quarterly]:
        np.testing.assert_array_equal(frame.drop(columns="National").sum(axis=1), frame.National)
        assert frame.National.sum() == 6303
    quarterly.index = pd.PeriodIndex(quarterly.index, freq="Q").year
    pd.testing.assert_frame_equal(quarterly.groupby(level=0).sum(), annual, check_names=False)
    climate = pd.read_csv(folder / "national_climate_interval_summary.csv").set_index("group")
    bins = pd.read_csv(folder / "national_climate_interval_distributions.csv")
    for group, summary in climate.iterrows():
        counts = bins.loc[bins.group.eq(group), "n_intervals"]
        assert counts.sum() == summary.n_intervals, group
    assert climate.loc["All", "n_intervals"] == 3372
    assert pd.read_csv(folder / "national_canton_ranking.csv").reports.sum() == 6303
    print(json.dumps(dict(status="passed", components=5, configurations=17, benchmarks=14,
                          matching="reference_pm1_interval_overlap", individual_records_accessed=False)))


if __name__ == "__main__":
    main()
