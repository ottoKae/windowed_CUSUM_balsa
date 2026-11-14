"""Verify published aggregate metrics without accessing individual study records."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT / "results/summary"
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
    print(json.dumps(dict(status="passed", components=5, configurations=17, benchmarks=14,
                          matching="reference_pm1_interval_overlap", individual_records_accessed=False)))


if __name__ == "__main__":
    main()
