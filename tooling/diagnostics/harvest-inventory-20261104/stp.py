"""Apply Spatial Timing Partitioning (STP) to frozen TDA inventory reports."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.stand import star_segmentation as stars

RUN = ROOT / "runs/validation/harvest_inventory_20261104_v1"
OLD = ROOT / "runs/validation/balsa_sts_episode_v4_1_20261102_v1/baselines/methods/R2_TIME"
A1 = ROOT / "runs/validation/balsa_stars_ablation_20261102_v1"
CHAIN = ROOT / "runs/validation/balsa_sts_method_order_20261103_v3/parity/REF_STARS"
MASTER = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2/master_units.parquet"
CONFIG = stars.Config(permutations=999, seed=20261102)
TABLE_KEYS = {"stands": "component_id", "zones": "zone_id", "rounds": "round_id",
              "node_ownership": "node_id", "tests": "component_id"}


def run_stp(reports, members, units, out_dir, inventory_ids, cohort_only=False):
    """Reuse verified old calibration; continue its original RNG for new reports."""
    out_dir = Path(out_dir)
    if (out_dir / "execution_receipt.json").exists():
        raise FileExistsError(f"Completed STP output exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    legacy_reports = pd.read_parquet(OLD / "reports.parquet")
    legacy_members = pd.read_parquet(OLD / "source_members.parquet")
    legacy_members = legacy_members.loc[legacy_members.component_id.isin(legacy_reports.component_id)].copy()
    frozen_tests = pd.read_parquet(A1 / "calibration_A1.parquet")
    members = members.loc[members.component_id.isin(reports.component_id)].copy()
    validate_inputs(reports, members, units, inventory_ids)
    input_parity = check_overlap(reports, members, units, inventory_ids, legacy_reports, legacy_members)
    input_parity.to_csv(out_dir / "input_overlap_parity.csv", index=False)
    cache = calibration_cache(frozen_tests)
    unit_lookup = units.set_index("unit_id", drop=False)
    legacy_unit_ids = legacy_members.unit_id.unique()
    missing_old_units = set(legacy_unit_ids) - set(unit_lookup.index)
    if missing_old_units:
        original_units = pd.read_parquet(MASTER).set_index("unit_id", drop=False)
        unit_lookup = pd.concat([unit_lookup, original_units.loc[sorted(missing_old_units), unit_lookup.columns]])
    rng = np.random.default_rng(CONFIG.seed)
    order = []
    legacy = segment_sequence(legacy_reports, legacy_members, unit_lookup, rng, cache, order, "frozen_cache")
    parity = compare_frozen_chain(legacy)
    parity.to_csv(out_dir / "cohort_parity.csv", index=False)
    old_q = legacy["stands"].set_index("component_id").q_value
    old_ids = set(legacy_reports.component_id)
    present_old_ids = old_ids & set(reports.component_id)
    retained = {name: table.loc[table.component_id.isin(present_old_ids)].copy() for name, table in legacy.items()}
    added = reports.loc[~reports.component_id.isin(old_ids)].sort_values(
        ["sample_id", "first_response_month", "representative_month", "component_id"]).copy()
    print(f"Frozen STP chain reproduced: {len(legacy_reports)} parents; new parents={len(added)}", flush=True)
    new_members = members.loc[members.component_id.isin(added.component_id)]
    if len(added):
        extension = segment_sequence(added, new_members, unit_lookup, rng, {}, order, "new_same_rule")
        result = {name: pd.concat([retained[name], extension[name]], ignore_index=True) for name in TABLE_KEYS}
    else:
        result = retained
    result["stands"]["q_value"] = stars.benjamini_hochberg(result["stands"].p_value)
    result["stands"]["q_value_frozen_cohort"] = result["stands"].component_id.map(old_q)
    differences = result["stands"].loc[result["stands"].component_id.isin(present_old_ids),
        ["component_id", "sample_id", "p_value", "q_value", "q_value_frozen_cohort"]].copy()
    differences["expected_scope_difference"] = ~np.isclose(
        differences.q_value, differences.q_value_frozen_cohort, equal_nan=True)
    differences["meaning"] = "BH family changed; p<=0.05 remains the frozen decision rule"
    differences.to_csv(out_dir / "expected_scope_differences.csv", index=False)
    check_source_conservation(result, reports, members)
    for name, table in result.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    source_rows = []
    for round_id, group in result["node_ownership"].groupby("round_id", sort=True):
        source_rows.append(dict(
            round_id=round_id, component_id=group.component_id.iloc[0], sample_id=group.sample_id.iloc[0],
            source_node_ids=json.dumps(sorted(group.node_id)),
            source_unit_ids=json.dumps(sorted(group.unit_id.unique())),
            source_node_count=len(group), response_pixels=group.unit_id.nunique()))
    pd.DataFrame(source_rows).to_parquet(out_dir / "child_sources.parquet", index=False)
    pd.DataFrame(order).to_csv(out_dir / "calibration_order.csv", index=False)
    write_json(out_dir / "protocol.json", dict(
        config=asdict(CONFIG), legacy_parent_order="Original frozen reports.parquet row order",
        extension_order=["sample_id", "first_response_month", "representative_month", "component_id"],
        random_stream="default_rng(20261102); replay each old admissible report's 999 permutation draws, then append new reports",
        cached_calibration="Exact component, node, native coordinate and denominator parity required before reuse",
        cache_scope="1216 frozen parents; reference labels are never inputs to calibration",
        q_value_scope="all returned inventory parents with finite p; descriptive only",
        q_value_frozen_cohort_scope="original 1216-parent family; retained for frozen provenance",
        decision="original p_value <= 0.05, min-zone and parent-support gates unchanged",
        cohort_only=cohort_only))
    return dict(tables=result, parity=parity, metadata=dict(
        elapsed_seconds=perf_counter() - started, input_parents=len(reports),
        frozen_parents_replayed=len(legacy_reports), frozen_parents_returned=len(present_old_ids),
        new_parents=len(added), child_reports=len(result["rounds"]),
        accepted_splits=int(result["stands"].destination_type.eq("zone_rounds").sum()),
        admissible=int(result["stands"].admissible.sum()), source_nodes=len(members),
        q_expected_scope_differences=int(differences.expected_scope_difference.sum()),
        frozen_chain_fields_passed=True, source_conservation_passed=True))


# Historical callers retain the original function spelling.
run_cstr = run_stp


def segment_sequence(reports, members, units, rng, cache, order, stage):
    """Use the unmodified component implementation and an explicit RNG sequence."""
    groups = dict(tuple(members.groupby("component_id", sort=False)))
    collected = {name: [] for name in TABLE_KEYS}
    started = perf_counter()
    for position, report in enumerate(reports.itertuples(index=False)):
        local = groups[report.component_id]
        local_units = units.loc[local.unit_id.unique()]
        cached = cache.get(report.component_id)
        before_state = state_digest(rng)
        begin = perf_counter()
        row, zones, rounds, owned, test = stars.segment_component(
            report.component_id, report.sample_id, report.prediction_id, local,
            local_units.reset_index(drop=True), int(report.total_pixels), CONFIG, rng, cached=cached)
        tested_track_size = 0
        if cached is not None and row["admissible"]:
            tested_track_size = int(local_units.groupby("track_key").size().max())
            # The original test permutes once per draw, before iterating tolerances.
            for _ in range(CONFIG.permutations):
                rng.permutation(tested_track_size)
        elif row["admissible"]:
            tested_track_size = int(local_units.groupby("track_key").size().max())
        order.append(dict(order=len(order), component_id=report.component_id, sample_id=report.sample_id,
            stage=stage, admissible=row["admissible"], tested_track_pixels=tested_track_size,
            permutation_draws=CONFIG.permutations if row["admissible"] else 0,
            rng_state_before=before_state, rng_state_after=state_digest(rng),
            elapsed_seconds=perf_counter() - begin))
        collected["stands"].append(row)
        collected["zones"].append(zones)
        collected["rounds"].append(rounds)
        collected["node_ownership"].append(owned)
        collected["tests"].append(dict(component_id=report.component_id, sample_id=report.sample_id,
                                      total_pixels=int(report.total_pixels), **test))
        if (position + 1) % 50 == 0:
            print(f"STP {stage}: {position + 1}/{len(reports)} parents; {perf_counter()-started:.1f}s", flush=True)
    result = {name: pd.DataFrame(rows) if name in ("stands", "tests") else pd.concat(rows, ignore_index=True)
              for name, rows in collected.items()}
    result["stands"] = result["stands"].reindex(columns=stars.STAND_COLUMNS)
    result["stands"]["q_value"] = stars.benjamini_hochberg(result["stands"].p_value)
    return result


def calibration_cache(frame):
    cache = {}
    for row in frame.to_dict("records"):
        chosen = row.pop("tolerance_months")
        chosen = None if pd.isna(chosen) else int(chosen)
        row.update(tolerance=chosen, observed={chosen: row.pop("observed_join_count")},
                   means={chosen: row.pop("null_join_mean")}, sds={chosen: row.pop("null_join_sd")})
        cache[row["component_id"]] = row
    return cache


def validate_inputs(reports, members, units, inventory_ids):
    if not reports.component_id.is_unique or not reports.prediction_id.is_unique:
        raise ValueError("TDA report component and prediction IDs must be unique")
    if not members.node_id.is_unique or members.duplicated(["component_id", "unit_id", "month_index"]).any():
        raise ValueError("TDA formal sources must preserve exclusive deduplicated nodes")
    if set(reports.component_id) != set(members.component_id):
        raise ValueError("Every formal parent must have sources, and every source a formal parent")
    if not units.unit_id.is_unique or not set(members.unit_id).issubset(units.unit_id):
        raise ValueError("Native units must be unique and cover every source member")
    if not set(reports.sample_id).issubset(inventory_ids):
        raise ValueError("Parent report outside the retained inventory")


def check_overlap(reports, members, units, inventory_ids, old_reports, old_members):
    required_old = old_reports.loc[old_reports.sample_id.isin(inventory_ids)]
    if not set(required_old.component_id).issubset(reports.component_id):
        raise ValueError("Inventory aggregation lost frozen parent reports within the inventory")
    records = []
    report_fields = [c for c in old_reports if c not in ("legacy_component_id", "legacy_prediction_id")]
    member_fields = [c for c in old_members if c not in ("legacy_component_id", "legacy_prediction_id")]
    pairs = [
        ("parent_reports", reports.loc[reports.component_id.isin(required_old.component_id)], required_old, "component_id", report_fields),
        ("parent_sources", members.loc[members.component_id.isin(required_old.component_id)],
         old_members.loc[old_members.component_id.isin(required_old.component_id)], "node_id", member_fields),
    ]
    original_units = pd.read_parquet(MASTER)
    overlapping_units = old_members.loc[old_members.component_id.isin(required_old.component_id)].unit_id.unique()
    pairs.append(("native_geometry", units.loc[units.unit_id.isin(overlapping_units)],
                  original_units.loc[original_units.unit_id.isin(overlapping_units)], "unit_id",
                  ["unit_id", "native_row", "native_col", "track_key"]))
    for name, actual, expected, key, fields in pairs:
        compare_table(actual, expected, key, fields)
        records.append(dict(table=name, rows=len(expected), fields=len(fields), passed=True))
    return pd.DataFrame(records)


def compare_frozen_chain(result):
    records = []
    for name, key in TABLE_KEYS.items():
        frozen = pd.read_parquet(CHAIN / f"{name}.parquet")
        compare_table(result[name], frozen, key, list(frozen.columns))
        records.append(dict(table=name, rows=len(frozen), fields=len(frozen.columns), passed=True,
                            scope="original_1216_parent_calibration_family"))
    for name in ("stands", "rounds"):
        frozen = pd.read_parquet(A1 / f"{name}_A1.parquet")
        compare_table(result[name], frozen, TABLE_KEYS[name], list(frozen.columns))
        records.append(dict(table=f"ablation_{name}", rows=len(frozen), fields=len(frozen.columns), passed=True,
                            scope="original_1216_parent_calibration_family"))
    return pd.DataFrame(records)


def compare_table(actual, expected, key, fields):
    pd.testing.assert_frame_equal(
        actual[fields].sort_values(key).reset_index(drop=True),
        expected[fields].sort_values(key).reset_index(drop=True), check_dtype=False, check_exact=True)


def check_source_conservation(result, reports, members):
    owned = result["node_ownership"]
    fields = ["node_id", "unit_id", "sample_id", "component_id", "month_index"]
    compare_table(owned, members, "node_id", fields)
    if not owned.node_id.is_unique or owned[["zone_id", "round_id"]].isna().any().any():
        raise AssertionError("Every source node must have exactly one spatial child")
    parent_ids = reports.set_index("component_id").prediction_id
    if not result["rounds"].parent_prediction_id.eq(result["rounds"].component_id.map(parent_ids)).all():
        raise AssertionError("Child parent IDs changed")
    split_ids = result["stands"].loc[result["stands"].destination_type.eq("zone_rounds"), "component_id"]
    if not result["rounds"].loc[result["rounds"].component_id.isin(split_ids), "parent_support_fraction"].ge(.2).all():
        raise AssertionError("Accepted spatial split violates the parent support gate")


def state_digest(rng):
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tda", "--octa", dest="tda", type=Path, default=RUN / "octa")
    parser.add_argument("--out", type=Path, default=RUN / "cstr")
    parser.add_argument("--cohort-only", action="store_true")
    args = parser.parse_args()
    input_dir = OLD if args.cohort_only else args.tda
    out_dir = args.out / "cohort_parity" if args.cohort_only else args.out
    if not args.cohort_only:
        tda_receipt = json.loads((args.tda / "execution_receipt.json").read_text(encoding="utf-8"))
        if tda_receipt.get("status") != "passed":
            raise ValueError("TDA must finish and pass before STP consumes its reports")
    paths = [input_dir / "reports.parquet", input_dir / "source_members.parquet", MASTER,
             OLD / "reports.parquet", OLD / "source_members.parquet", A1 / "calibration_A1.parquet",
             A1 / "stands_A1.parquet", A1 / "rounds_A1.parquet",
             *[CHAIN / f"{name}.parquet" for name in TABLE_KEYS]]
    inventory_path = (ROOT / "runs/validation/balsa_firstpost_national_20261030_v2/products/stand_summary.csv")
    paths.append(inventory_path)
    if not args.cohort_only:
        paths.append(args.tda / "execution_receipt.json")
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    code_paths = [Path(__file__), ROOT / "src/windowed_cusum/stand/star_segmentation.py"]
    code_before = {str(p.relative_to(ROOT)): digest(p) for p in code_paths}
    reports = pd.read_parquet(input_dir / "reports.parquet")
    members = pd.read_parquet(input_dir / "source_members.parquet")
    units = pd.read_parquet(MASTER)[["unit_id", "native_row", "native_col", "track_key"]]
    inventory_ids = set(pd.read_csv(inventory_path).sample_id)
    result = run_stp(reports, members, units, out_dir, inventory_ids, args.cohort_only)
    if before != {str(p.relative_to(ROOT)): digest(p) for p in paths}:
        raise RuntimeError("Frozen input changed during STP aggregation")
    if code_before != {str(p.relative_to(ROOT)): digest(p) for p in code_paths}:
        raise RuntimeError("STP execution source changed")
    write_json(out_dir / "execution_receipt.json", dict(
        status="passed", completed_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        config=asdict(CONFIG), **result["metadata"], input_sha256=before, code_sha256=code_before,
        protected_inputs_unchanged=True, source_code_unchanged=True,
        output_sha256={p.name: digest(p) for p in sorted(out_dir.iterdir()) if p.is_file()},
        detector_rerun=False, thresholds_changed=False, reference_labels_used_for_calibration=False,
        mode="frozen_cohort_parity" if args.cohort_only else "full_inventory_stp_aggregation"))
    print(json.dumps(result["metadata"]), flush=True)


if __name__ == "__main__":
    main()
