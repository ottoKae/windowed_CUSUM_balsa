"""Build the Sentinel-1 observation inventory used by the manuscript.

The frozen polygon cache identifies the tile--track groups actually used in
the analysis.  The S1-GRiTS catalogue supplies their archive metadata.  This
script joins those two sources, validates the frozen counts, and writes both a
40-row audit table and the four-row orbit-family summary reported in the text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


EXPECTED = {
    "catalog_rows": 112,
    "smonthly_rows": 56,
    "static_rows": 56,
    "used_tile_track_groups": 40,
    "used_mgrs_tiles": 31,
    "polygon_track_trajectories": 4001,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_string(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def track_token(track_key: str) -> str:
    match = re.search(r"_TK(.+)$", track_key)
    if not match:
        raise ValueError(f"Cannot parse track token from {track_key!r}")
    return match.group(1)


def summarize_family(group: pd.DataFrame) -> pd.Series:
    starts = pd.to_datetime(group["start_datetime"])
    ends = pd.to_datetime(group["end_datetime"])
    tokens = sorted(group["store_track_token"].unique(), key=lambda x: (len(x), x))
    return pd.Series(
        {
            "store_track_tokens": ", ".join(tokens),
            "tile_track_groups": int(len(group)),
            "mgrs_tiles": int(group["tile_id_catalog"].nunique()),
            "burst_footprints_min": int(group["n_bursts"].min()),
            "burst_footprints_max": int(group["n_bursts"].max()),
            "stored_monthly_composites_min": int(group["n_scenes"].min()),
            "stored_monthly_composites_max": int(group["n_scenes"].max()),
            "first_month_min": month_string(starts.min()),
            "first_month_max": month_string(starts.max()),
            "last_month_min": month_string(ends.min()),
            "last_month_max": month_string(ends.max()),
            "polygon_track_trajectories": int(group["polygon_tracks"].sum()),
        }
    )


def build_inventory(catalog_path: Path, cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    catalog = pd.read_parquet(catalog_path)
    cache = pd.read_parquet(cache_path)
    monthly = catalog.loc[catalog["product_type"].eq("smonthly")].copy()
    static = catalog.loc[catalog["product_type"].eq("static")].copy()

    checks = {
        "catalog_rows": len(catalog),
        "smonthly_rows": len(monthly),
        "static_rows": len(static),
        "catalog_all_complete": bool(catalog["status"].eq("complete").all()),
        "monthly_all_ascending": bool(monthly["flight_direction"].eq("ASCENDING").all()),
        "used_tile_track_groups": len(cache),
        "used_mgrs_tiles": int(cache["tile_id"].nunique()),
        "polygon_track_trajectories": int(cache["polygon_tracks"].sum()),
        "target_months": sorted(cache["target_months"].astype(int).unique().tolist()),
        "calendars": sorted(cache["calendar"].astype(str).unique().tolist()),
    }

    joined = cache.merge(
        monthly,
        left_on="track_key",
        right_on="geometry_group_id",
        how="left",
        validate="one_to_one",
        suffixes=("_cache", "_catalog"),
        indicator=True,
    )
    checks["unmatched_used_groups"] = int(joined["_merge"].ne("both").sum())
    checks["source_month_count_matches_catalog"] = bool(
        joined["source_months"].astype(int).eq(joined["n_scenes"].astype(int)).all()
    )
    checks["source_start_matches_catalog"] = bool(
        joined["source_start"].astype(str).eq(pd.to_datetime(joined["start_datetime"]).dt.strftime("%Y-%m")).all()
    )
    checks["source_end_matches_catalog"] = bool(
        joined["source_end"].astype(str).eq(pd.to_datetime(joined["end_datetime"]).dt.strftime("%Y-%m")).all()
    )

    expected_checks = {name: checks[name] == value for name, value in EXPECTED.items()}
    checks["expected_counts"] = EXPECTED
    checks["expected_count_checks"] = expected_checks
    checks["passed"] = bool(
        all(expected_checks.values())
        and checks["catalog_all_complete"]
        and checks["monthly_all_ascending"]
        and checks["unmatched_used_groups"] == 0
        and checks["source_month_count_matches_catalog"]
        and checks["source_start_matches_catalog"]
        and checks["source_end_matches_catalog"]
        and checks["target_months"] == [126]
        and checks["calendars"] == ["2016-01/2026-06"]
    )
    if not checks["passed"]:
        raise RuntimeError("Sentinel-1 observation inventory validation failed; inspect QA output")

    joined["store_track_token"] = joined["track_key"].map(track_token)
    joined["primary_track_family"] = joined["store_track_token"].str.split("-").str[0].astype(int)

    inventory_columns = [
        "track_key",
        "tile_id_catalog",
        "flight_direction",
        "primary_track_family",
        "store_track_token",
        "n_bursts",
        "n_scenes",
        "start_datetime",
        "end_datetime",
        "polygon_tracks",
        "ground_polygons",
        "native_pixels",
        "valid_fraction",
        "static_valid_fraction",
    ]
    inventory = joined[inventory_columns].rename(
        columns={
            "tile_id_catalog": "mgrs_tile",
            "n_bursts": "burst_footprints",
            "n_scenes": "stored_monthly_composites",
        }
    )
    inventory["start_datetime"] = pd.to_datetime(inventory["start_datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    inventory["end_datetime"] = pd.to_datetime(inventory["end_datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    inventory = inventory.sort_values(["primary_track_family", "mgrs_tile", "track_key"]).reset_index(drop=True)

    summary_rows = []
    for family, group in joined.groupby("primary_track_family", sort=True):
        row = summarize_family(group).to_dict()
        row["primary_track_family"] = family
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary = summary[["primary_track_family", *[c for c in summary.columns if c != "primary_track_family"]]]
    integer_columns = [
        "primary_track_family",
        "tile_track_groups",
        "mgrs_tiles",
        "burst_footprints_min",
        "burst_footprints_max",
        "stored_monthly_composites_min",
        "stored_monthly_composites_max",
        "polygon_track_trajectories",
    ]
    summary[integer_columns] = summary[integer_columns].astype(int)
    return inventory, summary, checks


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=repo / "data/local/paper_v1/s1grits/catalog.parquet")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=repo / "data/local/paper_v1/polygon_pixels/cache_manifest.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=repo / "results/data_inventory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory, summary, checks = build_inventory(args.catalog, args.cache_manifest)
    inventory_path = args.output_dir / "s1_tile_track_inventory.csv"
    summary_path = args.output_dir / "s1_orbit_family_summary.csv"
    qa_path = args.output_dir / "s1_observation_inventory_qa.json"
    complete_path = args.output_dir / "COMPLETE.json"

    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_path, index=False)
    checks.update(
        {
            "catalog": str(args.catalog.resolve()),
            "cache_manifest": str(args.cache_manifest.resolve()),
            "catalog_sha256": sha256(args.catalog),
            "cache_manifest_sha256": sha256(args.cache_manifest),
        }
    )
    qa_path.write_text(json.dumps(checks, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    complete_path.write_text(
        json.dumps(
            {
                "passed": True,
                "outputs": {
                    inventory_path.name: sha256(inventory_path),
                    summary_path.name: sha256(summary_path),
                    qa_path.name: sha256(qa_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
