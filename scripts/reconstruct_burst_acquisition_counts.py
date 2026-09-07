"""Count ASF RTC availability for LUT bursts in the frozen tile/track cohort.

Counts are acquisition availability, not proof of historical input admission.
Only metadata and Zarr time coordinates are read; no imagery is downloaded.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import asf_search as asf
import pandas as pd
import zarr


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True, help="MGRS-to-burst lookup Parquet")
    parser.add_argument("--archive", type=Path, required=True, help="S1-GRiTS archive root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "asf_metadata"
    cache_dir.mkdir(exist_ok=True)
    manifest = repo / "data/local/paper_v1/polygon_pixels/cache_manifest.parquet"
    spec = {"lookup_sha256": digest(args.lookup), "manifest_sha256": digest(manifest),
            "catalog_sha256": digest(args.archive / "catalog.parquet"),
            "start": "2016-01-01T00:00:00Z", "end": "2026-06-30T23:59:59Z",
            "processing_level": "RTC", "polarization": "VV+VH"}
    spec_path = args.output / "query_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Query/source signature changed; use a new output directory")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    lut = pd.read_parquet(args.lookup)
    used = pd.read_parquet(manifest)
    catalog = pd.read_parquet(args.archive / "catalog.parquet")
    catalog = catalog[catalog.product_type.eq("smonthly")].set_index("geometry_group_id")
    mappings, months = [], []
    for record in used.itertuples():
        tile, direction, token = record.track_key.split("_")
        tracks = [int(t) for t in token[2:].split("-")]
        subset = lut[lut.mgrs_tile_id.eq(tile) & lut.orbit_pass.eq(direction) & lut.track_number.isin(tracks)].copy()
        if subset.empty:
            raise ValueError(f"No LUT overlap for {record.track_key}")
        subset["track_key"], subset["family"] = record.track_key, tracks[0]
        mappings.append(subset)
        source = args.archive / tile / catalog.loc[record.track_key, "zarr_path"]
        time = zarr.open_group(str(source), mode="r")["time"][:]
        for month in pd.to_datetime(time).strftime("%Y-%m").unique():
            months.append({"track_key": record.track_key, "month": month})
    mapping = pd.concat(mappings, ignore_index=True).drop_duplicates(["track_key", "jpl_burst_id"])
    mapping.to_csv(args.output / "tile_track_burst_mapping.csv", index=False)
    observed = pd.DataFrame(months).drop_duplicates()
    observed.to_csv(args.output / "archive_group_months.csv", index=False)
    bursts = sorted(mapping.jpl_burst_id.unique())
    print(f"Querying {len(bursts)} unique bursts for {len(used)} used groups", flush=True)

    def query(burst):
        path = cache_dir / f"{burst}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["burst_id"] != burst:
                raise ValueError(f"Invalid cache identity: {path}")
            return saved["products"]
        result = asf.geo_search(operaBurstID=[burst.replace("-", "_")],
                                processingLevel="RTC", start=spec["start"], end=spec["end"])
        products = []
        for item in result:
            props = item.properties
            scene = props["sceneName"]
            if scene.split("_")[3] != burst:
                raise ValueError(f"Unexpected burst returned: {scene}")
            pol = props["polarization"]
            pol = pol if isinstance(pol, list) else str(pol).replace("+", ",").split(",")
            products.append({"jpl_burst_id": burst, "opera_id": scene,
                             "acquisition_time": props["startTime"],
                             "actual_track": props["pathNumber"],
                             "dual_vv_vh": set(p.strip() for p in pol) == {"VV", "VH"}})
        payload = {"burst_id": burst, "queried_at": datetime.now(timezone.utc).isoformat(), "products": products}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(path)
        return products

    all_products = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(query, burst): burst for burst in bursts}
        for n, future in enumerate(as_completed(futures), 1):
            all_products.extend(future.result())
            if n % 10 == 0 or n == len(bursts):
                print(f"Metadata {n}/{len(bursts)} bursts; {len(all_products)} product records", flush=True)
    products = pd.DataFrame(all_products)
    products.to_parquet(args.output / "asf_products.parquet", index=False)
    acquisitions = products[products.dual_vv_vh].copy()
    acquisitions["acquisition_time"] = pd.to_datetime(acquisitions.acquisition_time, utc=True)
    acquisitions = acquisitions.drop_duplicates(["jpl_burst_id", "acquisition_time"])
    acquisitions["month"] = acquisitions.acquisition_time.dt.strftime("%Y-%m")
    acquisitions.to_parquet(args.output / "unique_available_acquisitions.parquet", index=False)
    expanded = acquisitions.merge(mapping[["jpl_burst_id", "track_key", "family", "mgrs_tile_id"]], on="jpl_burst_id", validate="many_to_many")
    supported = expanded.merge(observed, on=["track_key", "month"], how="inner", validate="many_to_one")
    supported.to_parquet(args.output / "available_in_stored_group_months.parquet", index=False)
    summaries = []
    for family, group in expanded.groupby("family"):
        supported_group = supported[supported.family.eq(family)]
        count = lambda d: len(d.drop_duplicates(["jpl_burst_id", "acquisition_time"]))
        summaries.append({"family": int(family), "mgrs_tiles": int(group.mgrs_tile_id.nunique()),
                          "available_201601_202606": count(group),
                          "available_201701_202512": count(group[group.month.between("2017-01", "2025-12")]),
                          "available_in_stored_months_201601_202606": count(supported_group),
                          "available_in_stored_months_201701_202512": count(supported_group[supported_group.month.between("2017-01", "2025-12")])})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "orbit_family_counts.csv", index=False)
    supported.drop_duplicates(["family", "month", "jpl_burst_id", "acquisition_time"]).groupby(["family", "month"]).size().rename("available_acquisitions").to_csv(args.output / "orbit_month_counts.csv")
    national_supported = supported.drop_duplicates(["jpl_burst_id", "acquisition_time"])
    result = {"completed_at": datetime.now(timezone.utc).isoformat(),
              "query_completed": True, "all_bursts_queried": len(bursts),
              "used_groups": len(used), "lut_memberships": len(mapping),
              "products_returned": len(products), "unique_available_201601_202606": len(acquisitions),
              "unique_available_201701_202512": int(acquisitions.month.between("2017-01", "2025-12").sum()),
              "unique_available_in_stored_months_201601_202606": len(national_supported),
              "unique_available_in_stored_months_201701_202512": int(national_supported.month.between("2017-01", "2025-12").sum()),
              "historical_input_admission_verified": False,
              "meaning": "ASF dual-polarization RTC availability intersected with LUT and (where stated) stored group months; not historical successful-input count"}
    (args.output / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "COMPLETE.json")
    (args.output / "COMPLETE.json").write_text(json.dumps({"passed": True,
        "hashes": {str(p.relative_to(args.output)): digest(p) for p in files}}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
