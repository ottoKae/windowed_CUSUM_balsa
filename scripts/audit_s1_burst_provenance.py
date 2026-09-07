"""Read-only audit of retained burst provenance; logs are not usage manifests."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import pandas as pd


PRODUCT = re.compile(r"OPERA_L2_RTC-S1_(T\d{3}-\d{6}-IW[123])_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_(S1[ABC])_30_v[\d.]+")
FIELDS = ("opera_ids", "jpl_burst_ids", "opera_id", "acq_dt")


def present(value):
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist") and getattr(value, "ndim", 0) > 0:
        return len(value) > 0
    if isinstance(value, str):
        return value.strip() not in ("", "null", "[]", "None")
    return bool(pd.notna(value))


def metadata_files(root):
    """Stop at Zarr roots: no numerical chunks or raster arrays are traversed."""
    for directory, children, files in os.walk(root):
        if "zarr.json" in files or ".zgroup" in files:
            yield Path(directory) / ("zarr.json" if "zarr.json" in files else ".zattrs")
            children[:] = []
            continue
        for name in files:
            if name.endswith((".parquet", ".json")) or "catalog" in name:
                yield Path(directory) / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="S1-GRiTS archive root")
    parser.add_argument("--logs", type=Path, required=True, help="S1-GRiTS processing-log directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    roots = [args.archive] + [args.archive.parent / name for name in (
        "S1-GRiTS_Ecuador_40tiles_201601_202606_v3_qc_backups_20260822",
        "S1-GRiTS_Ecuador_merge_backups_20260817_101646",
        "S1-GRiTS_Ecuador_time_order_backups_20260817",
        "S1-GRiTS_Ecuador_remote_stage_20260816",
        "_cache",
    )]
    records = []
    for root in roots:
        for path in metadata_files(root):
            record = {"path": str(path), "scope": "current" if root == args.archive else "backup_or_cache"}
            try:
                if ".parquet" in path.name:
                    data = pd.read_parquet(path)
                    record.update(kind="parquet", rows=len(data), columns="|".join(data.columns))
                    for field in FIELDS:
                        record[field + "_nonempty_rows"] = int(data[field].map(present).sum()) if field in data else 0
                elif path.suffix == ".json" or path.name == ".zattrs":
                    data = json.loads(path.read_text(encoding="utf-8-sig"))
                    attrs = data.get("attributes", data)
                    record.update(kind="zarr_group" if path.name in ("zarr.json", ".zattrs") else "json",
                                  attribute_keys="|".join(sorted(attrs)),
                                  identity_fields="|".join(k for k in attrs if any(t in k.lower() for t in ("opera_id", "jpl_burst", "acquisit", "source"))))
                    if record["kind"] == "zarr_group":
                        record["array_names"] = "|".join(sorted(p.name for p in path.parent.iterdir() if p.is_dir()))
                else:
                    record["kind"] = "other_catalog_file"
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
        print(f"Scanned metadata: {root.name}", flush=True)
    pd.DataFrame(records).to_csv(args.output / "metadata_inventory.csv", index=False)

    # Retain identifiers only, never signed URLs, credentials or raw log lines.
    log_rows, identifiers = [], set()
    for path in sorted(args.logs.rglob("*.log")):
        ids, failure_ids = set(), set()
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                hits = {(m[0], m[1]) for m in PRODUCT.findall(line)}
                ids.update(hits)
                if any(word in line.lower() for word in ("failed", "retry", "error", "warning")):
                    failure_ids.update(hits)
        identifiers.update(ids)
        log_rows.append({"path": str(path), "bytes": path.stat().st_size,
                         "unique_burst_time_mentions": len(ids),
                         "burst_time_mentions_on_warning_or_failure_lines": len(failure_ids)})
    pd.DataFrame(log_rows).to_csv(args.output / "log_inventory.csv", index=False)
    pd.DataFrame(sorted(identifiers), columns=["burst_id", "acquisition_time"]).to_csv(
        args.output / "log_mentions_NOT_verified_inputs.csv", index=False)

    current = [r for r in records if r["scope"] == "current"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(args.archive),
        "roots_scanned": [str(p) for p in roots],
        "files_by_kind": dict(Counter(r.get("kind", "error") for r in records)),
        "current_files_by_kind": dict(Counter(r.get("kind", "error") for r in current)),
        "metadata_read_errors": [r for r in records if "error" in r],
        "parquet_rows_with_opera_ids": sum(r.get("opera_ids_nonempty_rows", 0) for r in records),
        "parquet_rows_with_jpl_burst_ids": sum(r.get("jpl_burst_ids_nonempty_rows", 0) for r in records),
        "log_files_scanned": len(log_rows),
        "distinct_burst_time_mentions_in_logs": len(identifiers),
        "log_mentions_are_verified_used_inputs": False,
        "formal_burst_acquisition_count": None,
        "count_status": "not_recoverable_from_a_complete_retained_input_manifest_in_scanned_sources",
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    files = sorted(args.output.iterdir())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output / "COMPLETE.json").write_text(json.dumps({
        "audit_completed": True, "formal_count_available": False, "outputs_sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
