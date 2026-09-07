#!/usr/bin/env python3
"""Build allowlisted public or restricted data ZIPs with embedded checksums."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import sys


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else []


def public_files() -> list[Path]:
    files = []
    for relative in (
        "data/demo", "data/schemas", "data/public",
    ):
        files.extend(files_under(ROOT / relative))
    for line in (ROOT / "data/manifests/PUBLIC_AGGREGATE_FILES.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            path = (ROOT / line).resolve()
            if not path.is_relative_to(ROOT / "results") or not path.is_file():
                raise ValueError(f"invalid aggregate allowlist entry: {line}")
            files.append(path)
    for relative in (
        "data/README.md", "data/DATA_DICTIONARY.md", "configs/paper_v1.json",
        "provenance/DATA_SOURCES.md", "provenance/PUBLIC_RELEASE_POLICY.md",
        "provenance/ID_AND_PRIVACY_POLICY.md",
    ):
        path = ROOT / relative
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def restricted_files() -> list[Path]:
    local = ROOT / "data/local/paper_v1"
    run = ROOT / "runs/windowed_cusum_balsa_paper_v1"
    prospective = ROOT / "runs/prospective_2026_n1"
    files = []
    for relative in (
        "inventory", "splits", "references", "polygon_pixels", "frozen_protocol",
    ):
        files.extend(files_under(local / relative))
    files.extend(files_under(local / "s1grits"))
    files.extend(files_under(run))
    files.extend(files_under(prospective))
    blocked_parts = {"basemap", "reviewer", "planet", "source.local.json"}
    blocked_names = {
        "prior_blinded_review_unblinded.csv",
        "review_overlap_audit.csv",
    }
    return sorted(
        set(
            path for path in files
            if path.name.lower() not in blocked_names
            and not any(part.lower() in blocked_parts for part in path.parts)
        )
    )


def archive_name(profile: str) -> str:
    return f"windowed_cusum_balsa_{profile}_paper_v1.zip"


def archive_path(path: Path, profile: str) -> str:
    if profile == "public":
        return path.relative_to(ROOT).as_posix()
    local = ROOT / "data/local/paper_v1"
    run = ROOT / "runs/windowed_cusum_balsa_paper_v1"
    prospective = ROOT / "runs/prospective_2026_n1"
    if path.is_relative_to(local):
        return "restricted/" + path.relative_to(local).as_posix()
    if path.is_relative_to(run):
        return "restricted/raw_to_result/" + path.relative_to(run).as_posix()
    return "restricted/prospective_2026/" + path.relative_to(prospective).as_posix()


def manifest_csv(records: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("public", "restricted", "real"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--acknowledge-sensitive-data", action="store_true")
    args = parser.parse_args()
    if args.profile == "real":
        sys.path.insert(0, str(ROOT / "src"))
        from lr_cusum.capsule import build, VERSION
        print(json.dumps(build(ROOT, args.destination or DIST / VERSION), indent=2))
        return
    if args.profile == "restricted" and not args.acknowledge_sensitive_data:
        raise SystemExit("restricted build requires --acknowledge-sensitive-data")

    files = public_files() if args.profile == "public" else restricted_files()
    if not files:
        raise SystemExit(f"no files found for {args.profile} release")
    DIST.mkdir(parents=True, exist_ok=True)
    output = DIST / archive_name(args.profile)
    records = [
        {
            "path": archive_path(path, args.profile),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    metadata = {
        "profile": args.profile,
        "dataset_version": "paper_v1",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "rights_boundary": (
            "Aggregate and synthetic files only" if args.profile == "public" else
            "Controlled exact-location capsule; do not redistribute without approval"
        ),
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, record in zip(files, records, strict=True):
            archive.write(path, record["path"])
        archive.writestr("MANIFEST.csv", manifest_csv(records))
        archive.writestr("RELEASE.json", json.dumps(metadata, indent=2))
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(json.dumps({**metadata, "archive": str(output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
