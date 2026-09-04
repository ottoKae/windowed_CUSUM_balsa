"""Audit the complete public package for integrity and disclosure risks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance" / "PRIVACY_AUDIT.json"
SCAN_ROOTS = (ROOT,)
EXCLUDED_DIRS = {
    ".git", ".pytest_cache", ".pytest_tmp", ".pytest_tmp_root", ".qa",
    ".ruff_cache", ".mypy_cache", "__pycache__", "build", "dist",
}
FORBIDDEN_SUFFIXES = {
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".gpkg", ".geojson",
    ".tif", ".tiff", ".zarr", ".xlsx", ".xls", ".wkb", ".wkt",
}
SENSITIVE_COLUMNS = {
    "polygon_id", "sample_id", "point_track_id", "objectid", "source_row",
    "longitude", "latitude", "geometry", "wkt", "wkb", "historical_id",
    "historical_point_id", "reference_polygon_id", "reviewer", "annotator",
    "evidence_file", "evidence_filename", "spatial_block_id", "hex_id",
    "tile_id", "track_key", "raster_row", "raster_col",
}
PRIVATE_PATTERNS = {
    "absolute_windows_path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])"),
    "workstation_username": re.compile(r"raokeyi", re.IGNORECASE),
    "person_name_token": re.compile(r"dongshixin", re.IGNORECASE),
    "private_inventory_id": re.compile(r"balsa:mag|mag_fc001", re.IGNORECASE),
    "private_hex_id": re.compile(r"r-\d{4,}_c-\d{4,}", re.IGNORECASE),
}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SECRET_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return path.is_file() and not any(
        part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> dict[str, object]:
    files = sorted(path for base in SCAN_ROOTS for path in base.rglob("*") if included(path))
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]

    sensitive_headers: dict[str, list[str]] = {}
    csv_parse_errors: dict[str, str] = {}
    for path in files:
        if path.suffix.lower() != ".csv":
            continue
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                header = next(csv.reader(stream), [])
        except Exception as exc:  # pragma: no cover - diagnostic path
            csv_parse_errors[path.relative_to(ROOT).as_posix()] = str(exc)
            continue
        found = sorted({column.strip().lower() for column in header} & SENSITIVE_COLUMNS)
        rel = path.relative_to(ROOT).as_posix()
        if found and not rel.startswith("data/demo/"):
            sensitive_headers[rel] = found

    pattern_hits: dict[str, list[str]] = {}
    text_suffixes = {
        ".csv", ".json", ".md", ".py", ".tex", ".toml", ".yml", ".yaml", ".ps1"
    }
    cjk_hits: list[str] = []
    secret_hits: dict[str, list[str]] = {}
    for path in files:
        # The audit source necessarily contains the prohibited search tokens.
        if path in {OUTPUT, Path(__file__).resolve()} or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        hits = [name for name, pattern in PRIVATE_PATTERNS.items() if pattern.search(text)]
        if hits:
            pattern_hits[rel] = hits
        if CJK.search(text):
            cjk_hits.append(rel)
        secrets = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
        if secrets:
            secret_hits[rel] = secrets

    annual = read_csv(ROOT / "results/national/annual_polygon_first_vv_events_2017_2025.csv")
    distribution = read_csv(ROOT / "results/national/polygon_event_count_distribution.csv")
    intervals = read_csv(ROOT / "results/national/adjacent_interval_distribution_6month_bins.csv")
    cohort = read_csv(ROOT / "results/polarization/cohort_counts_vh_pm2.csv")[0]
    oof = read_csv(ROOT / "results/formal_validation/nested_spatial_oof_metrics.csv")
    origins = read_csv(ROOT / "provenance/PUBLIC_FILE_ORIGINS.csv")
    ledger = json.loads((ROOT / "results/count_ledger.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "results/frozen_configuration.json").read_text(encoding="utf-8"))
    paper = json.loads((ROOT / "configs/paper_v1.json").read_text(encoding="utf-8"))

    origin_copy_checks: dict[str, bool] = {}
    for row in origins:
        public_path = ROOT / row["public_path"]
        origin_copy_checks[row["public_path"]] = (
            public_path.is_file()
            and public_path.stat().st_size == int(row["bytes"])
            and sha256(public_path) == row["source_sha256"]
        )

    demo_rows = read_csv(ROOT / "data/demo/monthly_s1_synthetic.csv")
    demo_ids = {row["ID"] for row in demo_rows}

    annual_sum = sum(int(row["annual_event_count"]) for row in annual)
    polygon_sum = sum(int(row["polygon_count"]) for row in distribution)
    event_sum_from_distribution = sum(
        int(row["event_count"]) * int(row["polygon_count"]) for row in distribution
    )
    interval_sum = sum(int(row["interval_count"]) for row in intervals)
    primary = next(
        row for row in oof
        if row["family"] == "lr_cusum" and row["protocol"] == "nested_5fold_spatial_oof"
    )
    route_counts = ledger["national_route_accounting"]
    paper_detector = paper["detector"]
    ledger_detector = ledger["paper_n1_operational_detector"]

    count_checks = {
        "annual_sum_is_6658": annual_sum == 6658,
        "polygon_distribution_sum_is_3498": polygon_sum == 3498,
        "weighted_polygon_event_sum_is_6658": event_sum_from_distribution == 6658,
        "interval_histogram_sum_is_3509": interval_sum == 3509,
        "polygon_first_plus_pixel_first_is_12899": (
            route_counts["polygon_first_vv_events"]["count"]
            + route_counts["pixel_first_vv_events"]["count"]
            == route_counts["route_concatenated_vv_rows"]["count"]
            == 12899
        ),
        "formal_primary_confusion_counts_match": (
            int(primary["tp"]) == 880
            and int(primary["fp"]) == 238
            and int(primary["fn"]) == 213
        ),
        "vh_supported_count_is_4061": int(cohort["vv_with_vh_support"]) == 4061,
        "vh_supported_interval_count_is_1385": int(cohort["supported_intervals"]) == 1385,
        "all_aggregate_copies_match_frozen_source_hashes": all(origin_copy_checks.values()),
        "demo_contains_only_synthetic_ids": bool(demo_ids) and all(
            item.startswith("DEMO_STAND_") for item in demo_ids
        ),
        "public_frozen_configuration_matches_paper_v1": (
            frozen["window_length_months"] == paper_detector["window_length"] == 48
            and frozen["window_step_months"] == paper_detector["step"] == 12
            and frozen["permutation_p_max"] == paper_detector["alpha"] == 0.025
            and frozen["initial_permutations"] == paper_detector["permutations"] == 3999
            and frozen["near_threshold_permutations"]
            == paper_detector["refined_permutations"]
            == 19999
            and ledger_detector["initial_permutations"] == 3999
            and ledger_detector["near_threshold_permutations"] == 19999
        ),
    }

    manuscript = ROOT / "manuscript"
    main_text = (manuscript / "main.tex").read_text(encoding="utf-8")
    section_matches = re.findall(r"\\input\{([^}]+)\}", main_text)
    figure_matches: list[str] = []
    for section in section_matches:
        section_path = manuscript / f"{section}.tex"
        if section_path.is_file():
            figure_matches.extend(
                re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", section_path.read_text(encoding="utf-8"))
            )
    manuscript_checks = {
        "all_input_sections_exist": all((manuscript / f"{item}.tex").is_file() for item in section_matches),
        "active_manuscript_references_eight_figures": len(figure_matches) == 8,
        "all_active_figure_assets_exist": all((manuscript / item).is_file() for item in figure_matches),
        "compiled_manuscript_pdf_exists": (manuscript / "main.pdf").is_file(),
    }

    json_errors: dict[str, str] = {}
    for path in files:
        if path.suffix.lower() == ".json" and path != OUTPUT:
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:  # pragma: no cover - diagnostic path
                json_errors[path.relative_to(ROOT).as_posix()] = str(exc)

    passed = not any(
        (
            forbidden_files,
            sensitive_headers,
            csv_parse_errors,
            pattern_hits,
            cjk_hits,
            secret_hits,
            json_errors,
        )
    ) and all(count_checks.values()) and all(manuscript_checks.values())
    return {
        "schema_version": "windowed-cusum-balsa.public-package-audit.v2",
        "scope": ["complete repository release, excluding generated build and QA files"],
        "file_count_scanned": len(files),
        "checks": count_checks,
        "aggregate_copy_hash_checks": origin_copy_checks,
        "forbidden_file_types": forbidden_files,
        "sensitive_tabular_headers": sensitive_headers,
        "csv_parse_errors": csv_parse_errors,
        "private_pattern_hits": pattern_hits,
        "cjk_text_hits": cjk_hits,
        "high_confidence_secret_hits": secret_hits,
        "json_parse_errors": json_errors,
        "manuscript_checks": manuscript_checks,
        "note": "Synthetic DEMO_STAND identifiers are invented and contain no source mapping.",
        "technical_passed": passed,
        "publication_gates": {
            "software_license_selected": (ROOT / "LICENSE").is_file(),
            "citation_metadata_finalized": (ROOT / "CITATION.cff").is_file(),
            "repository_url_and_archive_doi_finalized": False,
            "third_party_figure_permissions_confirmed": False,
        },
        "passed": passed,
    }


if __name__ == "__main__":
    report = audit()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
