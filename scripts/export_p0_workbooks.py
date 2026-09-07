#!/usr/bin/env python3
"""Export one public results workbook and one private protocol workbook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.DatetimeTZDtype):
            result[column] = result[column].astype(str)
        elif result[column].dtype == "object":
            result[column] = result[column].map(
                lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
            )
    return result


def unique_sheet(base: str, used: set[str]) -> str:
    clean = "".join(character if character.isalnum() else "_" for character in base).strip("_")
    clean = clean[:31] or "sheet"
    candidate = clean
    index = 2
    while candidate.lower() in used:
        suffix = f"_{index}"
        candidate = clean[: 31 - len(suffix)] + suffix
        index += 1
    used.add(candidate.lower())
    return candidate


def write_workbook(path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            safe = excel_safe(frame)
            sheet = unique_sheet(name, used)
            safe.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.book[sheet]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.iter_cols(min_row=1, max_row=min(200, worksheet.max_row)):
                width = max(len(str(cell.value or "")) for cell in cells)
                worksheet.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 10), 48)


def public_sheets() -> list[tuple[str, pd.DataFrame]]:
    sheets = []
    for path in sorted((ROOT / "results").rglob("*.csv")):
        name = path.relative_to(ROOT / "results").with_suffix("").as_posix().replace("/", "_")
        sheets.append((name, pd.read_csv(path)))
    ledger = json.loads((ROOT / "results/count_ledger.json").read_text(encoding="utf-8"))
    sheets.insert(0, ("count_ledger", pd.json_normalize(ledger)))
    return sheets


def private_sheets(data_root: Path, raw_run: Path, prospective: Path) -> list[tuple[str, pd.DataFrame]]:
    sheets = [
        ("outer_fold_map", pd.read_csv(data_root / "splits/outer_folds.csv")),
        ("oof_source_ledger", pd.read_csv(data_root / "frozen_protocol/oof_source_ledger.csv")),
    ]
    selections = data_root / "frozen_protocol/selections"
    predictions = data_root / "frozen_protocol/oof_predictions"
    for path in sorted(selections.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "method", path.stem)
        sheets.append((f"selection_{path.stem}", frame))
    combined = []
    for path in sorted(predictions.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame.insert(0, "method", path.stem)
        combined.append(frame)
    if combined:
        sheets.append(("all_oof_predictions", pd.concat(combined, ignore_index=True, sort=False)))
    for name, path in (
        ("cache_equivalence", raw_run / "cache_equivalence/track_cache_equivalence.csv"),
        ("cache_differences", raw_run / "cache_equivalence/difference_detail.csv"),
        ("prospective_review_summary", prospective / "review_overlap_summary.csv"),
    ):
        if path.is_file():
            sheets.append((name, pd.read_csv(path)))
    for name, path in (
        ("component_stage_metrics", raw_run / "component_ablation/stage_metrics.parquet"),
        ("component_fold_metrics", raw_run / "component_ablation/fold_metrics.parquet"),
        ("component_ci", raw_run / "component_ablation/bootstrap_contrast_summary.parquet"),
        ("prospective_histories", prospective / "candidate_histories.parquet"),
    ):
        if path.is_file():
            sheets.append((name, pd.read_parquet(path)))
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/local/paper_v1")
    parser.add_argument("--raw-run", type=Path, default=ROOT / "runs/windowed_cusum_balsa_paper_v1")
    parser.add_argument("--prospective", type=Path, default=ROOT / "runs/prospective_2026_n1")
    args = parser.parse_args()

    public = ROOT / "results/release/windowed_cusum_balsa_paper_results_v1.xlsx"
    private = args.data_root / "exports/windowed_cusum_balsa_private_protocol_v1.xlsx"
    write_workbook(public, public_sheets())
    write_workbook(private, private_sheets(args.data_root, args.raw_run, args.prospective))
    receipt = {
        "status": "complete",
        "public_workbook": str(public.relative_to(ROOT)),
        "public_sha256": sha256(public),
        "private_workbook": private.relative_to(ROOT).as_posix(),
        "private_sha256": sha256(private),
        "note": "The public workbook contains aggregates only; the private workbook contains exact fold and OOF records.",
    }
    receipt_path = ROOT / "results/release/WORKBOOK_EXPORT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
