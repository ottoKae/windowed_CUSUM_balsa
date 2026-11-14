"""Historical detectors; use scripts/run_current_pipeline.py for current first-post/TDA/STP."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .detection import DetectionConfig, detect_table
from .paper import PaperDetectionConfig, detect_table_n1, load_paper_config


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Monthly long table (CSV or Parquet).")
    parser.add_argument("output", type=Path, help="Candidate-event table to create.")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--time-column", default="time")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="protocol", required=True)

    paper = modes.add_parser(
        "paper",
        help="Run the historical N1 detector, not the current first-post/TDA/STP pipeline.",
    )
    _common(paper)
    paper.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit historical protocol JSON (for example configs/standard_w36.json).",
    )

    legacy = modes.add_parser(
        "legacy",
        help="Run the archived empirical-confidence LR-CuSum implementation.",
    )
    _common(legacy)
    legacy.add_argument("--bands", nargs="+", default=["VV_dB", "VH_dB"])
    legacy.add_argument("--window-length", type=int, default=36)
    legacy.add_argument("--step", type=int, default=12)
    legacy.add_argument("--confidence", type=float, default=0.95)
    legacy.add_argument("--bootstrap", type=int, default=800)
    legacy.add_argument("--seed", type=int, default=20)
    legacy.add_argument("--start-year", type=int, default=2017)
    legacy.add_argument("--end-year", type=int, default=2025)
    legacy.add_argument("--minimum-observations", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    frame = read_table(args.input)
    if args.protocol == "paper":
        config = load_paper_config(args.config) if args.config else PaperDetectionConfig()
        result = detect_table_n1(
            frame,
            id_column=args.id_column,
            time_column=args.time_column,
            config=config,
        )
    else:
        config = DetectionConfig(
            bands=tuple(args.bands),
            window_length=args.window_length,
            step=args.step,
            confidence_threshold=args.confidence,
            bootstrap_iterations=args.bootstrap,
            seed=args.seed,
            start_year=args.start_year,
            end_year=args.end_year,
            minimum_observations=args.minimum_observations,
        )
        result = detect_table(
            frame,
            id_column=args.id_column,
            time_column=args.time_column,
            config=config,
        )
    write_table(result, args.output)
    print(f"protocol={args.protocol} candidates={len(result)} output={args.output}")


if __name__ == "__main__":
    main()
