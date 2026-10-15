#!/usr/bin/env python3
"""Render the production representation/configuration figure."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "data/public/figure_inputs/fig06_representation_configuration"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.workflows import build_figure5_abcde_metric_encoding_v19 as formal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "runs/figure_sources/figure06_representation_configuration",
    )
    args = parser.parse_args()
    result = formal.main([
        "--trajectories",
        str(args.inputs / "aggregation_trajectories_exact.csv"),
        "--representation-metrics",
        str(args.inputs / "representation_metrics_figure_compat.csv"),
        "--n1-input",
        str(args.inputs / "n1"),
        "--window-input",
        str(args.inputs / "window"),
        "--output-dir",
        str(args.build_dir),
    ])
    if result != 0:
        raise SystemExit(result)
    source = args.build_dir / "figure_05_representation_and_sensitivity_abcde.pdf"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)
    for suffix in (".svg", ".png", ".tiff"):
        companion = source.with_suffix(suffix)
        if companion.exists():
            shutil.copy2(companion, args.output.with_suffix(suffix))


if __name__ == "__main__":
    main()
