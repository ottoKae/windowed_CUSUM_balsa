#!/usr/bin/env python3
"""Render the production population-signature figure from frozen local inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "analysis/workflows/build_population_s1_signature_figure_v11.py"
INPUTS = ROOT / "data/local/paper_v1/figure_inputs/fig04_population_signatures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "runs/figure_sources/figure04_population_signatures",
    )
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(FORMAL),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.build_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    source = args.build_dir / "figure_03_population_s1_signatures_v11.pdf"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)
    for suffix in (".svg", ".png", ".tiff"):
        companion = source.with_suffix(suffix)
        if companion.exists():
            shutil.copy2(companion, args.output.with_suffix(suffix))


if __name__ == "__main__":
    main()
