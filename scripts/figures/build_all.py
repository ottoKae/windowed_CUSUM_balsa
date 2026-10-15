#!/usr/bin/env python3
"""Build diagnostic JAG-v2 figures in manuscript reading order.

Only Figure 8 currently has a pixel-verified production wrapper. See the
figure reproducibility audit before treating other outputs as publication
assets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BUILDERS = {
    # The output filenames are retained because the current TeX sources refer
    # to them.  The integer keys now follow the actual compiled figure order.
    1: ("build_workflow.py", "fig03_workflow.pdf"),
    2: ("build_study_area.py", "fig02_study_area.pdf"),
    3: ("build_windowed_cusum_concept.py", "fig04_windowed_cusum_concept.pdf"),
    4: ("build_population_s1_signatures.py", "fig05_population_s1_signatures.pdf"),
    5: ("build_representative_timeseries.py", "fig06_representative_timeseries.pdf"),
    6: ("build_representation_configuration.py", "fig07_representation_configuration.pdf"),
    7: ("build_polarization_intervals.py", "fig08_polarization_intervals.pdf"),
    8: ("build_national_recurrence.py", "fig01_national_recurrence.pdf"),
}
PIXEL_VERIFIED = {8}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=BUILDERS, action="append")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures")
    parser.add_argument(
        "--publish", action="store_true",
        help="Write directly to manuscript_JAG_v2/figures; otherwise use runs/figures.",
    )
    args = parser.parse_args()
    destination = ROOT / "manuscript_JAG_v2/figures" if args.publish else args.output_dir
    selected = args.figure or list(BUILDERS)
    if args.publish:
        unverified = sorted(set(selected) - PIXEL_VERIFIED)
        if unverified:
            parser.error(
                "refusing to publish unverified figure wrappers: "
                + ", ".join(map(str, unverified))
            )
    for number in selected:
        script, filename = BUILDERS[number]
        command = [sys.executable, str(Path(__file__).with_name(script)), str(destination / filename)]
        print(f"building Figure {number}: {filename}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
