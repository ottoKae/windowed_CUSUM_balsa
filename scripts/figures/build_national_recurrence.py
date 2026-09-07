#!/usr/bin/env python3
"""Build the current manuscript Figure 8 with its frozen production layout.

The default reproduces the VV-primary national chronology figure. Passing
``--vh-evidence`` keeps the identical canvas and panel geometry while adding
one-to-one Polygon-first VH support (within +/-2 calendar months) to panels
(b) and (d). VH is never averaged into the primary VV trajectory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.workflows import build_figure6_recurrence_10km as formal  # noqa: E402


DATA = ROOT / "data/local/paper_v1"


def build(output: Path, *, data_root: Path, build_dir: Path, vh_evidence: bool) -> None:
    args = argparse.Namespace(
        events=data_root / "frozen_results/national/events/VV_dB.ground.parquet",
        vh_events=(
            data_root / "frozen_results/national/events/VH_dB.ground.parquet"
            if vh_evidence
            else None
        ),
        vh_tolerance_months=2,
        polygon_tracks=data_root / "polygon_pixels/polygon_tracks.parquet",
        balsa_shp=data_root / "inventory/balsa_validated.parquet",
        boundary=data_root / "admin/analysis_context/ecuador_mainland.parquet",
        amazonia_boundary=data_root / "admin/analysis_context/amazonia_sensu_stricto.parquet",
        basemap_cache=data_root / "basemap",
        study_area_config=ROOT / "configs/study_area_figure_v1.json",
        study_area_reference_pdf=ROOT / "manuscript_JAG_v2/figures/fig02_study_area.pdf",
        output_dir=build_dir,
        dpi=600,
        allow_count_mismatch=False,
    )
    result = formal.build(args)
    produced = Path(result["outputs"]["pdf"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, output)
    for suffix in (".svg", ".png", ".tiff"):
        source = produced.with_suffix(suffix)
        if source.exists():
            shutil.copy2(source, output.with_suffix(suffix))
    print(f"figure={output.resolve()}")
    print(f"source_bundle={build_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "runs/figure_sources/figure08_national_recurrence",
    )
    parser.add_argument("--vh-evidence", action="store_true")
    args = parser.parse_args()
    build(
        args.output,
        data_root=args.data,
        build_dir=args.build_dir,
        vh_evidence=args.vh_evidence,
    )


if __name__ == "__main__":
    main()
