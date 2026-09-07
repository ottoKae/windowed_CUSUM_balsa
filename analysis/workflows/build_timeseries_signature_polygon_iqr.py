"""Render the publication signature figure with native-pixel IQR shading.

This is an explicit companion entry point to
``build_timeseries_signature_polygon_median.py``. It preserves the median-only
workflow and enables Q25--Q75 shading from the exact polygon-interior pixel
cache used to construct the frozen Polygon-first series.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.workflows.build_timeseries_signature_polygon_median import (
    DEFAULT_V9_IQR_MANIFEST,
    IQR_MAP_HEIGHT_SCALE_FACTOR,
    main,
)


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                *sys.argv[1:],
                "--include-pixel-iqr",
                "--inside-iqr-legend",
                "--time-ymin",
                "-24",
                "--time-ymax",
                "-4",
                "--time-ytick-interval",
                "4",
                "--map-height-scale-factor",
                str(IQR_MAP_HEIGHT_SCALE_FACTOR),
                "--sar-stretch-manifest",
                str(DEFAULT_V9_IQR_MANIFEST),
            ]
        )
    )
