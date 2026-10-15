"""Backward-compatible imports; use :mod:`lr_cusum.baselines` in new code."""

from .baselines import (
    bfast_detect_seasonal,
    cusum_single,
    fpol_bocd_detect,
    is_decrease,
    recusum_detect,
)

__all__ = [
    "bfast_detect_seasonal",
    "cusum_single",
    "fpol_bocd_detect",
    "is_decrease",
    "recusum_detect",
]
