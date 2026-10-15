"""Windowed CUSUM detection for recurrent stand disturbances."""

from .core import cusum_amplitude, generate_windows, lr_cusum
from .detection import DetectionConfig, detect_group, detect_table
from .paper import PaperDetectionConfig, detect_series_n1, detect_table_n1

__all__ = [
    "DetectionConfig",
    "PaperDetectionConfig",
    "cusum_amplitude",
    "detect_group",
    "detect_series_n1",
    "detect_table",
    "detect_table_n1",
    "generate_windows",
    "lr_cusum",
]
