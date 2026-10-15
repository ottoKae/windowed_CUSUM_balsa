"""Backward-compatible imports; use :mod:`lr_cusum.spatial_fusion` in new code."""

from .spatial_fusion import space_time_dbscan_fusion, temporal_dbscan_fusion

__all__ = ["space_time_dbscan_fusion", "temporal_dbscan_fusion"]
