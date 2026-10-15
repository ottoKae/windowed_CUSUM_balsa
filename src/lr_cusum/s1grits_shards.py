"""Adapters from DOCC plantation Zarr shards to LR-CuSum long tables."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


CALENDAR = pd.date_range("2017-01-01", "2025-12-01", freq="MS")
DEFAULT_METADATA = (
    "point_track_id", "sample_id", "polygon_id", "sample_type", "role",
    "spatial_block_id", "tile_id", "track_key", "species_nco",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_stores(cohort_root: Path) -> list[Path]:
    report_path = cohort_root / "extraction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stores = []
    for track_key in sorted(report["shards"]):
        tile = track_key.split("_", 1)[0]
        store = cohort_root / tile / f"{track_key}.zarr"
        if not store.exists():
            raise FileNotFoundError(store)
        stores.append(store)
    return stores


def _metadata_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=metadata.index)
    for column in DEFAULT_METADATA:
        values = metadata[column] if column in metadata else pd.Series(pd.NA, index=metadata.index)
        output[column] = values.astype("string")
    if output.point_track_id.isna().any() or output.sample_id.isna().any():
        raise ValueError("point_track_id and sample_id are required")
    return output


def _long_frame(metadata, values, valid, n_obs, strategy):
    n, _, months = values.shape
    if months != len(CALENDAR):
        raise ValueError(f"expected 108 months, received {months}")
    base = _metadata_frame(metadata)
    output = pd.DataFrame({
        column: np.repeat(base[column].to_numpy(), months) for column in base.columns
    })
    for column in base.columns:
        output[column] = output[column].astype("string")
    output["time"] = np.tile(CALENDAR.to_numpy(), n)
    output["VV_dB"] = values[:, 0].reshape(-1).astype(np.float32)
    output["VH_dB"] = values[:, 1].reshape(-1).astype(np.float32)
    output["valid"] = valid.reshape(-1).astype(bool)
    output["n_obs"] = n_obs.reshape(-1).astype(np.uint16)
    output["pixel_strategy"] = strategy
    output.loc[~output.valid, ["VV_dB", "VH_dB"]] = np.nan
    return output


def convert_store(store: Path, output_root: Path, cohort: str, *, batch_size=256,
                  strategies=("center", "patch_median")) -> list[dict]:
    """Convert one point-track shard into partitioned long Parquet tables."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    strategies = tuple(strategies)
    unknown = set(strategies) - {"center", "patch_median"}
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    metadata_path = store / "metadata.parquet"
    metadata = pd.read_parquet(metadata_path).sort_values("array_index").reset_index(drop=True)
    if not np.array_equal(metadata.array_index.to_numpy(), np.arange(len(metadata))):
        raise ValueError(f"non-contiguous array_index in {store}")
    group = zarr.open_group(store, mode="r")
    if group["s1_value"].shape[1:] != (2, 108, 9, 9):
        raise ValueError(f"unexpected s1_value shape in {store}: {group['s1_value'].shape}")
    writers, paths, statistics = {}, {}, {strategy: {"valid": 0, "rows": 0} for strategy in strategies}
    tile = str(metadata.tile_id.iloc[0]); track = str(metadata.track_key.iloc[0])
    try:
        for start in range(0, len(metadata), batch_size):
            stop = min(len(metadata), start + batch_size)
            local_meta = metadata.iloc[start:stop]
            sar = np.asarray(group["s1_value"][start:stop], np.float32)
            valid = np.asarray(group["valid"][start:stop], bool)
            n_obs = np.asarray(group["n_obs"][start:stop], np.uint16)
            if "center" in strategies:
                center_valid = valid[:, :, 4, 4]
                center_values = sar[:, :, :, 4, 4]
                center_n_obs = n_obs[:, :, 4, 4]
                frame = _long_frame(local_meta, center_values, center_valid, center_n_obs, "center")
                statistics["center"]["valid"] += int(frame.valid.sum())
                statistics["center"]["rows"] += len(frame)
                if "center" not in writers:
                    path = output_root / "center" / cohort / tile / f"{track}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True); paths["center"] = path
                    writers["center"] = pq.ParquetWriter(path, pa.Table.from_pandas(frame, preserve_index=False).schema,
                                                          compression="zstd", use_dictionary=True)
                writers["center"].write_table(pa.Table.from_pandas(frame, preserve_index=False))
            if "patch_median" in strategies:
                masked = np.where(valid[:, None], sar, np.nan)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                    patch_values = np.nanmedian(masked, axis=(3, 4)).astype(np.float32)
                patch_count = valid.sum((2, 3)).astype(np.uint16)
                patch_valid = patch_count > 0
                frame = _long_frame(local_meta, patch_values, patch_valid, patch_count, "patch_median")
                frame.rename(columns={"n_obs": "valid_pixel_count"}, inplace=True)
                frame["valid_pixel_fraction"] = frame.valid_pixel_count.astype(np.float32) / 81.0
                statistics["patch_median"]["valid"] += int(frame.valid.sum())
                statistics["patch_median"]["rows"] += len(frame)
                if "patch_median" not in writers:
                    path = output_root / "patch_median" / cohort / tile / f"{track}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True); paths["patch_median"] = path
                    writers["patch_median"] = pq.ParquetWriter(
                        path, pa.Table.from_pandas(frame, preserve_index=False).schema,
                        compression="zstd", use_dictionary=True,
                    )
                writers["patch_median"].write_table(pa.Table.from_pandas(frame, preserve_index=False))
    finally:
        for writer in writers.values():
            writer.close()
    records = []
    for strategy, path in paths.items():
        records.append({
            "strategy": strategy, "cohort": cohort, "tile_id": tile, "track_key": track,
            "source_store": str(store.resolve()), "source_metadata_sha256": sha256(metadata_path),
            "output_path": str(path.resolve()), "output_sha256": sha256(path),
            "objects": len(metadata), "rows": statistics[strategy]["rows"],
            "valid_rows": statistics[strategy]["valid"],
            "valid_fraction": statistics[strategy]["valid"] / max(1, statistics[strategy]["rows"]),
        })
    return records


def build_object_index(cohorts: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for cohort, root in cohorts.items():
        for store in discover_stores(root):
            metadata = pd.read_parquet(store / "metadata.parquet")
            local = _metadata_frame(metadata)
            local["cohort"] = cohort
            local["source_store"] = str(store.resolve())
            local["array_index"] = metadata.array_index.to_numpy()
            rows.append(local)
    result = pd.concat(rows, ignore_index=True)
    if result.point_track_id.duplicated().any():
        raise ValueError("point_track_id is not unique across cohorts")
    return result.sort_values(["tile_id", "track_key", "point_track_id"]).reset_index(drop=True)
