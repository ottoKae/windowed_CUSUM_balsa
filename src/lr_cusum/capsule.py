"""Owner-approved B/C exports: no geometry, global coordinates or source imagery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil
import zipfile

import numpy as np
import pandas as pd

from .detection import stable_seed

VERSION = "balsa_bc_20260905_v1"
BANDS = ("VV_dB", "VH_dB")
SAFE = set("""
sample_id polygon_id polygon_track_id pixel_track_id track_key tile_id
event_id candidate_id series_key spatial_block_id outer_fold spatial_fold role
species_nco cohort_status validation_status support_mode is_fallback
interior_pixels extraction_pixels valid_fraction observed_months_min observed_months_max
series_index cache_index seed_VV_dB seed_VH_dB
event_month change_time reference_time detected_time Band band route
candidate_scope logical_window_id window_start_index window_end_index
decision_start_index decision_end_index window_start window_end decision_start decision_end
window_observations valid_observations clean_change_index local_change_index change_index
distance_to_window_edge_months cusum_amplitude pre_observations post_observations
pre_mean post_mean post_minus_pre permutation_confidence bootstrap_confidence
empirical_exceedance_fraction base_series_seed permutation_seed permutation_iterations
permutation_runtime_seconds passes_window_calendar passes_reporting_calendar
passes_minimum_observations passes_cusum_index passes_split candidate_valid passes_central
passes_decrease passes_permutation f4_rejection_reason null_p_value exceedance_count
permutations near_threshold_refined accepted_f0 accepted_f1 accepted_f2 accepted_f3
accepted_f4 accepted_r0 accepted_r1 accepted_r2 accepted_r3
selection_score track_support track_event_rows vh_supported vh_delta_months vh_change_time
vh_track_support support_pixels largest_component_pixels support_fraction
connected_support_fraction median_confidence median_amplitude
stage variant variant_id variant_name name method family protocol selected_name
selected_window_months selected_step_months window_months step_months window_length step
tolerance_months matcher_version tp fp fn precision recall f1 temporal_mae_months
median_signed_error_months signed_error_months absolute_error_months
reference_events detected_events objects event_count interval_months year
replicate contrast left right delta_precision delta_recall delta_f1 metric
estimate bootstrap_median ci_low ci_high probability_gt_zero replicates seed
familywise_ci_low familywise_ci_high admitted score n_obs
chi observations regularization confidence before after training_f1
trajectory_index segment raw_tBreak change_probability magnitude confirmed decrease
raw_tStart raw_tEnd available_through_month
null_seed parameter_id training_mae tie_break horizon_months as_of_cutoff evaluable
detected estimated_change_time candidate_events_in_reference_polygon final_estimated_change_time
stable_to_final polygon_first_time pixel_first_time route_delta_months polygon_track_support
pixel_track_support polygon_vh_supported pixel_vh_supported dual_route_dual_pol
vv_n_obs vh_n_obs common_n_obs
center_pixel_track_id precision_ci_low precision_ci_high recall_ci_low recall_ci_high
f1_ci_low f1_ci_high cluster_rows space_threshold_pixels
""".split())
PRIVATE_TEXT = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])|/mnt/|balsa:mag|mag_fc001|r-\d{4,}_c-\d{4,}", re.I)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temp.replace(path)


def frame_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


class PublicIDs:
    """Private lookup stays outside the exported directories."""

    def __init__(self, path: Path):
        self.path = path
        self.maps = json.loads(path.read_text()) if path.exists() else {}

    def register(self, kind: str, values, *, ordered: bool = False) -> None:
        values = sorted({str(x) for x in values if pd.notna(x)})
        existing = self.maps.setdefault(kind, {})
        missing = [x for x in values if x not in existing]
        if ordered and existing and missing:
            raise ValueError("block universe changed; create a new capsule version")
        if not ordered:
            secrets.SystemRandom().shuffle(missing)
        for x in missing:
            existing[x] = f"PUB_{kind}_{len(existing) + 1:07d}"

    def value(self, kind: str, value):
        if pd.isna(value):
            return None
        return self.maps[kind][str(value)]

    def save(self) -> None:
        write_json(self.path, self.maps)


def sanitize(frame: pd.DataFrame, ids: PublicIDs) -> pd.DataFrame:
    out = frame[[c for c in frame if c in SAFE]].copy()
    out.attrs = {}
    kinds = {"sample_id": "P", "polygon_id": "P", "polygon_track_id": "PT",
             "pixel_track_id": "PX", "track_key": "T", "tile_id": "TILE",
             "spatial_block_id": "B", "event_id": "E", "center_pixel_track_id":"PX"}
    for column, kind in kinds.items():
        if column in out:
            if kind == "E":
                ids.register(kind, out[column])
            out[column] = out[column].map(lambda x: ids.value(kind, x))
    for column in ("series_key", "candidate_id"):
        if column in out:
            def replace(value):
                if pd.isna(value):
                    return None
                key, *suffix = str(value).split("|")
                identifier, band = key.rsplit(":", 1)
                if band not in BANDS:
                    raise ValueError("unexpected series identity format")
                new = ids.maps["PT"].get(identifier, ids.maps.get("PX", {}).get(identifier))
                if new is None:
                    raise ValueError("unmapped candidate trajectory")
                return "|".join([f"{new}:{band}", *suffix])
            out[column] = out[column].map(replace)
    for column in out.select_dtypes(include=["object", "string"]):
        if out[column].dropna().astype(str).str.contains(PRIVATE_TEXT).any():
            raise ValueError(f"private value remains in approved field {column}")
    return out


def seal(folder: Path, metadata: dict) -> dict:
    records = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name not in {"MANIFEST.json", "COMPLETE.json"}:
            records.append({"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size,
                            "sha256": digest(p)})
    write_json(folder / "MANIFEST.json", {**metadata, "files": records})
    verify(folder)
    write_json(folder / "COMPLETE.json", {"passed": True, "version": VERSION,
                                          "files": len(records), "manifest_sha256": digest(folder / "MANIFEST.json")})
    archive = folder.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())
    temp.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="ascii")
    return {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": digest(archive)}


def verify(folder: Path) -> dict:
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    if (folder / "COMPLETE.json").exists():
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        if receipt["manifest_sha256"] != digest(folder / "MANIFEST.json"):
            raise ValueError("completion manifest checksum mismatch")
    listed = {r["path"] for r in manifest["files"]}
    if len(listed) != len(manifest["files"]):
        raise ValueError("duplicate manifest path")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual - listed - {"MANIFEST.json", "COMPLETE.json"}:
        raise ValueError("unexpected file in capsule")
    for record in manifest["files"]:
        rel = Path(record["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe manifest path")
        p = folder / rel
        if not p.resolve().is_relative_to(folder.resolve()):
            raise ValueError("manifest target escapes capsule")
        if not p.is_file() or p.stat().st_size != record["bytes"] or digest(p) != record["sha256"]:
            raise ValueError(f"checksum mismatch: {rel}")
        if p.suffix == ".parquet":
            frame = pd.read_parquet(p)
            if set(frame) - SAFE - {"local_row", "local_col"}:
                raise ValueError(f"unapproved columns: {rel}")
            for column in frame.select_dtypes(include=["object", "string"]):
                if frame[column].dropna().astype(str).str.contains(PRIVATE_TEXT).any():
                    raise ValueError(f"private values: {rel}: {column}")
        elif p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                if set(z.files) - {"values", "calendar", "valid", "n_obs", "source_month_present"}:
                    raise ValueError("unapproved array")
                if z["values"].shape[1:] != (2, 126) or len(z["calendar"]) != 126:
                    raise ValueError("invalid array calendar/shape")
        elif p.suffix in {".json", ".md", ".csv", ".txt"} or p.name in {"DATA_LICENSE", "CITATION.cff"}:
            text = p.read_text(encoding="utf-8")
            if p.suffix == ".json":
                def strings(value):
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, dict):
                        for key, item in value.items():
                            yield str(key)
                            yield from strings(item)
                    elif isinstance(value, list):
                        for item in value:
                            yield from strings(item)
                text = "\n".join(strings(json.loads(text)))
            if PRIVATE_TEXT.search(text):
                raise ValueError(f"private text: {rel}")
        else:
            raise ValueError(f"unapproved file type: {rel}")
    return {"passed": True, "files": len(listed)}


def build(root: Path, destination: Path) -> dict:
    """Build both immutable profiles from the SAME frozen input version."""
    local = root / "data/local/paper_v1"
    nrt = root / "runs/cusum_nrt_ccdc_20260905_v1"
    b, c = destination / "science", destination / "pixels"
    if b.exists() or c.exists():
        raise FileExistsError("choose a fresh destination; existing capsules are immutable")
    metadata = pd.read_parquet(local / "frozen_results/national/polygon_series_metadata.parquet")
    tracks = pd.read_parquet(local / "polygon_pixels/polygon_tracks.parquet")
    associations = pd.read_parquet(local / "polygon_pixels/pixel_associations.parquet")
    polygons = pd.read_parquet(local / "inventory/balsa_validated.parquet")
    roles = pd.read_parquet(local / "splits/polygon_roles.parquet")
    refs = pd.read_parquet(nrt / "references.parquet")
    if (len(polygons), len(tracks), len(metadata), len(associations), len(refs)) != (3498,4001,4001,176935,1093):
        raise ValueError("unexpected input cohort")
    blocks = refs[["spatial_block_id", "outer_fold"]].drop_duplicates()
    if len(blocks) != 57 or blocks.spatial_block_id.duplicated().any():
        raise ValueError("invalid reference folds")
    ids = PublicIDs(local / "exports" / f"{VERSION}.ids.local.json")
    ids.register("P", polygons.sample_id)
    for row in polygons[["sample_id", "polygon_id"]].itertuples(index=False):
        ids.maps["P"][str(row.polygon_id)] = ids.value("P", row.sample_id)
    ids.register("PT", tracks.polygon_track_id)
    ids.register("PX", associations.pixel_track_id)
    ids.register("T", tracks.track_key)
    ids.register("TILE", tracks.tile_id)
    # Sorted opaque block codes preserve the seeded resample ordering.
    ids.register("B", pd.concat([roles.spatial_block_id, refs.spatial_block_id]), ordered=True)
    ids.save()
    b.mkdir(parents=True); c.mkdir(parents=True)
    source_rows = []

    def export(source: Path, target: str, selected: pd.DataFrame | None = None):
        original = selected if selected is not None else (pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source))
        clean = sanitize(original, ids)
        if not len(clean.columns):
            raise ValueError(f"no approved columns in {source.name}")
        frame_write(b / target, clean)
        source_rows.append({"output": target, "input_sha256": digest(source), "rows": len(clean),
                            "omitted_columns": sorted(set(original) - set(clean))})

    export(local / "inventory/balsa_validated.parquet", "tables/polygons.parquet", polygons[["sample_id", "polygon_id", "species_nco"]])
    export(local / "splits/polygon_roles.parquet", "tables/polygon_roles.parquet", roles[["sample_id", "polygon_id", "spatial_block_id", "spatial_fold", "role"]])
    export(nrt / "references.parquet", "tables/references.parquet", refs[["event_id", "sample_id", "polygon_id", "event_month", "spatial_block_id", "outer_fold"]])
    export(nrt / "references.parquet", "tables/reference_blocks.parquet", blocks)
    track_meta = metadata.merge(tracks[["polygon_track_id", "spatial_block_id", "spatial_fold"]], on="polygon_track_id", validate="one_to_one")
    for band in BANDS:
        track_meta[f"seed_{band}"] = [stable_seed(f"{x}:{band}", 20260820) for x in track_meta.polygon_track_id]
    export(local / "frozen_results/national/polygon_series_metadata.parquet", "tables/polygon_tracks.parquet", track_meta)
    (b / "inputs").mkdir()
    shutil.copyfile(local / "frozen_results/national/polygon_series.npz", b / "inputs/stand_series.npz")
    source_rows.append({"output":"inputs/stand_series.npz",
                        "input_sha256":digest(local / "frozen_results/national/polygon_series.npz"),
                        "rows":len(metadata), "omitted_columns":[]})

    for part in ("component_ablation", "geometry", "truncation", "national"):
        for p in sorted((local / "frozen_results" / part).rglob("*.parquet")):
            if "by_track" not in p.parts:
                export(p, "evidence/frozen/" + p.relative_to(local / "frozen_results").as_posix())
    for p in sorted((local / "frozen_protocol/oof_predictions").glob("*.parquet")):
        export(p, "evidence/benchmark_legacy/" + p.name)
    refresh = root / "runs/numeric_refresh_20260905"
    for part in ("representations_n1_48_12", "benchmark_shared_oof"):
        for p in sorted((refresh / part).rglob("*")):
            if p.suffix in {".parquet",".csv"} and "pixel_events_by_track" not in p.parts:
                export(p,"evidence/numeric_refresh/" + p.relative_to(refresh).with_suffix(".parquet").as_posix())
    array_source = refresh / "representations_n1_48_12/representations.npz"
    with np.load(array_source, allow_pickle=False) as z:
        for i,name in enumerate(z["representation"]):
            if str(name) not in {"median","mean","mode","center"}:
                raise ValueError("unexpected stand representation")
            output = f"evidence/numeric_refresh/representations_n1_48_12/{name}.npz"
            np.savez_compressed(b / output,values=z["values"][:,i],calendar=z["calendar"])
            source_rows.append({"output":output,"input_sha256":digest(array_source),
                                "rows":len(z["values"]),"omitted_columns":[]})
    names = ["references.parquet", "trajectory_metadata.parquet", "input_support.parquet", "nrt_oof_events.parquet", "ccdc_oof_events.parquet", "recusum_frozen_oof.parquet", "oof_matches.parquet", "method_block_metrics.parquet", "method_fold_metrics.csv", "method_metrics.csv", "paired_block_replicates.parquet", "paired_contrasts.csv", "nrt_fold_selections.csv", "ccdc_fold_selections.csv"]
    for name in names:
        export(nrt / name, "evidence/nrt_ccdc/" + Path(name).with_suffix(".parquet").name)
    for name in ("method_metrics.csv", "method_block_metrics.parquet", "paired_contrasts.csv", "paired_block_replicates.parquet", "oof_matches.parquet"):
        export(nrt / "diagnostics" / name, "evidence/nrt_ccdc/direction_control/" + Path(name).with_suffix(".parquet").name)
    for sub, name in (("nrt_ablation_events", "NRT_decrease_filter.parquet"), ("band_ablation", "VV_VH_negative.events.parquet")):
        export(nrt / sub / name, "evidence/nrt_ccdc/" + name)
    export(nrt / "ground/nrt_b11_a3_tc92.parquet", "evidence/nrt_ccdc/nrt_source_defaults.parquet")
    # Official CCDC outputs contain artificial indices, not geographic pixels.
    for p in sorted((nrt / "ccdc_raw").rglob("*.json")):
        response = json.loads(p.read_text())
        if not isinstance(response, list):
            raise ValueError("unexpected official CCDC response")
        fields = {"trajectory_index", "tBreak", "tStart", "tEnd", "changeProb", "VV_magnitude", "numObs"}
        payload = [{k:v for k,v in row.items() if k in fields} for row in response]
        write_json(b / "evidence/nrt_ccdc/official_responses" / p.relative_to(nrt / "ccdc_raw"), payload)

    frame_write(c / "tables/polygon_tracks.parquet", sanitize(track_meta, ids))
    clean_associations = sanitize(associations, ids)
    for axis in ("row", "col"):
        values = associations[f"native_{axis}"]
        clean_associations[f"local_{axis}"] = values - values.groupby(associations.polygon_track_id).transform("min")
    frame_write(c / "tables/pixel_associations.parquet", clean_associations)
    pixel_count = 0
    pixel_sources = []
    for key in sorted(tracks.track_key.unique()):
        name = ids.value("T", key)
        p = local / f"polygon_pixels/cache/{key}.pixels.parquet"
        pixels = pd.read_parquet(p)
        for band in BANDS:
            pixels[f"seed_{band}"] = [stable_seed(f"{x}:{band}", 20260820) for x in pixels.pixel_track_id]
        frame_write(c / f"cache/{name}.pixels.parquet", sanitize(pixels, ids))
        with np.load(local / f"polygon_pixels/cache/{key}.npz", allow_pickle=False) as z:
            arrays = {k:z[k] for k in ("values", "calendar", "valid", "n_obs", "source_month_present")}
            pixel_count += len(arrays["values"])
            np.savez_compressed(c / f"cache/{name}.npz", **arrays)
        pixel_sources.append({"output":f"cache/{name}.npz",
                              "input_sha256":digest(local / f"polygon_pixels/cache/{key}.npz"),
                              "pixel_metadata_input_sha256":digest(p)})
    if pixel_count != 176907:
        raise ValueError("pixel count changed")
    ids.save()
    write_json(b / "SOURCE_HASHES.json", source_rows)
    write_json(c / "SOURCE_HASHES.json", pixel_sources)
    shutil.copyfile(root / "configs/cusum_nrt_ccdc_comparison_v1.json", b / "benchmark_config.json")
    authorization = {"owner_authorized": True, "approved_profiles": ["science", "pixels"],
                     "geometry_and_source_archive_authorized": False, "publication_performed": False,
                     "code_license": "MIT", "data_license": "CC-BY-4.0"}
    for folder in (b,c):
        write_json(folder / "AUTHORIZATION.json", authorization)
        shutil.copyfile(root / "configs/paper_v1.json", folder / "paper_v1.json")
        shutil.copyfile(root / "docs/REAL_CAPSULES.md", folder / "README.md")
        shutil.copyfile(root / "DATA_LICENSE", folder / "DATA_LICENSE")
        shutil.copyfile(root / "data/CITATION.cff", folder / "CITATION.cff")
        write_json(folder / "VERSION_BOUNDARIES.json", {
            "input": "original frozen pixel cache; NOT the later repaired raw-Zarr rebuild",
            "national_primary": "polygon_first VV, 6658 events; 3498 polygons; 3509 adjacent intervals",
            "legacy_pixel_route": "L0 6241 events; do not substitute N1 6197",
            "paired_polarization": "frozen 4061 VV-supported events; later N1 route summary 3657 is separate",
            "component_default": "fixed 48/12; method-comparison Windowed CUSUM uses held-out-selected 36/12",
            "raw_rebuild_difference": "39/40 tracks exact; 269 pixels affected in 2016-10 in one withheld-location track",
            "B_C_parity_required": True,
        })
    common = {"version": VERSION, "georeferenced": False, "owner_authorized": True,
              "polygons":3498, "polygon_tracks":4001, "reference_events":1093,
              "reference_blocks":57, "calendar_months":126}
    return {"science": seal(b, {**common, "profile":"science"}),
            "pixels": seal(c, {**common, "profile":"pixels", "unique_pixels":176907, "memberships":176935})}
