"""Publish frozen inventory report attributes and explicit spatial support layers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import shapely

# Old method keys are retained in archived tables; presentation uses current names.
METHOD_LABELS = {"BWA": "BWA", "OCTA": "TDA", "CSTR": "STP"}

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/validation/harvest_inventory_20261104_v1"
NATIONAL = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
INVENTORY = ROOT / "data/local/paper_v1/inventory/balsa_validated.parquet"
SCOPE = ROOT / "runs/validation/harvest_rotation_cycle_20261104_v1/scope"


def publish_products(tda_reports_path, tda_universe_path, stp_reports_path,
                     completion_receipts, out=RUN / "products", render=True):
    started = perf_counter()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = dict(inventory=INVENTORY, bwa_reports=NATIONAL / "national/stand_reports.parquet",
        inventory_status=NATIONAL / "products/stand_summary.csv", units=NATIONAL / "master_units.parquet",
        membership=NATIONAL / "national_membership.parquet", reference_index=SCOPE / "stand_index_459.csv",
        octa_reports=Path(tda_reports_path), octa_universe=Path(tda_universe_path),
        cstr_reports=Path(stp_reports_path))
    for index, path in enumerate(completion_receipts):
        paths[f"completion_receipt_{index}"] = Path(path)
    hashes = {key: digest(path) for key, path in paths.items()}
    for key, path in paths.items():
        if key.startswith("completion_receipt_"):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            assert receipt.get("status") == "passed" or receipt.get("passed") is True, f"Source stage incomplete: {path}"
    inventory = pd.read_parquet(paths["inventory"])
    status = pd.read_csv(paths["inventory_status"])
    bwa = pd.read_parquet(paths["bwa_reports"])
    tda = pd.read_parquet(paths["octa_reports"])
    universe = pd.read_parquet(paths["octa_universe"])
    children = pd.read_parquet(paths["cstr_reports"]).rename(columns={"round_id": "prediction_id"})
    units = pd.read_parquet(paths["units"])
    membership = pd.read_parquet(paths["membership"])
    reference_ids = set(pd.read_csv(paths["reference_index"]).sample_id)
    supported = set(status.loc[status.status.str.startswith("completed"), "sample_id"])
    validate_inputs(inventory, status, bwa, tda, children, universe, units, membership, supported)

    main_fields = ["sample_id", "source_row_original", "OBJECTID", "DPA_DESPRO", "DPA_DESCAN",
        "DPA_DESPAR", "species_nco", "validation_status", "geometry_sha256_frozen"]
    main = inventory[main_fields].merge(status[["sample_id", "native_cells", "status"]], on="sample_id", validate="one_to_one")
    main.rename(columns={"status": "native_support_status"}, inplace=True)
    main["in_reference_459"] = main.sample_id.isin(reference_ids)
    main["model_evaluable"] = main.sample_id.isin(supported)
    main["geometry_available"] = True
    main["report_start_month"] = "2017-01"
    main["report_end_month"] = "2025-12"
    # STP parent projection reuses each frozen TDA parent exactly once, including its date.
    parent_projection = tda.copy()
    interval_tables, stand_tables = [], []
    configurations = [("BWA", "report", "bwa", bwa), ("OCTA", "report", "octa", tda),
        ("CSTR", "parent_projection", "cstr", parent_projection), ("CSTR", "raw_child", "cstr_child", children)]
    for method, level, prefix, reports in configurations:
        stand_table, intervals = describe_reports(reports, main[["sample_id", "model_evaluable"]])
        stand_tables.append(stand_table.assign(method=method, report_level=level))
        interval_tables.append(intervals.assign(method=method, report_level=level))
        main = main.merge(stand_table.rename(columns={name: prefix+"_"+name for name in stand_table.columns if name != "sample_id"}),
            on="sample_id", validate="one_to_one")
    long_stands = pd.concat(stand_tables, ignore_index=True)
    intervals = pd.concat(interval_tables, ignore_index=True)
    assert main.bwa_count.sum() == len(bwa) and main.octa_count.sum() == len(tda)
    assert main.cstr_count.equals(main.octa_count)
    assert main.cstr_child_count.sum() == len(children)
    for prefix in ("bwa", "octa", "cstr", "cstr_child"):
        assert main.loc[~main.model_evaluable, prefix+"_count"].isna().all()
    main.to_csv(out / "stand_history.csv", index=False, encoding="utf-8-sig")
    long_stands.to_csv(out / "stand_method_summary.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(out / "adjacent_report_intervals.csv", index=False, encoding="utf-8-sig")

    event_fields = ["sample_id", "prediction_id", "representative_month", "first_response_month",
        "last_response_month", "response_months", "response_pixels", "total_pixels", "support_fraction", "source_unit_ids"]
    events = pd.concat([bwa[event_fields].assign(method="BWA"), tda[event_fields].assign(method="OCTA")], ignore_index=True)
    events["product_record_id"] = events.method+":"+events.prediction_id
    events["geometry_meaning"] = "fixed_inventory_polygon_not_harvest_extent"
    events["date_meaning"] = "algorithm_representative_month_not_verified_harvest_onset"
    assert events.product_record_id.is_unique
    events.to_csv(out / "event_reports.csv", index=False, encoding="utf-8-sig")
    child_fields = ["sample_id", "prediction_id", "parent_prediction_id", "round_index", "zone_count",
        "representative_month", "first_response_month", "last_response_month", "response_months",
        "response_pixels", "parent_pixels", "parent_support_fraction", "round_support_fraction", "source_node_count", "source_unit_ids"]
    child_table = children[child_fields].copy()
    child_table["method"] = "CSTR"
    child_table["n_siblings"] = child_table.parent_prediction_id.map(child_table.groupby("parent_prediction_id").size())
    child_table["is_spatial_split"] = child_table.n_siblings.gt(1)
    child_table["geometry_meaning"] = "union_of_full_native_response_cells_not_verified_harvest_extent"
    child_table["date_meaning"] = "algorithm_response_dates_not_verified_operation_duration"
    child_geometry, coordinate_receipt = build_child_geometry(child_table, units, membership)
    child_table.to_csv(out / "cstr_spatial_children.csv", index=False, encoding="utf-8-sig")
    mapping = child_table[["sample_id", "prediction_id", "parent_prediction_id", "n_siblings", "is_spatial_split"]].merge(
        tda[["prediction_id", "representative_month", "first_response_month", "last_response_month"]].rename(columns={
            "prediction_id": "parent_prediction_id", "representative_month": "parent_representative_month",
            "first_response_month": "parent_first_response_month", "last_response_month": "parent_last_response_month"}),
        on="parent_prediction_id", validate="many_to_one")
    mapping.to_csv(out / "cstr_parent_projection.csv", index=False, encoding="utf-8-sig")

    os.environ.setdefault("GDAL_DATA", str(Path(sys.prefix) / "Library/share/gdal"))
    os.environ.setdefault("PROJ_DATA", str(Path(sys.prefix) / "Library/share/proj"))
    import geopandas as gpd
    polygon_geometry = shapely.from_wkb(inventory.geometry.to_numpy())
    geometry_lookup = pd.Series(polygon_geometry, index=inventory.sample_id)
    layer_frames = {
        "stand_history": gpd.GeoDataFrame(main, geometry=main.sample_id.map(geometry_lookup), crs=32717),
        "event_reports": gpd.GeoDataFrame(events, geometry=events.sample_id.map(geometry_lookup), crs=32717),
        "cstr_spatial_children": gpd.GeoDataFrame(child_table, geometry=child_geometry, crs=32717),
    }
    gpkg = out / "harvest_inventory_2017_2025.gpkg"
    if gpkg.exists():
        raise FileExistsError(f"Publish to a fresh output directory: {gpkg}")
    for layer, frame in layer_frames.items():
        frame.to_file(gpkg, layer=layer, driver="GPKG", engine="fiona", index=False)
    layer_checks = audit_gpkg(gpkg, main, events, child_table, geometry_lookup)
    figures = render_figures(main, long_stands, intervals, layer_frames["stand_history"], out) if render else []
    write_guide(out, main, long_stands, intervals, events, child_table)
    assert all(digest(path) == hashes[key] for key, path in paths.items()), "Frozen publication inputs changed"
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        source_scope="full validated inventory; not restricted to the 459 reference cohort",
        report_period=["2017-01", "2025-12"], inventory_stands=len(main), supported_stands=int(main.model_evaluable.sum()),
        unsupported_stands=int((~main.model_evaluable).sum()), unsupported_original_polygons_retained=True,
        n_reports={"BWA": len(bwa), "OCTA": len(tda), "CSTR_parent_projection": len(parent_projection), "CSTR_raw_child": len(children)},
        method_display_names=METHOD_LABELS,
        counts_interpretation="STP stand-level count is parent projection; raw children are diagnostic and not independent rotation counts",
        interval_interpretation="adjacent model representative-month report intervals; not verified biological rotations",
        geometry=coordinate_receipt, gpkg_layer_checks=layer_checks, figures=figures,
        bootstrap_included=False, area_estimation=False, detector_rerun=False, aggregation_rerun=False,
        input_sha256={key:dict(path=str(path), sha256=hashes[key]) for key, path in paths.items()},
        script_sha256=digest(Path(__file__)), elapsed_seconds=perf_counter()-started,
        packages={name:importlib.metadata.version(name) for name in ("pandas", "shapely", "pyproj", "geopandas", "fiona", "matplotlib")},
        output_sha256={path.name:digest(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "product_receipt.json"})
    (out / "product_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return receipt


def validate_inputs(inventory, status, bwa, tda, children, universe, units, membership, supported):
    assert len(inventory) == 3498 and inventory.sample_id.is_unique
    assert set(status.sample_id) == set(inventory.sample_id) and status.sample_id.is_unique
    assert len(supported) == 3364 and len(inventory)-len(supported) == 134
    assert set(universe.sample_id) == set(inventory.sample_id) and universe.sample_id.is_unique
    assert set(universe.loc[universe.processing_status.eq("completed"), "sample_id"]) == supported
    assert universe.loc[~universe.sample_id.isin(supported), "report_count"].isna().all()
    assert units.unit_id.is_unique and membership.unit_id.is_unique
    assert set(membership.sample_id) == supported
    assert set(membership.unit_id) <= set(units.unit_id)
    counts = membership.groupby("sample_id").size()
    assert universe.loc[universe.sample_id.isin(supported)].set_index("sample_id").total_pixels.sort_index().equals(counts.sort_index().astype(universe.total_pixels.dtype))
    for reports in (bwa, tda, children):
        assert reports.prediction_id.is_unique and reports.sample_id.isin(supported).all()
        assert reports.representative_month.between("2017-01", "2025-12").all()
        assert (reports.first_response_month <= reports.last_response_month).all()
        pd.PeriodIndex(reports.representative_month, freq="M")
    assert set(children.parent_prediction_id) == set(tda.prediction_id)
    parent_samples = tda.set_index("prediction_id").sample_id
    assert children.parent_prediction_id.map(parent_samples).eq(children.sample_id).all()
    geometries = shapely.from_wkb(inventory.geometry.to_numpy())
    assert not shapely.is_missing(geometries).any() and not shapely.is_empty(geometries).any()
    assert shapely.is_valid(geometries).all()


def describe_reports(reports, panel):
    ordered = reports.sort_values(["sample_id", "representative_month", "prediction_id"])[
        ["sample_id", "prediction_id", "representative_month"]].copy()
    if "parent_prediction_id" in reports:
        ordered["parent_prediction_id"] = ordered.prediction_id.map(reports.set_index("prediction_id").parent_prediction_id)
    else:
        ordered["parent_prediction_id"] = ordered.prediction_id
    grouped = ordered.groupby("sample_id", sort=False)
    ordered["left_prediction_id"] = grouped.prediction_id.shift()
    ordered["left_month"] = grouped.representative_month.shift()
    ordered["left_parent_prediction_id"] = grouped.parent_prediction_id.shift()
    intervals = ordered.loc[ordered.left_prediction_id.notna()].copy()
    intervals.rename(columns={"prediction_id": "right_prediction_id", "representative_month": "right_month",
        "parent_prediction_id": "right_parent_prediction_id"}, inplace=True)
    intervals["interval_months"] = pd.PeriodIndex(intervals.right_month, freq="M").asi8-pd.PeriodIndex(intervals.left_month, freq="M").asi8
    intervals["same_parent_sibling"] = intervals.left_parent_prediction_id.eq(intervals.right_parent_prediction_id)
    assert intervals.interval_months.ge(0).all()
    result = panel[["sample_id"]].copy()
    result["count"] = result.sample_id.map(reports.groupby("sample_id").size()).fillna(0).astype("Int64")
    result["n_intervals"] = result.sample_id.map(intervals.groupby("sample_id").size()).fillna(0).astype("Int64")
    for suffix, q in (("q1", .25), ("median", .5), ("q3", .75)):
        values = intervals.groupby("sample_id").interval_months.quantile(q)
        result["interval_"+suffix+"_months"] = result.sample_id.map(values)
    result["first_report_month"] = result.sample_id.map(reports.groupby("sample_id").representative_month.min())
    result["last_report_month"] = result.sample_id.map(reports.groupby("sample_id").representative_month.max())
    result["status"] = np.where(panel.model_evaluable,
        np.where(result["count"].gt(0), "completed_with_reports", "completed_no_report"), "unsupported_spatial_no_native_centres")
    result.loc[~panel.model_evaluable, ["count", "n_intervals"]] = pd.NA
    return result, intervals


def build_child_geometry(children, units, membership):
    ownership = children[["sample_id", "prediction_id", "parent_prediction_id", "source_unit_ids"]].copy()
    ownership["unit_id"] = ownership.source_unit_ids.str.split(";")
    ownership = ownership.explode("unit_id").drop(columns="source_unit_ids")
    assert ownership.unit_id.ne("").all() and not ownership.duplicated(["parent_prediction_id", "unit_id"]).any()
    membership_lookup = membership.set_index("unit_id").sample_id
    assert ownership.unit_id.isin(membership_lookup.index).all()
    assert ownership.unit_id.map(membership_lookup).eq(ownership.sample_id).all()
    wanted = units.loc[units.unit_id.isin(ownership.unit_id)].copy()
    projected = {}
    source_crs = []
    for crs, group in wanted.groupby("track_crs", sort=True):
        geometry = shapely.from_wkb(group.native_geometry_wkb.to_numpy())
        if CRS.from_user_input(crs) != CRS.from_epsg(32717):
            transformer = Transformer.from_crs(crs, 32717, always_xy=True)
            geometry = shapely.transform(geometry, transformer.transform, interleaved=False)
        projected.update(zip(group.unit_id, geometry))
        source_crs.append(dict(crs=crs, n_unique_response_cells=len(group), transformed_to="EPSG:32717"))
    geometry_by_child = {key:shapely.union_all([projected[unit] for unit in group.unit_id])
        for key, group in ownership.groupby("prediction_id", sort=False)}
    geometry = np.array([geometry_by_child[key] for key in children.prediction_id], dtype=object)
    assert shapely.is_valid(geometry).all() and not shapely.is_empty(geometry).any()
    return geometry, dict(output_crs="EPSG:32717", source_crs=source_crs, native_cells_reprojected_before_union=True,
        n_unique_response_cells=len(wanted), native_cell_memberships=len(ownership),
        child_geometry="full_native_response_cell_union; no clipping to inventory and no harvest-area claim",
        parent_geometry="fixed_inventory_polygon", inventory_geometry="all 3498 original polygons including 134 unsupported")


def audit_gpkg(path, main, events, children, original_geometry):
    checks = []
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        layers = connection.execute("SELECT table_name,column_name,srs_id FROM gpkg_geometry_columns").fetchall()
        expected = {"stand_history":len(main), "event_reports":len(events), "cstr_spatial_children":len(children)}
        for layer, column, epsg in layers:
            count, geometry_count = connection.execute(f'SELECT COUNT(*), SUM("{column}" IS NOT NULL) FROM "{layer}"').fetchone()
            assert count == expected[layer] and geometry_count == count and epsg == 32717
            checks.append(dict(layer=layer, n_rows=count, nonnull_geometry=geometry_count, epsg=epsg))
        read_stands = pd.read_sql_query("SELECT sample_id,bwa_count,octa_count,cstr_count,cstr_child_count,model_evaluable FROM stand_history", connection).set_index("sample_id")
        source = main.set_index("sample_id")
        assert set(read_stands.index) == set(source.index)
        for col in ("bwa_count", "octa_count", "cstr_count", "cstr_child_count"):
            pd.testing.assert_series_equal(read_stands[col].astype("Int64").sort_index(), source[col].sort_index(), check_names=False)
        geometry_column = next(column for layer,column,_ in layers if layer == "stand_history")
        geometry_rows = pd.read_sql_query(f'SELECT sample_id,"{geometry_column}" AS geom FROM stand_history', connection)
        envelope_bytes = {0:0,1:32,2:48,3:48,4:64}
        wkb = [blob[8+envelope_bytes[(blob[3]>>1)&7]:] for blob in geometry_rows.geom]
        assert shapely.equals(shapely.from_wkb(wkb), original_geometry.loc[geometry_rows.sample_id].to_numpy()).all()
        checks.append(dict(check="stand_geometry_roundtrip_equals_original", n_rows=len(geometry_rows), passed=True))
        for layer, table in (("event_reports", events), ("cstr_spatial_children", children)):
            read_ids = pd.read_sql_query(f'SELECT sample_id,prediction_id FROM "{layer}"', connection)
            assert set(map(tuple, read_ids.to_numpy())) == set(map(tuple, table[["sample_id", "prediction_id"]].to_numpy()))
    return checks


def render_figures(main, summaries, intervals, spatial, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
        "font.size":10, "axes.spines.top":False, "axes.spines.right":False,
        "pdf.fonttype":42, "svg.fonttype":"none", "savefig.dpi":300})
    colors = ["#264653", "#2A9D8F", "#B66A45"]
    configs = [("BWA", "report"), ("OCTA", "report"), ("CSTR", "parent_projection")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    maximum = int(summaries["count"].max())
    for col, ((method, level), color) in enumerate(zip(configs, colors)):
        stands = summaries.loc[summaries.method.eq(method)&summaries.report_level.eq(level)]
        gaps = intervals.loc[intervals.method.eq(method)&intervals.report_level.eq(level), "interval_months"]
        axes[0,col].hist(stands["count"].dropna().astype(int), bins=np.arange(maximum+2)-.5, color=color, rwidth=.85)
        axes[0,col].set_title(METHOD_LABELS[method]+(" (parent projection)" if method == "CSTR" else ""))
        axes[0,col].set_xlabel("Reports per supported stand")
        axes[0,col].set_ylabel("Stands")
        axes[0,col].text(.98,.96,f"n = {stands['count'].notna().sum():,}; NA = {stands['count'].isna().sum():,}",
            transform=axes[0,col].transAxes,ha="right",va="top",fontsize=9)
        axes[1,col].hist(gaps, bins=np.arange(0,109,6), color=color, alpha=.9)
        axes[1,col].set_xlabel("Adjacent report interval (months)")
        axes[1,col].set_ylabel("Report pairs")
        axes[1,col].text(.98,.96,f"n = {len(gaps):,}; median = {gaps.median():g}",
            transform=axes[1,col].transAxes,ha="right",va="top",fontsize=9)
        if method == "CSTR":
            child_gaps = intervals.loc[intervals.method.eq("CSTR")&intervals.report_level.eq("raw_child"), "interval_months"]
            axes[1,col].hist(child_gaps, bins=np.arange(0,109,6), histtype="step", color="#6A4C93", linewidth=1.3, label="Raw children (diagnostic)")
            axes[1,col].legend(loc="upper right", bbox_to_anchor=(1,.87), fontsize=8)
    fig.suptitle("Inventory harvest-related report history, 2017–2025", y=.985, fontsize=14)
    fig.text(.5,.015,"134 unsupported stands retain NA. Intervals describe algorithm reports, not verified biological rotations.",ha="center",fontsize=9,color="#555555")
    fig.subplots_adjust(left=.065,right=.98,bottom=.11,top=.90,hspace=.38,wspace=.28)
    files = []
    for ext in ("png", "pdf"):
        name="inventory_report_distributions."+ext
        fig.savefig(out/name)
        files.append(name)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(11,7.8),sharex=True,sharey=True)
    maximum = int(max(main.bwa_count.max(),main.octa_count.max(),main.cstr_count.max()))
    norm=BoundaryNorm(np.arange(maximum+2)-.5,256)
    for ax, prefix, title in zip(axes,("bwa","octa","cstr"),("BWA","TDA","STP parent projection")):
        spatial.loc[spatial.model_evaluable].plot(column=prefix+"_count",ax=ax,cmap="viridis",norm=norm,linewidth=0)
        spatial.loc[~spatial.model_evaluable].plot(ax=ax,color="#BBBBBB",linewidth=.15,edgecolor="#777777")
        ax.set_title(title,fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m)")
        ax.ticklabel_format(style="sci",axis="both",scilimits=(0,0))
    axes[0].set_ylabel("Northing (m)")
    colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="viridis"),ax=axes.tolist(),orientation="horizontal",fraction=.035,pad=.12,shrink=.75,ticks=range(maximum+1))
    colorbar.set_label("Model report count; grey = unsupported (NA)")
    fig.suptitle("Fixed inventory polygons: report-count attributes",y=.96,fontsize=14)
    fig.text(.5,.015,"EPSG:32717 | All 3,498 original polygons | STP parent counts equal TDA by construction | No harvest-area estimate",ha="center",fontsize=8.5,color="#555555")
    for ext in ("png", "pdf"):
        name="inventory_report_count_attributes."+ext
        fig.savefig(out/name)
        files.append(name)
    plt.close(fig)
    return files


def write_guide(out, main, summaries, intervals, events, children):
    rows=[]
    for (method,level), stands in summaries.groupby(["method","report_level"],sort=False):
        gaps=intervals.loc[intervals.method.eq(method)&intervals.report_level.eq(level),"interval_months"]
        values=stands["count"].dropna()
        rows.append(f"| {METHOD_LABELS[method]} | {level} | {int(values.sum())} | {int(values.eq(0).sum())} | {values.median():g} [{values.quantile(.25):g}, {values.quantile(.75):g}] | {len(gaps)} | {gaps.median():g} [{gaps.quantile(.25):g}, {gaps.quantile(.75):g}] |")
    text="""# 全库存采伐相关报告：属性与空间图层使用说明

产品保留全部 3,498 个已验证库存多边形，报告期为 2017—2025 年。134 个无原生 SAR 像元中心支持的地块仍保留有效多边形，模型报告次数为 NA；完成计算但没有报告的地块为 0。以下四分位范围为描述性 IQR，不是置信区间。

| 方法 | 报告层级 | 报告数 | 已处理零报告林分 | 次数中位数 [Q1,Q3] | 相邻报告对 | 间隔中位数 [Q1,Q3]，月 |
|---|---|---:|---:|---|---:|---|
"""+"\n".join(rows)+"""

## 在 QGIS 中查看

将 `harvest_inventory_2017_2025.gpkg` 拖入 QGIS，按目的选择三个图层：

1. **`stand_history`：林分主属性图层。** 共 3,498 行，每个原库存多边形一行。比较主结果时选 `bwa_count`、`octa_count`、`cstr_count`。`bwa_*`、`octa_*`、`cstr_*` 分别保存次数、相邻报告间隔中位数/四分位数、首末代表月和处理状态。
2. **`event_reports`：BWA/TDA 父层报告。** 保留 `sample_id`、`prediction_id`、代表月、首末响应月与实际响应月。几何是固定库存多边形，不是已确认的实际采伐范围。STP 父投影直接引用 TDA 父报告，因此没有复制成另一套父事件图层。
3. **`cstr_spatial_children`：空间子报告诊断。** 保留子报告 `prediction_id` 和 `parent_prediction_id`，`is_spatial_split` 区分多子报告与原样保留。几何由其全部原生响应像元组成，先按各像元记录的原始轨道 CRS 转换到 EPSG:32717，再求并集；不裁切到库存边界。它表示算法空间支持，不是独立判读的采伐边界，也不用于面积估计。

## STP 次数与子报告怎么选

`cstr_count` 是父报告投影次数：每个冻结 TDA 父报告只计一次，日期也沿用原父报告。因而它与 TDA 一致是构造结果，不是 STP 提升精度的证据。`cstr_child_count` 及其他 `cstr_child_*` 字段仅用于查看空间细分输出，不应直接解释成新增轮伐次数。`cstr_parent_projection.csv` 提供每个子报告的父 ID 与父日期。不要将父层和子层报告数相加。

## 0、NA 和没有间隔

- 次数为 **0**：该林分已完成模型处理，但没有正式报告。
- 次数为 **NA**：无原生 SAR 中心支持，不能按零次采伐解释；原多边形仍可显示，属性图中为灰色。
- 间隔为空：不足两份报告，或林分不可评价。不能填成 0 个月；`*_n_intervals` 区分已处理但无相邻对的 0 与不可评价 NA。
- 间隔是在同一固定林分中按代表月及报告 ID 排序后计算的相邻报告月差，保留独立报告身份。汇总间隔按报告对统计，不是各林分间隔中位数的再次平均。

CSV 同步提供林分属性、各方法长表和逐对间隔。PNG/PDF 包括报告次数分布、间隔分布和原库存多边形属性图，图面英文不改变字段含义。该发布适配器没有计算 bootstrap 或面积；后续区间推断以独立统计模块为准。

报告间隔表示采伐相关模型响应的再现间隔。空间异步子报告不自动等于同一地表的新一轮采伐；本产品不能单独证明真实经营轮次、完整采伐历史、种植至收获周期或独立精度。

`product_receipt.json` 记录输入哈希、坐标转换、GIS 行数/ID/空值回读、软件版本和产品哈希。此适配器只读取冻结产物，不重新运行检测或分组。
"""
    text += "\n命名兼容：TDA 与 STP 是当前方法名称；旧版产品的 `octa_*`、`cstr_*` 字段及图层 ID 保持不变，分别对应 TDA 与 STP。\n"
    (out/"README.md").write_text(text,encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tda-reports", "--octa-reports", dest="tda_reports", type=Path,default=RUN/"octa/reports.parquet")
    parser.add_argument("--tda-universe", "--octa-universe", dest="tda_universe", type=Path,default=RUN/"octa/stand_universe.parquet")
    parser.add_argument("--stp-reports", "--cstr-reports", dest="stp_reports", type=Path,required=True)
    parser.add_argument("--completion-receipt",type=Path,action="append",required=True)
    parser.add_argument("--out",type=Path,default=RUN/"products")
    parser.add_argument("--no-figures",action="store_true")
    args=parser.parse_args()
    result=publish_products(args.tda_reports,args.tda_universe,args.stp_reports,args.completion_receipt,args.out,not args.no_figures)
    print(json.dumps({key:result[key] for key in ("passed","inventory_stands","supported_stands","unsupported_stands","n_reports")},ensure_ascii=False))
