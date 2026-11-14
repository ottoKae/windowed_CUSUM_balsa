"""Give the Archive cards more vertical space; preserve the v27 map and STS."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Polygon, Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from pypdf import PdfReader
from shapely.geometry import box, shape, LineString, Polygon as SpatialPolygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "runs/figures/stand_10387_sts_association_20261102_v1"
GEOMETRY = Path(__file__).with_name("geometry_evidence_v6.json")
OUT = ROOT / "runs/figures/workflow_sts_construction_20261108_v28_archive_pptx"
PDFTOPPM = "pdftoppm"
SIZE = np.array([190., 82.])
INK, MUTED, GRID = "#263F4C", "#627780", "#DEE5E8"
BURST, TILE = "#577DA0", "#9B6F6A"
SELECTED = {2: "#54278F", 39: "#8054A1", 60: "#8C78B5", 96: "#A755AC"}
MONTHS = ["2016-10", "2016-11", "2016-12", "2026-05", "2026-06"]
HEIGHTS = np.array([0., 7., 14., 22., 30.])
CMAP, NORM = plt.get_cmap("Spectral"), Normalize(-20, 0)
CELL = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]])
EXTENT = np.array([[-90, -60], [510, -60], [510, 540], [-90, 540]])
# The same XY viewing angles as v18, at two different spatial magnifications.
BASIS = np.array([[1., 1.], [.65, -.35]])
CUBE_ORIGIN, CUBE_SCALE = np.array([64.25, 28.]), .035
PANEL_SCALE = .72
STS_ORIGIN, STS_SCALE = np.array([122., 29.]), .125*PANEL_SCALE
STS_INK = "#263D4C"
CURVE_BOTTOM, CURVE_TOP = 29.+(np.array([56., 88.])-31.)*PANEL_SCALE
CURVE_WIDTH = 11.2*PANEL_SCALE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outline", action="store_true", help="Render geometry and connections without backscatter values.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [BASE / "pixel_association.csv", BASE / "VV_monthly_dB.csv", GEOMETRY]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    pixels = pd.read_csv(inputs[0]).sort_values("pixel_number_in_stand").reset_index(drop=True)
    raw = pd.read_csv(inputs[1]).set_index("unit_id").loc[pixels.unit_id]
    calendar = pd.period_range("2016-10", "2026-06", freq="M").astype(str).tolist()
    values = raw[calendar].to_numpy()
    assert values.shape == (106, 117) and np.isfinite(values).all()
    assert pixels.pixel_number_in_stand.tolist() == list(range(1, 107))
    xy = np.column_stack([pixels.x-660390., 9882450.-pixels.y])
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "Arial", "font.size": 8., "text.color": INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "sts-three-stage-v28"})
    fig = plt.figure(figsize=SIZE/25.4, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, SIZE[0]), ylim=(0, SIZE[1]), aspect="equal")
    ax.set_axis_off()
    for key, x, title in [("a", 3, "RTC burst archive"), ("b", 59, "Monthly data cubes"),
                          ("c", 122, "STS in a stand")]:
        label = ax.text(x, 77.5, f"({key}) {title}", fontsize=10.2, weight="bold", va="baseline")
        label.set_gid(f"title_{key}")

    archive(ax)
    footprint_ids = footprints(ax, geometry)
    arrow(ax, (22, 50.5), (22, 47.8))
    ax.text(25, 49.15, "Tile + track", fontsize=8, va="center")
    cube(ax, xy, raw, args.outline)
    sts(ax, xy, values, args.outline)

    arrow(ax, (41, 32), (58, 32))
    ax.text(49.5, 36, "Monthly\ncompositing", fontsize=8, ha="center", va="bottom", linespacing=1.15)
    arrow(ax, (93, 27), (118, 27))
    ax.text(109, 23.5, "Pixelwise\nextraction", fontsize=8, ha="center", va="top", linespacing=1.15)

    cax = fig.add_axes(np.array([68., 7.5, 29., 1.6])/SIZE[[0, 1, 0, 1]])
    if not args.outline:
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                          orientation="horizontal", ticks=[-20, -10, 0])
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, length=1.8, width=.5, pad=2)
        cb.ax.set_title("VV backscatter (dB)", fontsize=8, loc="center", pad=4)
    else:
        cax.set_axis_off()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for a in fig.axes for t in a.texts if t.get_text()]
    text_boxes = [t.get_window_extent(renderer).transformed(ax.transData.inverted()) for t in texts]
    assert all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= SIZE[0] and b.y1 <= SIZE[1] for b in text_boxes)
    assert all(t.get_fontfamily() == ["Arial"] for t in texts)
    stem = "layout_outline" if args.outline else "workflow_sts_construction"
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else
                             ({"Date": None} if ext == "svg" else None))
    fig.savefig(OUT / f"{stem}_print_size.png", dpi=110, facecolor="white")
    plt.close(fig)
    subprocess.run([str(PDFTOPPM), "-scale-to", "2300", "-singlefile", "-png",
                    str(OUT / f"{stem}.pdf"), str(OUT / f"{stem}_preview")], check=True)
    pdf = PdfReader(OUT / f"{stem}.pdf")
    assert len(pdf.pages) == 1 and len(pdf.pages[0].images) == 0
    assert np.allclose([float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)], SIZE*72/25.4)
    ids = [e.get("id") for e in ET.parse(OUT / f"{stem}.svg").iter() if e.get("id")]
    assert len(footprint_ids) == 31
    assert all(ids.count("local_burst_"+name) == 1 for name in footprint_ids)
    for month in MONTHS:
        assert ids.count(f"slice_{month}") == 1
        for number in SELECTED:
            assert ids.count(f"cube_pixel_{number}_{month}") == 1
    assert sum(x.startswith("stand_cell_") for x in ids) == 106
    assert all(ids.count(f"pixel_label_{n}") == 1 for n in range(1, 107))
    for number in SELECTED:
        assert ids.count(f"time_series_border_{number}") == 1
        if not args.outline:
            assert ids.count(f"VV_strip_{number}") == 1
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    summary = {"date": "2026-11-08", "outline": args.outline, "page_mm": SIZE.tolist(),
               "group": "17MPU_ASCENDING_TK18", "crs": "EPSG:32717", "stand": 10387,
               "members": 106, "selected_pixels": list(SELECTED), "months": calendar,
               "slice_months": MONTHS, "slice_offsets_mm": HEIGHTS.tolist(),
               "slice_spacing": "Schematic; the omitted middle years are not drawn to elapsed-time scale.",
               "cube_excerpt_utm_bounds": [660300, 9881910, 660900, 9882510],
               "cube_excerpt_m": [600, 600], "native_cell_m": [30, 30],
               "example_burst_footprints": footprint_ids,
               "footprint_semantics": "All 31 indexed TK18 footprints in the original v23 170 km display extent; static coverage, not monthly completeness.",
               "footprint_display_utm_bounds": [570000, 9760000, 740000, 9930000],
               "footprint_panel_reference": "v23 draw_local_grid; identical footprint geometries, order, colours and line widths",
               "selected_pixel_values": {str(n): {m: float(raw.iloc[n-1][m]) for m in MONTHS} for n in SELECTED},
               "curve_mapping": "117 original samples, common -20..0 dB width and common time axis; no interpolation or smoothing.",
               "STS_XY_basis": (BASIS*STS_SCALE).tolist(),
               "STS_panel_source": "workflow_sts_construction_20261107_v23_cube/workflow_ecuador_sts.pdf",
               "STS_uniform_scale_from_v23": PANEL_SCALE,
               "STS_origin_mapping_mm": {"v23": [190, 31], "current": STS_ORIGIN.tolist()},
               "STS_restored_features": ["106 projected pixel IDs", "four VV strips", "four profiles", "v23 XYT axes and cell colours"],
               "profile_display": "Curves raised into a common time band with spatial leaders; lower unlabelled T segment is not extra observation time.",
               "font": "Arial", "pdf_raster_images": 0, "all_text_inside_page": True,
               "inputs_sha256": hashes, "inputs_unchanged": True,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / f"{stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if not args.outline:
        (OUT / "README_ZH.md").write_text("""# RTC archive → 月度格网 → 固定位置 STS（v28）

2026-11-08。190 × 82 mm；Archive圆柱主体高度从14 mm增至18.5 mm，卡片与省略号重新居中，连接箭头和Tile + track文字相应下移；其余沿用v27。
以0.72倍等比例重绘v23的c，保持其几何、相对线宽、字号、全部编号、四条曲线及四条VV色带。原点由(190,31)映射至(122,29) mm。
保留此前版本和用户修改的PPT；没有修改TeX、算法或原始数据。

## 构图与真实数据

- a：Archive与局部空间索引。恢复v23的170 × 170 km范围（EPSG:32717：570000—740000 / 9760000—9930000），全部31条TK18 burst footprint及周边MGRS tiles，按原IW1/IW2/IW3顺序叠放。蓝色表示burst，灰红色表示MGRS tile；当前范围中的31条不等于17MPU原组的16条，亦不代表某月观测齐全。移除上一版的局部窗口定位圈，与v23示意一致。
- b展示该tile–track内600 × 600 m局部窗口，绝不是整块109.8 km的MGRS tile；a与b的空间尺度不同。
- b：30 m格网，五个真实月份依次为2016-10、2016-11、2016-12、2026-05、2026-06；中间年份以省略号表示。切片间距是示意布局，不编码实际时间间隔。只有林分成员的VV值着色，外围白色格网表示未展示数据，而非零值或已确认缺测。
- c：Stand 10387的106个固定成员及全部编号均保留，编号按XY平面倾斜投影；2、39、60、96以紫色突出。四条VV曲线和配套色带保留117个月，2016-10—2026-06；无平滑、无插值，统一时间轴和-20—0 dB幅度尺度。
- b/c共用v18的XY视角，采用不同放大倍率。同色边框/像元/曲线表示相同成员。
- 四个曲线框抬到同一高度展示，细线连接原始像元；T轴下方无年份的空间连接段不表示额外的时间范围。
- a/b文字Arial 8—10.2 pt，时间下标6.5 pt；c轴及年份字号为v23的12 pt乘0.72，即8.64 pt。106个格内编号为Arial矢量轮廓，保持v23的统一字形缩放与平面投影。PDF和SVG保持矢量。

## 推荐英文图注（未写入TeX）

Construction of fixed-location spatial time series from Sentinel-1 RTC observations.
(a) RTC burst products are indexed by MGRS tile and track. The spatial-index example shows 31 overlapping TK18 burst footprints in blue across a 170 × 170 km display extent, together with MGRS tile boundaries in muted red and an emphasized tile boundary. Footprints illustrate the spatial index rather than monthly observation availability.
(b) Track-specific monthly backscatter is organized on a common spatial grid. Five example layers illustrate the first three and last two months of the record; intermediate months are omitted, and layer spacing is schematic. The displayed 600 × 600 m window is an excerpt of the tile, with backscatter shown only for stand-member cells.
(c) Fixed cell locations link the monthly observations to individual spatial time series within Stand 10387. All 106 member cells are numbered. Four highlighted members are traced through the data cube and shown as complete VV trajectories from October 2016 to June 2026, using common time and amplitude scales. The narrow colour strip beside each trajectory uses the same backscatter scale as the monthly layers. Purple colours identify individual cells.

## 文件和复现

- `workflow_sts_construction.pdf`：论文用矢量图。
- `workflow_sts_construction.svg`：可编辑矢量图。
- `workflow_sts_construction.png`：600 dpi PNG。
- `workflow_sts_construction_preview.png`：从PDF实际渲染的预览。
- `workflow_sts_construction_print_size.png`：按110 dpi输出的紧凑阅读检查图；实际屏幕毫米大小取决于显示器。
- `layout_outline.*`：结构检查稿，无后向散射数值/曲线。
- `workflow_sts_construction_summary.json`：输入哈希、真实月份、所选像元值与显示语义。

在算法仓库目录执行：

```powershell
python -X utf8 'tooling/diagnostics/workflow-sts-construction-20261105/draw_workflow_v28_archive_pptx.py'
```

同一命令末尾加 `--outline` 可以重新生成结构稿。脚本使用仓库中已有的CSV和几何缓存，离开仓库复制单个脚本不会包含这些数据。
""", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "stem": stem, "page_mm": SIZE.tolist()}))


def arrow(ax, start, end, color=INK, width=.85):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=width, color=color, mutation_scale=8,
                                shrinkA=0, shrinkB=0), zorder=30)


def project(points, origin, scale, height=0.):
    return np.asarray(points) @ (BASIS*scale).T + origin + [0, height]


def archive(ax):
    center, width, bottom, top = 22., 28., 53.5, 72.
    ax.add_patch(Ellipse((center, bottom), width, 4.5, fc="#EDF3F6", ec=BURST, lw=.7))
    ax.add_patch(Rectangle((center-width/2, bottom), width, top-bottom, fc="#EDF3F6", ec="none"))
    ax.plot([center-width/2]*2, [bottom, top], color=BURST, lw=.7)
    ax.plot([center+width/2]*2, [bottom, top], color=BURST, lw=.7)
    ax.add_patch(Ellipse((center, top), width, 4.5, fc="#DCE8EF", ec=BURST, lw=.7))
    ax.text(center, top, "Archive", ha="center", va="center", fontsize=8.5)
    for x in [15.5, 28.5]:
        for y in [56.8, 65.9]:
            vertices = np.array([[x-5.3, y-2.45], [x+5.3, y-1.45], [x+5.3, y+2.45], [x-5.3, y+1.45]])
            ax.add_patch(Polygon(vertices, fc="#DCE8EF", ec=BURST, lw=.45))
            ax.text(x, y, "burst", ha="center", va="center", fontsize=8)
        ax.scatter([x]*3, [60.45, 61.35, 62.25], s=.85, color=BURST, linewidths=0)


def footprints(ax, geometry):
    extent = box(570000, 9760000, 740000, 9930000)
    bounds = np.array(extent.bounds)
    origin, scale = np.array([4., 12.]), 35./170000
    def mapped(coords):
        return (np.asarray(coords)-bounds[:2])*scale+origin
    selected = sorted(geometry["bursts"], key=lambda b: (b["jpl_burst_id"].rsplit("-", 1)[1], b["jpl_burst_id"]))
    for b in selected:
        polygon = shape(b["geometry_utm"]).intersection(extent)
        subswath = b["jpl_burst_id"].rsplit("-", 1)[1]
        patch = Polygon(mapped(polygon.exterior.coords),
                        fc={"IW1": "#E6EFF4", "IW2": "#D1E3ED", "IW3": "#BCD5E3"}[subswath],
                        ec=BURST, lw=.35)
        patch.set_gid("local_burst_"+b["jpl_burst_id"])
        ax.add_patch(patch)
    for i, tile in enumerate(geometry["neighbor_tiles"]):
        polygon = shape(tile["geometry_utm"]).intersection(extent)
        if not polygon.is_empty:
            patch = Polygon(mapped(polygon.exterior.coords), fill=False, ec=TILE, lw=.55)
            patch.set_gid(f"local_mgrs_{i}")
            ax.add_patch(patch)
    target = shape(geometry["target_tile"]["geometry_utm"])
    target_patch = Polygon(mapped(target.exterior.coords), fill=False, ec=TILE, lw=1.15)
    target_patch.set_gid("selected_mgrs_tile")
    ax.add_patch(target_patch)
    for x, label, color in [(4, "Burst", BURST), (22, "MGRS tile", TILE)]:
        ax.add_patch(Rectangle((x, 5), 3, 2, fill=False, ec=color, lw=.75))
        ax.text(x+4.5, 6, label, va="center", fontsize=8)
    return [b["jpl_burst_id"] for b in selected]


def cube(ax, xy, raw, outline):
    grid = np.array([(x, y) for y in np.arange(-45, 540, 30) for x in np.arange(-75, 510, 30)])
    for layer, (height, month) in enumerate(zip(HEIGHTS, MONTHS)):
        z = 2+layer*5
        for point in grid:
            ax.add_patch(Polygon(project(point+CELL, CUBE_ORIGIN, CUBE_SCALE, height),
                                 fc="white", ec=GRID, lw=.2, zorder=z))
        for i, point in enumerate(xy):
            ax.add_patch(Polygon(project(point+CELL, CUBE_ORIGIN, CUBE_SCALE, height),
                                 fc="#DCE5E9" if outline else CMAP(NORM(raw.iloc[i][month])),
                                 ec=GRID, lw=.2, zorder=z+1))
        for number, color in SELECTED.items():
            p = Polygon(project(xy[number-1]+CELL, CUBE_ORIGIN, CUBE_SCALE, height),
                        fill=False, ec=color, lw=.7, zorder=z+2)
            p.set_gid(f"cube_pixel_{number}_{month}")
            ax.add_patch(p)
        p = Polygon(project(EXTENT, CUBE_ORIGIN, CUBE_SCALE, height), fill=False, ec=INK, lw=.65, zorder=z+2)
        p.set_gid(f"slice_{month}")
        ax.add_patch(p)
    faces = unary_union([SpatialPolygon(project(EXTENT, CUBE_ORIGIN, CUBE_SCALE, h)) for h in HEIGHTS])
    mask = faces.buffer(.12)
    # Only two light alignment guides; projected opaque faces hide every rear segment.
    for index in [0, 2]:
        line = LineString([project(EXTENT[index], CUBE_ORIGIN, CUBE_SCALE, h) for h in [0, HEIGHTS[-1]]])
        visible = line.difference(mask)
        parts = list(visible.geoms) if visible.geom_type == "MultiLineString" else [visible]
        for part in parts:
            if not part.is_empty and part.length > .2:
                coords = np.asarray(part.coords)
                ax.plot(coords[:, 0], coords[:, 1], color=MUTED, lw=.4, ls=(0, (2.5, 3)), zorder=28)
    ax.text(80, 72.3, "Tile excerpt · one track", fontsize=8, ha="center", color=MUTED)
    # Symbolic layer indices avoid suggesting that displayed slice gaps encode elapsed time.
    for h, suffix in [(0, "1"), (7, "2"), (14, "3"), (22, "T-1"), (30, "T")]:
        pt = project(EXTENT[2], CUBE_ORIGIN, CUBE_SCALE, h)
        ax.text(pt[0]+1.5, pt[1], "t", fontsize=8, fontstyle="italic", va="center", color=MUTED)
        ax.text(pt[0]+2.35, pt[1]-.65, suffix, fontsize=6.5, va="center", color=MUTED)
    omission_y = project(EXTENT[2], CUBE_ORIGIN, CUBE_SCALE, 18)[1]
    ax.scatter([104.1]*3, omission_y+np.array([-.75, 0., .75]), s=.6, color=MUTED, linewidths=0)


def sts(ax, xy, values, outline):
    s = PANEL_SCALE
    plane_basis = BASIS*STS_SCALE
    for i in np.argsort(xy[:, 1]):
        number = i+1
        p = Polygon(project(xy[i]+CELL, STS_ORIGIN, STS_SCALE),
                    fc=SELECTED.get(number, "#C8DCE7"), ec="white", lw=.45*s, zorder=2)
        p.set_gid(f"stand_cell_{i+1}")
        ax.add_patch(p)
        glyph = TextPath((0, 0), str(number), size=1,
                         prop=font_manager.FontProperties(family="Arial"))
        bounds = glyph.get_extents()
        local = Affine2D().translate(-bounds.x0-bounds.width/2, -bounds.y0-bounds.height/2).scale(15.)
        u, v = project(xy[i], STS_ORIGIN, STS_SCALE)
        plane = Affine2D.from_values(*plane_basis[:, 0], *(-plane_basis[:, 1]), u, v)
        label = PathPatch(glyph, transform=local+plane+ax.transData,
                          fc="white" if number in SELECTED else STS_INK, ec="none", zorder=7)
        label.set_gid(f"pixel_label_{number}")
        ax.add_patch(label)
    for end, label, offset in [(project([280, 0], STS_ORIGIN, STS_SCALE), "X", [0, 4.5]),
                               (project([0, 550], STS_ORIGIN, STS_SCALE), "Y", [4, -2]),
                               (np.array([STS_ORIGIN[0], CURVE_TOP]), "T", [3, 2.5])]:
        ax.annotate("", xy=end, xytext=STS_ORIGIN,
                    arrowprops=dict(arrowstyle="-|>", lw=1.1*s, color=STS_INK, shrinkA=0, shrinkB=0,
                                    mutation_scale=12*s), zorder=3)
        ax.text(*(end+np.asarray(offset)*s), label, fontsize=12*s, weight="bold",
                ha="center", va="center", color=STS_INK)
    for year, y in [("2016", CURVE_BOTTOM+.6*s), ("2026", CURVE_TOP-.6*s)]:
        ax.plot(STS_ORIGIN[0]+np.array([-.6, .6])*s, [y, y], color=STS_INK, lw=.75*s)
        ax.text(STS_ORIGIN[0]-3.3*s, y, year, fontsize=12*s, rotation=90,
                ha="center", va="center", color=STS_INK)
    for number, color in SELECTED.items():
        anchor = project(xy[number-1], STS_ORIGIN, STS_SCALE)
        x = anchor[0]
        cell_top = 15*STS_SCALE
        ax.plot([x, x], [anchor[1]+cell_top, CURVE_BOTTOM], color=color, lw=.5*s, alpha=.7, zorder=3)
        p = Rectangle((x, CURVE_BOTTOM), CURVE_WIDTH, CURVE_TOP-CURVE_BOTTOM,
                      fc="white", ec="#B4A6BC", lw=.55*s, zorder=4)
        p.set_gid(f"time_series_border_{number}")
        ax.add_patch(p)
        ax.plot([x, x], [CURVE_BOTTOM, CURVE_TOP], color=color, lw=s, solid_capstyle="butt", zorder=6)
        if not outline:
            t = np.linspace(CURVE_BOTTOM+.6*s, CURVE_TOP-.6*s, values.shape[1])
            edges = np.r_[CURVE_BOTTOM+.6*s, (t[:-1]+t[1:])/2, CURVE_TOP-.6*s]
            strips = [[(x+.4*s, a), (x+1.5*s, a), (x+1.5*s, b), (x+.4*s, b)]
                      for a, b in zip(edges[:-1], edges[1:])]
            band = PolyCollection(strips, fc=CMAP(NORM(values[number-1])), ec="none", zorder=5)
            band.set_gid(f"VV_strip_{number}")
            ax.add_collection(band)
            line, = ax.plot(x+2.3*s+NORM(values[number-1])*(CURVE_WIDTH-3*s), t,
                            color=color, lw=.75*s, zorder=6)
            line.set_gid(f"VV_trajectory_{number}")
        ax.scatter(x, anchor[1]+cell_top+.38*s, s=6*s*s, c=color,
                   edgecolors="white", linewidths=.4*s, zorder=8)


if __name__ == "__main__":
    main()
