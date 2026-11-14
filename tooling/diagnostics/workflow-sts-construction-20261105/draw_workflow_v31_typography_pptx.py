"""Unify typography and use black MGRS boundaries in the user's edited layout."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, Collection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Ellipse, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.text import Text, Annotation
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pypdf import PdfReader

import draw_workflow_v28_archive_pptx as source


ROOT = source.ROOT
OUT = ROOT / "runs/figures/workflow_sts_construction_20261109_v31_typography_pptx"
INPUT = ROOT / "runs/figures/workflow_sts_construction_20261108_v29_ppt_layout/input/edited_workflow_v28_snapshot.pptx"
TITLE_PT, BODY_PT, TICK_PT, SUBSCRIPT_PT = 10., 7.5, 6.5, 6.
MODULE_ARTISTS = {}


def main():
    MODULE_ARTISTS.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    prs = Presentation(INPUT)
    page = np.array([prs.slide_width, prs.slide_height])/36000
    manifest = json.loads((source.OUT / "pptx_manifest.json").read_text(encoding="utf-8"))
    assets = {hashlib.sha256(p.read_bytes()).hexdigest(): p.stem
              for p in (source.OUT / "pptx_assets").glob("*.svg")}
    leaves = list(flatten(prs.slides[0].shapes, Affine2D()))
    for shape, parent in leaves:
        assert shape.rotation == 0, shape.name
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            assert picture_key(shape, assets) in {"footprints", "cube", "sts", "legend"}
    inputs = [INPUT, source.BASE / "pixel_association.csv", source.BASE / "VV_monthly_dB.csv",
              source.GEOMETRY, Path(source.__file__), source.OUT / "pptx_manifest.json"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    pixels = pd.read_csv(source.BASE / "pixel_association.csv").sort_values("pixel_number_in_stand").reset_index(drop=True)
    raw = pd.read_csv(source.BASE / "VV_monthly_dB.csv").set_index("unit_id").loc[pixels.unit_id]
    calendar = pd.period_range("2016-10", "2026-06", freq="M").astype(str).tolist()
    values = raw[calendar].to_numpy()
    xy = np.column_stack([pixels.x-660390., 9882450.-pixels.y])
    geometry = json.loads(source.GEOMETRY.read_text(encoding="utf-8"))
    assert values.shape == (106, 117) and np.isfinite(values).all()
    plt.rcParams.update({"font.family": "Arial", "font.size": BODY_PT, "text.color": source.INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "sts-ppt-layout-v31"})
    fig = plt.figure(figsize=page/25.4, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, page[0]), ylim=(0, page[1]), aspect="equal")
    ax.set_axis_off()
    records = []
    for index, (shape, parent) in enumerate(leaves):
        z = 100*index
        corners = parent.transform(np.array([[shape.left, shape.top], [shape.left+shape.width, shape.top+shape.height]])/36000)
        left, top = corners[0]
        width, height = corners[1]-corners[0]
        record = {"shape_id": shape.shape_id, "name": shape.name,
                  "box_ppt_mm": [float(left), float(top), float(width), float(height)]}
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            key = picture_key(shape, assets)
            box = np.array(manifest["module_bounds_inches"][key])*25.4
            crop = np.array([shape.crop_left, shape.crop_top, shape.crop_right, shape.crop_bottom])
            full_width, full_height = box[2:]-box[:2]
            cropped = box+np.array([crop[0]*full_width, crop[3]*full_height,
                                   -crop[2]*full_width, -crop[1]*full_height])
            sx, sy = np.array([width, height])/(cropped[2:]-cropped[:2])
            mapper = (Affine2D().translate(-cropped[0], -cropped[1]).scale(sx, sy)
                      .translate(left, page[1]-top-height))
            record.update(module=key, crop_fractions=crop.tolist(), scale_xy=[float(sx), float(sy)])
            if key == "legend":
                original_bar = np.array([[68., 7.5], [97., 9.1]])
                target = mapper.transform(original_bar)
                cax = fig.add_axes([target[0, 0]/page[0], target[0, 1]/page[1],
                                    (target[1, 0]-target[0, 0])/page[0], (target[1, 1]-target[0, 1])/page[1]])
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=source.NORM, cmap=source.CMAP), cax=cax,
                                  orientation="horizontal", ticks=[-20, -10, 0])
                cb.solids.set_rasterized(False); cb.solids.set_edgecolor("face")
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=TICK_PT, length=1.8, width=.5, pad=2)
                cb.ax.set_title("VV backscatter (dB)", fontsize=BODY_PT, loc="center", pad=4)
            else:
                before = set(ax.get_children())
                if key == "footprints":
                    source.footprints(ax, geometry)
                elif key == "cube":
                    source.cube(ax, xy, raw, False)
                    # This nearest corner is in front of the lower faces, unlike the rear guides.
                    for gap, (lower, upper) in enumerate(zip(source.HEIGHTS[:-1], source.HEIGHTS[1:])):
                        ends = np.array([source.project(source.EXTENT[3], source.CUBE_ORIGIN,
                                                       source.CUBE_SCALE, h) for h in [lower, upper]])
                        ends[:, 1] += [.12, -.12]
                        guide, = ax.plot(ends[:, 0], ends[:, 1], color=source.MUTED,
                                         lw=.4, ls=(0, (2.5, 3)), zorder=28)
                        guide.set_gid(f"cube_front_guide_{gap}")
                else:
                    source.sts(ax, xy, values, False)
                    for artist in list(ax.get_children()):
                        if artist in before:
                            continue
                        if (isinstance(artist, Line2D) and len(artist.get_xdata()) == 2
                                and np.allclose(artist.get_xdata(), source.STS_ORIGIN[0]
                                                + np.array([-.6, .6])*source.PANEL_SCALE)):
                            artist.remove()
                        elif isinstance(artist, Text) and artist.get_text() in {"2016", "2026"}:
                            year = artist.get_text()
                            y = source.STS_ORIGIN[1] if year == "2016" else source.CURVE_TOP
                            artist.set_y(y)
                            artist.set_va("bottom" if year == "2016" else "top")
                            artist.set_gid(f"time_axis_year_{year}")
                    time_endpoints = mapper.transform([source.STS_ORIGIN,
                                                       [source.STS_ORIGIN[0], source.CURVE_TOP]])
                children = [a for a in ax.get_children() if a not in before]
                clip = Rectangle((left, page[1]-top-height), width, height, transform=ax.transData)
                for artist in children:
                    transform_artist(artist, mapper, ax, np.sqrt(sx*sy))
                    if isinstance(artist, Text):
                        artist.set_fontsize(SUBSCRIPT_PT if key == "cube" and artist.get_fontsize() < 7 else BODY_PT)
                    if (key == "footprints" and isinstance(artist, Patch)
                            and np.allclose(artist.get_edgecolor(), to_rgba(source.TILE))):
                        artist.set_edgecolor("black")
                    artist.set_zorder(z+artist.get_zorder())
                    artist.set_clip_on(True)
                    artist.set_clip_path(clip)
                    if key == "footprints" and isinstance(artist, Text):
                        artist.set_clip_on(False)
                MODULE_ARTISTS[key] = children
            records.append(record)
            continue
        draw_native_shape(ax, shape, parent, page[1], z)
        if shape.has_text_frame and shape.text:
            draw_text(ax, shape, parent, page[1], z+1)
            record["text"] = shape.text
        records.append(record)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    year_bounds = {}
    for artist in ax.texts:
        if artist.get_gid() in {"time_axis_year_2016", "time_axis_year_2026"}:
            bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
            year_bounds[artist.get_text()] = [bounds.y0, bounds.y1]
    assert np.isclose(year_bounds["2016"][0], time_endpoints[0, 1])
    assert np.isclose(year_bounds["2026"][1], time_endpoints[1, 1])
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"workflow_sts_construction.{extension}", dpi=600, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else
                             ({"Date": None} if extension == "svg" else None))
    fig.savefig(OUT / "print_size_preview.png", dpi=110, facecolor="white")
    plt.close(fig)
    subprocess.run([str(source.PDFTOPPM), "-scale-to", "2300", "-singlefile", "-png",
                    str(OUT / "workflow_sts_construction.pdf"), str(OUT / "preview")], check=True)
    pdf = PdfReader(OUT / "workflow_sts_construction.pdf")
    assert len(pdf.pages[0].images) == 0
    svg_ids = [node.get("id") for node in ET.parse(OUT / "workflow_sts_construction.svg").iter() if node.get("id")]
    assert all(svg_ids.count(f"pixel_label_{i}") == 1 for i in range(1, 107))
    assert all(svg_ids.count(f"VV_trajectory_{i}") == 1 for i in source.SELECTED)
    assert all(svg_ids.count(f"cube_front_guide_{i}") == 1 for i in range(4))
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT / "layout_from_ppt.json").write_text(json.dumps({"page_mm": page.tolist(), "objects": records}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({
        "input_snapshot": str(INPUT), "input_sha256": hashes, "inputs_unchanged": True,
        "source_generator": str(Path(__file__)), "page_mm": page.tolist(),
        "data_redrawn_from_source": True, "ppt_contains_original_svg_modules": True,
        "members": 106, "curve_months": 117, "pdf_raster_images": 0,
        "typography_pt": {"title": TITLE_PT, "body_axes_years_legend_title": BODY_PT,
                          "VV_colourbar_ticks": TICK_PT, "slice_subscripts": SUBSCRIPT_PT,
                          "projected_cell_numbers": "original uniform Arial glyph scale retained"},
        "MGRS_boundary_colour": "#000000",
        "time_axis_endpoints_mm": time_endpoints.tolist(), "year_bounds_mm": year_bounds,
        "refinements": ["four visible front-corner guide segments", "no T-axis cross ticks",
                        "2016 text bottom aligned with XYT origin", "2026 text top aligned with T arrow tip"],
        "changes_reproduced": ["map and its legend reduced", "shared VV colourbar moved",
                               "cube subtitle cropped out", "B-C arrow replaced by brace",
                               "extraction caption moved", "c title moved", "Archive arrow extended behind cylinder"]
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "README_ZH.md").write_text("""# 字号统一与MGRS黑色边框（v31）

2026-11-09。保留v30布局，所有文字采用Arial，标题10 pt，说明、Archive和burst、XYZ/时间轴、年份、图例标题均7.5 pt；VV色标数字缩为6.5 pt，切片时间下标6 pt。106个倾斜编号继续采用原来一致且能容纳的矢量字形尺寸。MGRS边框和对应图例方框改黑色，burst仍为蓝色。

依据用户修改后的第1页，保留地图及其图例缩放、VV色标位置、b顶部裁剪、右尖大括号、说明和c标题的新位置，以及藏在Archive后方的长箭头。

- b补上朝向读者的最近角点连线，分成四段连接相邻切片；最近角点在下层切片前方。背部虚线仍按原来的面片遮挡处理。
- c删除T轴两处横刻线，2016文字包围框的底部与XYT原点对齐，2026文字包围框的顶部与T箭头端点对齐。
- 保留用户PPT的模块位置及画布。c下方的留白用于像元与曲线的对应关系，不额外添加装饰。
- T轴标明整个记录覆盖的年份，属于示意布局。四个上方曲线框各自包含完整117个月；曲线框下方细线表示像元对应关系，不表示额外观测月份。

- 格网、31个burst、106个编号、四条117个月曲线均由既有Python绘图函数和原始CSV/几何数据重绘。
- 从PPT快照读取位置、尺寸、裁剪比例、文字、原生路径和对象叠放顺序。未将PPT截图嵌入成图。
- 原始用户PPT保持不变；读取v29中的 `input/edited_workflow_v28_snapshot.pptx` 固定快照。
- 画布约190.015 × 82.021 mm，保留PowerPoint保存的实际尺寸；PDF/SVG保持矢量，PNG为600 dpi。
- `preview.png` 是Python生成PDF的实际渲染。
- `layout_from_ppt.json` 记录各对象的PPT坐标、缩放和裁剪，`summary.json` 记录输入哈希。

## 复现

在算法仓库目录执行：

```powershell
python -X utf8 'tooling/diagnostics/workflow-sts-construction-20261105/draw_workflow_v31_typography_pptx.py'
```

复现绘图无需启动PowerPoint；它只读取固定PPT快照及仓库中已有数据。没有修改TeX。
""", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "page_mm": page.tolist(), "objects": len(records)}))


def flatten(shapes, parent):
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            x = shape._element.grpSpPr.xfrm
            offset = np.array([x.off.x, x.off.y])/36000
            child = np.array([x.chOff.x, x.chOff.y])/36000
            scale = np.array([x.ext.cx/x.chExt.cx, x.ext.cy/x.chExt.cy])
            group = Affine2D().translate(*(-child)).scale(*scale).translate(*offset)+parent
            yield from flatten(shape.shapes, group)
        else:
            yield shape, parent


def picture_key(shape, assets):
    svg = next(node for node in shape._element.iter() if node.tag.endswith("}svgBlip"))
    blob = shape.part.related_part(svg.get(qn("r:embed"))).blob
    return assets[hashlib.sha256(blob).hexdigest()]


def transform_artist(artist, mapper, ax, scale):
    # Remove the live data-transform branch; a frozen inverse breaks when export changes DPI.
    if isinstance(artist, Annotation):
        artist.xy = mapper.transform(artist.xy)
        artist.set_position(mapper.transform(artist.get_position()))
        artist.set_fontsize(artist.get_fontsize()*scale)
        if artist.arrow_patch:
            artist.arrow_patch.set_linewidth(artist.arrow_patch.get_linewidth()*scale)
            artist.arrow_patch.set_mutation_scale(artist.arrow_patch.get_mutation_scale()*scale)
    elif isinstance(artist, Patch):
        artist.set_transform((artist.get_data_transform()-ax.transData)+mapper+ax.transData)
        artist.set_linewidth(artist.get_linewidth()*scale)
    elif isinstance(artist, PathCollection):
        artist.set_offset_transform((artist.get_offset_transform()-ax.transData)+mapper+ax.transData)
        artist.set_sizes(artist.get_sizes()*scale**2)
        artist.set_linewidths(artist.get_linewidths()*scale)
    else:
        artist.set_transform((artist.get_transform()-ax.transData)+mapper+ax.transData)
        if isinstance(artist, Text):
            artist.set_fontsize(artist.get_fontsize()*scale)
        elif isinstance(artist, Line2D):
            artist.set_linewidth(artist.get_linewidth()*scale)
        elif isinstance(artist, Collection):
            artist.set_linewidths(artist.get_linewidths()*scale)


def draw_native_shape(ax, shape, parent, page_height, z):
    properties = shape._element.spPr
    geom = properties.find(qn("a:custGeom"))
    preset = properties.find(qn("a:prstGeom"))
    if geom is None and (preset is None or preset.get("prst") != "ellipse"):
        return
    fill = properties.find(qn("a:solidFill"))
    face = "#"+fill.find(qn("a:srgbClr")).get("val") if fill is not None else "none"
    line = properties.find(qn("a:ln"))
    edge, width = "none", 0.
    if line is not None and line.find(qn("a:noFill")) is None:
        color = line.find(qn("a:solidFill"))
        edge = "#"+color.find(qn("a:srgbClr")).get("val")
        width = int(line.get("w", "12700"))/12700
    def points(coords):
        output = parent.transform(np.asarray(coords))
        output[:, 1] = page_height-output[:, 1]
        return output
    if geom is None:
        center = points([[shape.left/36000+shape.width/72000, shape.top/36000+shape.height/72000]])[0]
        scale = np.diag(parent.get_matrix())[:2]
        ax.add_patch(Ellipse(center, shape.width/36000*scale[0], shape.height/36000*scale[1], fc=face, ec=edge, lw=width, zorder=z))
        return
    for item in geom.find(qn("a:pathLst")):
        vertices, codes = [], []
        scale = np.array([shape.width/int(item.get("w")), shape.height/int(item.get("h"))])/36000
        offset = np.array([shape.left, shape.top])/36000
        for segment in item:
            kind = segment.tag.split("}")[-1]
            if kind == "close":
                vertices.append(vertices[0]); codes.append(MplPath.CLOSEPOLY)
            else:
                code = {"moveTo": MplPath.MOVETO, "lnTo": MplPath.LINETO,
                        "cubicBezTo": MplPath.CURVE4, "quadBezTo": MplPath.CURVE3}[kind]
                for pt in segment:
                    vertices.append(offset+np.array([int(pt.get("x")), int(pt.get("y"))])*scale)
                    codes.append(code)
        vertices = points(vertices)
        if line is not None and line.find(qn("a:tailEnd")) is not None:
            ax.add_patch(FancyArrowPatch(vertices[0], vertices[-1], arrowstyle="-|>",
                                         mutation_scale=8, shrinkA=0, shrinkB=0, color=edge, lw=width, zorder=z))
        else:
            patch = PathPatch(MplPath(vertices, codes), fc=face if item.get("fill") != "none" else "none",
                              ec=edge, lw=width, joinstyle="round", capstyle="round", zorder=z)
            patch.set_gid(f"ppt_shape_{shape.shape_id}")
            ax.add_patch(patch)


def draw_text(ax, shape, parent, page_height, z):
    tf = shape.text_frame
    first = next(run for p in tf.paragraphs for run in p.runs if run.text)
    font = first.font
    anchor = tf.paragraphs[0].alignment
    ha = "center" if str(anchor).startswith("CENTER") else "left"
    x = shape.left+(shape.width/2 if ha == "center" else tf.margin_left)
    y = shape.top+shape.height/2
    x, y = parent.transform([x/36000, y/36000])
    color = "#"+str(font.color.rgb)
    size = TITLE_PT if shape.text.startswith(("(a)", "(b)", "(c)")) else BODY_PT
    ax.text(x, page_height-y, shape.text, fontsize=size, family="Arial",
            weight="bold" if font.bold else "normal", color=color, ha=ha, va="center",
            linespacing=1.15, zorder=z)


if __name__ == "__main__":
    main()
