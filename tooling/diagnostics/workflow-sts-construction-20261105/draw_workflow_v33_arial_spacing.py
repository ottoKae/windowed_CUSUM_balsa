"""Use Arial exclusively and give the MGRS map more room above its legend."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Polygon, PathPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pypdf import PdfReader

import draw_workflow_v31_typography_pptx as prior


ROOT = prior.ROOT
OUT = ROOT / "runs/figures/workflow_sts_construction_20261109_v33_arial_spacing"
INPUT = ROOT / "runs/figures/workflow_sts_construction_20261109_v32_ppt_border/input/edited_workflow_v31_snapshot.pptx"
MAP_UP_MM = 2.


class CapturedFigure(Exception):
    def __init__(self, figure):
        self.figure = figure


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    prs = Presentation(INPUT)
    page = np.array([prs.slide_width, prs.slide_height])/36000
    manifest = json.loads((prior.OUT / "pptx_manifest.json").read_text(encoding="utf-8"))
    assets = {hashlib.sha256(p.read_bytes()).hexdigest(): p.stem
              for p in (prior.OUT / "pptx_assets").glob("*.svg")}
    save = Figure.savefig
    def capture(fig, *args, **kwargs):
        raise CapturedFigure(fig)
    Figure.savefig = capture
    try:
        prior.main()
    except CapturedFigure as result:
        fig = result.figure
    finally:
        Figure.savefig = save
    plt.rcParams["svg.hashsalt"] = "sts-ppt-border-v33"
    ax, colorbar = fig.axes
    module_artists = prior.MODULE_ARTISTS
    retained = {a for group in module_artists.values() for a in group}
    for artist in [*ax.patches, *ax.lines, *ax.collections, *ax.texts]:
        if artist not in retained:
            artist.remove()
    colorbar.title.set_visible(False)
    for label in colorbar.get_xticklabels():
        label.set_visible(False)
    borders, records = {}, []
    for index, (shape, parent) in enumerate(prior.flatten(prs.slides[0].shapes, Affine2D())):
        z = index*100
        corners = parent.transform(np.array([[shape.left, shape.top],
                                             [shape.left+shape.width, shape.top+shape.height]])/36000)
        left, top = corners[0]
        width, height = corners[1]-corners[0]
        record = {"id": shape.shape_id, "name": shape.name,
                  "ppt_box_mm": [float(left), float(top), float(width), float(height)],
                  "rotation": shape.rotation}
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            key = prior.picture_key(shape, assets)
            box = np.array(manifest["module_bounds_inches"][key])*25.4
            crop = np.array([shape.crop_left, shape.crop_top, shape.crop_right, shape.crop_bottom])
            w0, h0 = box[2:]-box[:2]
            cropped = box+np.array([crop[0]*w0, crop[3]*h0, -crop[2]*w0, -crop[1]*h0])
            sx, sy = np.array([width, height])/(cropped[2:]-cropped[:2])
            mapper = (Affine2D().translate(*(-cropped[:2])).scale(sx, sy)
                      .translate(left, page[1]-top-height))
            if key == "legend":
                b = colorbar.get_position()
                target = mapper.transform([[b.x0*page[0], b.y0*page[1]],
                                           [b.x1*page[0], b.y1*page[1]]])
                colorbar.set_position([target[0, 0]/page[0], target[0, 1]/page[1],
                                       (target[1, 0]-target[0, 0])/page[0],
                                       (target[1, 1]-target[0, 1])/page[1]])
            else:
                clip = Rectangle((left, page[1]-top-height), width, height, transform=ax.transData)
                for artist in module_artists[key]:
                    prior.transform_artist(artist, mapper, ax, np.sqrt(sx*sy))
                    artist.set_zorder(z+artist.get_zorder()%100)
                    artist.set_clip_on(True)
                    artist.set_clip_path(clip)
                    if key == "footprints" and not isinstance(artist, (Rectangle, Text)):
                        prior.transform_artist(artist, Affine2D().translate(0, MAP_UP_MM), ax, 1.)
                        artist.set_clip_path(Rectangle((left, page[1]-top-height+MAP_UP_MM),
                                                       width, height, transform=ax.transData))
            record.update(module=key, scale_xy=[float(sx), float(sy)], crop=crop.tolist())
        else:
            geometry = shape._element.spPr.find(qn("a:prstGeom"))
            if geometry is not None and geometry.get("prst") in {"roundRect", "rightArrow", "bentConnector3"}:
                patch = draw_preset(ax, shape, parent, page[1], z)
                if geometry.get("prst") == "roundRect":
                    borders[shape.shape_id] = {"ppt_box_mm": record["ppt_box_mm"],
                                               "bottom_y_mm": page[1]-top-height,
                                               "adj": 3843, "artist": patch}
            else:
                # The lowest rectangle is fully covered by the opaque panel border.
                prior.draw_native_shape(ax, shape, parent, page[1], z)
            if shape.has_text_frame and shape.text:
                draw_text(ax, shape, parent, page[1], z+1)
                record["text"] = shape.text
        records.append(record)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    add_step_circles(ax, renderer)
    legend = [a for a in module_artists["footprints"]
              if isinstance(a, Rectangle) or isinstance(a, Text)]
    y_label = [a for a in module_artists["sts"] if isinstance(a, Text) and a.get_text() == "Y"]
    adjustments = []
    for name, group, border_id in [("Burst / MGRS tile legend", legend, 35), ("Y axis mark", y_label, 36)]:
        before = Bbox.union([a.get_window_extent(renderer) for a in group]).transformed(ax.transData.inverted())
        dy = max(0., borders[border_id]["bottom_y_mm"]+.8-before.y0)
        for artist in group:
            prior.transform_artist(artist, Affine2D().translate(0, dy), ax, 1.)
        adjustments.append({"component": name, "up_mm": float(dy), "before_bounds_mm": before.extents.tolist(),
                            "panel_id": border_id})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for entry, group in zip(adjustments, [legend, y_label]):
        bounds = Bbox.union([a.get_window_extent(renderer) for a in group]).transformed(ax.transData.inverted())
        entry["after_bounds_mm"] = bounds.extents.tolist()
        assert bounds.y0 >= borders[entry["panel_id"]]["bottom_y_mm"]+.79
    for shape_id, border in borders.items():
        l, t, w, h = border["ppt_box_mm"]
        expected = [l, page[1]-t-h, l+w, page[1]-t]
        actual = border.pop("artist").get_path().get_extents().extents
        assert np.allclose(actual, expected, atol=1e-9)
        border["rendered_bounds_mm"] = actual.tolist()
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"workflow_sts_construction.{extension}", dpi=600, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else
                             ({"Date": None} if extension == "svg" else None))
    fig.savefig(OUT / "print_size_preview.png", dpi=110, facecolor="white")
    plt.close(fig)
    subprocess.run([str(prior.source.PDFTOPPM), "-scale-to", "2300", "-singlefile", "-png",
                    str(OUT / "workflow_sts_construction.pdf"), str(OUT / "preview")], check=True)
    pdf = PdfReader(OUT / "workflow_sts_construction.pdf")
    fonts = sorted({str(font.get_object()["/BaseFont"])
                    for font in pdf.pages[0]["/Resources"]["/Font"].get_object().values()})
    assert all("Arial" in font for font in fonts), fonts
    ids = [n.get("id", "") for n in ET.parse(OUT / "workflow_sts_construction.svg").iter()]
    assert len(pdf.pages[0].images) == 0
    assert all(ids.count(f"pixel_label_{i}") == 1 for i in range(1, 107))
    assert sum(i.startswith("local_burst_") for i in ids) == 31
    assert all(ids.count(f"step_circle_{i}") == 1 for i in [1, 2, 3])
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == digest
    report = {"input_snapshot": str(INPUT), "input_sha256": digest, "page_mm": page.tolist(),
              "fixed_borders": borders, "adjustments": adjustments, "objects": records,
              "source_generator": str(Path(__file__)), "inputs_unchanged": True,
              "fonts": fonts, "all_fonts_Arial": True, "MGRS_map_up_mm": MAP_UP_MM,
              "step_markers": "Arial digits enclosed by vector circles; no fallback font",
              "members": 106, "trajectories": [2, 39, 60, 96], "curve_months": 117,
              "pdf_raster_images": 0}
    (OUT / "layout_and_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_ZH.md").write_text(
        "# Arial统一与MGRS图例间距（v33）\n\n"
        "按用户修改后的v31 PPT第1页绘制。两个圆角边框的位置、尺寸与圆角参数直接读取PPT；"
        "保留新的标题、字号、竖排步骤、折线连接及模块位置。所有模块从既有Python绘图与原数据重绘，未使用截图。\n\n"
        "保留v32的图例及Y标记位置。a中的地图整体上移2 mm，肘线末端随地图左边中部上移2 mm；"
        "Archive、图例和两个外框不动，地图与图例之间增加2 mm间距。\n\n"
        "全部字体为Arial。①②③改为Arial数字1/2/3加矢量圆圈，避免混入微软雅黑等替代字体。"
        "PDF嵌入字体已逐项检查，仅含Arial、Arial Bold及Arial Italic。\n\n"
        "保留190.015×82.021 mm原画布；PDF/SVG为矢量，PNG为600 dpi。原PPT、旧图和TeX均未改。\n\n"
        "在算法仓库目录执行：\n```powershell\n"
        "python -X utf8 "
        "'tooling/diagnostics/workflow-sts-construction-20261105/draw_workflow_v33_arial_spacing.py'\n```\n",
        encoding="utf-8")
    print(json.dumps({"output": str(OUT), "adjustments": adjustments, "fixed_borders": borders}))


def draw_text(ax, shape, parent, page_height, z):
    tf = shape.text_frame
    first = next(run for p in tf.paragraphs for run in p.runs if run.text)
    font = first.font
    ha = "center" if str(tf.paragraphs[0].alignment).startswith("CENTER") else "left"
    box = np.array([shape.left, shape.top, shape.width, shape.height])/36000
    point = np.array([box[0]+(box[2]/2 if ha == "center" else tf.margin_left/36000), box[1]+box[3]/2])
    rotation = Affine2D().rotate_deg_around(*(box[:2]+box[2:]/2), shape.rotation)
    x, y = (rotation+parent).transform(point)
    colour = "#"+str(font.color.rgb) if font.color.type is not None else "black"
    label = shape.text.translate(str.maketrans({"①": "1", "②": "2", "③": "3"}))
    if shape.text[:1] in {"①", "②", "③"}:
        label = label[0]+" "+label[1:]
    text = ax.text(x, page_height-y, label, fontsize=font.size.pt, family="Arial",
                   weight="bold" if font.bold else "normal", color=colour,
                   ha=ha, va="center", linespacing=1., rotation=-shape.rotation,
                   rotation_mode="anchor", zorder=z)
    text.set_gid(f"ppt_text_{shape.shape_id}")


def add_step_circles(ax, renderer):
    for text in ax.texts:
        if text.get_gid() not in {"ppt_text_30", "ppt_text_69", "ppt_text_32"}:
            continue
        digit = text.get_text()[0]
        # Match the first glyph baseline after the existing 90-degree text rotation.
        line = text._get_layout(renderer)[1][0]
        baseline = text.get_transform().transform(text.get_position())+np.array(line[2:])
        baseline = ax.transData.inverted().transform(baseline)
        bounds = TextPath((0, 0), digit, size=text.get_fontsize(),
                          prop=FontProperties(family="Arial")).get_extents()
        glyph_center = (bounds.min+bounds.max)/2*25.4/72
        center = baseline+Affine2D().rotate_deg(text.get_rotation()).transform(glyph_center)
        circle = Circle(center, text.get_fontsize()*.48*25.4/72, fill=False,
                        ec=text.get_color(), lw=.45, zorder=text.get_zorder())
        circle.set_gid(f"step_circle_{digit}")
        ax.add_patch(circle)


def draw_preset(ax, shape, parent, page_height, z):
    sp = shape._element.spPr
    preset = sp.find(qn("a:prstGeom"))
    kind = preset.get("prst")
    l, t, w, h = np.array([shape.left, shape.top, shape.width, shape.height])/36000
    line = sp.find(qn("a:ln"))
    width = int(line.get("w", "9525"))/12700
    def mapped(points):
        points = parent.transform(np.array(points)+[l, t])
        points[:, 1] = page_height-points[:, 1]
        return points
    if kind == "roundRect":
        adjustment = int(preset.find(qn("a:avLst"))[0].get("fmla").split()[-1])
        r, k = min(w, h)*adjustment/100000, .5522847498307936
        vertices = [(r, 0), (w-r, 0), (w-r+k*r, 0), (w, r-k*r), (w, r),
                    (w, h-r), (w, h-r+k*r), (w-r+k*r, h), (w-r, h),
                    (r, h), (r-k*r, h), (0, h-r+k*r), (0, h-r),
                    (0, r), (0, r-k*r), (r-k*r, 0), (r, 0), (r, 0)]
        codes = [MplPath.MOVETO]+[MplPath.LINETO, *([MplPath.CURVE4]*3)]*4+[MplPath.CLOSEPOLY]
        patch = PathPatch(MplPath(mapped(vertices), codes), fc="white", ec="black", lw=width, zorder=z)
        patch.set_gid(f"fixed_border_{shape.shape_id}")
        ax.add_patch(patch)
        return patch
    if kind == "rightArrow":
        neck = w-min(w, h)/2
        points = [(0, h/4), (neck, h/4), (neck, 0), (w, h/2), (neck, h), (neck, 3*h/4), (0, 3*h/4)]
        patch = Polygon(mapped(points), fc="white", ec="black", lw=width, joinstyle="miter", zorder=z)
        ax.add_patch(patch)
        return patch
    adjustment = int(preset.find(qn("a:avLst"))[0].get("fmla").split()[-1])/100000
    points = np.array([[0, 0], [w*adjustment, 0], [w*adjustment, h], [w, h]])
    xfrm = sp.find(qn("a:xfrm"))
    if xfrm.get("flipV") == "1":
        points[:, 1] = h-points[:, 1]
    if xfrm.get("flipH") == "1":
        points[:, 0] = w-points[:, 0]
    points = Affine2D().rotate_deg_around(w/2, h/2, shape.rotation).transform(points)
    points = mapped(points)
    points[-2:, 1] += MAP_UP_MM
    ax.plot(points[:, 0], points[:, 1], color="black", lw=width, solid_joinstyle="miter", zorder=z)
    ax.add_patch(Circle(points[0], 3.5*25.4/72/2, fc="black", ec="none", zorder=z+1))
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", color="black",
                                 mutation_scale=10, lw=width, shrinkA=0, shrinkB=0, zorder=z+1))


if __name__ == "__main__":
    main()
