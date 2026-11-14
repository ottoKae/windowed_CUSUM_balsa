"""Render the native objects of an editable stand workflow with Matplotlib."""
from pathlib import Path
import argparse
import colorsys
import hashlib
import json
import math
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from pptx import Presentation
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pypdf import PdfReader

from draw_technical_workflow import ROOT, PDFTOPPM

DEFAULT = ROOT / "runs/figures/stand_22736_technical_workflow_20261107_v3"
EMU_MM = 36000
PT_MM = 25.4 / 72


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", type=Path, default=DEFAULT / "real_stand_workflow_editable.pptx")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    assert len(prs.slides) == 1, "This figure renderer expects its single workflow slide."
    width, height = prs.slide_width / EMU_MM, prs.slide_height / EMU_MM
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=144, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()
    inventory = []
    master = prs.slides[0].slide_layout.slide_master
    theme = parse_xml(master.part.part_related_by(RT.THEME).blob)
    colors = {}
    for entry in theme.find(".//" + qn("a:clrScheme")):
        value = entry[0]
        colors[entry.tag.rsplit("}", 1)[-1]] = value.get("lastClr") or value.get("val")
    for alias, key in master.element.find(qn("p:clrMap")).attrib.items():
        colors[alias] = colors[key]
    line_widths = [float(line.get("w")) / 12700
                   for line in theme.find(".//" + qn("a:lnStyleLst"))]
    render_shapes(prs.slides[0].shapes, ax, renderer, (1, 1, 0, 0), inventory,
                  colors, line_widths)
    stem = args.out_dir / "real_stand_workflow_python"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=350)
    plt.close(fig)
    for suffix, pixels in (("preview", 2400), (f"{round(width)}mm", round(width/25.4*150))):
        subprocess.run([str(PDFTOPPM), "-scale-to-x", str(pixels), "-scale-to-y", "-1",
                        "-png", "-singlefile", str(stem.with_suffix(".pdf")),
                        str(stem) + "_" + suffix], check=True)
    pdf = PdfReader(stem.with_suffix(".pdf"))
    svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
    assert len(pdf.pages) == 1 and not pdf.pages[0].images and "<image" not in svg
    manifest = {
        "pptx_source": str(args.pptx.resolve()),
        "pptx_sha256": hashlib.sha256(args.pptx.read_bytes()).hexdigest(),
        "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canvas_mm": [width, height], "layout_source": "Native PowerPoint object positions and styles",
        "new_inference": False, "pdf_raster_images": 0, "svg_raster_images": 0,
        "objects": inventory,
    }
    (args.out_dir / "python_render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(stem.with_suffix(".pdf"))


def render_shapes(shapes, ax, renderer, transform, inventory, colors, line_widths):
    sx, sy, tx, ty = transform
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            xf = shape.element.grpSpPr.xfrm
            gx, gy = xf.ext.cx / xf.chExt.cx, xf.ext.cy / xf.chExt.cy
            child = (sx * gx, sy * gy,
                     tx + sx * (xf.off.x - gx * xf.chOff.x),
                     ty + sy * (xf.off.y - gy * xf.chOff.y))
            inventory.append({"name": shape.name, "kind": "group", "children": len(shape.shapes)})
            render_shapes(shape.shapes, ax, renderer, child, inventory, colors, line_widths)
            continue
        x, y = (tx + sx * shape.left) / EMU_MM, (ty + sy * shape.top) / EMU_MM
        w, h = sx * shape.width / EMU_MM, sy * shape.height / EMU_MM
        record = {"name": shape.name, "kind": str(shape.shape_type), "box_mm": [x, y, w, h]}
        inventory.append(record)
        if shape.has_text_frame and shape.text:
            record["text"] = shape.text
            render_text(shape, ax, renderer, x, y, w, h, math.sqrt(sx * sy), colors)
            continue
        props = shape.element.spPr
        line = props.find(qn("a:ln"))
        stroke = color_from(line, "#263D4C", colors) if line is not None else "none"
        line_ref = shape.element.find(".//" + qn("a:lnRef"))
        default_width = line_widths[int(line_ref.get("idx")) - 1] if line_ref is not None else .5
        lw = (float(line.get("w")) / 12700 if line is not None and line.get("w") is not None
              else default_width) * math.sqrt(sx * sy)
        dash = shape.line.dash_style
        linestyle = (0, (3, 2)) if dash in (MSO_LINE_DASH_STYLE.DASH, MSO_LINE_DASH_STYLE.DASH_DOT,
                                           MSO_LINE_DASH_STYLE.LONG_DASH) else "-"
        fill = color_from(props, "none", colors)
        common = dict(facecolor=fill, edgecolor=stroke, linewidth=lw, linestyle=linestyle)
        geom = props.find(qn("a:custGeom"))
        xf = props.find(qn("a:xfrm"))
        flip_x = xf is not None and xf.get("flipH") in ("1", "true")
        flip_y = xf is not None and xf.get("flipV") in ("1", "true")
        angle = shape.rotation
        rotate = Affine2D().rotate_deg_around(x + w / 2, y + h / 2, angle) + ax.transData
        if geom is not None:
            for path in geom.find(qn("a:pathLst")):
                # PowerPoint omits the zero path dimension on horizontal/vertical lines.
                pw, ph = float(path.get("w", "0")), float(path.get("h", "0"))
                points, codes = [], []
                for command in path:
                    tag = command.tag.rsplit("}", 1)[-1]
                    if tag == "close":
                        points.append(points[0]); codes.append(MplPath.CLOSEPOLY)
                        continue
                    if tag not in ("moveTo", "lnTo"):
                        raise ValueError(f"Unsupported path command {tag} in {shape.name}")
                    p = command[0]
                    px, py = float(p.get("x")) / (pw or 1), float(p.get("y")) / (ph or 1)
                    points.append((x + w * (1 - px if flip_x else px),
                                   y + h * (1 - py if flip_y else py)))
                    codes.append(MplPath.MOVETO if tag == "moveTo" else MplPath.LINETO)
                style = dict(common)
                if path.get("fill") == "none": style["facecolor"] = "none"
                if path.get("stroke") in ("0", "false"): style["edgecolor"] = "none"
                patch = PathPatch(MplPath(points, codes), transform=rotate,
                                  joinstyle="round", capstyle="butt", **style)
                patch.set_gid(f"ppt_{shape.shape_id}"); ax.add_patch(patch)
        elif shape.shape_type == MSO_SHAPE_TYPE.LINE:
            x0, x1 = (x + w, x) if flip_x else (x, x + w)
            y0, y1 = (y + h, y) if flip_y else (y, y + h)
            ax.plot([x0, x1], [y0, y1], color=stroke, lw=lw, ls=linestyle,
                    transform=rotate, solid_capstyle="butt", gid=f"ppt_{shape.shape_id}")
        else:
            preset = props.find(qn("a:prstGeom"))
            kind = preset.get("prst") if preset is not None else "rect"
            if kind == "rect":
                patch = Rectangle((x, y), w, h, transform=rotate,
                                  antialiased=fill == "none" or stroke != "none", **common)
            elif kind == "roundRect":
                radius = min(w, h) * shape.adjustments[0]
                patch = FancyBboxPatch((x, y), w, h,
                                      boxstyle=f"round,pad=0,rounding_size={radius}",
                                      transform=rotate, **common)
            elif kind == "ellipse": patch = Ellipse((x + w/2, y + h/2), w, h, transform=rotate, **common)
            elif kind == "rightArrow":
                shaft_inset = h * (1 - shape.adjustments[0]) / 2
                head = min(w, h) * shape.adjustments[1]
                local = [(0, shaft_inset), (w-head, shaft_inset), (w-head, 0),
                         (w, h/2), (w-head, h), (w-head, h-shaft_inset),
                         (0, h-shaft_inset), (0, shaft_inset)]
                points = [(x + (w-px if flip_x else px), y + (h-py if flip_y else py))
                          for px, py in local]
                patch = PathPatch(MplPath(points, [1, 2, 2, 2, 2, 2, 2, 79]),
                                  transform=rotate, joinstyle="miter", **common)
            elif kind == "diamond":
                points = [(x + w/2, y), (x + w, y + h/2), (x + w/2, y + h),
                          (x, y + h/2), (x + w/2, y)]
                patch = PathPatch(MplPath(points, [1, 2, 2, 2, 79]), transform=rotate, **common)
            else: raise ValueError(f"Unsupported native shape {kind}: {shape.name}")
            patch.set_gid(f"ppt_{shape.shape_id}"); ax.add_patch(patch)


def color_from(element, default, colors):
    if element is None: return default
    if element.find(qn("a:noFill")) is not None: return "none"
    solid = element.find(qn("a:solidFill"))
    if solid is None: return default
    value = solid[0]
    kind = value.tag.rsplit("}", 1)[-1]
    if kind == "srgbClr": hex_color = value.get("val")
    elif kind == "schemeClr": hex_color = colors[value.get("val")]
    else: raise ValueError(f"Unsupported color type {kind}")
    if not len(value): return "#" + hex_color
    rgb = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    opacity = 1.0
    for adjustment in value:
        kind = adjustment.tag.rsplit("}", 1)[-1]
        amount = float(adjustment.get("val")) / 100000
        if kind == "lumMod": lightness *= amount
        elif kind == "lumOff": lightness += amount
        elif kind == "alpha": opacity = amount
        else: raise ValueError(f"Unsupported color adjustment {kind}")
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    if opacity != 1.0: return (*rgb, opacity)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def render_text(shape, ax, renderer, x, y, w, h, scale, colors):
    frame = shape.text_frame
    paragraphs = frame.paragraphs
    nonempty = [p for p in paragraphs if p.text]
    assert len(nonempty) == 1, f"Use one line per editable textbox: {shape.name}"
    paragraph = nonempty[0]
    parts = []
    for run in paragraph.runs:
        font = run.font
        size = (font.size.pt if font.size is not None else 7) * scale
        prop = FontProperties(family=font.name or "Arial", size=size,
                              weight="bold" if font.bold else "normal",
                              style="italic" if font.italic else "normal")
        advance = renderer.get_text_width_height_descent(run.text, prop, False)[0] * 25.4 / ax.figure.dpi
        rpr = run._r.rPr
        baseline = float(rpr.get("baseline", "0")) / 100000 if rpr is not None else 0
        parts.append((run.text, prop, size, advance, baseline, color_from(rpr, "#263D4C", colors)))
    total = sum(p[3] for p in parts)
    margin_l, margin_r = frame.margin_left / EMU_MM * scale, frame.margin_right / EMU_MM * scale
    align = int(paragraph.alignment) if paragraph.alignment is not None else 1
    left = x + (w - total)/2 if align == 2 else x + w - margin_r - total if align == 3 else x + margin_l
    max_size = max(p[2] for p in parts)
    base_y = y + h/2 + max_size * PT_MM * .34
    # PowerPoint rotation is clockwise; data y also increases downward.
    angle = math.radians(shape.rotation)
    cx, cy = x + w/2, y + h/2
    for index, (value, prop, size, advance, baseline, color) in enumerate(parts):
        py = base_y - baseline * size * PT_MM
        dx, dy = left - cx, py - cy
        px = cx + math.cos(angle)*dx - math.sin(angle)*dy
        py = cy + math.sin(angle)*dx + math.cos(angle)*dy
        ax.text(px, py, value, fontproperties=prop, color=color,
                ha="left", va="baseline", rotation=-shape.rotation, rotation_mode="anchor",
                gid=f"ppt_{shape.shape_id}_run{index}")
        left += advance


if __name__ == "__main__":
    main()
