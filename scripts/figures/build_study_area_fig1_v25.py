"""Replace field photos with the frozen coverage map; remove the Stand 39 locator."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from build_study_area_fig1_v7 import group_span
from update_study_area_validation_panel_a import digest, SVG_RECT

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/figures/fig1_harvest_concept_v24_full_mgrs_20261029"
COVERAGE = ROOT / "runs/figures/s1_cumulative_coverage_20261029_v25"
OUT = ROOT / "runs/figures/fig1_harvest_concept_v25_coverage_panel_20261029"
STEM = "study_area_fig1_v25"
PT = 72/25.4
NS = {"s": "http://www.w3.org/2000/svg"}
COLORS = {91: "#552A7A", 18: "#075E65", 120: "#872D58", 47: "#30343B"}


def remove_group(svg, name):
    a, b = group_span(svg, name)
    return svg[:a]+svg[b:]


def svg_body(svg, prefix):
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', svg)
    svg = re.sub(r'url\(#([^\)]+)\)', lambda m: f'url(#{prefix}{m[1]})', svg)
    svg = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{prefix}{m[1]}"', svg)
    a = svg.index(">", svg.index("<svg"))+1
    return svg[a:svg.rfind("</svg>")]


def key_overlay(page_rect, panel, map_rect, source, out, *, title="No.Observations", swatch_width=14, swatch_linewidth=.65, centered_swatch=False, compact=False):
    """Keep the original colour ramp, with labels readable at final printed size."""
    bar_clip = fitz.Rect(245.3, 359.65, 386.4, 366.5)
    ramp = np.array(Image.open(io.BytesIO(source[0].get_pixmap(dpi=600, clip=bar_clip, alpha=False).tobytes("png"))))
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6.5})
    fig = plt.figure(figsize=(page_rect.width/72, page_rect.height/72))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, page_rect.width)
    ax.set_ylim(page_rect.height, 0)
    ax.set_axis_off()
    ax.add_patch(Rectangle((panel.x0, map_rect.y1), panel.width, panel.y1-map_rect.y1, facecolor="white", edgecolor="none"))
    left, right = panel.x0+10, panel.x1-10
    y = map_rect.y1
    ax.text(left, y+9, title, fontsize=7, fontweight="bold", va="baseline")
    ax.imshow(ramp, extent=(left, right, y+19, y+13), origin="upper", interpolation="nearest", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((left, y+13), right-left, 6, fill=False, edgecolor="#777777", linewidth=.4, zorder=3))
    for value in range(0, 901, 100):
        x = left+(right-left)*value/900
        ax.plot([x, x], [y+19, y+21], color="#777777", linewidth=.4)
        ax.text(x, y+28, str(value), ha="center", va="baseline", fontsize=6.5)
    key_y, name_y = (35, 44) if compact else (39, 50)
    ax.text(left, y+key_y if centered_swatch else y+43, "Burst in", fontsize=6.5, fontweight="bold", va="center" if centered_swatch else "baseline")
    for x, family in zip(np.linspace(left+50, right-20, 4), COLORS, strict=True):
        height = swatch_width/4
        ax.add_patch(Rectangle((x-swatch_width/2, y+key_y-height/2 if centered_swatch else y+35), swatch_width, height, fill=False, edgecolor=COLORS[family], linewidth=swatch_linewidth))
        ax.text(x, y+name_y if centered_swatch else y+47, f"TK{family}", fontsize=6.5, ha="center", va="baseline")
    assert y+(46 if compact else 49) < panel.y1, "Legend does not fit below the undistorted map"
    ax.add_patch(Rectangle((panel.x0, panel.y0), panel.width, panel.height, fill=False, edgecolor="#333333", linewidth=.65))
    ax.text(panel.x0+3, panel.y0+11, "(b)", fontsize=11, fontweight="bold", va="baseline",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
    fig.savefig(out / "qa/panel_b_labels.pdf", transparent=True, dpi=600)
    fig.savefig(out / "qa/panel_b_labels.svg", transparent=True, dpi=600)
    plt.close(fig)


def main():
    pdf_source, svg_source = SOURCE / "study_area_fig1_v24.pdf", SOURCE / "study_area_fig1_v24.svg"
    cp = COVERAGE / "s1_cumulative_ascending_201601_202606.pdf"
    cs = cp.with_suffix(".svg")
    if (OUT / f"{STEM}.pdf").exists():
        raise FileExistsError("Existing export retained")
    protected = {p: digest(p) for p in [pdf_source, svg_source, cp, cs,
                 *(ROOT / "manuscript_JAG").rglob("*.tex"), *(ROOT / "manuscript_JAG/figures").glob("*.pdf")]}
    (OUT / "qa").mkdir(parents=True, exist_ok=True)
    layout = json.loads((ROOT / "runs/figures/fig1_harvest_concept_v22_scale_boundary_20261028/QA.json").read_text())
    original = fitz.open(pdf_source)
    document = fitz.open(pdf_source)
    page = document[0]
    # These eight independent streams are exactly the locator frame, circles, leader and label.
    contents = page.get_contents()
    assert len(contents) == 11 and b"Fig1V8ArialBold" in document.xref_stream(contents[8])
    for xref in contents[1:9]:
        document.update_stream(xref, b"")
    for record in layout["panels"][:2]:
        x, y, w, h = record["rectangle_mm"]
        rect = fitz.Rect(x, y, x+w, y+h)*PT
        page.add_redact_annot(rect+(-.5, -.5, .5, .5), fill=(1, 1, 1), cross_out=False)
    page.apply_redactions(images=2, graphics=2, text=0)

    sx, sy, sw, sh = SVG_RECT
    translation = layout["national_translation_y_pt"]
    a_rect = fitz.Rect(sx+.011875, sy+translation+.011875, sx+sw+.011875, sy+sh+translation+.011875)
    g = next(p["rectangle_mm"] for p in layout["panels"] if p["panel"] == "g")
    panel = fitz.Rect(a_rect.x1+layout["clear_column_gap_pt"]+.65, a_rect.y0, (g[0]+g[2])*PT, a_rect.y1)
    coverage_svg = cs.read_text(encoding="utf-8")
    root = ET.fromstring(coverage_svg)
    clip = root.find('.//s:clipPath[@id="p4fc5b4d1eb"]/s:rect', NS)
    x, y, w, h = (float(clip.get(k)) for k in ("x", "y", "width", "height"))
    source_rect = fitz.Rect(x, y, x+w, y+h)
    scale = panel.width/source_rect.width
    map_rect = fitz.Rect(panel.x0, panel.y0, panel.x1, panel.y0+source_rect.height*scale)
    assert np.isclose(map_rect.width/source_rect.width, map_rect.height/source_rect.height)
    with fitz.open(cp) as coverage:
        key_overlay(page.rect, panel, map_rect, coverage, OUT)
        stream_id = coverage[0].get_contents()[0]
        data = coverage.xref_stream(stream_id)
        marker = b"Q q /A5 gs 1 g 1 g\n\n231.375118 42.995854 m"
        assert data.count(marker) == 1
        data = data[:data.index(marker)]+b"Q\n"
        assert len(re.findall(rb"\bq\b", data)) == len(re.findall(rb"\bQ\b", data))
        coverage.update_stream(stream_id, data)
        assert not coverage[0].get_text().strip()
        coverage.save(OUT / "qa/coverage_map_without_legend.pdf", deflate=True)
        page.show_pdf_page(map_rect, coverage, 0, clip=source_rect, keep_proportion=True)
    with fitz.open(OUT / "qa/panel_b_labels.pdf") as labels:
        page.show_pdf_page(page.rect, labels, 0)
    pdf = OUT / f"{STEM}.pdf"
    document.save(pdf, garbage=4, deflate=True, no_new_id=True)
    document.close()

    svg = svg_source.read_text(encoding="utf-8")
    for group in ("stand39_locator_v10", "v10_axes_1", "v10_axes_2"):
        svg = remove_group(svg, group)
    coverage_svg = remove_group(coverage_svg, "axes_2")
    crop = layout["page_top_crop_pt"]
    dx, dy = map_rect.x0-scale*source_rect.x0, map_rect.y0-scale*source_rect.y0
    map_body = svg_body(coverage_svg, "coverage_")
    labels = svg_body((OUT / "qa/panel_b_labels.svg").read_text(encoding="utf-8"), "coverage_label_")
    insert = (f'<g id="coverage_panel_b_v25" transform="translate(0 {crop:.9f})">'
              f'<defs><clipPath id="coverage_target"><rect x="{map_rect.x0}" y="{map_rect.y0}" width="{map_rect.width}" height="{map_rect.height}"/></clipPath></defs>'
              f'<g clip-path="url(#coverage_target)"><g transform="translate({dx:.9f} {dy:.9f}) scale({scale:.12f})">{map_body}</g></g>'
              f'{labels}</g>')
    svg = svg.replace("</svg>", insert+"</svg>")
    (OUT / f"{STEM}.svg").write_text(svg, encoding="utf-8")
    ET.fromstring(svg)

    with fitz.open(pdf) as current:
        assert current[0].rect == original[0].rect
        text = current[0].get_text().replace("\u00a0", " ")
        assert "Stand 39" not in text and "(c)" not in text and text.count("(b)") == 1
        assert all(text.count(f"({letter})") == 1 for letter in "adefghijk")
        assert text.count("250 m") == 8
        # Keep all eight lower panels identical at their final physical coordinates.
        lower_top = layout["panels"][2]["rectangle_mm"][1]*PT-1
        clip = fitz.Rect(0, lower_top, current[0].rect.width, current[0].rect.height)
        images = [d[0].get_pixmap(dpi=300, clip=clip, alpha=False) for d in (original, current)]
        lower_delta = np.abs(np.frombuffer(images[0].samples, np.uint8).astype(int)-np.frombuffer(images[1].samples, np.uint8).astype(int))
        # PDF redaction rewrites numeric operators; allow only tiny antialias differences.
        assert lower_delta.max() <= 2, "Lower image panels changed"
        lower_image_records = [[i for i in d[0].get_image_info(hashes=True) if i["bbox"][1] > lower_top] for d in (original, current)]
        assert len(lower_image_records[0]) == len(lower_image_records[1]) == 8
        for p, q in zip(*lower_image_records, strict=True):
            assert p["digest"] == q["digest"] and p["bbox"] == q["bbox"]
        def lower_text(d):
            return [(s["text"], s["font"], s["size"], s["bbox"]) for b in d[0].get_text("dict")["blocks"]
                    for line in b.get("lines", []) for s in line["spans"] if s["bbox"][1] > lower_top]
        assert lower_text(original) == lower_text(current)
        pixmaps = [d[0].get_pixmap(dpi=300, alpha=False) for d in (original, current)]
        arrays = [np.frombuffer(p.samples, np.uint8).reshape(p.height, p.width, p.n) for p in pixmaps]
        delta = np.max(np.abs(arrays[0].astype(int)-arrays[1].astype(int)), axis=2)
        # Only the small locator area and right-hand panel may change.
        for region in (fitz.Rect(81, 101, 141, 128), fitz.Rect(panel.x0-1, panel.y0-1, panel.x1+1, panel.y1+1)):
            x0, y0, x1, y1 = np.rint(np.array(region)*300/72).astype(int)
            delta[y0:y1, x0:x1] = 0
        assert delta.max() <= 2, "Unexpected edits outside requested regions"
        pixmaps[1].save(OUT / "preview.png")
        current[0].get_pixmap(dpi=200, clip=fitz.Rect(0, 0, current[0].rect.width, a_rect.y1+1), alpha=False).save(OUT / "top_panels.png")
    original.close()
    assert all(digest(p) == v for p, v in protected.items())
    qa = {"status": "passed", "source_sha256": {str(p): v for p, v in protected.items()},
          "script_sha256": digest(Path(__file__)), "pdf_sha256": digest(pdf), "svg_sha256": digest(OUT / f"{STEM}.svg"),
          "a_frame_pt": list(a_rect), "b_frame_pt": list(panel), "b_map_frame_pt": list(map_rect),
          "map_isotropic_scale": scale, "map_height_fraction_of_b": map_rect.height/panel.height,
          "a_b_clear_gap_pt": panel.x0-a_rect.x1-.65, "legend_font_pt": 6.5,
          "coverage_values_recomputed": False, "lower_images_text_and_positions_identical": True,
          "lower_render_max_channel_delta_300_dpi": int(lower_delta.max()),
          "outside_requested_regions_max_channel_delta": int(delta.max()),
          "render_note": "Redaction normalizes PDF numbers; only <=2/255 antialias differences allowed outside changed regions.",
          "stand39_locator_removed": True,
          "panel_labels": ["a", "b", "d", "e", "f", "g", "h", "i", "j", "k"], "manuscript_changed": False}
    (OUT / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(f"""# Fig1 v25: inventory and cumulative Sentinel-1 coverage

Panel (a) retains the v24 MGRS map and framed legend. Only the Stand 39 annotation, locator frame, circles and leader are removed; its inventory/reference markers remain.

The two field photographs are replaced by one panel (b) using the frozen supplied v25 coverage PDF. Its top and bottom align with (a), and its right edge aligns with (g)/(k). The clear a-b gap matches the existing approximately 5 pt spacing. The map is scaled uniformly by {scale:.6f}; its full geographic extent and complete mainland are retained. The map uses {map_rect.height/panel.height:.1%} of panel height, with the original colour ramp and track colours rearranged below it at readable 6.5-7 pt type. No coverage values are recomputed or colours reclassified.

The lower eight panels retain d-g and h-k labels and their layout. Numbering has not been shifted automatically. All eight embedded image hashes, positions, text and font sizes are unchanged. PDF operator normalization introduces only tiny antialias differences (at most 2/255 per channel at 300 dpi); checks reject larger changes outside the edited regions. The TeX, installed manuscript figures, v24 figure and standalone coverage figure are unchanged. Original coverage-count interpretation and limitations remain as documented beside that source PDF.

Reproduce: `scripts/figures/build_study_area_fig1_v25.py`. Cropped map and new legend intermediates are in `qa/`; hashes, alignment and preservation checks are in `QA.json`.
""", encoding="utf-8")
    (OUT / "index.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Fig1 v25</title><style>body{{max-width:1100px;margin:24px auto;background:#eee;font:16px Arial}}img{{width:100%}}</style><h1>Fig1: inventory and Sentinel-1 coverage</h1><a href="{STEM}.pdf">PDF</a> | <a href="{STEM}.svg">SVG</a><p><img src="preview.png" alt="Updated Fig1 with one coverage panel b"></p></html>', encoding="utf-8")
    print(json.dumps({"pdf": str(pdf), "map_height_fraction": qa["map_height_fraction_of_b"], "status": "passed"}))


if __name__ == "__main__":
    main()
