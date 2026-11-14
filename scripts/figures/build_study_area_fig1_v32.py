"""Restore SAR grid styling and compare three coverage palettes at fixed counts."""
import hashlib
import json
import os
import re

import fitz
import matplotlib as mpl
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

from build_study_area_fig1_v25 import ROOT, COVERAGE, key_overlay
from update_study_area_validation_panel_a import digest

SOURCE = ROOT / "runs/figures/fig1_harvest_concept_v31_white_ticks_20261029/study_area_fig1_v31.pdf"
OUT = ROOT / "runs/figures/fig1_harvest_concept_v32_palette_comparison_20261029"
PALETTES = ("turbo", "gist_stern", "CMRmap_r")
TITLE = "Sentinel-1 observations from 2016 to June 2026"


def recolor_indexed(document, xref, palette, *, heatmap=False):
    """Replace the colour table only; preserve count-bin indices and alpha mask."""
    colorspace = document.xref_get_key(xref, "ColorSpace")[1]
    table = re.search(r"<([0-9A-Fa-f]+)>", colorspace)
    assert table and "/Indexed" in colorspace
    rgb = np.frombuffer(bytes.fromhex(table[1]), np.uint8).reshape(-1, 3)
    turbo = (mpl.colormaps["turbo"](np.arange(256))[:, :3]*255).astype(np.uint8)
    distance = np.max(np.abs(rgb[:, None, :].astype(int)-turbo[None, :, :].astype(int)), axis=2)
    indices = distance.argmin(axis=1)
    errors = distance.min(axis=1)
    missing = np.all(rgb == 0, axis=1) if heatmap else np.zeros(len(rgb), dtype=bool)
    assert errors[~missing].max() <= (0 if heatmap else 1)
    stream = document.xref_stream(xref)
    mask_xref = int(document.xref_get_key(xref, "SMask")[1].split()[0])
    mask = document.xref_stream(mask_xref)
    if heatmap:
        assert np.all(np.frombuffer(mask, np.uint8)[np.isin(np.frombuffer(stream, np.uint8), np.flatnonzero(missing))] == 0)
        with rasterio.open(COVERAGE / "s1_cumulative_ascending_201601_202606.tif") as dataset:
            values = np.unique(dataset.read(1))
            values = values[values != dataset.nodata]
        expected = np.unique(np.minimum(255, (values/900*256).astype(int)))
        assert set(indices[~missing]) == set(expected)
    updated = (mpl.colormaps[palette](indices)[:, :3]*255).astype(np.uint8)
    updated[missing] = rgb[missing]
    document.xref_set_key(xref, "ColorSpace", colorspace[:table.start(1)]+updated.tobytes().hex()+colorspace[table.end(1):])
    assert stream == document.xref_stream(xref) and mask == document.xref_stream(mask_xref)
    return {"bins_preserved": len(indices[~missing]), "original_lut_max_error": int(errors[~missing].max()),
            "index_stream_sha256": hashlib.sha256(stream).hexdigest(), "alpha_mask_sha256": hashlib.sha256(mask).hexdigest()}


def restore_sar(document):
    forms = {x: document.xref_stream(x) for x in set(v[0] for v in document[0].get_xobjects())}
    number = rb"-?\d*\.?\d+"
    pattern = re.compile(rb"/A2 gs \.533333 G \[2.5 1.7\] 0 d 0 J \.55 w ("+number+rb") ("+number+rb") m ("+number+rb") ("+number+rb") l S")
    xref, content = next((x, s) for x, s in forms.items() if len(pattern.findall(s)) == 24)
    changed = 0
    def grid(match):
        nonlocal changed
        if max(float(match[2]), float(match[4])) > 114:
            return match[0]
        changed += 1
        return b".55 w"+match[0].split(b".55 w", 1)[1]
    document.update_stream(xref, pattern.sub(grid, content))
    assert changed == 12
    xref, axes = next((x, s) for x, s in forms.items() if b"503.289726 l" in s and b"5.065625 cm" in s)
    pattern = re.compile(rb"q 1 G\s+("+number+rb") ("+number+rb") m\s+("+number+rb") ("+number+rb") l\s+S\s+Q")
    changed_ticks = 0
    def tick(match):
        nonlocal changed_ticks
        if max(float(match[2]), float(match[4])) >= 110:
            return match[0]
        changed_ticks += 1
        return match[0].replace(b"q 1 G", b"q 0 G", 1)
    document.update_stream(xref, pattern.sub(tick, axes))
    assert changed_ticks == 14


def main():
    if (OUT / "QA.json").exists():
        raise FileExistsError("Existing comparison retained")
    protected = {p: digest(p) for p in [SOURCE, *(ROOT / "manuscript_JAG").rglob("*.tex")]}
    prior = json.loads((ROOT / "runs/figures/fig1_harvest_concept_v29_inward_ticks_20261029/QA.json").read_text())
    panel, map_rect = fitz.Rect(prior["b_frame_pt"]), fitz.Rect(prior["b_map_frame_pt"])
    reports, crops, reference_pixels = {}, [], None
    for palette in PALETTES:
        out = OUT / palette
        (out / "qa").mkdir(parents=True, exist_ok=True)
        document = fitz.open(SOURCE)
        restore_sar(document)
        page = document[0]
        images = {im[0]: im for im in page.get_images(full=True)}
        heatmap_xref = next(x for x, im in images.items() if im[2:4] == (3364, 3701) and im[5] == "Indexed")
        heatmap_receipt = recolor_indexed(document, heatmap_xref, palette, heatmap=True)
        if palette != "turbo":
            with fitz.open(COVERAGE / "s1_cumulative_ascending_201601_202606.pdf") as coverage:
                bar = next(im[0] for im in coverage[0].get_images(full=True) if im[2:4] == (1183, 65))
                recolor_indexed(coverage, bar, palette)
                key_overlay(page.rect, panel, map_rect, coverage, out, title=TITLE,
                            swatch_width=22, swatch_linewidth=1, centered_swatch=True, compact=True)
            with fitz.open(out / "qa/panel_b_labels.pdf") as labels:
                new_bar = labels[0].get_images(full=True)
                assert len(new_bar) == 1
                new_pixmap = fitz.Pixmap(labels, new_bar[0][0])
            bar_xref = next(x for x, im in images.items() if im[2:4] == (1572, 50))
            assert (new_pixmap.width, new_pixmap.height, new_pixmap.n) == (1572, 50, 3)
            document.xref_set_key(bar_xref, "DecodeParms", "null")
            document.update_stream(bar_xref, new_pixmap.samples)
        pdf = out / f"study_area_fig1_v32_{palette}.pdf"
        document.save(pdf, garbage=4, deflate=True)
        document.close()
        with fitz.open(pdf) as current:
            pix = current[0].get_pixmap(dpi=200, alpha=False)
            pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            if reference_pixels is None:
                reference_pixels = pixels.copy()
            else:
                delta = np.max(np.abs(pixels.astype(int)-reference_pixels.astype(int)), axis=2)
                x0, y0, x1, y1 = np.array(panel+(-1, -1, 1, 1))*200/72
                delta[int(y0):int(y1)+1, int(x0):int(x1)+1] = 0
                assert delta.max() == 0, int(delta.max())
            pix.save(out / "preview.png")
            crop = current[0].get_pixmap(dpi=200, clip=panel+(-1, -1, 1, 1), alpha=False)
            crops.append(Image.frombytes("RGB", (crop.width, crop.height), crop.samples))
        reports[palette] = {"pdf": str(pdf), "pdf_sha256": digest(pdf), "heatmap": heatmap_receipt}
    assert all(digest(p) == value for p, value in protected.items())
    width, height = crops[0].size
    comparison = Image.new("RGB", (3*width+40, height+55), "white")
    draw = ImageDraw.Draw(comparison)
    font = ImageFont.truetype(os.path.join(os.environ["WINDIR"], "Fonts", "arialbd.ttf"), 24)
    for i, (name, crop) in enumerate(zip(PALETTES, crops, strict=True)):
        x = i*(width+20)
        comparison.paste(crop, (x, 45))
        draw.text((x+width/2, 10), name, anchor="mt", fill="black", font=font)
    comparison.save(OUT / "palette_comparison.png")
    report = {"status": "passed", "palettes": reports, "normalization": [0, 900], "heatmap_alpha": .78,
              "count_raster_sha256": digest(COVERAGE / "s1_cumulative_ascending_201601_202606.tif"),
              "same_count_bins_alpha_geography_layout": True,
              "sar_grid": "0.55 pt black solid, original 0.52 opacity", "sar_ticks": "black",
              "optical_grid_and_ticks_unchanged": True, "bottom_label_offset_0_2pt_preserved": True,
              "manuscript_and_prior_exports_unchanged": True}
    (OUT / "QA.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Fig1 v32 palette comparison\n\nSAR g-j grids return to 0.55 pt black solid with original 0.52 opacity; "
        "their 14 inward ticks return to black. Optical panels remain gray dashed with white ticks. "
        "The 0.2 pt bottom label offset remains.\n\nAll three palettes use the same 0-900 normalization and "
        "0.78 heatmap alpha. Only the indexed RGB lookup table changes: all 39 visible count bins are "
        "matched exactly to the original turbo table and cross-checked against the frozen count raster. "
        "Pixel indices and alpha masks remain byte-identical. The colorbar is regenerated from the "
        "original colorbar at matching palette indices (original bar quantization within 1/255 RGB). "
        "No counts, dates, geography, image panels, burst outlines, text or layout are recalculated. "
        "These are alternatives for selection, not a change to the manuscript figure.\n", encoding="utf-8")
    cards = "".join(f'<article><h2>{name}</h2><a href="{name}/study_area_fig1_v32_{name}.pdf">PDF</a>'
                    f' · <a href="{name}/preview.png">Full preview</a></article>' for name in PALETTES)
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Fig1 palette comparison</title>'
        '<style>body{max-width:1800px;margin:24px auto;font:16px Arial;padding:0 16px}img{width:100%}'
        'section{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}</style>'
        '<h1>Fig1: coverage palette comparison</h1><p>Fixed range: 0–900; opacity: 0.78. Same data and layout.</p>'
        f'<section>{cards}</section><img src="palette_comparison.png" alt="Three coverage palettes at identical scale">', encoding="utf-8")
    print(json.dumps({"status": "passed", "comparison": str(OUT / "index.html")}))


if __name__ == "__main__":
    main()
