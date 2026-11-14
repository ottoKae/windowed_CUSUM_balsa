"""Update only Fig. 1(a)'s validation layer; retain the accepted SVG/PDF artwork.

Run with an environment containing geopandas, matplotlib, PyMuPDF and pandas.
All outputs are local. The manuscript PDF is installed separately after visual QA.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "runs/figures/study_area_bursts_v5"
OUT = ROOT / "runs/figures/study_area_bursts_v6_validation_a_20261026"
STEM = "study_area_bursts_validation_a"
FORMAL = ROOT / "manuscript_JAG/figures/fig01_study_area.pdf"
REF = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_v3/evaluation/reference_matches.csv"
SAMPLE = ROOT / "runs/validation/b20_spatial_probability_review_20260928_v1/sampling_frame.csv"
SCORES = ROOT / "runs/validation/jag_reference_score_closeout_20261015_v1/scoring/report_scores.csv"
INVENTORY = ROOT / "data/local/paper_v1/inventory/balsa_validated.parquet"
SHAPE = ROOT.parent / "LR-CuSum_v100/shp/balsa_polygons/FC001_PLANTACION_FORESTAL_2025_A_BALSA/FC001_PLANTACION_FORESTAL_2025_A_BALSA_ID.shp"
OLD_REF = ROOT.parent / "LR-CuSum_v100/outputs/balsa_validation_closure_v2/reference_events.parquet"
EXTENT = (-81.20287031905998, -75.06919372793989, -5.145511022719896, 1.5992886867201852)
SVG_RECT = (18.61625, 18.61625, 256.29, 281.960219)
EXPECTED_PDF = "20d7564041475c1ef61db9f9cb2e831f24bc93f494a9424a3762acd8d4ba0484"
EXPECTED_SVG = "dbeba130c9e8b959c6bb8ef0b9ad9c797b684b7d6e2b7f7ca632c940e73c7f36"
NS = {"s": "http://www.w3.org/2000/svg"}
NUMBER = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_span(text: str, name: str) -> tuple[int, int]:
    start = text.index(f'<g id="{name}">')
    level = 0
    for match in re.finditer(r"</?g\b[^>]*>", text[start:]):
        level += -1 if match[0].startswith("</") else 1
        if level == 0:
            return start, start + match.end()
    raise ValueError(f"Unclosed SVG group: {name}")


def circle_path(x: float, y: float, radius: float, svg: bool) -> str:
    transform = Affine2D().scale(radius, -radius if svg else radius).translate(x, y)
    pieces = []
    for vertices, code in MplPath.unit_circle().iter_segments(transform, simplify=False):
        op = {MplPath.MOVETO: "M", MplPath.CURVE4: "C", MplPath.CLOSEPOLY: "z"}[code]
        values = " ".join(f"{v:.6f}".rstrip("0").rstrip(".") for v in vertices)
        if code == MplPath.CLOSEPOLY:
            pieces.append("z" if svg else "h\n\nB")
        else:
            pieces.append(f"{op} {values}" if svg else f"{values} {op.lower()}")
    return "\n".join(pieces)


def coordinates(frame: pd.DataFrame, rectangle: tuple, svg: bool) -> np.ndarray:
    x0, y0, width, height = rectangle
    xmin, xmax, ymin, ymax = EXTENT
    x = x0 + (frame.longitude.to_numpy() - xmin) / (xmax - xmin) * width
    fraction = (frame.latitude.to_numpy() - ymin) / (ymax - ymin)
    y = y0 + (1 - fraction if svg else fraction) * height
    return np.column_stack((x, y))


def read_stands() -> tuple[pd.DataFrame, pd.DataFrame, set[int], dict]:
    references = pd.read_csv(REF)
    references = references.loc[(references.method == "windowed_w36_d12_s12") & (references.tolerance_months == 3)]
    sample = pd.read_csv(SAMPLE)
    sample = sample.loc[sample.selected.eq(True)]
    scores = pd.read_csv(SCORES)
    scores = scores.loc[scores.tolerance_months == 3]
    assert len(references) == references.event_id.nunique() == 1059
    assert len(sample) == sample.prediction_id.nunique() == len(scores) == 307
    assert set(sample.prediction_id) == set(scores.prediction_id)
    month = references.loc[references.timing_type == "month"]
    interval = references.loc[references.timing_type == "interval"]
    assert (len(month), month.sample_id.nunique(), len(interval), interval.sample_id.nunique()) == (1049, 459, 10, 9)
    ids = set(references.sample_id) | set(sample.sample_id)
    assert len(ids) == 467 and sample.sample_id.nunique() == 239
    assert len(set(month.sample_id) & set(sample.sample_id)) == 236

    inventory = gpd.read_parquet(INVENTORY).to_crs(32717)
    assert inventory.sample_id.is_unique
    selected = inventory.loc[inventory.sample_id.isin(ids), ["sample_id", "geometry"]].copy()
    assert len(selected) == 467
    official = gpd.read_file(SHAPE).to_crs(32717)
    official["fc001_id"] = pd.to_numeric(official.ID).astype(int)
    assert len(official) == 3890 and official.fc001_id.is_unique
    points = selected.copy()
    points.geometry = selected.geometry.representative_point()
    joined = gpd.sjoin(points, official[["fc001_id", "geometry"]], predicate="within", how="left")
    assert len(joined) == 467 and joined.sample_id.is_unique and joined.fc001_id.notna().all()
    assert joined.fc001_id.is_unique
    official_geometry = official.set_index("fc001_id").geometry
    for row in joined.itertuples():
        assert selected.loc[row.Index, "geometry"].equals(official_geometry.loc[row.fc001_id])

    official["area_ha"] = official.geometry.area / 10000
    centers = gpd.GeoSeries(official.geometry.representative_point(), crs=32717).to_crs(4326)
    official["longitude"], official["latitude"] = centers.x, centers.y
    lower, upper = official.area_ha.quantile([0.05, 0.95])
    sizes = 2.5 + (official.area_ha.clip(lower, upper) - lower) / (upper - lower) * 17.5
    official["radius_pt"] = np.sqrt(np.maximum(sizes * 1.6, 7)) / 2
    official_table = pd.DataFrame(official.drop(columns="geometry"))
    members = official_table[["fc001_id", "area_ha", "longitude", "latitude", "radius_pt"]].merge(
        joined[["sample_id", "fc001_id"]], on="fc001_id", how="inner", validate="one_to_one", sort=False)
    for name, table in [("month_reference", month), ("interval_reference", interval), ("probability_review", sample)]:
        members[f"{name}_count"] = members.sample_id.map(table.groupby("sample_id").size()).fillna(0).astype(int)
        members[f"is_{name}"] = members[f"{name}_count"].gt(0)
    assert len(members) == 467

    old_events = pd.read_parquet(OLD_REF)
    old_samples = old_events.sort_values(["sample_id", "event_id"]).drop_duplicates("sample_id")
    old_points = gpd.GeoDataFrame(old_samples[["sample_id"]], geometry=gpd.points_from_xy(old_samples.longitude, old_samples.latitude), crs=4326).to_crs(32717)
    old_join = gpd.sjoin(old_points, official[["fc001_id", "geometry"]], predicate="within", how="left")
    assert len(old_join) == 400 and old_join.fc001_id.notna().all() and old_join.fc001_id.is_unique
    old_ids = set(old_join.fc001_id)
    new_ids = set(members.fc001_id)
    assert (len(old_ids & new_ids), len(new_ids - old_ids), len(old_ids - new_ids)) == (391, 76, 9)
    counts = {"month_events": 1049, "month_stands": 459, "interval_events": 10, "interval_stands": 9,
              "probability_reports": 307, "probability_stands": 239, "yellow_stands": 467,
              "green_inventory_stands": 3890, "retained": 391, "added": 76, "removed_from_yellow": 9,
              "all_T_F_U_retained": True, "unmatched_reference_records_retained": int((~references.found).sum())}
    return members, official_table, old_ids, counts


def main() -> None:
    FONT = Path(os.environ["WINDIR"]) / "Fonts/arial.ttf"
    assert digest(OLD / "study_area_bursts.pdf") == EXPECTED_PDF
    assert digest(OLD / "study_area_bursts.svg") == EXPECTED_SVG
    assert FONT.exists(), "Existing Arial font is required for the new legend."
    OUT.mkdir(parents=True, exist_ok=True)
    backup = OUT / "original"
    backup.mkdir(exist_ok=True)
    for source in [OLD / "study_area_bursts.svg", OLD / "study_area_bursts.pdf", FORMAL]:
        target = backup / source.name
        if not target.exists():
            shutil.copy2(source, target)
    protected = {p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "manuscript_JAG").rglob("*.tex")}
    protected.update({p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "manuscript_JAG/figures").glob("*.pdf")})
    members, official, old_ids, counts = read_stands()
    old_members = official.loc[official.fc001_id.isin(old_ids)]
    new_ids = set(members.fc001_id)
    members.to_csv(OUT / "validation_stands.csv", index=False)
    changes = pd.DataFrame([{"fc001_id": i, "old_yellow": i in old_ids, "current_yellow": i in new_ids,
                             "change": "retained" if i in old_ids & new_ids else "added" if i in new_ids else "removed_from_yellow"}
                            for i in sorted(old_ids | new_ids)])
    changes.to_csv(OUT / "validation_membership_changes.csv", index=False)

    svg = (OLD / "study_area_bursts.svg").read_text(encoding="utf-8")
    start, end = group_span(svg, "PathCollection_2")
    old_svg_paths = ET.fromstring(svg).find('.//*[@id="PathCollection_2"]').findall("s:path", NS)
    assert len(old_svg_paths) == 400
    svg_old_xy = coordinates(old_members, SVG_RECT, True)
    for original, (x, y), radius in zip(old_svg_paths, svg_old_xy, old_members.radius_pt, strict=True):
        expected = circle_path(x, y, radius, True)
        assert np.allclose(np.array(re.findall(NUMBER, original.attrib["d"]), float), np.array(re.findall(NUMBER, expected), float), atol=1.2e-6, rtol=0)
    svg_paths = []
    for (x, y), radius in zip(coordinates(members, SVG_RECT, True), members.radius_pt, strict=True):
        svg_paths.append(f'<path d="{circle_path(x, y, radius, True)}" clip-path="url(#p9d84063698)" style="fill: #f2c744; fill-opacity: 0.95; stroke: #246b45; stroke-opacity: 0.95; stroke-width: 0.3"/>')
    new_group = '<g id="PathCollection_2">\n' + "\n".join(svg_paths) + "\n</g>"
    legend = '''<g id="validation_legend_a_20261026">
<circle cx="182" cy="32" r="2.2" style="fill: #f2c744; fill-opacity: 0.95; stroke: #246b45; stroke-opacity: 0.95; stroke-width: 0.3"/>
<text x="188" y="34.5" style="font-size: 7.5px; font-family: Arial, Helvetica, sans-serif; fill: #222222">Validation stands (a)</text>
<text x="188" y="44.5" style="font-size: 7.5px; font-family: Arial, Helvetica, sans-serif; fill: #222222">n = 467</text>
</g>'''
    updated_svg = svg[:start] + new_group + "\n" + legend + svg[end:]
    assert updated_svg.replace(new_group + "\n" + legend, "LAYER", 1) == svg[:start] + "LAYER" + svg[end:]
    protected_groups = {}
    for name in ["PathCollection_1", "axes_2", "axes_3", "axes_4", "axes_5", "axes_6", "axes_7", "axes_8"]:
        a, b = group_span(svg, name)
        c, d = group_span(updated_svg, name)
        assert svg[a:b] == updated_svg[c:d], name
        protected_groups[name] = hashlib.sha256(svg[a:b].encode()).hexdigest()
    assert len(ET.fromstring(updated_svg).find('.//*[@id="PathCollection_1"]')) == 3890
    assert len(ET.fromstring(updated_svg).find('.//*[@id="PathCollection_2"]')) == 467
    (OUT / f"{STEM}.svg").write_text(updated_svg, encoding="utf-8", newline="\n")

    original = fitz.open(OLD / "study_area_bursts.pdf")
    document = fitz.open(OLD / "study_area_bursts.pdf")
    page = document[0]
    assert len(page.get_contents()) == 1
    content_xref = page.get_contents()[0]
    content = document.xref_stream(content_xref)
    color_at = content.index(b"0.9490196078 0.7803921569 0.2666666667 rg")
    geometry_start = content.index(b"\n\n", color_at) + 2
    geometry_end = content.index(b"\nQ q /A10 gs", geometry_start)
    old_geometry = content[geometry_start:geometry_end].decode("ascii")
    assert len(re.findall(r"\bm\b", old_geometry)) == len(re.findall(r"\bB\b", old_geometry)) == 400
    assert len(re.findall(r"\bc\b", old_geometry)) == 3200
    clip = re.search(rb"q ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re W n /A9 gs", content)
    assert clip is not None
    pdf_rect = tuple(map(float, clip.groups()))
    pdf_old_xy = coordinates(old_members, pdf_rect, False)
    expected_old = "\n\n".join(circle_path(x, y, r, False) for (x, y), r in zip(pdf_old_xy, old_members.radius_pt, strict=True))
    assert np.allclose(np.array(re.findall(NUMBER, old_geometry), float), np.array(re.findall(NUMBER, expected_old), float), atol=1.2e-6, rtol=0)
    pdf_xy = coordinates(members, pdf_rect, False)
    geometry = "\n\n".join(circle_path(x, y, r, False) for (x, y), r in zip(pdf_xy, members.radius_pt, strict=True)).encode()
    updated_content = content[:geometry_start] + geometry + content[geometry_end:]
    document.update_stream(content_xref, updated_content)
    assert document.xref_stream(content_xref) == updated_content
    # The old Arial subset lacks V and =; add a separate font without changing it.
    offset_x = pdf_rect[0] - SVG_RECT[0]
    offset_y = page.rect.height - pdf_rect[1] - pdf_rect[3] - SVG_RECT[1]
    page.draw_circle((182 + offset_x, 32 + offset_y), 2.2, color=(36/255, 107/255, 69/255),
                     fill=(242/255, 199/255, 68/255), width=0.3, fill_opacity=0.95, stroke_opacity=0.95)
    for text, y in [("Validation stands (a)", 34.5), ("n = 467", 44.5)]:
        page.insert_text((188 + offset_x, y + offset_y), text, fontsize=7.5,
                         fontname="Fig1ValidationArial", fontfile=str(FONT), color=(34/255,)*3)
    pdf_path = OUT / f"{STEM}.pdf"
    document.save(pdf_path, deflate=True)
    document.close()
    updated = fitz.open(pdf_path)
    assert updated[0].rect == original[0].rect
    assert updated.xref_stream(content_xref) == updated_content
    extracted_text = updated[0].get_text().replace("\u00a0", " ")
    assert "Validation stands (a)" in extracted_text
    assert "n = 467" in extracted_text

    # Compare actual PDF renders; no change may extend outside symbols or legend.
    dpi = 600
    scale = dpi / 72
    old_pix = original[0].get_pixmap(dpi=dpi, alpha=False)
    new_pix = updated[0].get_pixmap(dpi=dpi, alpha=False)
    old_image = np.frombuffer(old_pix.samples, np.uint8).reshape(old_pix.height, old_pix.width, 3)
    new_image = np.frombuffer(new_pix.samples, np.uint8).reshape(new_pix.height, new_pix.width, 3)
    changed = np.any(old_image != new_image, axis=2)
    right_changes = int(changed[:, int(280 * scale):].sum())
    assert right_changes == 0, f"Protected right panels changed at {right_changes} pixels"
    allowed = np.zeros(changed.shape, bool)
    for xy, radii in [(pdf_old_xy, old_members.radius_pt), (pdf_xy, members.radius_pt)]:
        for (x, y), radius in zip(xy, radii, strict=True):
            top = original[0].rect.height - y
            margin = radius + 0.5 + 2/scale
            left, right = max(0, int((x-margin)*scale)), min(allowed.shape[1], int(np.ceil((x+margin)*scale)))
            low, high = max(0, int((top-margin)*scale)), min(allowed.shape[0], int(np.ceil((top+margin)*scale)))
            allowed[low:high, left:right] = True
    allowed[int(26*scale):int(49*scale), int(178*scale):int(273*scale)] = True
    outside_changes = int(changed[~allowed].sum())
    assert outside_changes == 0, f"Unexpected changes outside yellow symbols/legend: {outside_changes}"
    for name, clip_rect in [("preview", None), ("panel_a", fitz.Rect(0, 0, 280, updated[0].rect.height))]:
        updated[0].get_pixmap(dpi=300, alpha=False, clip=clip_rect).save(OUT / f"{name}.png")
    for path, value in protected.items():
        assert digest(ROOT / path) == value, f"Protected manuscript file modified: {path}"
    assert digest(OLD / "study_area_bursts.pdf") == EXPECTED_PDF
    assert digest(OLD / "study_area_bursts.svg") == EXPECTED_SVG
    inputs = [REF, SAMPLE, SCORES, INVENTORY, OLD_REF, *sorted(SHAPE.parent.glob(SHAPE.stem + ".*"))]
    qa = {"status": "data_and_render_checks_passed_pending_visual_review", "counts": counts,
          "method": "Direct PDF vector path splice and SVG group splice; no panel rerender or compositing",
          "source_pdf_sha256": EXPECTED_PDF, "source_svg_sha256": EXPECTED_SVG,
          "source_input_hashes": {str(p): digest(p) for p in inputs},
          "font_file_sha256": digest(FONT), "script_sha256": digest(Path(__file__)),
          "pdf_sha256": digest(pdf_path), "svg_sha256": digest(OUT / f"{STEM}.svg"),
          "protected_svg_group_sha256": protected_groups, "protected_manuscript_hashes": protected,
          "unique_geometry_matches": 467, "missing_or_ambiguous_matches": 0,
          "original_yellow_paths_reproduced": 400, "pdf_geometry_outside_layer_unchanged": True,
          "pdf_render_dpi": dpi, "changed_pixels_right_of_280pt": right_changes,
          "changed_pixels_outside_symbols_and_legend": outside_changes,
          "total_changed_pixels": int(changed.sum()),
          "page_size_pt": list(updated[0].rect), "tex_modified": False,
          "caption_note": "Panel (a) now uses current validation stands; (b)-(d) retain the original dated-event subset."}
    (OUT / "QA_panel_preservation.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (OUT / "caption_suggestion.md").write_text(
        "# Suggested caption clarification\n\n"
        "(a) Distribution of the official balsa plantation inventory and the 467 stands represented in the current validation datasets. "
        "These datasets comprise 1,049 month-dated reference events, 10 interval-dated reference events assessed separately, "
        "and 307 probability-sampled detection reports. Stands shared by these datasets are shown once. "
        "(b)-(d) retain the original regional detail maps and dated-event subset. "
        "Blue boundaries indicate static ascending Sentinel-1 burst footprints. "
        "Hillshade background from Esri World Hillshade (© Esri).\n\n"
        "This is a suggestion for the author; the manuscript TeX has not been edited.\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": counts, "right_panel_changed_pixels": right_changes,
                      "unexpected_changed_pixels": outside_changes}, indent=2))


if __name__ == "__main__":
    main()
