"""Fig. 1 v7: move the two-row inventory legend to panel (a).

Start from accepted v6 artwork, retaining all mapped samples and regional panels.
The v6 script and assets remain archived; manuscript installation is separate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import numpy as np

from update_study_area_validation_panel_a import digest, mpl

mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/figures/study_area_bursts_v6_validation_a_20261026"
OUT = ROOT / "runs/figures/study_area_fig1_v7_20261026"
STEM = "study_area_fig1_v7"
LABELS = (("Official inventory", "#2f8f5b", 32.0), ("Dated subset", "#f2c744", 44.0))


def group_span(text: str, name: str) -> tuple[int, int]:
    start = text.index(f'<g id="{name}">')
    level = 0
    for match in re.finditer(r"</?g\b[^>]*>", text[start:]):
        # Empty groups do not add nesting; the accepted SVG contains several.
        if match[0].endswith("/>"):
            continue
        level += -1 if match[0].startswith("</") else 1
        if level == 0:
            return start, start + match.end()
    raise ValueError(f"Unclosed SVG group: {name}")


def main() -> None:
    previous = json.loads((SOURCE / "QA_panel_preservation.json").read_text())
    source_pdf = SOURCE / "study_area_bursts_validation_a.pdf"
    source_svg = SOURCE / "study_area_bursts_validation_a.svg"
    assert digest(source_pdf) == previous["pdf_sha256"]
    assert digest(source_svg) == previous["svg_sha256"]
    backup = SOURCE / "source_snapshot/update_study_area_validation_panel_a.py"
    assert digest(backup) == previous["script_sha256"]
    assert digest(ROOT / "scripts/figures/update_study_area_validation_panel_a.py") == digest(backup)
    protected = {p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "manuscript_JAG").rglob("*.tex")}
    protected.update({p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "manuscript_JAG/figures").glob("*.pdf")})
    OUT.mkdir(parents=True, exist_ok=True)

    original_svg = source_svg.read_text(encoding="utf-8")
    svg = original_svg
    a, b = group_span(svg, "validation_legend_a_20261026")
    legend = ['<g id="inventory_dated_legend_a_v7">']
    for label, color, y in LABELS:
        legend.append(f'<rect x="179.12" y="{y-3.06:g}" width="5.76" height="6.12" style="fill: {color}; stroke: #246b45; stroke-width: 1"/>')
        legend.append(f'<text x="188" y="{y+2.5:g}" style="font-size: 7.5px; font-family: Arial, Helvetica, sans-serif; fill: #222222">{label}</text>')
    legend.append("</g>")
    svg = svg[:a] + "\n".join(legend) + svg[b:]
    a, b = group_span(svg, "axes_8")
    svg = svg[:a] + '<g id="axes_8"></g>' + svg[b:]
    protected_groups = {}
    for name in ["PathCollection_1", "PathCollection_2", "axes_2", "axes_3", "axes_4", "axes_5", "axes_6", "axes_7"]:
        a, b = group_span(original_svg, name)
        c, d = group_span(svg, name)
        assert original_svg[a:b] == svg[c:d], name
        protected_groups[name] = True
    (OUT / f"{STEM}.svg").write_text(svg, encoding="utf-8", newline="\n")

    original = fitz.open(source_pdf)
    document = fitz.open(source_pdf)
    page = document[0]
    streams = page.get_contents()
    main_xref = max(streams, key=lambda x: len(document.xref_stream(x)))
    content = document.xref_stream(main_xref)
    start = content.rfind(b"\n0.1843137255 0.5607843137 0.3568627451 rg 0 j ")
    assert start > 0 and content.endswith(b"\nQ\n")
    removed = content[start:]
    assert "balsa inventory".encode("utf-16-be") in removed
    assert "Dated event subset".encode("utf-16-be") in removed
    new_content = content[:start] + b"\nQ\n"
    assert len(re.findall(rb"\bq\b", new_content)) == len(re.findall(rb"\bQ\b", new_content))
    document.update_stream(main_xref, new_content)
    # Accepted v6 appends the circle and two text streams after its wrapped map.
    local = streams[streams.index(main_xref)+2:]
    assert len(local) == 3
    assert b"Fig1ValidationArial" in document.xref_stream(local[1])
    assert b"Fig1ValidationArial" in document.xref_stream(local[2])
    for xref in local:
        document.update_stream(xref, b"")
    clip = re.search(rb"q ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re W n /A9 gs", content)
    assert clip is not None
    x, y, width, height = map(float, clip.groups())
    dx = x - 18.61625
    dy = page.rect.height - y - height - 18.61625
    for label, color, cy in LABELS:
        fill = tuple(int(color[k:k+2], 16)/255 for k in (1, 3, 5))
        page.draw_rect(fitz.Rect(179.12+dx, cy-3.06+dy, 184.88+dx, cy+3.06+dy),
                       color=(36/255, 107/255, 69/255), fill=fill, width=1)
        page.insert_text((188+dx, cy+2.5+dy), label, fontsize=7.5,
                         fontname="Fig1ValidationArial", color=(34/255,)*3)
    pdf = OUT / f"{STEM}.pdf"
    document.save(pdf, deflate=True, no_new_id=True)
    document.close()
    updated = fitz.open(pdf)
    assert updated[0].rect == original[0].rect
    assert updated.xref_stream(main_xref) == new_content
    text = updated[0].get_text().replace("\u00a0", " ")
    assert text.count("Official inventory") == text.count("Dated subset") == 1
    assert all(old not in text for old in ("Official balsa inventory", "Dated event subset", "Validation stands (a)", "n = 467"))

    dpi = 600
    scale = dpi/72
    before = original[0].get_pixmap(dpi=dpi, alpha=False)
    after = updated[0].get_pixmap(dpi=dpi, alpha=False)
    shape = (before.height, before.width, 3)
    changed = np.any(np.frombuffer(before.samples, np.uint8).reshape(shape) != np.frombuffer(after.samples, np.uint8).reshape(shape), axis=2)
    allowed = np.zeros(changed.shape, bool)
    allowed[int(26*scale):int(51*scale), int(177*scale):int(273*scale)] = True
    allowed[int(291*scale):int(304*scale), int(284*scale):int(497*scale)] = True
    unexpected = int(changed[~allowed].sum())
    regional = int(changed[:int(291*scale), int(280*scale):].sum())
    assert unexpected == regional == 0, (unexpected, regional)
    updated[0].get_pixmap(dpi=300, alpha=False).save(OUT / "preview.png")
    updated[0].get_pixmap(dpi=300, alpha=False, clip=fitz.Rect(0, 0, 280, updated[0].rect.height)).save(OUT / "panel_a.png")
    for path, value in protected.items():
        assert digest(ROOT / path) == value, path
    qa = {"status": "data_and_render_checks_passed_pending_visual_review",
          "date": "2026-10-26", "change": "Move shared legend to upper-right of (a), one column and two rows; shorten labels",
          "source_pdf_sha256": digest(source_pdf), "source_svg_sha256": digest(source_svg),
          "pdf_sha256": digest(pdf), "svg_sha256": digest(OUT / f"{STEM}.svg"),
          "script_sha256": digest(Path(__file__)), "v6_script_backup_sha256": digest(backup),
          "protected_manuscript_hashes": protected, "protected_svg_groups_identical": protected_groups,
          "yellow_stands": 467, "green_inventory_stands": 3890, "sample_membership_unchanged": True,
          "pdf_render_dpi": dpi, "changed_pixels_outside_legend_regions": unexpected,
          "changed_pixels_in_bcd_maps": regional, "total_changed_pixels": int(changed.sum()),
          "legend_labels": [row[0] for row in LABELS], "legend_font_size_pt": 7.5,
          "tex_modified": False,
          "caption_note": "Dated subset in (a) retains the current 467 stands; regional maps retain their previous dated subsets. No sample selection changed."}
    (OUT / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "bcd_changed_pixels": regional, "unexpected_changed_pixels": unexpected}, indent=2))


if __name__ == "__main__":
    main()
