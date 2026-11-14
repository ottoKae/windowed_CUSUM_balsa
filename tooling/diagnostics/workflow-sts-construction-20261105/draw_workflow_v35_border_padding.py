"""Export v33 with 0.8 mm white margins above and below the black panel borders."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import numpy as np
from pypdf import PdfReader

import draw_workflow_v33_arial_spacing as source


OUT = source.ROOT / "runs/figures/workflow_sts_construction_20261109_v35_border_padding"


class CapturedFigure(Exception):
    def __init__(self, figure):
        self.figure = figure


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save = Figure.savefig
    def capture(fig, *args, **kwargs):
        raise CapturedFigure(fig)
    Figure.savefig = capture
    try:
        source.main()
    except CapturedFigure as captured:
        fig = captured.figure
    finally:
        Figure.savefig = save
    ax = fig.axes[0]
    page_mm = np.array(fig.get_size_inches())*25.4
    borders = [patch for patch in ax.patches if str(patch.get_gid()).startswith("fixed_border_")]
    # Measure padding from the outside of the centred border stroke.
    bottom_mm = min(patch.get_path().get_extents().y0-patch.get_linewidth()/2*25.4/72
                    for patch in borders)
    top_mm = max(patch.get_path().get_extents().y1+patch.get_linewidth()/2*25.4/72
                 for patch in borders)
    padding_mm = 0.8
    crop_bottom_mm, crop_top_mm = bottom_mm-padding_mm, top_mm+padding_mm
    crop = Bbox.from_extents(0, crop_bottom_mm/25.4, page_mm[0]/25.4, crop_top_mm/25.4)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"workflow_sts_construction.{extension}", bbox_inches=crop, pad_inches=0,
                    dpi=600, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else
                             ({"Date": None} if extension == "svg" else None))
    plt.close(fig)
    pdf = PdfReader(OUT / "workflow_sts_construction.pdf")
    exported_mm = np.array([float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)])*25.4/72
    assert np.allclose(exported_mm, [page_mm[0], crop_top_mm-crop_bottom_mm])
    fonts = sorted({str(font.get_object()["/BaseFont"])
                    for font in pdf.pages[0]["/Resources"]["/Font"].get_object().values()})
    assert all("Arial" in font for font in fonts)
    assert len(pdf.pages[0].images) == 0
    ids = [n.get("id", "") for n in ET.parse(OUT / "workflow_sts_construction.svg").iter()]
    assert all(ids.count(f"pixel_label_{i}") == 1 for i in range(1, 107))
    assert all(ids.count(f"fixed_border_{i}") == 1 for i in [35, 36])
    subprocess.run([str(source.prior.source.PDFTOPPM), "-scale-to", "2300", "-singlefile", "-png",
                    str(OUT / "workflow_sts_construction.pdf"), str(OUT / "preview")], check=True)

    manuscript = source.ROOT / "manuscript_JAG"
    tex_files = [manuscript / "main.tex", *sorted((manuscript / "sections").glob("*.tex"))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tex_files}
    target = manuscript / "figures/fig03_sts_construction.pdf"
    target.write_bytes((OUT / "workflow_sts_construction.pdf").read_bytes())
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tex_files}
    report = {"source_generator": str(Path(__file__)), "original_page_mm": page_mm.tolist(),
              "cropped_page_mm": exported_mm.tolist(), "bottom_removed_mm": crop_bottom_mm,
              "top_extended_mm": crop_top_mm-page_mm[1], "vertical_padding_mm": padding_mm,
              "border_stroke_preserved": True, "left_and_right_edges_unchanged": True,
              "content_scale": 1., "fonts": fonts, "pdf_raster_images": 0,
              "manuscript_figure": str(target), "tex_and_caption_unchanged": True,
              "tex_sha256": hashes}
    (OUT / "crop_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "README_ZH.md").write_text(
        "# Border padding (v35)\n\n"
        "The black panel borders have 0.8 mm white margins above and below their outer edges. "
        "Content scale, panel positions, fonts and horizontal page bounds are unchanged. "
        "PDF/SVG remain vector exports; PNG is 600 dpi.\n\n"
        "The manuscript figure is updated without editing any TeX or caption.\n\n"
        "Reproduce: run draw_workflow_v35_border_padding.py in the figure diagnostics folder.\n",
        encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tex_sha256"}, indent=2))


if __name__ == "__main__":
    main()
