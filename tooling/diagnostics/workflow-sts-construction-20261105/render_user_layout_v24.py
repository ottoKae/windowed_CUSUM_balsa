"""Reproduce the user's v23 PowerPoint layout and correct STP legend years."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

from pptx import Presentation

from draw_technical_workflow import ROOT

OUT = ROOT / "runs/figures/stand_recurrent_workflow_20261108_v24"
SOURCE = OUT / "user_layout_source.pptx"
CAPTION = (
    "Windowed CUSUM detection and spatiotemporal aggregation for Stand 22301 "
    "(3.62 ha; 39 pixels). (a) Monthly Sentinel-1 VV backscatter, with Pixel 17 "
    "highlighted in green; dashed lines mark reference harvest months from optical "
    "interpretation. (b) Monthly detected-pixel counts and temporal detection "
    "aggregation (TDA) reports. (c) First-detection months within TDA3 "
    "(January–July 2024). (d) Spatial timing partitioning (STP) of TDA3 into two "
    "groups; legend intervals span each group's first and last detections. "
    "Purple borders associate (c) and (d) with TDA3. Grey map cells indicate no "
    "detection within TDA3. STP groups remain within one stand-level report, "
    "and their detection ranges do not establish harvest duration."
)


def main():
    prs = Presentation(SOURCE)
    edits = []
    for shape in leaves(prs.slides[0].shapes):
        if shape.name in ("STP1_label", "STP2_label"):
            before = shape.text
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = run.text.replace("2026", "2024")
            edits.append({"shape": shape.name, "before": before, "after": shape.text})
    prs.slides[0].notes_slide.notes_text_frame.text = CAPTION
    target = OUT / "real_stand_workflow_editable.pptx"
    prs.save(target)
    (OUT / "CAPTION_EN.md").write_text(CAPTION + "\n", encoding="utf-8")
    (OUT / "layout_adaptation.json").write_text(json.dumps({
        "layout_source": str(SOURCE.relative_to(ROOT)),
        "layout_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "corrections": edits,
        "positions_and_object_styles_changed": False,
        "caption_updated": True,
        "new_inference": False,
        "TeX_modified": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("draw_workflow_from_pptx.py")),
                    "--pptx", str(target), "--out-dir", str(OUT)], check=True)


def leaves(shapes):
    for shape in shapes:
        if shape.shape_type == 6:
            yield from leaves(shape.shapes)
        else:
            yield shape


if __name__ == "__main__":
    main()
