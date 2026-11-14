"""Preserve the user's Windowed CuSum label and recolor latest detections with Spectral."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb
from pptx import Presentation
from pptx.dml.color import RGBColor

from draw_technical_workflow import ROOT, INK
from render_user_layout_v24 import leaves

OUT = ROOT / "runs/figures/stand_recurrent_workflow_20261109_v28"
SOURCE = OUT / "user_layout_source.pptx"
EVIDENCE = ROOT / "runs/figures/stand_recurrent_workflow_20261107_v4/evidence.json"


def main():
    prs = Presentation(SOURCE)
    e = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    lo, hi = e["calendar"].index("2017-01"), e["calendar"].index("2025-12")
    latest = {}
    for event in e["events"]:
        month = event["detection_month"]
        if "2017-01" <= month <= "2025-12":
            pid = event["pixel_id"]
            latest[pid] = max(latest.get(pid, month), month)
    cmap = colormaps["Spectral"].resampled(4096)
    fills = {pid: to_hex(cmap((e["calendar"].index(month)-lo)/(hi-lo)))
             for pid, month in latest.items()}
    changes = []
    for shape in leaves(prs.slides[0].shapes):
        if shape.name.startswith("latest_detection_pixel_"):
            pid = int(shape.name.rsplit("_", 1)[-1])
            color = fills.get(pid, "#E3EAF0")
            shape.fill.fore_color.rgb = RGBColor.from_string(color[1:])
            changes.append({"shape_id": shape.shape_id, "name": shape.name, "fill": color})
        elif shape.name.startswith("latest_color_"):
            index = int(shape.name.rsplit("_", 1)[-1])
            color = to_hex(cmap(index/63))
            shape.fill.fore_color.rgb = RGBColor.from_string(color[1:])
            changes.append({"shape_id": shape.shape_id, "name": shape.name, "fill": color})
        elif shape.name.startswith("latest_pixel_number_"):
            pid = int(shape.name.rsplit("_", 1)[-1])
            r, g, b = to_rgb(fills.get(pid, "#E3EAF0"))
            color = "FFFFFF" if .2126*r+.7152*g+.0722*b < .49 else INK[1:]
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.color.rgb = RGBColor.from_string(color)
    label = next(s for s in leaves(prs.slides[0].shapes)
                 if s.has_text_frame and s.text == "Windowed CuSum")
    target = OUT / "real_stand_workflow_editable.pptx"
    prs.save(target)
    caption = prs.slides[0].notes_slide.notes_text_frame.text
    (OUT / "CAPTION_EN.md").write_text(caption.strip()+"\n", encoding="utf-8")
    (OUT / "layout_adaptation.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "label": {"text": label.text, "shape_id": label.shape_id,
                  "box_mm": [v/36000 for v in (label.left,label.top,label.width,label.height)],
                  "preserved_without_changes": True},
        "colormap": "Spectral", "date_bounds": ["2017-01", "2025-12"],
        "fill_changes": changes, "new_inference": False, "TeX_modified": False,
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("draw_workflow_from_pptx.py")),
                    "--pptx", str(target), "--out-dir", str(OUT)], check=True)


if __name__ == "__main__":
    main()
