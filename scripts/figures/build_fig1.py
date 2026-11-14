"""Canonical Fig1 export: approved turbo map and black SAR grids/ticks."""
import argparse
import json
import os
import re
from pathlib import Path

import fitz

from build_study_area_fig1_v32 import SOURCE, restore_sar, recolor_indexed
from build_study_area_fig1_v25 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/figures/fig1_final_20261029/Fig1.pdf")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing export retained; specify a new --output path")
    with fitz.open(SOURCE) as document:
        restore_sar(document)
        image = next(im[0] for im in document[0].get_images(full=True)
                     if im[2:4] == (3364, 3701) and im[5] == "Indexed")
        recolor_indexed(document, image, "turbo", heatmap=True)
        # Move only the four bottom longitude labels 0.8 pt farther from their frames.
        axes = [(x, document.xref_stream(x)) for x in set(v[0] for v in document[0].get_xobjects())
                if b"5.065625 cm" in document.xref_stream(x)]
        assert len(axes) == 1 and axes[0][1].count(b"5.065625 cm") == 4
        axis_content = axes[0][1].replace(b"5.065625 cm", b"4.265625 cm")
        latitudes = {s["text"]: fitz.Rect(s["bbox"]) for block in document[0].get_text("dict")["blocks"]
                     for line in block.get("lines", []) for s in line["spans"]
                     if s["bbox"][0] > 489 and s["bbox"][1] < 240}
        assert len(latitudes) == 7
        rotated = []
        def rotate_latitude(match):
            raw = re.search(rb"\[ \((.*?)\) \] TJ", match[0], re.S)
            text = raw[1].decode("utf-16-be") if raw else ""
            if text not in latitudes:
                return match[0]
            rect = latitudes[text]
            cx, cy = (rect.x0+rect.x1)/2, document[0].rect.height-(rect.y0+rect.y1)/2
            rotated.append(text)
            return f"q -1 0 0 -1 {2*cx:.8f} {2*cy:.8f} cm\n".encode()+match[0]+b"\nQ"
        axis_content = re.sub(rb"q\s+-0 -1 1 -0 [^\n]+ cm\s+BT.*?ET\s+Q", rotate_latitude, axis_content, flags=re.S)
        assert len(rotated) == 7
        document.update_stream(axes[0][0], axis_content)
        forms = {x: document.xref_stream(x) for x in set(v[0] for v in document[0].get_xobjects())}
        xref, content = next((x, s) for x, s in forms.items() if b"179.13188 273.1335" in s)
        start = b"q 0 G/A9 gs 1 g .46666668 G .4 w 173.21276 31.919173 m"
        end = b"q Q 0 g 0 G 1 j 1 w/A10 gs"
        assert content.count(start) == content.count(end) == 1
        # Lift the inset just enough to leave a clear attribution row beneath it.
        inset_start = content.index(b"q 0 G/A1 gs .8627451 .93333336 .96862748 rg 214.68997 15.688762 m")
        inset_end = content.index(start)
        content = content[:inset_start]+b"q 1 0 0 1 0 6 cm "+content[inset_start:inset_end]+b" Q "+content[inset_end:]
        # Align the Area frame top with the inset; retain the inter-legend gap.
        content = content.replace(start, b"q 1 0 0 1 0 19.4153595 cm "+start[2:])
        content = content.replace(end, b"1 0 0 1 0 -19.4153595 cm "+end)
        assert content.count(b"168.17427 230.87772 Tm") == 1
        content = content.replace(b"168.17427 230.87772 Tm", b"168.17427 250.2930795 Tm")
        document.update_stream(xref, content)
        layout = json.loads((ROOT / "runs/figures/fig1_harvest_concept_v29_inward_ticks_20261029/QA.json").read_text())
        frame = fitz.Rect(layout["a_frame_pt"])
        text = "Hillshade: Esri World Hillshade \u00a9 Esri"
        font = fitz.Font(fontfile=os.path.join(os.environ["WINDIR"], "Fonts", "arial.ttf"))
        position = (frame.x1-3-font.text_length(text, fontsize=5.5), frame.y1-4)
        document[0].insert_text(position, text, fontsize=5.5,
                                fontname="Fig1CreditArial", fontfile=os.path.join(os.environ["WINDIR"], "Fonts", "arial.ttf"),
                                color=(1, 1, 1), fill=(.25, .25, .25),
                                render_mode=2, border_width=.25)
        document[0].insert_text(position, text, fontsize=5.5,
                                fontname="Fig1CreditArial", color=(.25, .25, .25))
        assert len(document) == 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        document.save(args.output, garbage=4, deflate=True)
    print(args.output)


if __name__ == "__main__":
    main()
