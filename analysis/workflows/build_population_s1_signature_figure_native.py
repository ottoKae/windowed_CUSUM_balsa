"""Render the 978-event Fig. 5 using the approved three-panel v11 layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.workflows import build_population_s1_signature_figure_v4 as v4
from analysis.workflows import build_population_s1_signature_figure_v11 as v11

RUN = ROOT / "runs/figures/fig05_native_signatures_20261109_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RUN / "data")
    parser.add_argument("--output-dir", type=Path, default=RUN)
    args = parser.parse_args()
    inputs = args.input_dir.resolve()
    output = args.output_dir.resolve()
    provenance = json.loads((inputs / "provenance.json").read_text(encoding="utf-8"))
    validation = json.loads((inputs / "validation.json").read_text(encoding="utf-8"))
    paths = {
        "events": inputs / "events.csv",
        "summary": inputs / "table_summary.csv",
        "curve": inputs / "curve.csv",
        "provenance": inputs / "provenance.json",
        "validation": inputs / "validation.json",
        "background": ROOT / provenance["archived_background"]["values_path"],
    }
    events = pd.read_csv(paths["events"])
    summary = pd.read_csv(paths["summary"])
    curve = pd.read_csv(paths["curve"])
    with np.load(paths["background"]) as arrays:
        background = arrays["values"].copy()

    # The data-preparation step checks these estimators against the frozen P3 run.
    counts = events.groupby("band").event_id.nunique().to_dict()
    if counts != {"VH": 978, "VV": 978} or len(events) != 1956:
        raise ValueError(f"Expected the paired 978-event cohort, got {counts}")
    if events.block_id.nunique() != 61 or background.shape != (3498, 2, 126):
        raise ValueError("Spatial blocks or archived background disagree with the release")
    if not np.isfinite(events[["pre_db", "post_db", "post_delta_db"]]).all().all():
        raise ValueError("Eligible event distributions must contain finite values")
    if len(curve) != 50 or set(curve.relative_month) != set(range(-12, 13)):
        raise ValueError("Expected 25 relative months for each polarization")

    fig, layout = v4.render_figure(background, events, summary, curve, v11.DEFAULT_SEED)
    layout.update(v11.apply_v11_layout(fig, curve, events, summary))
    layout["axis_positions"] = [list(ax.get_position().bounds) for ax in fig.axes]
    layout["axis_limits"] = [
        {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())} for ax in fig.axes
    ]
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "fig05_population_s1_signatures"
    fig.savefig(
        stem.with_suffix(".pdf"), bbox_inches="tight", dpi=600,
        metadata={"Title": "Sentinel-1 backscatter responses to 978 reference harvest events"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)

    manifest = {
        "workflow": "native_reference_signatures_with_archived_descriptive_background",
        "eligible_events": 978,
        "spatial_blocks": 61,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 20261105,
        "bootstrap_recomputed": False,
        "layout": layout,
        "uncertainty_bands": {
            "light_alpha_0.18": "interquartile range across eligible events",
            "dark_alpha_0.25": "pointwise 95% spatial-block bootstrap CI of median",
        },
        "background_note": (
            "Archived polygon-month medians are descriptive context, not native-cell "
            "detection inputs or undisturbed controls. Full distributions enter the "
            "density estimates; panel a retains its original -20 to -3 dB display range."
        ),
        "background_display_coverage": validation["archived_background_statistics"],
        "inputs": {str(path.relative_to(ROOT)): v11.sha256(path) for path in paths.values()},
        "render_sources": {
            str(path.relative_to(ROOT)): v11.sha256(path)
            for path in sorted(Path(__file__).parent.glob("build_population_s1_signature_figure_*.py"))
        },
        "outputs": {
            suffix: v11.sha256(stem.with_suffix(suffix)) for suffix in (".pdf", ".svg", ".png")
        },
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Rendered {stem}.pdf/.svg/.png; 978 events, 61 spatial blocks; v11 layout retained.")


if __name__ == "__main__":
    main()
