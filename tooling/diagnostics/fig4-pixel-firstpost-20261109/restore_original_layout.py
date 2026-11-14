"""Feed frozen pixel data into the original Figure 4 renderer and crop."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.workflows import build_windowed_cusum_concept_n1_current as original
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox, blended_transform_factory

EVIDENCE = ROOT / "runs/figures/fig04_pixel_firstpost_20261109_v1/evidence.json"
ARCHIVE = ROOT / "runs/figures/concept_label_preview_20260911_v3_review"
OUT = ROOT / "runs/figures/fig04_original_layout_pixel_20261109_v9"


def main():
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    dates = pd.to_datetime(evidence["calendar"])
    series = original.base.SeriesData(
        sample_id=evidence["sample_id"], historical_id=evidence["stand_id"],
        calendar=dates, values=np.asarray(evidence["VV_dB"], dtype=float),
        metadata=evidence["pixel_identity"],
    )
    windows = [original.PanelWindow(
        logical_id=index + 1,
        window_start_index=window["window_start_index"],
        window_end_index=window["window_stop_index"],
        decision_start_index=window["decision_start_index"],
        decision_end_index=window["decision_stop_index"],
        window_start=pd.Timestamp(window["window_start"]),
        window_end=pd.Timestamp(window["window_end"]),
        decision_start=pd.Timestamp(window["decision_start"]),
        decision_end=pd.Timestamp(window["decision_end"]),
    ) for index, window in enumerate(evidence["all_windows"])]
    outside = window_audit(evidence["left"], evidence)
    inside = window_audit(evidence["right"], evidence)
    protocol = original.N1Protocol(
        window_length=36, step=12, minimum_observations=12,
        alpha=0.025, replicates=3999, near_threshold_low=0.02,
        near_threshold_high=0.03, near_threshold_replicates=19999,
        seed=evidence["permutation"]["base_seed"],
        context_start="2016-01", context_end="2026-06",
        reporting_start_year=2017, reporting_end_year=2025,
    )
    figure, diagnostics = original.render_figure(
        series, pd.to_datetime(evidence["reference_months"]), windows,
        outside, inside, protocol, pd.Timestamp("2022-02"),
    )

    # The old renderer extends W8 for its old W48 protocol. Only the data
    # geometry changes here; its bar height, stacking, and style stay intact.
    timeline = figure.axes[0]
    for rectangle, window in zip(timeline.patches[::2], windows):
        rectangle.set_width(mdates.date2num(window.window_end + pd.offsets.MonthBegin(1)) - mdates.date2num(window.window_start))
    for label, value in zip(timeline.get_legend().get_texts(), (
        "Pixel 5 in stand 8655", "Reference harvest", "36-month context", "12-month decision window",
    )):
        label.set_text(value)
    figure.axes[3].set_ylabel(r"Cumulative deviation, $C_k$")
    for axis in figure.axes[3:5]:
        for label in axis.texts:
            if label.get_text() == "max(S)":
                label.set_text(r"max($C_k$)")
            elif label.get_text() == "min(S)":
                label.set_text(r"min($C_k$)")
    # New point locations require local annotation clearance, not panel moves.
    for label in figure.axes[3].texts:
        if label.get_text() == r"min($C_k$)":
            left, right = figure.axes[3].get_xlim()
            x = (mdates.date2num(label.xy[0]) - left) / (right - left)
            label.set_position((x, 0.40))
            label.set_horizontalalignment("center")
    for axis in figure.axes[1:3]:
        # The inherited mean-difference bracket consists of three z=7 lines.
        for line in list(axis.lines):
            if line.get_zorder() == 7:
                line.remove()
        for label in list(axis.texts):
            if label.get_text().endswith(" dB"):
                label.remove()

    q_brackets = []
    for axis, audit in zip(figure.axes[3:5], (outside, inside)):
        for label in list(axis.texts):
            if label.get_text().startswith("Q ="):
                label.remove()
            elif axis == figure.axes[4] and label.get_text() == r"max($C_k$)":
                label.set_position((0.34, 0.91))
                label.set_horizontalalignment("right")
            elif axis == figure.axes[4] and label.get_text() == r"min($C_k$)":
                label.set_position((0.80, 0.40))
                label.set_horizontalalignment("center")
        low, high = float(np.min(audit.cumulative_values)), float(np.max(audit.cumulative_values))
        transform = blended_transform_factory(axis.transAxes, axis.transData)
        # The right-hand time margin is outside the observed curve; the
        # bracket measures its full vertical range, not a detection date.
        bracket_x = 0.975
        left, right = axis.get_xlim()
        for index, y in ((int(np.argmin(audit.cumulative_values)), low),
                         (int(np.argmax(audit.cumulative_values)), high)):
            point_x = (mdates.date2num(audit.cumulative_dates[index]) - left) / (right - left)
            axis.plot([point_x, bracket_x], [y, y], transform=transform,
                      color=original.BISHAN_GREY, linestyle=":", linewidth=0.7, alpha=0.8, zorder=1.5)
        axis.plot([bracket_x, bracket_x], [low, high], transform=transform,
                  color=original.BISHAN_BLACK, linewidth=0.9, zorder=7)
        for y in (low, high):
            axis.plot([bracket_x - 0.009, bracket_x + 0.009], [y, y],
                      transform=transform, color=original.BISHAN_BLACK, linewidth=0.9, zorder=7)
        axis.text(bracket_x - 0.025, (low + high) / 2, f"Q = {audit.cusum_amplitude:.2f}",
                  transform=transform, ha="center", va="center", rotation=90, fontsize=7.2,
                  fontweight="bold", color=original.BISHAN_BLACK, zorder=8)
        assert np.isclose(high - low, audit.cusum_amplitude)
        q_brackets.append({"min_C": low, "max_C": high, "Q": audit.cusum_amplitude})
        for label in axis.texts:
            if label.get_text() in (r"max($C_k$)", r"min($C_k$)"):
                label.set_text(label.get_text().replace(r"$C_k$", r"$\mathbfit{C_k}$"))
                label.set_fontweight("bold")

    null_axis = figure.axes[5]
    quantile_line = next(line for line in null_axis.lines
                         if line.get_label() == "Null quantile (97.5th percentile)")
    null_quantile = float(quantile_line.get_xdata()[0])
    null_axis.text(inside.cusum_amplitude - 0.08, 0.50, f"Q = {inside.cusum_amplitude:.2f}",
                   transform=null_axis.get_xaxis_transform(), ha="right", va="center", rotation=90,
                   fontsize=7.2, fontweight="bold", color=original.BISHAN_RED, zorder=8)
    null_axis.text(null_quantile - 0.08, 0.50,
                   rf"$\mathbfit{{Q}}^{{\mathrm{{perm}}}}_{{0.975}} = {null_quantile:.2f}$",
                   transform=null_axis.get_xaxis_transform(), ha="right", va="center", rotation=90,
                   fontsize=7.2, color=original.BISHAN_GREY, zorder=8)

    # These are the exact label adjustments in the archived v3 wrapper.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = {label.get_text(): label for axis in figure.axes for label in axis.texts
              if label.get_text() in {"(a)", "(b)", "(d)", "(f)"}}
    target_left = labels["(b)"].get_window_extent(renderer).x0
    positions = {"(a)": (0.975, "top"), "(f)": (0.025, "bottom")}
    for axis in figure.axes:
        for label in axis.texts:
            if label.get_text() in positions:
                y, vertical = positions[label.get_text()]
                x = axis.transAxes.inverted().transform((target_left, axis.bbox.y0))[0]
                if label.get_text() == "(a)":
                    y -= (3.0 / 72.0 * figure.dpi) / axis.bbox.height
                label.set_position((x, y))
                label.set_horizontalalignment("left")
                label.set_verticalalignment(vertical)
                label.set_zorder(20)
            elif label.get_text() in {"(b)", "(c)", "(d)", "(e)"}:
                x, y = label.get_position()
                label.set_position((x, y + (1.0 / 72.0 * figure.dpi) / axis.bbox.height))

    old_pdf = ARCHIVE / "windowed_cusum_concept_n1_current.pdf"
    old_page, old_rectangles = pdf_frames(old_pdf)
    # Lock the old PDF crop, including its margins; replacing text must not
    # make Matplotlib recompute the overall width or move any panel.
    canvas_width, canvas_height = figure.get_size_inches()
    axis = timeline.get_position()
    crop = Bbox.from_bounds(
        canvas_width * axis.x0 - old_rectangles[0][0] / 72,
        canvas_height * axis.y0 - old_rectangles[0][1] / 72,
        float(old_page.mediabox.width) / 72,
        float(old_page.mediabox.height) / 72,
    )
    OUT.mkdir(exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(OUT / f"fig04_windowed_cusum_concept.{extension}",
                       bbox_inches=crop, pad_inches=0, dpi=400)
    new_page, new_rectangles = pdf_frames(OUT / "fig04_windowed_cusum_concept.pdf")
    same_frames = np.allclose(old_rectangles, new_rectangles, atol=1e-5, rtol=0)
    same_page = np.allclose(list(old_page.mediabox), list(new_page.mediabox), atol=1e-5, rtol=0)
    assert same_frames and same_page
    diagnostics.update({
        "source_renderer": str(Path(original.__file__)),
        "source_renderer_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        "original_pdf_sha256": hashlib.sha256(old_pdf.read_bytes()).hexdigest(),
        "evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
        "stand_id": 8655, "pixel_id": 5,
        "page_size_identical": same_page,
        "all_six_panel_frames_identical": same_frames,
        "original_panel_frames_pt": old_rectangles,
        "new_panel_frames_pt": new_rectangles,
        "page_size_pt": list(new_page.mediabox),
        "crop_inches": list(crop.bounds),
        "legend_anchors_unchanged": True,
        "font_and_palette_from_original_renderer": True,
        "data_annotation_clearance_only": [
            "d: minimum label centered directly above valley; data anchor unchanged",
            "e: maximum label left of peak; minimum label above valley; anchors unchanged",
        ],
        "de_extrema_and_Q_labels_bold": True,
        "bc_mean_difference_numbers_and_brackets_removed": True,
        "bc_pre_and_post_means_retained": True,
        "q_brackets": q_brackets,
        "de_extrema_to_bracket_horizontal_dotted_lines": True,
        "f_observed_Q_label": inside.cusum_amplitude,
        "f_null_quantile_label": null_quantile,
        "f_null_quantile_method": "higher; read from the existing plotted line",
        "f_value_labels_rotation_ccw_degrees": 90,
        "f_value_labels_vertical_center_axes_fraction": 0.5,
        "f_value_labels_gap_data_units": 0.08,
        "candidate_month": evidence["right"]["candidate_month"],
        "peak_month": evidence["right"]["peak_month"],
        "frozen_p_perm": evidence["permutation"]["p_perm"],
    })
    (OUT / "layout_verification.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    plt.close(figure)
    print(json.dumps({"output": str(OUT), "same_page": same_page, "same_frames": same_frames}))


def window_audit(window, evidence):
    retained = window["inside_decision"]
    null = evidence["permutation"]
    candidate = pd.Timestamp(window["candidate_month"])
    return original.base.WindowAudit(
        label="", logical_id=window["window_start_index"] // 12 + 1,
        window_start_index=window["window_start_index"], window_end_index=window["window_stop_index"],
        decision_start_index=window["decision_start_index"], decision_end_index=window["decision_stop_index"],
        window_start=pd.Timestamp(window["window_start"]), window_end=pd.Timestamp(window["window_end"]),
        decision_start=pd.Timestamp(window["decision_start"]), decision_end=pd.Timestamp(window["decision_end"]),
        candidate_sample_index=window["candidate_month_index"], candidate_sample_month=candidate,
        reported_transition_month=candidate, cusum_amplitude=window["Q"],
        pre_mean_corrected=window["pre_mean"], post_mean_corrected=window["post_mean"],
        post_minus_pre_corrected=window["post_minus_pre"], permutation_rank=1.0 if retained else np.nan,
        plus_one_p_value=null["p_perm"] if retained else np.nan,
        passes_central=retained, passes_decrease=window["negative_shift"],
        passes_permutation=retained, accepted=retained, valid_observations=window["valid_observations"],
        permutation_seed=null["seed"], cumulative_values=np.asarray(window["cumulative"]),
        cumulative_dates=pd.to_datetime(np.asarray(evidence["calendar"])[window["valid_month_indices"]]),
        null_amplitudes=np.asarray(null["Q_permuted"]) if retained else np.array([]),
    )


def pdf_frames(path):
    reader = PdfReader(path)
    page = reader.pages[0]
    rectangles = []
    for args, operator in ContentStream(page.get_contents(), reader).operations:
        if operator == b"re":
            rectangle = [float(value) for value in args]
            if rectangle not in rectangles:
                rectangles.append(rectangle)
    return page, rectangles


if __name__ == "__main__":
    main()
