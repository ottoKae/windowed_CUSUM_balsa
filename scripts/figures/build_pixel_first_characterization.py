"""Render new reference-cohort previews from saved, real source tables only.

No detection, parameter selection, reference matching or uncertainty estimation
is performed here. Missing source tables are reported, never simulated. The
public entry point is ``build(run_dir)``; all products stay beneath that run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


GREEN = "#2f6f56"
YELLOW = "#d5b14c"
CHARCOAL = "#343b3b"
GREY = "#b8bfbc"
PALE = "#dce6df"
STAGES = ("B0", "B1", "P1", "P2", "P3")
MM = 25.4
WIDTH_MM = 183

CONTRACT = """# Pixel-first figure contract

These are new development-analysis previews, not replacement manuscript figures
or a claim of journal submission compliance. Python/matplotlib is the saved and
exclusive drawing backend. All numerical marks come from saved run tables.

## Scientific question and evidence chain

Figure A asks whether P3 changes catalogue-conditional event agreement relative
to B0, and whether any change is robust or merely a timing/recall trade-off.
Panel a: pooled P/Recall/F1 and saved 95% paired spatial-block intervals, B0/B1/P1/P2/P3.
Panel b: all five descriptive outer-fold F1 differences and the primary pooled
P3-minus-B0 paired interval. Fold estimates are not independent significance tests.
Panel c: paired signed/absolute error differences on shared reference events,
retaining the original matching and separating median/Q25/mode. Positive signed
error means later than the reference; the difference is P3 minus B0.
An interval spanning zero does not prove equivalence or supported superiority.

Figure B asks what the inferred response description means at native-pixel scale.
The frozen pilot is not selected for performance. Every selected polygon remains
represented, even when it has no accepted episode. Display tracks are frozen
label-independently; when no episode is frozen, display the earliest accepted P3
track episode, with stable ID as tie-breaker, never the closest reference event.
Each page combines a raw display-track VV median/IQR trajectory, independently
dated ground references, inferred track episodes, a calibrated pixel date map,
and a one-vote histogram with cumulative response / fixed N_observable.
No track pixels are pooled into ground area. Fallback extraction is not a map.
The multi-track pilot supplement shows every track's accepted P3 episode dates,
Q10–Q90 response spans and responder fractions beside independent reference dates.
It uses saved event ledgers, not alternate-track raw VV trajectories. Geometric
and cached cell counts remain separate; no cross-track pixels are added as area.

Figure C asks where response descriptions and catalogue agreement are limited.
Panels distinguish selected-ratio patterns, all-accepted-episode Q10–Q90 spread,
response support versus observability, and catalogue-conditioned subgroup errors.
The reference catalogue is not an exhaustive absence census; unmatched detections
are not labelled verified false disturbances. Response fractions/spreads do not
validate disturbed area, physical onset/duration, propagation, or real-time delay.

## Visual/export contract

A is a 183 mm wide three-panel quantitative grid. B uses readable 183 mm wide
individual polygon pages rather than compressing 12 cases onto one page. C is a
183 mm wide compact quantitative grid. White background; green #2f6f56, yellow
#d5b14c, charcoal and neutral greys, with redundant markers/line styles; no red.
Body/tick/legend text is 8 pt; panel labels are 9 pt, bold lowercase. All rendered
glyphs must be at least 5 pt. PDF TrueType and SVG text stay editable; PNG is 600
dpi at the same physical size. No tight bounding-box crop changes page width.

## Integrity and QA

Saved-table filters and unavailable inputs are enumerated in FIGURE_QA.md.
Loaded numerical source tables are copied to figures/source_data, unmodified.
No observations are downsampled. Missing numeric values remain missing; missing
or ineligible map cells are hatched separately from grey eligible nondetections.
Date colours indicate observed response months, with integer calendar months.
Spatial geometry requires calibrated projected metre coordinates and a real
native affine cell basis (or an exactly verified basis from saved centres).
Pixels are not resized to look like a larger area. No basemap or external data.

Intervals are the saved 5,000 paired spatial-block bootstrap intervals, conditional
on locked OOF predictions, not full selection uncertainty. Quantitative subgroup
points without supplied intervals are explicitly descriptive. Timing sample sizes
are conditional shared-event counts, not all matched detections. No p-value or
significance stars are inferred. Missing panels are documented and never filled
with fabricated marks. Automatic font/canvas checks do not replace visual QA;
final-size panel-by-panel visual review and exported PDF glyph audit remain
required before publication use.
"""


class MissingSource(ValueError):
    """An honest unavailable figure, distinct from a renderer programming error."""


class Sources:
    def __init__(self, root: Path):
        self.root = root
        self.output = root / "figures"
        self.output.mkdir(exist_ok=True)
        (self.output / "source_data").mkdir(exist_ok=True)
        self.notes: list[str] = []
        self.products: list[str] = []
        self.panels: list[dict] = []

    def read(self, relative: str, required=()) -> pd.DataFrame:
        path = self.root / relative
        if not path.is_file():
            raise MissingSource(f"Missing {relative}")
        try:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except pd.errors.EmptyDataError as error:
            raise MissingSource(f"Empty source without schema: {relative}") from error
        require(frame, required, relative)
        source = self.output / "source_data" / relative.replace("/", "__")
        if path.suffix == ".parquet":
            frame.to_parquet(source, index=False)
        else:
            frame.to_csv(source, index=False)
        self.notes.append(f"Source {relative}: {len(frame)} rows; preserved unchanged in source_data.")
        return frame

    def select(self, frame: pd.DataFrame, mask, rule: str) -> pd.DataFrame:
        selected = frame.loc[mask].copy()
        n_before, n_after = len(frame), len(selected)
        self.notes.append(f"Filter {rule}: {n_before} -> {n_after} rows.")
        return selected

    def save(self, fig, name: str, panel_roles: list[str]):
        # Locators may create unused out-of-range tick artists. Remove those
        # artists before auditing canvas bounds without changing visible ticks.
        for ax in fig.axes:
            if not ax.axison:
                continue
            for ticks, limits, setter in ((ax.get_xticks(), ax.get_xlim(), ax.set_xticks),
                                          (ax.get_yticks(), ax.get_ylim(), ax.set_yticks)):
                lower, upper = sorted(limits)
                kept = [tick for tick in ticks if lower <= tick <= upper]
                if kept and len(kept) != len(ticks):
                    setter(kept)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [x for x in fig.findobj(Text) if x.get_visible() and x.get_text()]
        minimum = min((x.get_fontsize() for x in texts), default=8)
        if minimum < 5:
            raise ValueError(f"{name}: rendered source text below 5 pt")
        outside = []
        for item in texts:
            bounds = item.get_window_extent(renderer)
            if bounds.width and bounds.height and (
                bounds.x0 < -1 or bounds.y0 < -1
                or bounds.x1 > fig.bbox.width + 1 or bounds.y1 > fig.bbox.height + 1
            ):
                outside.append(item.get_text().replace("\n", " "))
        if outside:
            self.notes.append(f"Canvas review {name}: text possibly outside page: {outside}")
        fig.savefig(self.output / f"{name}.pdf", dpi=600, facecolor="white")
        fig.savefig(self.output / f"{name}.svg", dpi=600, facecolor="white")
        fig.savefig(self.output / f"{name}.png", dpi=600, facecolor="white")
        for suffix in ("pdf", "svg", "png"):
            destination = self.output / f"{name}.{suffix}"
            self.products.append(str(destination.relative_to(self.root)))
        for number, role in enumerate(panel_roles):
            self.panels.append(dict(figure=name, panel=chr(97 + number), role=role,
                                    source_font_min_pt=minimum, visual_review="pending"))
        plt.close(fig)


def require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise MissingSource(f"{name}: required columns absent: {', '.join(sorted(missing))}")


def primary(frame, sources: Sources, name: str):
    if "tolerance_months" in frame:
        return sources.select(frame, frame.tolerance_months.eq(2), f"{name}: primary ±2 months")
    return frame


def month_dates(values):
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def month_numbers(values):
    dates = pd.to_datetime(values)
    return dates.dt.year * 12 + dates.dt.month


def flag(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(("true", "1", "yes"))


def title(ax, letter, text):
    ax.set_title(text, loc="left", fontsize=8, pad=8)
    ax.text(-0.10, 1.07, letter, transform=ax.transAxes, weight="bold", fontsize=9,
            va="bottom", ha="right")


def interval(ax, center, low, high, position, color, marker="o", horizontal=True):
    values = np.asarray([center, low, high], float)
    if not np.isfinite(values).all() or low > high:
        raise MissingSource("Nonfinite or reversed saved confidence interval")
    # Draw endpoints directly: percentile intervals need not contain the estimate.
    if horizontal:
        ax.hlines(position, low, high, color=color, linewidth=1)
        ax.plot(center, position, marker=marker, color=color, markersize=4)
    else:
        ax.vlines(position, low, high, color=color, linewidth=1)
        ax.plot(position, center, marker=marker, color=color, markersize=4)


def one(frame, **keys):
    local = frame
    for column, value in keys.items():
        local = local.loc[local[column].eq(value)]
    if len(local) != 1:
        raise MissingSource(f"Expected one saved row for {keys}; found {len(local)}")
    return local.iloc[0]


def timing_rows(sources):
    summary = primary(sources.read("evaluation/timing_summary.csv"), sources, "timing_summary")
    intervals = primary(sources.read("evaluation/timing_intervals.csv"), sources, "timing_intervals")
    require(intervals, ("metric", "estimate", "ci_low", "ci_high"), "timing_intervals")
    if "left" in intervals and "right" in intervals:
        intervals = sources.select(intervals, intervals.left.astype(str).str.startswith("P3") & intervals.right.eq("B0"),
                                   "timing: P3/median/Q25/mode minus B0, identical shared-reference cohort")
    elif "contrast" in intervals:
        intervals = sources.select(intervals, intervals.contrast.astype(str).str.match(r"^P3(?:_Q25|_mode)?_minus_B0$"),
                                   "timing: P3 timing contrasts versus B0 only")
    rows = []
    for item in intervals.to_dict("records"):
        metric = str(item["metric"]).lower()
        if "signed" in metric:
            kind = "Signed"
        elif "absolute" in metric or "mae" in metric:
            kind = "Absolute"
        else:
            continue
        label = str(item.get("date_summary", item.get("variant", item.get("left", item.get("contrast", "Median")))))
        label = "Q25" if "q25" in label.lower() else "Mode" if "mode" in label.lower() else "Median"
        count = item.get("shared_events", item.get("n_shared", item.get("n", None)))
        if count is None:
            local = summary
            for column in ("contrast", "date_summary", "variant", "metric"):
                if column in item and column in local:
                    local = local.loc[local[column].eq(item[column])]
            for column in ("shared_events", "n_shared", "n"):
                if column in local and local[column].nunique() == 1:
                    count = local[column].iloc[0]
                    break
        if count is None or pd.isna(count):
            raise MissingSource("Shared-event timing rows require saved sample counts")
        rows.append(dict(item, date_label=label, kind=kind, count=int(count)))
    if not rows:
        raise MissingSource("No saved paired signed/absolute timing differences")
    result = pd.DataFrame(rows)
    if result.duplicated(["date_label", "kind"]).any():
        raise MissingSource("Ambiguous timing contrasts; require one P3-minus-B0 row per summary/metric")
    return result.sort_values(["date_label", "kind"], key=lambda s: s.map(
        {"Median": 0, "Q25": 1, "Mode": 2, "Signed": 0, "Absolute": 1}))


def figure_a(sources):
    metrics = primary(sources.read("evaluation/metrics.csv", ("stage", "precision", "recall", "f1")), sources, "metrics")
    cis = primary(sources.read("evaluation/intervals.csv", ("stage", "metric", "ci_low", "ci_high")), sources, "intervals")
    contrasts = primary(sources.read("evaluation/contrasts.csv", ("contrast", "metric", "estimate", "ci_low", "ci_high")), sources, "contrasts")
    folds = primary(sources.read("evaluation/fold_metrics.csv", ("stage", "outer_fold", "f1")), sources, "fold_metrics")
    timing = timing_rows(sources)
    for stage in STAGES:
        one(metrics, stage=stage)
        for metric in ("precision", "recall", "f1"):
            one(cis, stage=stage, metric=metric)
    pooled = one(contrasts, contrast="P3_minus_B0", metric="f1")
    fold_values = folds.loc[folds.stage.isin(("B0", "P3"))].pivot(index="outer_fold", columns="stage", values="f1")
    if len(fold_values) != 5 or fold_values.isna().any().any():
        raise MissingSource("Figure A requires all five B0/P3 outer-fold estimates")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM / MM, 92 / MM))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.83, wspace=.58)
    ax = axes[0]
    for shift, (metric, color, marker) in zip((-.22, 0, .22),
            (("precision", YELLOW, "s"), ("recall", CHARCOAL, "^"), ("f1", GREEN, "o"))):
        for position, stage in enumerate(STAGES):
            point = one(metrics, stage=stage)
            ci = one(cis, stage=stage, metric=metric)
            interval(ax, point[metric], ci.ci_low, ci.ci_high, position + shift, color, marker, False)
    ax.set(xticks=np.arange(5), xticklabels=STAGES, ylim=(0, 1.04), ylabel="Catalogue agreement")
    ax.legend(handles=[Line2D([], [], marker=m, color=c, linestyle="none", label=l)
                      for l, c, m in (("P", YELLOW, "s"), ("Recall", CHARCOAL, "^"), ("F1", GREEN, "o"))],
              loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3, columnspacing=.65, handletextpad=.25)
    title(ax, "a", "Agreement and trade-off")
    ax = axes[1]
    differences = fold_values.P3 - fold_values.B0
    ax.axvline(0, color=GREY, linestyle=":", linewidth=.9)
    ax.plot(differences.to_numpy(), np.arange(5), "o", color=CHARCOAL, markersize=4)
    interval(ax, pooled.estimate, pooled.ci_low, pooled.ci_high, 6, GREEN, "D")
    ax.set(yticks=[0, 1, 2, 3, 4, 6], yticklabels=[f"Fold {x}" for x in differences.index] + ["Pooled"],
           xlabel="ΔF1 (P3 − B0)", ylim=(6.7, -.7))
    title(ax, "b", "Spatial consistency")
    ax = axes[2]
    ax.axvline(0, color=GREY, linestyle=":", linewidth=.9)
    for position, item in enumerate(timing.itertuples()):
        interval(ax, item.estimate, item.ci_low, item.ci_high, position,
                 GREEN if item.kind == "Signed" else YELLOW, "o" if item.kind == "Signed" else "s")
    labels = [f"{x.date_label} {x.kind.lower()}\nn = {x.count}" for x in timing.itertuples()]
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Paired error Δ (months)",
           ylim=(len(labels) - .4, -.7))
    title(ax, "c", "Shared-event timing")
    fig.text(.075, .035, "±2 months; 95% paired block intervals. Fold points: descriptive. Timing: shared references only.", fontsize=8)
    sources.save(fig, "figure_A_method_comparison", ["P/Recall/F1 trade-off", "Fold robustness and primary paired contrast", "Shared-reference timing"])


def selected_episodes(sources):
    episodes = sources.read("descriptions/episodes.parquet", ("track_event_id", "polygon_track_id", "representative_month"))
    if "route" in episodes and episodes.route.astype(str).str.upper().isin(STAGES).any():
        episodes = sources.select(episodes, episodes.route.astype(str).str.upper().eq("P3"), "episodes: P3 only")
    if "accepted" in episodes:
        episodes = sources.select(episodes, flag(episodes.accepted), "episodes: accepted only")
    if "selected_ratio" not in episodes and "ratio" in episodes:
        # describe() already restricts to each outer fold's selected candidate.
        episodes = episodes.rename(columns={"ratio": "selected_ratio"})
    return episodes


def figure_c(sources):
    episodes = selected_episodes(sources)
    require(episodes, ("selected_ratio", "outer_fold", "q10_month", "q90_month", "response_fraction", "observable_coverage"), "episodes")
    if episodes.empty:
        raise MissingSource("No accepted P3 episodes for cohort diagnostics")
    strata_source = next((name for name in ("evaluation/description_strata_display.csv", "evaluation/description_strata.csv", "evaluation/event_strata.csv")
                         if (sources.root / name).exists()), "evaluation/event_strata.csv")
    strata = primary(sources.read(strata_source), sources, "event_strata")
    if "stratum_type" not in strata and "stratum_variable" in strata:
        strata = strata.rename(columns={"stratum_variable": "stratum_type"})
    if "stage" in strata:
        strata = sources.select(strata, strata.stage.eq("P3"), "cohort errors: P3")
    require(strata, ("stratum_type", "stratum", "recall"), "event_strata")
    chosen = episodes[["outer_fold", "selected_ratio"]].drop_duplicates()
    if chosen.outer_fold.duplicated().any():
        raise MissingSource("Multiple selected P3 ratios inside one held-out fold")
    relevant = strata.stratum_type.astype(str).str.contains("support|area|region|tile|validity|opportunity", case=False, regex=True)
    errors = sources.select(strata, relevant, "subgroups: prespecified geometry/coverage/region categories")
    if strata_source.endswith("description_strata_display.csv"):
        sources.notes.append("Figure C d uses the prespecified area/validity display strata, not outcome-ranked rows. Full region and conditional-support tables accompany the preview and are not additional primary tests.")
        for relative in ("evaluation/description_strata.csv", "evaluation/conditional_support_strata.csv"):
            if (sources.root / relative).exists():
                sources.read(relative)
    if errors.empty:
        errors = strata.copy()
        sources.notes.append("Figure C d: support/area/region errors unavailable; showing saved known-event-position/count strata instead. This requested diagnostic remains incomplete.")
    if errors.empty:
        raise MissingSource("No saved subgroup error summaries")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / MM, 146 / MM))
    fig.subplots_adjust(left=.11, right=.97, bottom=.13, top=.9, hspace=.67, wspace=.5)
    ax = axes[0, 0]
    chosen = chosen.sort_values("outer_fold")
    ax.plot(chosen.outer_fold, chosen.selected_ratio, "s", color=GREEN, markersize=5)
    ax.set(xlabel="Held-out fold", ylabel="Training-selected ratio", xticks=chosen.outer_fold,
           yticks=[.2, .3, .4, .5], ylim=(.17, .53))
    title(ax, "a", "Threshold selection")
    spread = month_numbers(episodes.q90_month) - month_numbers(episodes.q10_month)
    sources.notes.append(f"Q10–Q90 spread: {len(spread)} episodes; {int(spread.isna().sum())} missing, excluded only from histogram.")
    if not spread.notna().any():
        raise MissingSource("All accepted response spreads are missing")
    if (spread.dropna() < 0).any():
        raise ValueError("Q90 precedes Q10")
    ax = axes[0, 1]
    maximum = int(spread.max())
    ax.hist(spread.dropna(), bins=np.arange(-.5, maximum + 1.5), color=GREEN, edgecolor="white")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.set(xlabel="Q10–Q90 response spread (months)", ylabel="Accepted track episodes")
    title(ax, "b", "Inferred temporal spread")
    ax = axes[1, 0]
    good = episodes[["observable_coverage", "response_fraction"]].notna().all(axis=1)
    valid = sources.select(episodes, good, "support/coverage: finite pairs")
    if ((valid[["observable_coverage", "response_fraction"]] < 0) | (valid[["observable_coverage", "response_fraction"]] > 1)).any().any():
        raise ValueError("Support or coverage outside [0, 1]")
    ax.scatter(valid.observable_coverage, valid.response_fraction, s=8, color=GREEN, alpha=.35, edgecolors="none", rasterized=True)
    ax.set(xlabel="Observable / geometric pixels", ylabel="Responders / observable pixels", xlim=(0, 1.03), ylim=(0, 1.03))
    title(ax, "c", "Support is not observed area")
    ax = axes[1, 1]
    display_types = {"event_position": "Position", "known_event_count": "Known count",
                     "pixel_support": "Pixels", "area": "Area", "region": "Region", "tile": "Tile"}
    labels = [f"{display_types.get(str(x.stratum_type), str(x.stratum_type))}: {x.stratum}"
              + (f"\nn = {int(x.reference_events)}" if hasattr(x, "reference_events") else "") for x in errors.itertuples()]
    if len(labels) > 9:
        raise MissingSource("More than nine subgroup rows: supply a compact prespecified subgroup display table; no rows silently dropped")
    ax.plot(1 - errors.recall.to_numpy(float), np.arange(len(errors)), "s", color=YELLOW, markersize=4)
    ax.set(yticks=np.arange(len(errors)), yticklabels=labels, xlabel="Known-reference miss fraction\n(1 − recall)", xlim=(0, 1), ylim=(len(errors) - .4, -.6))
    title(ax, "d", "Catalogue-conditioned failures")
    fig.text(.11, .025, f"a–c: {len(episodes)} accepted P3 track episodes. d: known references; descriptive subgroups, no subgroup tests.", fontsize=8)
    sources.save(fig, "figure_C_cohort_diagnostics", ["Selected ratios", "Response spread, all accepted", "Support and observability, all accepted", "Reference-conditioned subgroup errors"])


def stratified_pages(sources, conditional=False):
    """Paginate every saved regional/support category, never rank by outcomes."""
    relative = ("evaluation/conditional_support_strata.csv" if conditional
                else "evaluation/description_strata.csv")
    frame = primary(sources.read(relative, ("stage", "stratum_type", "stratum")), sources, relative)
    frame = sources.select(frame, frame.stage.eq("P3"), "supplementary diagnostics: P3 only")
    if not conditional:
        frame = sources.select(frame, frame.stratum_type.astype(str).str.contains("region|tile"),
                               "regional pages: all province/canton/tile strata")
    metric = "temporal_mae_months" if conditional else "recall"
    count_column = "matched_events" if conditional else "reference_events"
    require(frame, (metric, count_column), relative)
    pages = []
    for kind, group in frame.groupby("stratum_type", sort=True):
        group = group.sort_values("stratum", kind="stable")
        for start in range(0, len(group), 8):
            pages.append((str(kind), group.iloc[start:start + 8], start, len(group)))
    if not pages:
        raise MissingSource(f"No saved rows for {relative}")
    sources.notes.append(f"Supplementary {relative}: every one of {len(frame)} P3 rows is paginated in stable category order, at most eight rows per panel; none ranked or omitted by outcome.")
    for page_start in range(0, len(pages), 2):
        chunks = pages[page_start:page_start + 2]
        fig, axes = plt.subplots(1, len(chunks), squeeze=False, figsize=(WIDTH_MM / MM, 113 / MM))
        fig.subplots_adjust(left=.25 if len(chunks) == 1 else .21, right=.97, top=.77, bottom=.22, wspace=1.15)
        roles = []
        for offset, (kind, group, first, total) in enumerate(chunks):
            ax = axes[0, offset]
            values = group[metric].to_numpy(float)
            if not conditional:
                values = 1 - values
            labels = [textwrap.fill(str(item.stratum).replace("_", " "), width=18, break_long_words=False, break_on_hyphens=False)
                      + f"\nn = {int(getattr(item, count_column))}" for item in group.itertuples()]
            valid = np.isfinite(values)
            ax.plot(values[valid], np.arange(len(group))[valid], "s" if conditional else "o",
                    color=GREEN if conditional else YELLOW, markersize=4)
            for position in np.where(~valid)[0]:
                ax.text(.98, position, "Unavailable", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
            ax.set(yticks=np.arange(len(group)), yticklabels=labels, ylim=(len(group) - .3, -.7),
                   xlabel="Matched-event MAE (months)" if conditional else "Known-reference miss fraction")
            if conditional:
                ax.set_xlim(0, max(2, np.nanmax(values) * 1.1) if valid.any() else 2)
            else:
                ax.set_xlim(0, 1)
            label = kind.replace("_", " ")
            title(ax, chr(97 + offset), f"{label}\nRows {first + 1}–{first + len(group)} of {total}")
            roles.append("Matched-event-conditioned support errors" if conditional else "All-reference regional error concentration")
        heading = "Support-conditioned timing: original P3 matches only" if conditional else "Regional diagnostics: all known references"
        fig.text(.10, .94, heading, fontsize=9, weight="bold")
        fig.text(.10, .87, "Descriptive estimates; no subgroup significance tests. Missing denominators remain unavailable.", fontsize=8)
        footer = ("n = matched ground events; support uses one label-independent display track.\nUnknown/fallback groups are retained. No recall or disturbed-area claim is made."
                  if conditional else "n = known reference events. Every saved region/tile category is shown across these pages.\nThe catalogue is not an exhaustive no-disturbance census.")
        fig.text(.10, .055, footer, fontsize=8)
        kind_name = "support_conditioned" if conditional else "regions"
        sources.save(fig, f"figure_C_{kind_name}_{page_start // 2 + 1:02d}", roles)


def concordance_pages(sources):
    """Compare all native tracks of frozen multi-track pilots without new extraction."""
    pilots = sources.read("descriptions/pilot_polygons.csv", ("sample_id", "polygon_track_id"))
    tracks = sources.read("inputs/polygon_tracks.parquet", ("sample_id", "polygon_track_id", "track_key"))
    audit = sources.read("inputs/observation_audit.parquet",
                         ("polygon_track_id", "n_geometry", "extracted_pixels", "cached_geometry_coverage"))
    references = sources.read("inputs/references.parquet", ("sample_id", "event_month"))
    episodes = selected_episodes(sources)
    require(episodes, ("q10_month", "q90_month", "N_response", "N_observable", "response_fraction", "ground_event_id"), "episodes")
    cohort = tracks.loc[tracks.sample_id.isin(pilots.sample_id)].copy()
    ids = sorted(cohort.groupby("sample_id").size().loc[lambda x: x > 1].index)
    if not ids:
        raise MissingSource("No multi-track polygons in frozen pilot")
    cohort = sources.select(cohort, cohort.sample_id.isin(ids), "concordance: every track of every multi-track frozen pilot")
    cohort = cohort.merge(audit[["polygon_track_id", "n_geometry", "extracted_pixels", "cached_geometry_coverage"]],
                          on="polygon_track_id", validate="one_to_one").sort_values(["sample_id", "track_key", "polygon_track_id"])
    cohort["frozen_display_track"] = cohort.polygon_track_id.isin(pilots.polygon_track_id)
    events = sources.select(episodes, episodes.polygon_track_id.isin(cohort.polygon_track_id), "concordance: all accepted P3 episodes on included tracks")
    counts = events.groupby("polygon_track_id").size()
    cohort["accepted_P3_episodes"] = cohort.polygon_track_id.map(counts).fillna(0).astype(int)
    cohort.to_csv(sources.output / "source_data/pilot_concordance_tracks.csv", index=False)
    events.to_csv(sources.output / "source_data/pilot_concordance_episodes.csv", index=False)
    sources.notes.append(f"Concordance: {len(ids)} frozen pilot polygons, {len(cohort)} tracks, {len(events)} accepted P3 episodes; all tracks, no-event rows and reference dates retained. Source tables include ground-event membership IDs. No alternate-track raw VV is plotted or newly extracted.")
    cmap = LinearSegmentedColormap.from_list("concordance_support", [YELLOW, GREEN])
    norm = matplotlib.colors.Normalize(0, 1)
    for start in range(0, len(ids), 2):
        page_ids = ids[start:start + 2]
        fig, axes = plt.subplots(len(page_ids), 1, squeeze=False, figsize=(WIDTH_MM / MM, 133 / MM))
        fig.subplots_adjust(left=.37, right=.97, top=.80, bottom=.23, hspace=.95)
        roles = []
        for offset, sample_id in enumerate(page_ids):
            ax = axes[offset, 0]
            group = cohort.loc[cohort.sample_id.eq(sample_id)]
            ref_dates = pd.to_datetime(references.loc[references.sample_id.eq(sample_id), "event_month"])
            ax.plot(ref_dates, np.zeros(len(ref_dates)), "v", color=CHARCOAL, markersize=4, linestyle="none", zorder=4)
            labels = [f"Reference dates (n = {len(ref_dates)})"]
            for y, track in enumerate(group.itertuples(), 1):
                local = events.loc[events.polygon_track_id.eq(track.polygon_track_id)].sort_values(["representative_month", "track_event_id"])
                ax.axhline(y, color=GREY, linewidth=.5, alpha=.6, zorder=0)
                for event in local.itertuples():
                    ax.plot(pd.to_datetime([event.q10_month, event.q90_month]), [y, y], color=CHARCOAL, linewidth=1.4, solid_capstyle="butt", zorder=2)
                    fraction = float(event.response_fraction)
                    if np.isfinite(fraction) and not 0 <= fraction <= 1:
                        raise ValueError("Concordance response fraction outside [0, 1]")
                    ax.scatter(pd.Timestamp(event.representative_month), y, marker="s", s=24,
                               color=cmap(norm(fraction)) if np.isfinite(fraction) else "white",
                               edgecolors=CHARCOAL, linewidths=.45, zorder=3)
                if local.empty:
                    ax.text(.5, y, "No accepted P3 episode", transform=ax.get_yaxis_transform(), ha="center", va="center")
                marker = "*" if track.frozen_display_track else ""
                labels.append(f"{track.track_key}{marker}; {len(local)} events\n"
                              f"G {int(track.n_geometry)}; cached {int(track.extracted_pixels)} ({track.cached_geometry_coverage:.0%})")
            ax.set(yticks=np.arange(len(labels)), yticklabels=labels, ylim=(len(labels) - .5, -.55),
                   xlim=(pd.Timestamp("2016-01-01"), pd.Timestamp("2026-06-01")))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=6)
            short_id = ":".join(str(sample_id).split(":")[-2:])
            title(ax, chr(97 + offset), f"Polygon {short_id}")
            roles.append("All-track P3 date/support concordance with independent references")
        fig.text(.05, .95, "Multi-track pilots: response-date concordance", fontsize=9, weight="bold")
        fig.text(.05, .90, "All accepted P3 episodes; common calendar. Different native grids remain separate.", fontsize=8)
        fig.legend(handles=[Line2D([], [], color=CHARCOAL, marker="v", linestyle="none", label="Reference"),
                            Line2D([], [], color=CHARCOAL, marker="s", markerfacecolor=GREEN, label="Median / Q10–Q90")],
                   loc="upper left", bbox_to_anchor=(.05, .185), ncol=2, borderaxespad=0, handlelength=1.5)
        color_ax = fig.add_axes([.73, .146, .22, .017])
        colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal", ticks=[0, .5, 1])
        colorbar.ax.set_title("Responders / observable", fontsize=8, pad=4)
        fig.text(.05, .035, "* Frozen display track. G = full geometric cells; cached = ever-observed cells.\n"
                 "Spans describe responding pixels, not duration. No alternate-track raw VV or pooled area is shown.", fontsize=8)
        sources.save(fig, f"figure_B_track_concordance_{start // 2 + 1:02d}", roles)


def affine_basis(pixels, track):
    """Return the real native cell basis; never assume map pixels are 30 m."""
    candidates = ("transform", "affine", "native_transform")
    for column in candidates:
        value = track.get(column, None)
        if value is None or (np.isscalar(value) and not isinstance(value, str) and pd.isna(value)):
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)]
        array = np.asarray(value, dtype=float).ravel()
        if len(array) >= 6 and np.isfinite(array[:6]).all():
            expected = np.column_stack((array[0] * (pixels.native_col + .5) + array[1] * (pixels.native_row + .5) + array[2],
                                        array[3] * (pixels.native_col + .5) + array[4] * (pixels.native_row + .5) + array[5]))
            if not np.allclose(expected, pixels[["x_m", "y_m"]].to_numpy(float), atol=.001, rtol=0):
                raise ValueError("Saved metric centres disagree with native affine pixel-centre coordinates")
            return np.array([array[0], array[3]]), np.array([array[1], array[4]])
    columns = ("a", "b", "c", "d", "e", "f")
    if all(column in track for column in columns):
        return np.array([track.a, track.d], float), np.array([track.b, track.e], float)
    # x_m/y_m are explicitly metre-valued saved centres. Recover the affine only
    # when at least three non-collinear native cells determine it exactly.
    design = np.column_stack((pixels.native_col, pixels.native_row, np.ones(len(pixels))))
    targets = pixels[["x_m", "y_m"]].to_numpy(float)
    if np.linalg.matrix_rank(design) < 3:
        raise MissingSource("Calibrated map requires an affine basis for collinear/small interiors")
    coeff, *_ = np.linalg.lstsq(design, targets, rcond=None)
    if not np.allclose(design @ coeff, targets, atol=.001, rtol=0):
        raise MissingSource("Saved metric pixel centres do not follow one native affine grid")
    return coeff[0], coeff[1]


def map_panel(ax, pixels, episode, track, fig):
    require(pixels, ("pixel_track_id", "native_row", "native_col", "x_m", "y_m", "eligible_episode", "response_month"), "pilot_pixels")
    if pixels.pixel_track_id.duplicated().any():
        raise ValueError("Pilot map must have exactly one row per pixel for its displayed episode")
    if pixels.empty:
        raise MissingSource("No saved native pixels for display episode")
    col_basis, row_basis = affine_basis(pixels, track)
    lengths = [np.linalg.norm(col_basis), np.linalg.norm(row_basis)]
    if not np.allclose(lengths, 30, atol=.1, rtol=0):
        raise MissingSource(f"Expected the calibrated native 30 m grid; found basis lengths {lengths}")
    if np.isclose(np.linalg.det(np.column_stack((col_basis, row_basis))), 0):
        raise ValueError("Degenerate native affine cell geometry")
    points = pixels[["x_m", "y_m"]].to_numpy(float)
    if not np.isfinite(points).all():
        raise MissingSource("Nonfinite map coordinates")
    origin = points.min(axis=0)
    offsets = np.array([-.5 * col_basis - .5 * row_basis, .5 * col_basis - .5 * row_basis,
                        .5 * col_basis + .5 * row_basis, -.5 * col_basis + .5 * row_basis])
    cells = points[:, None, :] - origin + offsets[None, :, :]
    eligible = flag(pixels.eligible_episode).to_numpy()
    response = pd.to_datetime(pixels.response_month, errors="raise")
    responding = response.notna().to_numpy()
    if np.any(responding & ~eligible):
        raise ValueError("Displayed responder is outside fixed observable cohort")
    if int(eligible.sum()) != int(episode.N_observable) or int(responding.sum()) != int(episode.N_response):
        raise ValueError("Pilot map counts disagree with frozen episode denominators")
    for mask, face, hatch in ((~eligible, "white", "///"), (eligible & ~responding, GREY, None)):
        if mask.any():
            ax.add_collection(PolyCollection(cells[mask], facecolors=face, edgecolors=GREY, linewidths=.25, hatch=hatch))
    if responding.any():
        observed = sorted(response.loc[responding].dt.to_period("M").unique())
        gradient = LinearSegmentedColormap.from_list("response_months", [YELLOW, GREEN])
        cmap = ListedColormap([GREEN] if len(observed) == 1 else gradient(np.linspace(0, 1, len(observed))))
        lookup = {value: index for index, value in enumerate(observed)}
        colors = [cmap(lookup[value]) for value in response.loc[responding].dt.to_period("M")]
        ax.add_collection(PolyCollection(cells[responding], facecolors=colors, edgecolors="white", linewidths=.15))
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=BoundaryNorm(np.arange(-.5, len(observed) + .5), len(observed)), cmap=cmap),
                           ax=ax, fraction=.045, pad=.025, ticks=np.arange(len(observed)))
        bar.ax.set_yticklabels([str(value) for value in observed])
        bar.set_label("Assigned response month", fontsize=8)
    lower, upper = cells.min(axis=(0, 1)), cells.max(axis=(0, 1))
    width = upper[0] - lower[0]
    bar_length = next((size for size in (300, 150, 90, 60, 30) if size <= width * .4), 30)
    margin = max(30, (upper[1] - lower[1]) * .17)
    ax.set(xlim=(lower[0] - 15, upper[0] + 15), ylim=(lower[1] - margin, upper[1] + 15), aspect="equal")
    ax.plot([lower[0], lower[0] + bar_length], [.10, .10], transform=ax.get_xaxis_transform(), color=CHARCOAL, linewidth=2)
    ax.annotate(f"{bar_length} m", xy=(lower[0] + bar_length / 2, .10), xycoords=ax.get_xaxis_transform(),
                xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=8)
    ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"Support {int(episode.N_response)}/{int(episode.N_observable)}; coverage {episode.observable_coverage:.0%}", labelpad=10)
    ax.legend(handles=[Patch(facecolor=GREY, label="Eligible; no response"), Patch(facecolor="white", edgecolor=GREY, hatch="///", label="Missing / ineligible")],
              loc="upper left", bbox_to_anchor=(.11, .13), bbox_transform=fig.transFigure,
              borderaxespad=0, ncol=1, handlelength=1.2)


def profile_panel(ax, local, episode):
    if "response_count" not in local and "response_pixels" in local:
        local = local.rename(columns={"response_pixels": "response_count"})
    require(local, ("month", "response_count", "cumulative_response_fraction", "N_observable"), "profiles")
    local = local.sort_values("month")
    if local.empty or local.month.duplicated().any():
        raise MissingSource("Missing or duplicate monthly profile bins")
    if local.N_observable.nunique() != 1 or int(local.N_observable.iloc[0]) != int(episode.N_observable):
        raise ValueError("Cumulative curve denominator must stay fixed at episode N_observable")
    counts = local.response_count.to_numpy(float)
    denominator = int(episode.N_observable)
    fractions = local.cumulative_response_fraction.to_numpy(float)
    if denominator <= 0 or not np.allclose(fractions, np.cumsum(counts) / denominator):
        raise ValueError("Saved cumulative profile is not fixed-N assigned responses")
    if np.any(counts < 0) or not np.allclose(counts, np.rint(counts)) or int(counts.sum()) != int(episode.N_response):
        raise ValueError("Histogram does not contain exactly one vote per response pixel")
    dates = month_dates(local.month)
    ax.bar(dates, counts, width=20, color=PALE, edgecolor=GREEN, linewidth=.5)
    ax.set_ylabel("Responding pixels")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    twin = ax.twinx()
    twin.step(dates, fractions, where="post", color=GREEN, linewidth=1.3)
    twin.plot(dates.iloc[-1], fractions[-1], "o", color=GREEN, markersize=4)
    twin.set(ylim=(0, 1.05), ylabel="Cumulative / fixed observable N")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right"); label.set_rotation_mode("anchor")
    ax.set_xlabel(f"Fixed N = {denominator}; endpoint {episode.response_fraction:.0%}")


def pilot_pages(sources, diagnostic=False):
    selection_file = "diagnostic_polygons.csv" if diagnostic else "pilot_polygons.csv"
    pilot = sources.read(f"descriptions/{selection_file}", ("sample_id", "polygon_track_id"))
    if pilot.sample_id.duplicated().any():
        raise ValueError("Frozen pilot must contain each selected polygon once")
    if diagnostic:
        sources.notes.append(f"Separate diagnostic case list has {len(pilot)} polygons; never substitutes for the frozen pilot.")
    else:
        sources.notes.append(f"Frozen pilot has {len(pilot)} polygons; target 12; no score-based substitutions.")
    series = sources.read("descriptions/pilot_series.parquet", ("polygon_track_id", "month", "vv_median", "vv_q25", "vv_q75"))
    references = sources.read("inputs/references.parquet", ("sample_id",))
    reference_date = next((x for x in ("event_month", "reference_month", "reference_time", "change_time", "reference_date") if x in references), None)
    if reference_date is None:
        raise MissingSource("Reference table needs reference_month/reference_time/change_time/reference_date")
    tracks = sources.read("inputs/polygon_tracks.parquet", ("polygon_track_id",))
    episodes = selected_episodes(sources)
    pixels = sources.read("descriptions/pilot_pixels.parquet", ("polygon_track_id",))
    profiles = sources.read("descriptions/profiles.parquet", ("track_event_id",))
    ground_methods = {route: sources.read(f"oof/{route}.parquet", ("sample_id", "change_time"))
                      for route in ("B0", "P3")} if diagnostic else {}
    for index, case in enumerate(pilot.to_dict("records"), 1):
        sample = str(case["sample_id"]); identifier = str(case["polygon_track_id"])
        raw = sources.select(series, series.polygon_track_id.astype(str).eq(identifier), f"pilot {index}: frozen display-track raw VV").sort_values("month")
        if raw.empty:
            sources.notes.append(f"ABSENT pilot {index}: no raw series; selected polygon {sample} remains in frozen table.")
            continue
        if raw.month.duplicated().any():
            raise ValueError("Pilot trajectory has duplicated months")
        track = one(tracks, polygon_track_id=identifier)
        local = sources.select(episodes, episodes.polygon_track_id.astype(str).eq(identifier), f"pilot {index}: accepted display-track episodes")
        selected_id = case.get("track_event_id", None)
        if selected_id is not None and pd.notna(selected_id):
            chosen = local.loc[local.track_event_id.eq(selected_id)]
        else:
            chosen = local.sort_values(["representative_month", "track_event_id"]).iloc[:1]
        fig = plt.figure(figsize=(WIDTH_MM / MM, 163 / MM))
        grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.4), width_ratios=(1.06, 1),
                               left=.11, right=.87, bottom=.18, top=.84, hspace=.65, wspace=.63)
        trajectory = fig.add_subplot(grid[0, :]); map_ax = fig.add_subplot(grid[1, 0]); profile_ax = fig.add_subplot(grid[1, 1])
        dates = month_dates(raw.month)
        trajectory.fill_between(dates, raw.vv_q25.to_numpy(float), raw.vv_q75.to_numpy(float), color=PALE, linewidth=0)
        trajectory.plot(dates, raw.vv_median, color=CHARCOAL, linewidth=.9)
        refs = references.loc[references.sample_id.astype(str).eq(sample)]
        for value in month_dates(refs[reference_date]):
            trajectory.axvline(value, color=CHARCOAL, linestyle=":", linewidth=.8)
            trajectory.plot(value, .95, marker="v", color=CHARCOAL, markersize=5, transform=trajectory.get_xaxis_transform())
        detection_legend = []
        if diagnostic:
            for route, color, marker, line_style, level in (("B0", YELLOW, "D", "-.", .17), ("P3", GREEN, "^", "--", .05)):
                ground = ground_methods[route]
                displayed = sources.select(ground, ground.sample_id.astype(str).eq(sample), f"diagnostic {index}: {route} ground events")
                for value in month_dates(displayed.change_time):
                    trajectory.axvline(value, color=color, linestyle=line_style, linewidth=.8)
                    trajectory.plot(value, level, marker=marker, color=color, markersize=4, transform=trajectory.get_xaxis_transform())
                detection_legend.append(Line2D([], [], color=color, linestyle=line_style, marker=marker, label=f"{route} ground event"))
        else:
            for value in month_dates(local.representative_month):
                trajectory.axvline(value, color=GREEN, linestyle="--", linewidth=.8)
                trajectory.plot(value, .05, marker="^", color=GREEN, markersize=5, transform=trajectory.get_xaxis_transform())
            detection_legend.append(Line2D([], [], color=GREEN, linestyle="--", marker="^", label="P3 track episode"))
        trajectory.axvline(pd.Timestamp("2017-01-01"), color=GREY, linestyle="-.", linewidth=.7)
        trajectory.axvline(pd.Timestamp("2026-01-01"), color=GREY, linestyle="-.", linewidth=.7)
        trajectory.set_ylabel("VV (dB)")
        trajectory.xaxis.set_major_locator(mdates.YearLocator(2)); trajectory.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        trajectory.legend(handles=[Line2D([], [], color=CHARCOAL, label="Raw VV median"), Patch(color=PALE, label="Pixel IQR"),
                                   Line2D([], [], color=CHARCOAL, linestyle=":", marker="v", label="Reference")] + detection_legend,
                          loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, columnspacing=1.2)
        title(trajectory, "a", "Raw display-track trajectory and independently dated references")
        title(map_ax, "b", "Native-pixel response dates")
        title(profile_ax, "c", "Assigned pixel responses")
        fallback = str(track.get("support_mode", "")).lower() == "representative_fallback" or flag(pd.Series([case.get("fallback_extraction", case.get("fallback", False))])).iloc[0]
        if chosen.empty:
            for ax in (map_ax, profile_ax):
                ax.set_axis_off(); ax.text(.5, .5, "No accepted P3 episode\non the frozen display track", transform=ax.transAxes, ha="center", va="center")
            sources.notes.append(f"Pilot {index}: no accepted P3 display-track episode; map/profile unavailable, not zeros.")
        else:
            episode = chosen.iloc[0]
            if fallback:
                map_ax.set_axis_off(); map_ax.text(.5, .5, "Representative-pixel fallback\nNo within-stand spatial map", transform=map_ax.transAxes, ha="center", va="center")
                sources.notes.append(f"Pilot {index}: fallback extraction; spatial map intentionally unavailable.")
            else:
                local_pixels = pixels.loc[pixels.polygon_track_id.astype(str).eq(identifier)]
                if "track_event_id" in local_pixels:
                    local_pixels = local_pixels.loc[local_pixels.track_event_id.eq(episode.track_event_id)]
                try:
                    map_panel(map_ax, local_pixels, episode, track, fig)
                except MissingSource as error:
                    map_ax.set_axis_off(); map_ax.text(.5, .5, "Map unavailable\nSee figure QA", transform=map_ax.transAxes, ha="center", va="center")
                    sources.notes.append(f"Pilot {index} map ABSENT: {error}")
            local_profile = profiles.loc[profiles.track_event_id.eq(episode.track_event_id)]
            try:
                profile_panel(profile_ax, local_profile, episode)
            except MissingSource as error:
                profile_ax.set_axis_off(); profile_ax.text(.5, .5, "Profile unavailable\nSee figure QA", transform=profile_ax.transAxes, ha="center", va="center")
                sources.notes.append(f"Pilot {index} profile ABSENT: {error}")
        category = str(case.get("pilot_category", case.get("pilot_stratum", case.get("stratum", "Frozen pilot"))))
        page_label = "Diagnostic" if diagnostic else "Pilot"
        category = {"single_track_single_reference": "Single track / one known event", "repeated_references": "Repeated known events",
                    "repeated_reference": "Repeated known events", "multi_track": "Multiple tracks",
                    "limited_support": "Limited coverage / support", "low_coverage_or_small": "Limited coverage / small interior",
                    "diagnostic_method_disagreement": "B0/P3 ground-event disagreement",
                    "diagnostic_missed_known_reference": "Missed known reference",
                    "diagnostic_long_chain": "Long-chain parent / bounded P3 child"}.get(category, category.replace("_", " "))
        fig.text(.11, .97, f"{page_label} {index:02d} | {category}", fontsize=9, weight="bold")
        fig.text(.11, .94, f"Polygon: {sample}", fontsize=8)
        episode_label = "none" if chosen.empty else pd.Timestamp(chosen.iloc[0].representative_month).strftime("%Y-%m")
        fig.text(.11, .91, f"Display track: {track.get('track_key', identifier)}; map/profile episode: {episode_label}", fontsize=8)
        fig.text(.11, .035, "Inferred response timing/support, not validated disturbance duration or area.\nRaw trajectory gaps remain missing; reference dates do not define episode membership.", fontsize=8)
        sources.save(fig, f"figure_B_{page_label.lower()}_{index:02d}", ["Raw VV and distinct reference/detection markers", "Calibrated native-pixel response map", "Responder histogram and fixed-cohort cumulative profile"])


def build(run_dir: str | Path) -> dict:
    """Create previews and their contract/QA inside an existing experiment run."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": CHARCOAL, "axes.labelcolor": CHARCOAL, "xtick.color": CHARCOAL, "ytick.color": CHARCOAL})
    (root / "FIGURE_CONTRACT.md").write_text(CONTRACT, encoding="utf-8")
    sources = Sources(root)
    unavailable = {}
    for name, function in (("A", figure_a), ("B", pilot_pages), ("C", figure_c)):
        try:
            function(sources)
        except MissingSource as error:
            unavailable[name] = str(error)
            sources.notes.append(f"Figure {name} ABSENT or incomplete: {error}")
            plt.close("all")
    for conditional, relative in ((False, "evaluation/description_strata.csv"), (True, "evaluation/conditional_support_strata.csv")):
        if (root / relative).exists():
            try:
                stratified_pages(sources, conditional=conditional)
            except MissingSource as error:
                sources.notes.append(f"Supplementary stratified pages unavailable: {error}")
                plt.close("all")
    if (root / "descriptions/diagnostic_polygons.csv").exists():
        try:
            pilot_pages(sources, diagnostic=True)
        except MissingSource as error:
            unavailable["B_diagnostics"] = str(error)
            sources.notes.append(f"Separate diagnostic pages ABSENT or incomplete: {error}")
            plt.close("all")
    else:
        sources.notes.append("Separate disagreement/missed-reference/long-chain diagnostic pages ABSENT: no saved diagnostic_polygons.csv. Frozen pilot cases were not replaced.")
    try:
        concordance_pages(sources)
    except MissingSource as error:
        unavailable["B_concordance"] = str(error)
        sources.notes.append(f"Multi-track concordance pages ABSENT or incomplete: {error}")
        plt.close("all")
    pd.DataFrame(sources.panels).to_csv(sources.output / "panel_qa.csv", index=False)
    notes = ["# Figure QA notes", "", "Status: data-driven previews; final-size human visual review and exported PDF glyph audit remain pending.",
             "The requested raster delivery is 600-dpi PNG, not TIFF; the source validator's TIFF warning is expected for this preview contract.",
             "", "No source observations were simulated or downsampled. Missing panels remain unavailable, not false zero-valued plots.",
             "", "## Sources, filters and exclusions", ""] + [f"- {note}" for note in sources.notes]
    notes += ["", "## Interpretation", "", "These are spatially held-out development comparisons. Precision/F1 are catalogue-conditional.",
              "No operational precision, full rotation recovery, pixel accuracy, true onset/duration or disturbed area is established.",
              "IQR/histograms are descriptive distributions, not confidence intervals. Fold/subgroup points are descriptive unless saved intervals are shown.",
              "", "## Render review required", "", "Review each panel's labels, legend clearance, missingness, intervals and denominator at final size.",
              "Confirm every PDF text run is at least 5 pt; SVG text must remain editable. Check P3 salience and grayscale encoding.",
              "", "## Products", ""] + [f"- {name}" for name in sources.products]
    (root / "FIGURE_QA.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    result = dict(products=sources.products, unavailable=unavailable, visual_qa="pending",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  products_sha256={relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                                   for relative in sources.products})
    (sources.output / "figure_build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.run_dir), indent=2))
