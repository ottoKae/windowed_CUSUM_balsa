"""Auditable component engine for the Windowed CUSUM ablation experiment.

This module deliberately does not replace :func:`windowed_cusum.detector.core.lr_cusum`.
Instead, it evaluates the production mathematical primitives once for a global
trajectory and once for the common sliding-window geometry, records every gate
needed by the experiment, and derives detector variants by Boolean selection.

The complete forward stage ``F4`` preserves the legacy implementation's
calendar, candidate-index, clipped-window-start seed, direction, and empirical
confidence semantics.  Keeping the production wrapper untouched makes exact
parity an explicit experiment gate rather than an assumption introduced by a
refactor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from windowed_cusum.detector.core import _calendar_align, bootstrap_confidence, cusum_amplitude, generate_windows


@dataclass(frozen=True)
class ComponentAblationConfig:
    """Frozen detector settings shared by every component variant.

    ``seed`` is the experiment-level default.  Callers reproducing grouped
    production inference should pass the already-derived object-and-band seed
    through ``series_seed`` in :func:`build_component_ledgers`.
    """

    window_length: int = 48
    step: int = 12
    confidence_threshold: float = 0.975
    bootstrap_iterations: int = 800
    minimum_observations: int = 12
    seed: int = 20260820
    start_year: int | None = 2017
    end_year: int | None = 2025

    def __post_init__(self) -> None:
        if self.window_length < 3:
            raise ValueError("window_length must be >= 3")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be >= 1")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must not exceed end_year")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for run manifests."""

        return asdict(self)


@dataclass(frozen=True)
class VariantDefinition:
    """Boolean admission rule applied to one of the two master ledgers."""

    variant_id: str
    short_name: str
    source_ledger: str
    require_central: bool = False
    require_decrease: bool = False
    require_permutation: bool = False
    interpretation: str = ""

    def __post_init__(self) -> None:
        if self.source_ledger not in {"global", "sliding"}:
            raise ValueError("source_ledger must be 'global' or 'sliding'")
        if self.source_ledger == "global" and self.require_central:
            raise ValueError("the central-window gate is not defined for a global trajectory")


FORWARD_VARIANTS: tuple[VariantDefinition, ...] = (
    VariantDefinition(
        "F0",
        "global_raw",
        "global",
        interpretation="Exact CUSUM range statistic applied once to the complete trajectory.",
    ),
    VariantDefinition(
        "F1",
        "window_raw",
        "sliding",
        interpretation="One unfiltered candidate per eligible sliding window.",
    ),
    VariantDefinition(
        "F2",
        "window_central",
        "sliding",
        require_central=True,
        interpretation="Sliding candidates restricted to the central decision interval.",
    ),
    VariantDefinition(
        "F3",
        "window_central_decrease",
        "sliding",
        require_central=True,
        require_decrease=True,
        interpretation="Central candidates restricted to negative post-minus-pre changes.",
    ),
    VariantDefinition(
        "F4",
        "window_central_decrease_permutation",
        "sliding",
        require_central=True,
        require_decrease=True,
        require_permutation=True,
        interpretation="Complete frozen Windowed CUSUM detector.",
    ),
)


REMOVAL_VARIANTS: tuple[VariantDefinition, ...] = (
    VariantDefinition(
        "R0",
        "full_global",
        "global",
        require_decrease=True,
        require_permutation=True,
        interpretation="Global context with the final direction and permutation gates.",
    ),
    VariantDefinition(
        "R1",
        "full_no_central",
        "sliding",
        require_decrease=True,
        require_permutation=True,
        interpretation="Complete sliding detector without the central-decision gate.",
    ),
    VariantDefinition(
        "R2",
        "full_no_decrease",
        "sliding",
        require_central=True,
        require_permutation=True,
        interpretation="Complete sliding detector without the decrease-direction gate.",
    ),
    VariantDefinition(
        "R3",
        "full_no_permutation",
        "sliding",
        require_central=True,
        require_decrease=True,
        interpretation="Complete sliding detector without empirical admission; identical to F3.",
    ),
)


VARIANT_REGISTRY: Mapping[str, VariantDefinition] = MappingProxyType(
    {variant.variant_id: variant for variant in FORWARD_VARIANTS + REMOVAL_VARIANTS}
)


_BASE_LEDGER_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "series_key",
    "candidate_scope",
    "logical_window_id",
    "window_start_index",
    "window_end_index",
    "decision_start_index",
    "decision_end_index",
    "window_start",
    "window_end",
    "decision_start",
    "decision_end",
    "window_observations",
    "valid_observations",
    "valid_fraction",
    "clean_change_index",
    "local_change_index",
    "change_index",
    "change_time",
    "distance_to_window_edge_months",
    "cusum_amplitude",
    "pre_observations",
    "post_observations",
    "pre_mean",
    "post_mean",
    "post_minus_pre",
    "permutation_confidence",
    "bootstrap_confidence",
    "empirical_exceedance_fraction",
    "base_series_seed",
    "permutation_seed",
    "permutation_iterations",
    "permutation_runtime_seconds",
    "passes_window_calendar",
    "passes_reporting_calendar",
    "passes_minimum_observations",
    "passes_cusum_index",
    "passes_split",
    "candidate_valid",
    "passes_central",
    "passes_decrease",
    "passes_permutation",
    "f4_rejection_reason",
)

_ACCEPTANCE_COLUMNS: tuple[str, ...] = tuple(
    f"accepted_{variant.variant_id.lower()}"
    for variant in FORWARD_VARIANTS + REMOVAL_VARIANTS
)

GLOBAL_LEDGER_COLUMNS: tuple[str, ...] = _BASE_LEDGER_COLUMNS + _ACCEPTANCE_COLUMNS
SLIDING_LEDGER_COLUMNS: tuple[str, ...] = GLOBAL_LEDGER_COLUMNS


def _legacy_window_calendar_gate(
    dates: pd.DatetimeIndex,
    decision_start: int,
    decision_end: int,
    config: ComponentAblationConfig,
) -> bool:
    """Reproduce the production window-level year eligibility checks exactly."""

    if decision_start >= decision_end or not len(dates):
        return False
    if config.start_year is not None and dates[decision_end - 1].year < config.start_year:
        return False
    if config.end_year is not None and dates[decision_start].year > config.end_year:
        return False
    return True


def _reporting_calendar_gate(
    candidate_time: pd.Timestamp | pd.NaT,
    config: ComponentAblationConfig,
) -> bool:
    if pd.isna(candidate_time):
        return False
    timestamp = pd.Timestamp(candidate_time)
    if config.start_year is not None and timestamp.year < config.start_year:
        return False
    if config.end_year is not None and timestamp.year > config.end_year:
        return False
    return True


def _empty_record(
    *,
    series_key: str,
    scope: str,
    logical_window_id: int,
    window_start: int,
    window_end: int,
    decision_start: int,
    decision_end: int,
    dates: pd.DatetimeIndex,
    series_seed: int,
    config: ComponentAblationConfig,
    metadata: Mapping[str, object],
    passes_window_calendar: bool,
) -> dict[str, object]:
    window_count = max(0, window_end - window_start)
    record: dict[str, object] = dict(metadata)
    record.update({
        "candidate_id": f"{series_key}|{scope}|w{logical_window_id}",
        "series_key": str(series_key),
        "candidate_scope": scope,
        "logical_window_id": int(logical_window_id),
        "window_start_index": int(window_start),
        "window_end_index": int(window_end),
        "decision_start_index": int(decision_start),
        "decision_end_index": int(decision_end),
        "window_start": pd.Timestamp(dates[window_start]) if window_count else pd.NaT,
        "window_end": pd.Timestamp(dates[window_end - 1]) if window_count else pd.NaT,
        "decision_start": pd.Timestamp(dates[decision_start])
        if decision_start < len(dates) else pd.NaT,
        "decision_end": pd.Timestamp(dates[decision_end - 1])
        if decision_start < decision_end <= len(dates) else pd.NaT,
        "window_observations": int(window_count),
        "valid_observations": 0,
        "valid_fraction": 0.0,
        "clean_change_index": pd.NA,
        "local_change_index": pd.NA,
        "change_index": pd.NA,
        "change_time": pd.NaT,
        "distance_to_window_edge_months": pd.NA,
        "cusum_amplitude": np.nan,
        "pre_observations": 0,
        "post_observations": 0,
        "pre_mean": np.nan,
        "post_mean": np.nan,
        "post_minus_pre": np.nan,
        "permutation_confidence": np.nan,
        "bootstrap_confidence": np.nan,
        "empirical_exceedance_fraction": np.nan,
        "base_series_seed": int(series_seed),
        # Production uses the clipped window start, not the logical window ID.
        "permutation_seed": int(series_seed + window_start),
        "permutation_iterations": int(config.bootstrap_iterations),
        "permutation_runtime_seconds": 0.0,
        "passes_window_calendar": bool(passes_window_calendar),
        "passes_reporting_calendar": False,
        "passes_minimum_observations": False,
        "passes_cusum_index": False,
        "passes_split": False,
        "candidate_valid": False,
        "passes_central": False,
        "passes_decrease": False,
        "passes_permutation": False,
        "f4_rejection_reason": "not_evaluated",
    })
    for column in _ACCEPTANCE_COLUMNS:
        record[column] = False
    return record


def _scan_candidate(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    series_key: str,
    scope: str,
    logical_window_id: int,
    window_start: int,
    window_end: int,
    decision_start: int,
    decision_end: int,
    series_seed: int,
    config: ComponentAblationConfig,
    metadata: Mapping[str, object],
    passes_window_calendar: bool,
) -> dict[str, object]:
    """Evaluate one candidate window and retain every component gate."""

    record = _empty_record(
        series_key=series_key,
        scope=scope,
        logical_window_id=logical_window_id,
        window_start=window_start,
        window_end=window_end,
        decision_start=decision_start,
        decision_end=decision_end,
        dates=dates,
        series_seed=series_seed,
        config=config,
        metadata=metadata,
        passes_window_calendar=passes_window_calendar,
    )
    window = values[window_start:window_end]
    valid_indices = np.flatnonzero(np.isfinite(window))
    valid_count = int(len(valid_indices))
    record["valid_observations"] = valid_count
    record["valid_fraction"] = float(valid_count / len(window)) if len(window) else 0.0
    enough = valid_count >= config.minimum_observations
    record["passes_minimum_observations"] = enough
    if not enough:
        record["f4_rejection_reason"] = _full_rejection_reason(record)
        return record

    clean = np.asarray(window[valid_indices], dtype=np.float64)
    amplitude, clean_index = cusum_amplitude(clean)
    record["cusum_amplitude"] = float(amplitude)
    record["clean_change_index"] = int(clean_index) if clean_index >= 0 else pd.NA
    if clean_index < 0:
        record["f4_rejection_reason"] = _full_rejection_reason(record)
        return record
    record["passes_cusum_index"] = True

    local_index = int(valid_indices[int(clean_index)])
    global_index = int(window_start + local_index)
    record["local_change_index"] = local_index
    record["change_index"] = global_index
    record["change_time"] = pd.Timestamp(dates[global_index])
    record["passes_reporting_calendar"] = _reporting_calendar_gate(
        record["change_time"], config
    )
    record["distance_to_window_edge_months"] = int(
        min(local_index, len(window) - 1 - local_index)
    )
    record["passes_central"] = bool(decision_start <= global_index < decision_end)

    # This split test and the pre/post means intentionally match core.lr_cusum.
    split_valid = local_index != 0 and local_index < len(window)
    if not split_valid:
        record["f4_rejection_reason"] = _full_rejection_reason(record)
        return record
    before_window = window[:local_index]
    after_window = window[local_index:]
    pre_count = int(np.isfinite(before_window).sum())
    post_count = int(np.isfinite(after_window).sum())
    record["pre_observations"] = pre_count
    record["post_observations"] = post_count
    if not pre_count or not post_count:
        record["f4_rejection_reason"] = _full_rejection_reason(record)
        return record
    before = float(np.nanmean(before_window))
    after = float(np.nanmean(after_window))
    if not (np.isfinite(before) and np.isfinite(after)):
        record["f4_rejection_reason"] = _full_rejection_reason(record)
        return record
    record["passes_split"] = True
    record["candidate_valid"] = True
    record["pre_mean"] = before
    record["post_mean"] = after
    record["post_minus_pre"] = after - before
    record["passes_decrease"] = bool(after < before)

    # Evaluate once for the master ledger. Extra evaluations for candidates
    # rejected by an earlier gate do not affect F4 because the primitive resets
    # its RNG from the deterministic seed on every invocation.
    started = perf_counter()
    confidence = float(
        bootstrap_confidence(
            clean,
            float(amplitude),
            config.bootstrap_iterations,
            int(series_seed + window_start),
        )
    )
    record["permutation_runtime_seconds"] = float(perf_counter() - started)
    record["permutation_confidence"] = confidence
    record["bootstrap_confidence"] = confidence
    record["empirical_exceedance_fraction"] = 1.0 - confidence
    record["passes_permutation"] = bool(confidence >= config.confidence_threshold)
    record["f4_rejection_reason"] = _full_rejection_reason(record)
    return record


def _full_rejection_reason(record: Mapping[str, object]) -> str:
    """Return the first failed gate in production F4 order."""

    if not bool(record["passes_window_calendar"]):
        return "outside_window_calendar"
    if not bool(record["passes_minimum_observations"]):
        return "insufficient_observations"
    if not bool(record["passes_cusum_index"]):
        return "invalid_cusum_index"
    if not bool(record["passes_split"]):
        return "invalid_split"
    if not bool(record["passes_central"]):
        return "outside_central_decision"
    if not bool(record["passes_decrease"]):
        return "non_decrease"
    if not bool(record["passes_permutation"]):
        return "below_permutation_threshold"
    return "accepted"


def _acceptance_mask(
    frame: pd.DataFrame,
    definition: VariantDefinition,
    *,
    reporting_only: bool,
) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    mask = frame["passes_window_calendar"].fillna(False).astype(bool)
    mask &= frame["candidate_valid"].fillna(False).astype(bool)
    if definition.require_central:
        mask &= frame["passes_central"].fillna(False).astype(bool)
    if definition.require_decrease:
        mask &= frame["passes_decrease"].fillna(False).astype(bool)
    if definition.require_permutation:
        mask &= frame["passes_permutation"].fillna(False).astype(bool)
    if reporting_only:
        mask &= frame["passes_reporting_calendar"].fillna(False).astype(bool)
    return mask


def _rejection_reason(
    row: Mapping[str, object],
    definition: VariantDefinition,
    *,
    reporting_only: bool,
) -> str:
    if not bool(row["passes_window_calendar"]):
        return "outside_window_calendar"
    if not bool(row["passes_minimum_observations"]):
        return "insufficient_observations"
    if not bool(row["passes_cusum_index"]):
        return "invalid_cusum_index"
    if not bool(row["passes_split"]):
        return "invalid_split"
    if definition.require_central and not bool(row["passes_central"]):
        return "outside_central_decision"
    if definition.require_decrease and not bool(row["passes_decrease"]):
        return "non_decrease"
    if definition.require_permutation and not bool(row["passes_permutation"]):
        return "below_permutation_threshold"
    if reporting_only and not bool(row["passes_reporting_calendar"]):
        return "outside_reporting_calendar"
    return "accepted"


def _with_acceptance_columns(frame: pd.DataFrame, source_ledger: str) -> pd.DataFrame:
    output = frame.copy()
    for definition in FORWARD_VARIANTS + REMOVAL_VARIANTS:
        column = f"accepted_{definition.variant_id.lower()}"
        if definition.source_ledger == source_ledger:
            output[column] = _acceptance_mask(output, definition, reporting_only=False)
        else:
            output[column] = False
    return output


def _frame_from_records(
    records: list[dict[str, object]],
    *,
    metadata_columns: tuple[str, ...],
    source_ledger: str,
) -> pd.DataFrame:
    columns = metadata_columns + GLOBAL_LEDGER_COLUMNS
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame = _with_acceptance_columns(frame, source_ledger)
    ordered = list(dict.fromkeys(columns))
    extras = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, ordered + extras]


def derive_variant_events(
    global_candidates: pd.DataFrame,
    sliding_candidates: pd.DataFrame,
    variant_id: str,
    *,
    reporting_only: bool = False,
) -> pd.DataFrame:
    """Select one detector variant from persisted master candidate ledgers."""

    try:
        definition = VARIANT_REGISTRY[str(variant_id).upper()]
    except KeyError as error:
        raise KeyError(f"unknown variant_id {variant_id!r}") from error
    source = global_candidates if definition.source_ledger == "global" else sliding_candidates
    mask = _acceptance_mask(source, definition, reporting_only=reporting_only)
    selected = source.loc[mask].copy()
    selected.insert(0, "variant_id", definition.variant_id)
    selected.insert(1, "variant_name", definition.short_name)
    return selected.reset_index(drop=True)


def variant_admission_table(
    global_candidates: pd.DataFrame,
    sliding_candidates: pd.DataFrame,
    *,
    include_removals: bool = True,
    reporting_only: bool = False,
) -> pd.DataFrame:
    """Return one candidate-by-variant audit table without rescanning series."""

    definitions = FORWARD_VARIANTS + (REMOVAL_VARIANTS if include_removals else ())
    rows: list[pd.DataFrame] = []
    for definition in definitions:
        source = global_candidates if definition.source_ledger == "global" else sliding_candidates
        if source.empty:
            continue
        mask = _acceptance_mask(source, definition, reporting_only=reporting_only)
        columns = [
            column for column in (
                "sample_id",
                "polygon_id",
                "polygon_track_id",
                "tile_id",
                "track_key",
                "Band",
                "role",
                "species_nco",
                "candidate_id",
                "series_key",
                "candidate_scope",
                "logical_window_id",
                "change_time",
                "cusum_amplitude",
                "post_minus_pre",
                "permutation_confidence",
            ) if column in source
        ]
        local = source.loc[:, columns].copy()
        local.insert(0, "variant_id", definition.variant_id)
        local.insert(1, "variant_name", definition.short_name)
        local["accepted"] = mask.to_numpy(dtype=bool)
        local["rejection_reason"] = [
            _rejection_reason(row, definition, reporting_only=reporting_only)
            for row in source.to_dict(orient="records")
        ]
        rows.append(local)
    if not rows:
        return pd.DataFrame(columns=[
            "variant_id", "variant_name", "candidate_id", "accepted", "rejection_reason"
        ])
    return pd.concat(rows, ignore_index=True)


@dataclass
class ComponentLedgers:
    """Global and sliding candidate ledgers derived from one input series."""

    global_candidates: pd.DataFrame
    sliding_candidates: pd.DataFrame
    config: ComponentAblationConfig

    def stage_frame(self, variant_id: str, *, reporting_only: bool = False) -> pd.DataFrame:
        return derive_variant_events(
            self.global_candidates,
            self.sliding_candidates,
            variant_id,
            reporting_only=reporting_only,
        )

    def admission_table(
        self,
        *,
        include_removals: bool = True,
        reporting_only: bool = False,
    ) -> pd.DataFrame:
        return variant_admission_table(
            self.global_candidates,
            self.sliding_candidates,
            include_removals=include_removals,
            reporting_only=reporting_only,
        )

    def assert_forward_nesting(self) -> None:
        """Raise when the pre-fusion F1--F4 candidate flow is not nested."""

        identifiers = {
            variant_id: set(self.stage_frame(variant_id)["candidate_id"].astype(str))
            for variant_id in ("F1", "F2", "F3", "F4")
        }
        for left, right in (("F2", "F1"), ("F3", "F2"), ("F4", "F3")):
            if not identifiers[left].issubset(identifiers[right]):
                difference = sorted(identifiers[left] - identifiers[right])[:5]
                raise AssertionError(f"{left} is not a subset of {right}: {difference}")

    def legacy_f4_events(self) -> pd.DataFrame:
        """Return admitted F4 rows with backward-compatible event fields."""

        return self.stage_frame("F4")


def build_component_ledgers(
    series,
    dates,
    *,
    config: ComponentAblationConfig = ComponentAblationConfig(),
    series_key: str = "",
    series_seed: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ComponentLedgers:
    """Scan one trajectory twice and materialize all F0--F4 component gates.

    Parameters
    ----------
    series, dates:
        One monthly trajectory and its calendar.  Missing observations remain
        ``NaN`` and retain their calendar positions.  Calendar alignment is the
        same as the production wrapper.
    config:
        Frozen component geometry and empirical-admission settings.
    series_key:
        Stable identifier used only to construct auditable candidate IDs.
    series_seed:
        Object-and-band seed passed to the production wrapper.  When omitted,
        ``config.seed`` is used directly.
    metadata:
        Optional scalar identifiers (for example sample, polygon, track, tile,
        role, and band) copied to every candidate row.
    """

    values, aligned_dates = _calendar_align(series, dates)
    values = np.asarray(values, dtype=np.float64)
    aligned_dates = pd.DatetimeIndex(aligned_dates)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if len(values) != len(aligned_dates):
        raise ValueError("aligned series and dates must have equal lengths")
    local_metadata = dict(metadata or {})
    reserved = set(GLOBAL_LEDGER_COLUMNS) | {"variant_id", "variant_name"}
    collisions = sorted(reserved.intersection(local_metadata))
    if collisions:
        raise ValueError(f"metadata keys collide with ledger schema: {collisions}")
    metadata_columns = tuple(local_metadata)
    resolved_seed = int(config.seed if series_seed is None else series_seed)

    global_records: list[dict[str, object]] = []
    if len(values):
        global_records.append(_scan_candidate(
            values,
            aligned_dates,
            series_key=str(series_key),
            scope="global",
            logical_window_id=0,
            window_start=0,
            window_end=len(values),
            decision_start=0,
            decision_end=len(values),
            series_seed=resolved_seed,
            config=config,
            metadata=local_metadata,
            passes_window_calendar=_legacy_window_calendar_gate(
                aligned_dates, 0, len(values), config
            ),
        ))

    sliding_records: list[dict[str, object]] = []
    for logical_window_id, (window_start, window_end, decision_start, decision_end) in enumerate(
        generate_windows(len(values), config.window_length, config.step)
    ):
        sliding_records.append(_scan_candidate(
            values,
            aligned_dates,
            series_key=str(series_key),
            scope="sliding",
            logical_window_id=logical_window_id,
            window_start=window_start,
            window_end=window_end,
            decision_start=decision_start,
            decision_end=decision_end,
            series_seed=resolved_seed,
            config=config,
            metadata=local_metadata,
            passes_window_calendar=_legacy_window_calendar_gate(
                aligned_dates, decision_start, decision_end, config
            ),
        ))

    ledgers = ComponentLedgers(
        global_candidates=_frame_from_records(
            global_records,
            metadata_columns=metadata_columns,
            source_ledger="global",
        ),
        sliding_candidates=_frame_from_records(
            sliding_records,
            metadata_columns=metadata_columns,
            source_ledger="sliding",
        ),
        config=config,
    )
    ledgers.assert_forward_nesting()
    return ledgers


__all__ = [
    "ComponentAblationConfig",
    "ComponentLedgers",
    "FORWARD_VARIANTS",
    "GLOBAL_LEDGER_COLUMNS",
    "REMOVAL_VARIANTS",
    "SLIDING_LEDGER_COLUMNS",
    "VARIANT_REGISTRY",
    "VariantDefinition",
    "build_component_ledgers",
    "derive_variant_events",
    "variant_admission_table",
]
