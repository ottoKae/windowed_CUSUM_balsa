"""Deterministic evaluation helpers for LR-CuSum component ablations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .validation_closure import (
    EVENT_MATCHER_OPTIMAL_V2,
    match_events_detailed,
)


_COUNT_COLUMNS = ("tp", "fp", "fn")
_DERIVED_METRICS = ("precision", "recall", "f1")
_ALL_METRICS = _COUNT_COLUMNS + _DERIVED_METRICS


@dataclass(frozen=True)
class ComponentEvaluation:
    """Tables produced by :func:`evaluate_event_stages`."""

    stage_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    block_metrics: pd.DataFrame
    matches: pd.DataFrame


@dataclass(frozen=True)
class BlockBootstrapResult:
    """Common-index paired spatial-block bootstrap outputs."""

    stage_replicates: pd.DataFrame
    contrast_replicates: pd.DataFrame
    stage_summary: pd.DataFrame
    contrast_summary: pd.DataFrame


@dataclass(frozen=True)
class RandomizationResult:
    """Paired block label-swap randomization outputs."""

    replicates: pd.DataFrame
    summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _normalise_tolerances(tolerances: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(tolerances, (int, np.integer)):
        values = (int(tolerances),)
    else:
        values = tuple(dict.fromkeys(int(value) for value in tolerances))
    if not values:
        raise ValueError("at least one tolerance is required")
    if any(value < 0 for value in values):
        raise ValueError("tolerances must be non-negative")
    return values


def _counts_to_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _sorted_unique(values) -> list:
    return sorted(pd.unique(values), key=lambda value: str(value))


def evaluate_event_stages(
    references: pd.DataFrame,
    stage_events: Mapping[str, pd.DataFrame],
    *,
    matcher_version: str = EVENT_MATCHER_OPTIMAL_V2,
    tolerances: int | Iterable[int] = (2,),
    block_column: str = "spatial_block_id",
    fold_column: str | None = None,
    folds: Iterable | None = None,
) -> ComponentEvaluation:
    """Evaluate named ground-event tables on a common reference population.

    Complete samples are assigned to their reference spatial block and optional
    fold.  Events outside selected folds are removed before scoring.  An event
    whose ``sample_id`` is absent from the full reference table is rejected,
    because it cannot be assigned to a paired spatial-block resampling unit.
    """
    if not stage_events:
        raise ValueError("stage_events must contain at least one named stage")
    _require_columns(
        references,
        ("sample_id", "event_month", block_column),
        "references",
    )
    if fold_column is not None:
        _require_columns(references, (fold_column,), "references")
    elif folds is not None:
        raise ValueError("fold_column is required when folds are selected")

    tolerance_values = _normalise_tolerances(tolerances)
    full_references = references.copy()
    full_references["_sample_key"] = full_references.sample_id.astype(str)
    if full_references[block_column].isna().any():
        raise ValueError(f"references contains missing {block_column} values")

    metadata_columns = ["_sample_key", block_column]
    if fold_column is not None:
        metadata_columns.append(fold_column)
    metadata = full_references[metadata_columns].drop_duplicates()
    cardinality = metadata.groupby("_sample_key").size()
    if (cardinality > 1).any():
        conflicting = ", ".join(cardinality[cardinality > 1].index[:5])
        raise ValueError(
            "each sample_id must map to exactly one block/fold; conflicts: " + conflicting
        )

    selected_references = full_references
    selected_folds = None
    if fold_column is not None:
        available_folds = _sorted_unique(full_references[fold_column].dropna())
        selected_folds = available_folds if folds is None else list(dict.fromkeys(folds))
        unknown_folds = set(selected_folds) - set(available_folds)
        if unknown_folds:
            raise ValueError(f"unknown folds requested: {sorted(unknown_folds, key=str)}")
        selected_references = full_references.loc[
            full_references[fold_column].isin(selected_folds)
        ].copy()
    if selected_references.empty:
        raise ValueError("the selected reference population is empty")

    selected_metadata = selected_references[metadata_columns].drop_duplicates()
    selected_keys = set(selected_metadata._sample_key)
    full_keys = set(metadata._sample_key)
    evaluation_references = selected_references.copy()
    evaluation_references["sample_id"] = evaluation_references._sample_key

    normalised_events: dict[str, pd.DataFrame] = {}
    for stage, events in stage_events.items():
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage names must be non-empty strings")
        if events.empty and not {"sample_id", "change_time"}.issubset(events.columns):
            local = pd.DataFrame(columns=["sample_id", "change_time"])
        else:
            _require_columns(events, ("sample_id", "change_time"), f"stage {stage!r}")
            local = events[["sample_id", "change_time"]].copy()
        local["_sample_key"] = local.sample_id.astype(str)
        unknown_samples = set(local._sample_key) - full_keys
        if unknown_samples:
            example = ", ".join(sorted(unknown_samples)[:5])
            raise ValueError(
                f"stage {stage!r} contains sample_id values absent from references: {example}"
            )
        local = local.loc[local._sample_key.isin(selected_keys)].copy()
        local["sample_id"] = local._sample_key
        normalised_events[stage] = local[["sample_id", "change_time"]]

    stage_rows = []
    fold_rows = []
    block_rows = []
    match_frames = []
    stage_names = list(normalised_events)
    blocks = _sorted_unique(selected_metadata[block_column])

    for tolerance in tolerance_values:
        for stage in stage_names:
            events = normalised_events[stage]
            metrics, matches = match_events_detailed(
                evaluation_references,
                events,
                tolerance_months=tolerance,
                matcher_version=matcher_version,
            )
            stage_rows.append({
                "stage": stage,
                "tolerance_months": tolerance,
                "matcher_version": matcher_version,
                **metrics,
            })

            if len(matches):
                annotated = matches.merge(
                    selected_metadata,
                    left_on="sample_id",
                    right_on="_sample_key",
                    how="left",
                    validate="many_to_one",
                ).drop(columns="_sample_key")
                annotated.insert(0, "stage", stage)
                annotated.insert(1, "tolerance_months", tolerance)
                annotated.insert(2, "matcher_version", matcher_version)
                match_frames.append(annotated)

            for block in blocks:
                local_references = evaluation_references.loc[
                    evaluation_references[block_column].eq(block)
                ]
                local_keys = set(local_references._sample_key)
                local_events = events.loc[events.sample_id.isin(local_keys)]
                local_metrics, _ = match_events_detailed(
                    local_references,
                    local_events,
                    tolerance_months=tolerance,
                    matcher_version=matcher_version,
                )
                row = {
                    "stage": stage,
                    "tolerance_months": tolerance,
                    "matcher_version": matcher_version,
                    block_column: block,
                    **local_metrics,
                }
                if fold_column is not None:
                    block_fold = selected_metadata.loc[
                        selected_metadata[block_column].eq(block), fold_column
                    ].drop_duplicates()
                    if len(block_fold) != 1:
                        raise ValueError(
                            f"each {block_column} must map to one {fold_column}"
                        )
                    row[fold_column] = block_fold.iloc[0]
                block_rows.append(row)

            if fold_column is not None:
                for fold in selected_folds:
                    local_references = evaluation_references.loc[
                        evaluation_references[fold_column].eq(fold)
                    ]
                    local_keys = set(local_references._sample_key)
                    local_events = events.loc[events.sample_id.isin(local_keys)]
                    local_metrics, _ = match_events_detailed(
                        local_references,
                        local_events,
                        tolerance_months=tolerance,
                        matcher_version=matcher_version,
                    )
                    fold_rows.append({
                        "stage": stage,
                        "tolerance_months": tolerance,
                        "matcher_version": matcher_version,
                        fold_column: fold,
                        **local_metrics,
                    })

    matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame(
        columns=[
            "stage", "tolerance_months", "matcher_version", "sample_id",
            "reference_time", "detected_time", "signed_error_months",
            "absolute_error_months", block_column,
        ] + ([fold_column] if fold_column is not None else [])
    )
    return ComponentEvaluation(
        stage_metrics=pd.DataFrame(stage_rows),
        fold_metrics=pd.DataFrame(fold_rows),
        block_metrics=pd.DataFrame(block_rows),
        matches=matches,
    )


def _count_cube(
    block_metrics: pd.DataFrame,
    *,
    tolerance_months: int,
    block_column: str,
) -> tuple[list[str], list, dict[str, np.ndarray]]:
    _require_columns(
        block_metrics,
        ("stage", "tolerance_months", block_column) + _COUNT_COLUMNS,
        "block_metrics",
    )
    local = block_metrics.loc[
        block_metrics.tolerance_months.eq(tolerance_months)
    ].copy()
    if local.empty:
        raise ValueError(f"no block metrics found for tolerance {tolerance_months}")
    duplicate = local.duplicated(["stage", block_column], keep=False)
    if duplicate.any():
        raise ValueError("block_metrics has duplicate stage/block rows")
    stages = _sorted_unique(local.stage.astype(str))
    blocks = _sorted_unique(local[block_column])
    cube = {}
    for stage in stages:
        selected = local.loc[local.stage.astype(str).eq(stage)].set_index(block_column)
        missing = set(blocks) - set(selected.index)
        if missing:
            raise ValueError(
                f"stage {stage!r} is missing paired blocks: {sorted(missing, key=str)}"
            )
        cube[stage] = selected.loc[blocks, list(_COUNT_COLUMNS)].to_numpy(dtype=np.int64)
    return stages, blocks, cube


def _validate_contrasts(
    contrasts: Mapping[str, tuple[str, str]],
    stages: Sequence[str],
) -> list[tuple[str, str, str]]:
    stage_set = set(stages)
    rows = []
    for name, pair in contrasts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("contrast names must be non-empty strings")
        if len(pair) != 2:
            raise ValueError(f"contrast {name!r} must contain (left, right)")
        left, right = str(pair[0]), str(pair[1])
        missing = {left, right} - stage_set
        if missing:
            raise ValueError(
                f"contrast {name!r} references unknown stages: {sorted(missing)}"
            )
        rows.append((name, left, right))
    return rows


def common_index_paired_block_bootstrap(
    block_metrics: pd.DataFrame,
    contrasts: Mapping[str, tuple[str, str]],
    *,
    tolerance_months: int = 2,
    replicates: int = 5000,
    seed: int = 20260823,
    metrics: Sequence[str] = _DERIVED_METRICS,
    block_column: str = "spatial_block_id",
) -> BlockBootstrapResult:
    """Bootstrap all stages and contrasts with one shared block index per draw."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    metric_names = tuple(dict.fromkeys(metrics))
    unknown_metrics = set(metric_names) - set(_ALL_METRICS)
    if unknown_metrics:
        raise ValueError(f"unknown metrics: {sorted(unknown_metrics)}")
    stages, blocks, cube = _count_cube(
        block_metrics,
        tolerance_months=tolerance_months,
        block_column=block_column,
    )
    contrast_rows = _validate_contrasts(contrasts, stages)
    point = {
        stage: _counts_to_metrics(*cube[stage].sum(axis=0))
        for stage in stages
    }

    rng = np.random.default_rng(seed)
    stage_replicates = []
    contrast_replicates = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(blocks), len(blocks))
        replicate_metrics = {}
        for stage in stages:
            values = _counts_to_metrics(*cube[stage][indices].sum(axis=0))
            replicate_metrics[stage] = values
            stage_replicates.append({
                "replicate": replicate,
                "stage": stage,
                **values,
            })
        for name, left, right in contrast_rows:
            contrast_replicates.append({
                "replicate": replicate,
                "contrast": name,
                "left": left,
                "right": right,
                **{
                    f"delta_{metric}": (
                        replicate_metrics[left][metric] - replicate_metrics[right][metric]
                    )
                    for metric in metric_names
                },
            })

    stage_frame = pd.DataFrame(stage_replicates)
    contrast_frame = pd.DataFrame(contrast_replicates)
    stage_summary = []
    for stage in stages:
        selected = stage_frame.loc[stage_frame.stage.eq(stage)]
        for metric in metric_names:
            values = selected[metric].to_numpy(dtype=float)
            stage_summary.append({
                "stage": stage,
                "metric": metric,
                "estimate": float(point[stage][metric]),
                "bootstrap_median": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "replicates": int(replicates),
                "seed": int(seed),
            })

    contrast_summary = []
    for name, left, right in contrast_rows:
        selected = contrast_frame.loc[contrast_frame.contrast.eq(name)]
        for metric in metric_names:
            values = selected[f"delta_{metric}"].to_numpy(dtype=float)
            contrast_summary.append({
                "contrast": name,
                "left": left,
                "right": right,
                "metric": metric,
                "estimate": float(point[left][metric] - point[right][metric]),
                "bootstrap_median": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_gt_zero": float(np.mean(values > 0)),
                "replicates": int(replicates),
                "seed": int(seed),
            })
    return BlockBootstrapResult(
        stage_replicates=stage_frame,
        contrast_replicates=contrast_frame,
        stage_summary=pd.DataFrame(stage_summary),
        contrast_summary=pd.DataFrame(contrast_summary),
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values; NaNs remain NaN."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    if np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = valid[np.argsort(values[valid], kind="mergesort")]
    family_size = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (family_size - rank) * float(values[index])
        running = max(running, candidate)
        result[index] = min(1.0, running)
    return result


def paired_block_label_swap_test(
    block_metrics: pd.DataFrame,
    contrasts: Mapping[str, tuple[str, str]],
    *,
    tolerance_months: int = 2,
    metric: str = "f1",
    permutations: int = 10000,
    seed: int = 20260823,
    block_column: str = "spatial_block_id",
) -> RandomizationResult:
    """Run paired two-sided block label-swap tests for named stage contrasts."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if metric not in _ALL_METRICS:
        raise ValueError(f"unknown metric: {metric}")
    stages, blocks, cube = _count_cube(
        block_metrics,
        tolerance_months=tolerance_months,
        block_column=block_column,
    )
    contrast_rows = _validate_contrasts(contrasts, stages)
    observed_stage = {
        stage: _counts_to_metrics(*cube[stage].sum(axis=0))
        for stage in stages
    }

    rng = np.random.default_rng(seed)
    replicate_rows = []
    for permutation in range(permutations):
        swap = rng.integers(0, 2, len(blocks), dtype=np.int8).astype(bool)
        for name, left, right in contrast_rows:
            left_counts = cube[left]
            right_counts = cube[right]
            permuted_left = left_counts.copy()
            permuted_right = right_counts.copy()
            permuted_left[swap] = right_counts[swap]
            permuted_right[swap] = left_counts[swap]
            left_metric = _counts_to_metrics(*permuted_left.sum(axis=0))[metric]
            right_metric = _counts_to_metrics(*permuted_right.sum(axis=0))[metric]
            replicate_rows.append({
                "permutation": permutation,
                "contrast": name,
                "left": left,
                "right": right,
                "metric": metric,
                "null_delta": float(left_metric - right_metric),
            })
    replicate_frame = pd.DataFrame(replicate_rows)

    summaries = []
    for name, left, right in contrast_rows:
        observed = float(observed_stage[left][metric] - observed_stage[right][metric])
        null = replicate_frame.loc[
            replicate_frame.contrast.eq(name), "null_delta"
        ].to_numpy(dtype=float)
        p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15)) / (
            permutations + 1
        )
        summaries.append({
            "contrast": name,
            "left": left,
            "right": right,
            "metric": metric,
            "observed_delta": observed,
            "p_value": float(p_value),
            "permutations": int(permutations),
            "seed": int(seed),
        })
    summary = pd.DataFrame(summaries)
    if len(summary):
        summary["p_value_holm"] = holm_adjust(summary.p_value.to_numpy())
    else:
        summary["p_value_holm"] = pd.Series(dtype=float)
    return RandomizationResult(replicates=replicate_frame, summary=summary)

