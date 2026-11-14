"""Tatbul range precision/recall for inclusive, integer-month intervals.

Implements alpha=0, gamma(k)=1/k, uniform precision and uniform/front recall
from Tatbul et al., "Precision and Recall for Time Series", equations 3--9:
https://arxiv.org/abs/1803.03639

Intervals are compared only within a sample_id. Overlapping and duplicate
intervals remain separate objects: there is no union or one-to-one matching.
In particular, exact duplicate predictions can still score P=R=1. These
scores measure the supplied intervals, not independently verified harvest
identity, completeness of reference history, or spatial correspondence.
"""
from numbers import Integral

import numpy as np
import pandas as pd


OVERLAP_COLUMNS = [
    "sample_id", "reference_id", "prediction_id", "overlap_start_month",
    "overlap_end_month", "n_months", "reference_omega", "prediction_omega",
]


def evaluate_interval_metrics(references: pd.DataFrame, predictions: pd.DataFrame,
                              recall_bias: str = "uniform"):
    """Return summary, per-reference scores, per-prediction scores and overlaps.

    Each input has sample_id, reference_id/prediction_id, start_month and
    end_month. Months must be integer absolute month indices, with both bounds
    inclusive. IDs are unique within a sample; different IDs may have identical
    intervals. Front recall uses weights L, L-1, ..., 1 within each reference.

    Recall is the mean over all reference objects; precision is the mean over
    all prediction objects, including samples absent from the other table.
    F1 is calculated from those global means, not averaged over samples.
    An empty denominator gives NaN; two nonempty but disjoint inputs give F1=0.
    """
    if recall_bias not in ("uniform", "front"):
        raise ValueError("recall_bias must be 'uniform' or 'front'")
    references = _validated_intervals(references, "reference_id")
    predictions = _validated_intervals(predictions, "prediction_id")
    reference_counts = np.zeros(len(references), dtype=np.int64)
    prediction_counts = np.zeros(len(predictions), dtype=np.int64)
    reference_rewards = np.zeros(len(references))
    prediction_rewards = np.zeros(len(predictions))
    by_sample = {}
    for index, prediction in enumerate(predictions.itertuples(index=False)):
        by_sample.setdefault(prediction.sample_id, []).append((index, prediction))

    overlaps = []
    for i, reference in enumerate(references.itertuples(index=False)):
        ref_start, ref_end = int(reference.start_month), int(reference.end_month)
        ref_length = ref_end - ref_start + 1
        for j, prediction in by_sample.get(reference.sample_id, []):
            pred_start, pred_end = int(prediction.start_month), int(prediction.end_month)
            start, end = max(ref_start, pred_start), min(ref_end, pred_end)
            if start > end:
                continue
            n_months = end - start + 1
            if recall_bias == "front":
                reference_omega = (n_months * (2 * ref_end - start - end + 2)
                                   / (ref_length * (ref_length + 1)))
            else:
                reference_omega = n_months / ref_length
            prediction_omega = n_months / (pred_end - pred_start + 1)
            reference_counts[i] += 1
            prediction_counts[j] += 1
            reference_rewards[i] += reference_omega
            prediction_rewards[j] += prediction_omega
            overlaps.append(dict(
                sample_id=reference.sample_id, reference_id=reference.reference_id,
                prediction_id=prediction.prediction_id, overlap_start_month=start,
                overlap_end_month=end, n_months=n_months,
                reference_omega=reference_omega, prediction_omega=prediction_omega))

    reference_scores = _object_scores(references, reference_counts, reference_rewards)
    prediction_scores = _object_scores(predictions, prediction_counts, prediction_rewards)
    recall = float(reference_scores.score.mean())
    precision = float(prediction_scores.score.mean())
    if not len(references) or not len(predictions):
        f1 = np.nan
    elif recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    summary = dict(
        n_references=len(references), n_predictions=len(predictions),
        recall=recall, precision=precision, f1=float(f1),
        n_overlap_pairs=len(overlaps),
        n_references_multiple_overlaps=int((reference_counts > 1).sum()),
        n_predictions_multiple_overlaps=int((prediction_counts > 1).sum()),
        alpha=0.0, gamma="1/x", recall_bias=recall_bias, precision_bias="uniform")
    return (summary, reference_scores, prediction_scores,
            pd.DataFrame(overlaps, columns=OVERLAP_COLUMNS))


def _validated_intervals(frame, id_column):
    columns = ["sample_id", id_column, "start_month", "end_month"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Interval table is missing columns: {sorted(missing)}")
    identities = frame[["sample_id", id_column]]
    if identities.isna().any().any() or identities.eq("").any().any():
        raise ValueError("Sample and interval IDs must be present and nonempty")
    if frame.duplicated(["sample_id", id_column]).any():
        raise ValueError(f"{id_column} must be unique within each sample_id")
    for column in ("start_month", "end_month"):
        if any(not isinstance(value, Integral) or isinstance(value, bool)
               for value in frame[column]):
            raise ValueError("Interval months must be integers, not strings, floats or missing values")
    if (frame.start_month > frame.end_month).any():
        raise ValueError("Interval start_month must not exceed end_month")
    return frame[columns].copy().reset_index(drop=True)


def _object_scores(frame, counts, rewards):
    # Equation 7 sets the factor to 1 for zero or one overlap, then gamma(k).
    factors = 1.0 / np.maximum(counts, 1)
    return frame.assign(overlap_count=counts, overlap_reward_sum=rewards,
                        cardinality_factor=factors, score=factors * rewards)
