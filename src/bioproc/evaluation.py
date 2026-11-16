"""Scoring with baselines, batch-aware splits and a stated detection limit.

Ground rules section 1: a number is not a result until it can be interpreted. Every
model score here is returned next to what a no-information model scores on the same
split, so a figure a trivial predictor beats is visibly not evidence.

Baselines used:

- regression: the mean of the training target, and persistence (carry the last
  measured value forward) where the data is a time series;
- classification: the majority class, and a label shuffle;
- soft sensor: the offline measurement it is meant to replace, interpolated.

Splits are grouped by batch. A random split over time points from the same run leaks
nearly-identical neighbouring samples into the test set and flatters every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def rmse(y_true, y_predicted) -> float:
    """Root-mean-square error.

    >>> round(rmse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]), 4)
    0.5774
    """
    y_true, y_predicted = np.asarray(y_true, dtype=float), np.asarray(y_predicted, dtype=float)
    if y_true.shape != y_predicted.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_predicted.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))


def normalised_rmse_percent(y_true, y_predicted) -> float:
    """RMSE as a percentage of the observed range, so steps can be compared."""
    y_true = np.asarray(y_true, dtype=float)
    spread = float(np.ptp(y_true))
    if spread == 0:
        raise ValueError("the target has zero range; a percentage is undefined")
    return 100.0 * rmse(y_true, y_predicted) / spread


def regression_scores(y_true, y_predicted, y_train=None) -> dict[str, float]:
    """RMSE, normalised RMSE, R^2 and the mean-predictor baseline on the same test set.

    ``skill_vs_mean`` is ``1 - MSE_model / MSE_mean``: positive means the model beats
    predicting the training mean, zero or negative means it does not.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = {
        "rmse": rmse(y_true, y_predicted),
        "nrmse_percent": normalised_rmse_percent(y_true, y_predicted),
        "r2": float(r2_score(y_true, y_predicted)),
        "n_test": int(y_true.size),
    }
    reference = float(np.mean(y_train if y_train is not None else y_true))
    baseline = rmse(y_true, np.full_like(y_true, reference))
    scores["rmse_mean_baseline"] = baseline
    scores["skill_vs_mean"] = float(1.0 - (scores["rmse"] / baseline) ** 2) if baseline > 0 else float("nan")
    return scores


def persistence_baseline(y_true) -> dict[str, float]:
    """Carry the previous value forward. The baseline to beat for a time series.

    The first point has no predecessor and is dropped, so ``n_test`` is one smaller.
    """
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size < 2:
        raise ValueError("persistence needs at least 2 points")
    return {"rmse_persistence": rmse(y_true[1:], y_true[:-1]), "n_test": int(y_true.size - 1)}


def classification_scores(y_true, y_predicted, *, seed: int = 0, shuffles: int = 200) -> dict[str, float]:
    """MCC and balanced accuracy, with majority-class and label-shuffle baselines.

    Raw accuracy is deliberately not the headline: batch-failure data is imbalanced
    (about 10% failures in the Gangadharan et al. 2021 set), where always predicting
    "pass" scores 90%.
    """
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    if y_true.shape != y_predicted.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_predicted.shape}")
    values, counts = np.unique(y_true, return_counts=True)
    majority = values[int(np.argmax(counts))]
    generator = np.random.default_rng(seed)
    shuffled = [matthews_corrcoef(y_true, generator.permutation(y_true)) for _ in range(shuffles)]
    return {
        "mcc": float(matthews_corrcoef(y_true, y_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_predicted)),
        "accuracy": float(np.mean(y_true == y_predicted)),
        "mcc_majority_baseline": float(matthews_corrcoef(y_true, np.full_like(y_true, majority))),
        "balanced_accuracy_majority_baseline": float(balanced_accuracy_score(y_true, np.full_like(y_true, majority))),
        "mcc_shuffled_mean": float(np.mean(shuffled)),
        "mcc_shuffled_p95": float(np.percentile(shuffled, 95)),
        "majority_class_fraction": float(counts.max() / counts.sum()),
        "n_test": int(y_true.size),
    }


def confusion(y_true, y_predicted, labels=None) -> pd.DataFrame:
    """Confusion matrix as a labelled frame (reported next to every MCC)."""
    labels = list(labels) if labels is not None else sorted(set(np.asarray(y_true).tolist()))
    matrix = confusion_matrix(y_true, y_predicted, labels=labels)
    return pd.DataFrame(
        matrix, index=[f"true {label}" for label in labels], columns=[f"pred {label}" for label in labels]
    )


def grouped_splits(groups, n_splits: int | None = None):
    """Batch-aware cross-validation splits.

    Leave-one-batch-out when there are few batches, grouped k-fold otherwise. Returns
    ``(splitter, name)``.

    Raises
    ------
    ValueError
        If there are fewer than two batches to split on.
    """
    unique = pd.unique(np.asarray(groups))
    if unique.size < 2:
        raise ValueError(f"need at least 2 batches to split on, got {unique.size}")
    if n_splits is None:
        n_splits = min(5, unique.size)
    if unique.size <= 6:
        return LeaveOneGroupOut(), "leave-one-batch-out"
    return GroupKFold(n_splits=n_splits), f"grouped {n_splits}-fold"


def cross_validated_predictions(model, x, y, groups) -> tuple[np.ndarray, str]:
    """Out-of-fold predictions using batch-aware splits. Returns (predictions, split name).

    Every test point is predicted by a model that never saw its batch.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if not len(x) == len(y) == len(groups):
        raise ValueError("x, y and groups must have the same length")
    splitter, name = grouped_splits(groups)
    predictions = np.full(y.shape, np.nan)
    from sklearn.base import clone

    for train_index, test_index in splitter.split(x, y, groups):
        fitted = clone(model).fit(x[train_index], y[train_index])
        predictions[test_index] = fitted.predict(x[test_index])
    if np.isnan(predictions).any():
        raise ValueError("some samples were never in a test fold")
    return predictions, name


def detection_limit(values, *, confidence: float = 0.95) -> dict[str, float]:
    """Smallest difference in means this sample size could resolve.

    Reported with every "no significant difference": a two-sample t-test's minimum
    detectable difference at 80% power is about ``2.8 * sd * sqrt(2/n)``.

    Raises
    ------
    ValueError
        If there are fewer than two replicates.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        raise ValueError("a detection limit needs at least 2 replicates")
    sd = float(np.std(values, ddof=1))
    return {
        "n": int(n),
        "sd": sd,
        "min_detectable_difference": 2.8 * sd * np.sqrt(2.0 / n),
        "min_detectable_percent": 2.8 * sd * np.sqrt(2.0 / n) / abs(float(np.mean(values))) * 100.0,
        "confidence": confidence,
    }


def report_table(named_scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Scores for several models as one frame, models as rows, baselines as columns."""
    if not named_scores:
        raise ValueError("no scores to report")
    return pd.DataFrame(named_scores).T
