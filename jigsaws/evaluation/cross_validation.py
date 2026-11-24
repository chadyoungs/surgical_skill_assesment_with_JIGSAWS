"""LOSO cross-validation with a small, dependency-free grid search.

Methods supply a scikit-learn style estimator factory and a parameter grid;
fold construction, name-to-row indexing, metric aggregation and reporting are
identical for every method (and for any new method added later).
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from ..dataset.splits import Fold
from ..errors import DataError
from .metrics import BinaryMetrics

LOGGER = logging.getLogger(__name__)


class Estimator(Protocol):
    def fit(self, x, y): ...

    def predict(self, x): ...


EstimatorFactory = Callable[..., Estimator]


@dataclass(frozen=True, slots=True)
class FoldScore:
    fold: str
    n_train: int
    n_test: int
    metrics: BinaryMetrics


@dataclass(frozen=True, slots=True)
class SearchResult:
    best_params: dict[str, object]
    mean_accuracy: float
    folds: tuple[FoldScore, ...]
    grid: tuple[tuple[dict[str, object], float], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "best_params": self.best_params,
            "mean_accuracy": self.mean_accuracy,
            "n_folds": len(self.folds),
            "folds": [
                {
                    "fold": score.fold,
                    "n_train": score.n_train,
                    "n_test": score.n_test,
                    **score.metrics.as_dict(),
                }
                for score in self.folds
            ],
            "grid": [
                {"params": params, "mean_accuracy": mean}
                for params, mean in self.grid
            ],
        }

    def write(self, path: str | Path, *, extra: Mapping[str, object] | None = None) -> Path:
        payload: dict[str, object] = {"result": self.as_dict()}
        if extra:
            payload.update(extra)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    def log(self) -> None:
        LOGGER.info(
            "LOSO result: mean accuracy %.4f over %d folds; params=%s",
            self.mean_accuracy,
            len(self.folds),
            self.best_params,
        )
        for score in self.folds:
            LOGGER.info(
                "  %-10s train=%3d test=%3d  %s",
                score.fold,
                score.n_train,
                score.n_test,
                score.metrics.format(),
            )


def _index_rows(names: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, name in enumerate(names):
        if name in mapping:
            raise DataError(f"Duplicate row name '{name}' in feature table")
        mapping[name] = index
    return mapping


def _param_combinations(
    param_grid: Mapping[str, Sequence[object]],
) -> list[dict[str, object]]:
    keys = tuple(param_grid)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(param_grid[key] for key in keys))
    ]
    return combinations or [{}]


def grid_search_loso_per_fold(
    folds: Sequence[Fold],
    fold_features: Mapping[str, tuple[Sequence[str], np.ndarray]],
    labels_by_name: Mapping[str, int],
    estimator_factory: EstimatorFactory,
    param_grid: Mapping[str, Sequence[object]],
    *,
    log_progress: bool = True,
) -> SearchResult:
    """LOSO search where every fold has its own feature matrix.

    Needed when a method's representation depends on the fold (e.g. BoF
    vocabularies trained per fold). Fold construction and metrics stay shared.
    """
    prepared: list[tuple[Fold, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold in folds:
        if fold.name not in fold_features:
            raise DataError(f"No feature matrix supplied for fold {fold.name}")
        names, features = fold_features[fold.name]
        rows = _index_rows(names)
        missing = [name for name in fold.all_trials if name not in rows]
        if missing:
            raise DataError(
                f"Fold {fold.name} references trials without features: "
                + ", ".join(missing[:5])
            )
        absent_labels = [name for name in fold.all_trials if name not in labels_by_name]
        if absent_labels:
            raise DataError(
                "No skill label for trials: " + ", ".join(absent_labels[:5])
            )
        train_idx = np.array([rows[name] for name in fold.train])
        test_idx = np.array([rows[name] for name in fold.test])
        if train_idx.size == 0 or test_idx.size == 0:
            raise DataError(f"Fold {fold.name} has an empty train or test set")
        train_labels = np.array([labels_by_name[name] for name in fold.train])
        test_labels = np.array([labels_by_name[name] for name in fold.test])
        prepared.append(
            (fold, np.asarray(features), train_idx, test_idx, train_labels, test_labels)
        )

    best_mean = -1.0
    best_params: dict[str, object] = {}
    best_scores: tuple[FoldScore, ...] = ()
    grid_history: list[tuple[dict[str, object], float]] = []

    for params in _param_combinations(param_grid):
        fold_scores: list[FoldScore] = []
        for fold, features, train_idx, test_idx, train_labels, test_labels in prepared:
            estimator = estimator_factory(**params)
            estimator.fit(features[train_idx], train_labels)
            predictions = np.asarray(estimator.predict(features[test_idx]))
            fold_scores.append(
                FoldScore(
                    fold=fold.name,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    metrics=BinaryMetrics.compute(test_labels, predictions),
                )
            )

        mean = float(np.mean([score.metrics.accuracy for score in fold_scores]))
        grid_history.append((params, mean))
        if log_progress:
            LOGGER.debug("params=%s mean_accuracy=%.4f", params, mean)
        if mean > best_mean:
            best_mean = mean
            best_params = params
            best_scores = tuple(fold_scores)

    return SearchResult(
        best_params=best_params,
        mean_accuracy=best_mean,
        folds=best_scores,
        grid=tuple(grid_history),
    )


def grid_search_loso(
    folds: Sequence[Fold],
    names: Sequence[str],
    features: np.ndarray,
    labels: np.ndarray,
    estimator_factory: EstimatorFactory,
    param_grid: Mapping[str, Sequence[object]],
    *,
    log_progress: bool = True,
) -> SearchResult:
    """Fit/evaluate every parameter combination on every fold.

    Selection is by mean test accuracy (the legacy code selected by the sum,
    which is equivalent for a constant fold count).
    """
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if features.shape[0] != len(names):
        raise DataError(
            f"Feature rows ({features.shape[0]}) do not match row names ({len(names)})"
        )
    if labels.shape[0] != len(names):
        raise DataError(f"Labels ({labels.shape[0]}) do not match rows ({len(names)})")

    rows = _index_rows(names)
    keys = tuple(param_grid)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(param_grid[key] for key in keys))
    ]
    if not combinations:
        combinations = [{}]

    best_mean = -1.0
    best_params: dict[str, object] = {}
    best_scores: tuple[FoldScore, ...] = ()
    grid_history: list[tuple[dict[str, object], float]] = []

    for params in combinations:
        fold_scores: list[FoldScore] = []
        for fold in folds:
            missing_train = [name for name in fold.train if name not in rows]
            missing_test = [name for name in fold.test if name not in rows]
            if missing_train or missing_test:
                raise DataError(
                    f"Fold {fold.name} references trials without features: "
                    + ", ".join((missing_train + missing_test)[:5])
                )
            train_idx = [rows[name] for name in fold.train]
            test_idx = [rows[name] for name in fold.test]
            if not train_idx or not test_idx:
                raise DataError(f"Fold {fold.name} has an empty train or test set")

            estimator = estimator_factory(**params)
            estimator.fit(features[train_idx], labels[train_idx])
            predictions = np.asarray(estimator.predict(features[test_idx]))
            fold_scores.append(
                FoldScore(
                    fold=fold.name,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    metrics=BinaryMetrics.compute(labels[test_idx], predictions),
                )
            )

        mean = float(np.mean([score.metrics.accuracy for score in fold_scores]))
        grid_history.append((params, mean))
        if log_progress:
            LOGGER.debug("params=%s mean_accuracy=%.4f", params, mean)
        if mean > best_mean:
            best_mean = mean
            best_params = params
            best_scores = tuple(fold_scores)

    return SearchResult(
        best_params=best_params,
        mean_accuracy=best_mean,
        folds=best_scores,
        grid=tuple(grid_history),
    )
