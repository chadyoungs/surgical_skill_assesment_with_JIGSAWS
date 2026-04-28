#!/usr/bin/env python3
"""
STIP + BoF Step 3 – Classification.

Loads pre-computed BoF histogram vectors and trains/evaluates SVM classifiers
for each LOSO fold, performing a grid search over C and gamma.

Pipeline overview
-----------------
Step 1 (clustering.py): BoF vocabulary training and histogram prediction.
Step 2 (feature_processing.py): Feature extraction and normalisation.
Step 3 (this script): SVM / kNN / Bayes classification.
Step 4: Sliding-window evaluation.

Note: This method is post-hoc analysis, not real-time.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC

import Global_Var
from data_extract_stip import DataExtract


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_LIST: list[str] = ["Suturing", "Knot_Tying", "Needle_Passing"]
TASK_SYMBOL: int = Global_Var.TASK_SYMBOL
N_CLUSTERS: int = Global_Var.CLUSTERS

# SVM hyper-parameter search grid
_GAMMA_VALUES: list[float] = [0.001, 0.01, 0.1, 1, 10, 100]
_C_VALUES: list[float] = [0.001, 0.01, 0.1, 1, 10, 100]


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _build_index_map(
    extractor: DataExtract,
    name_lists: list[list[str]],
) -> list[list[int]]:
    """Convert per-fold surgery-name lists to sorted index lists.

    Args:
        extractor: A populated :class:`DataExtract` instance.
        name_lists: Per-fold lists of surgery names (from ``train_sum``/``test_sum``).

    Returns:
        Per-fold sorted lists of integer meta-data indices.
    """
    return [
        sorted(extractor.metaData_index[name] for name in fold)
        for fold in name_lists
    ]


def _index_to_name(extractor: DataExtract) -> dict[int, str]:
    """Build a reverse mapping from meta-data index → surgery name."""
    return {v: k for k, v in extractor.metaData_index.items()}


def _collect_vectors(
    time_series: list[list[np.ndarray]],
    index_lists: list[list[int]],
    n_clusters: int,
) -> list[np.ndarray]:
    """Stack BoF vectors for the requested indices in each fold.

    Args:
        time_series: Per-fold lists of histogram vectors (shape ``(n_clusters,)``).
        index_lists: Per-fold sorted lists of integer indices into *time_series*.
        n_clusters: Histogram dimensionality.

    Returns:
        Per-fold arrays of shape ``(n_selected, n_clusters)``.
    """
    result: list[np.ndarray] = []
    for fold_idx, indices in enumerate(index_lists):
        fold_vectors = np.zeros((1, n_clusters))
        for i, (hog, hof) in enumerate(
            zip(time_series[0][fold_idx], time_series[1][fold_idx])
        ):
            if i in indices:
                fold_vectors = np.vstack((fold_vectors, np.reshape(hog, (1, n_clusters))))
        fold_vectors = np.delete(fold_vectors, 0, axis=0)
        result.append(np.reshape(fold_vectors, (-1, n_clusters)))
    return result


def _collect_targets(
    extractor: DataExtract,
    idx_to_name: dict[int, str],
    index_lists: list[list[int]],
) -> list[list[int]]:
    """Build per-fold target label lists.

    Args:
        extractor: A populated :class:`DataExtract` instance.
        idx_to_name: Reverse index→surgery-name mapping.
        index_lists: Per-fold sorted lists of integer meta-data indices.

    Returns:
        Per-fold lists of integer skill labels (0 = novice, 2 = expert).
    """
    return [
        [extractor.metaData_score[idx_to_name[j]][1] for j in fold if j in idx_to_name]
        for fold in index_lists
    ]


# ---------------------------------------------------------------------------
# SVM grid search
# ---------------------------------------------------------------------------

def svm_grid_search(
    train_vectors: list[np.ndarray],
    test_vectors: list[np.ndarray],
    train_targets: list[list[int]],
    test_targets: list[list[int]],
) -> tuple[float, dict]:
    """Run a C × gamma grid search and return the best average score.

    Args:
        train_vectors: Per-fold training feature matrices.
        test_vectors: Per-fold test feature matrices.
        train_targets: Per-fold training labels.
        test_targets: Per-fold test labels.

    Returns:
        A 2-tuple ``(best_score, best_parameters)`` where *best_score* is the
        **sum** of per-fold scores (not the average) for the best parameter
        combination, and *best_parameters* is ``{'C': ..., 'gamma': ...}``.
    """
    best_score: float = 0.0
    best_parameters: dict = {}

    for gamma in _GAMMA_VALUES:
        for C in _C_VALUES:
            fold_score_sum = sum(
                SVC(gamma=gamma, C=C).fit(train_vectors[i], train_targets[i]).score(
                    test_vectors[i], test_targets[i]
                )
                for i in range(len(train_vectors))
            )
            print(f"gamma: {gamma} and C: {C}")
            print(f"average score for current parameters: {fold_score_sum / len(train_vectors):.4f}")
            if fold_score_sum > best_score:
                best_score = fold_score_sum
                best_parameters = {"C": C, "gamma": gamma}

    return best_score, best_parameters


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load BoF histograms, run SVM grid search, and print results."""
    task = TASK_LIST[TASK_SYMBOL]
    time_series_dir = Path(".") / "time_series_data" / task

    extractor = DataExtract()
    extractor.get_category()
    extractor.get_txt_index()
    extractor.get_score()

    train_list = extractor.train_sum()
    test_list = extractor.test_sum()
    n_folds = len(train_list)

    # Load pre-computed BoF histograms for each fold
    time_series_hog: list[list[np.ndarray]] = []
    time_series_hof: list[list[np.ndarray]] = []
    for i in range(n_folds):
        hog_path = time_series_dir / f"time_series_hog_set{i + 1}_{N_CLUSTERS}.pkl"
        hof_path = time_series_dir / f"time_series_hof_set{i + 1}_{N_CLUSTERS}.pkl"
        time_series_hog.append(joblib.load(hog_path))
        time_series_hof.append(joblib.load(hof_path))

    train_indices = _build_index_map(extractor, train_list)
    test_indices = _build_index_map(extractor, test_list)
    idx_to_name = _index_to_name(extractor)

    # Build feature matrices
    train_hog_vectors: list[np.ndarray] = []
    train_hof_vectors: list[np.ndarray] = []
    test_hog_vectors: list[np.ndarray] = []
    test_hof_vectors: list[np.ndarray] = []

    for fold_idx in range(n_folds):
        train_hog = np.zeros((1, N_CLUSTERS))
        train_hof = np.zeros((1, N_CLUSTERS))
        test_hog = np.zeros(N_CLUSTERS)
        test_hof = np.zeros(N_CLUSTERS)

        for count, (hog, hof) in enumerate(zip(time_series_hog[fold_idx], time_series_hof[fold_idx])):
            hog_row = np.reshape(hog, (1, N_CLUSTERS))
            hof_row = np.reshape(hof, (1, N_CLUSTERS))
            if count in train_indices[fold_idx]:
                train_hog = np.vstack((train_hog, hog_row))
                train_hof = np.vstack((train_hof, hof_row))
            if count in test_indices[fold_idx]:
                test_hog = np.vstack((test_hog, hog_row))
                test_hof = np.vstack((test_hof, hof_row))

        train_hog = np.delete(train_hog, 0, axis=0)
        train_hof = np.delete(train_hof, 0, axis=0)
        test_hog = np.delete(test_hog, 0, axis=0)
        test_hof = np.delete(test_hof, 0, axis=0)

        train_hog_vectors.append(np.reshape(np.hstack((train_hog, train_hof)), (-1, 2 * N_CLUSTERS)))
        test_hog_vectors.append(np.reshape(np.hstack((test_hog, test_hof)), (-1, 2 * N_CLUSTERS)))

    train_targets = _collect_targets(extractor, idx_to_name, train_indices)
    test_targets = _collect_targets(extractor, idx_to_name, test_indices)

    print("Start to classify the histogram")
    best_score, best_params = svm_grid_search(
        train_hog_vectors, test_hog_vectors, train_targets, test_targets
    )

    print(f"best score: {best_score:.4f}")
    print(f"best parameters: {best_params}")
    print(f"Test set average score for {n_folds} sets: {best_score / n_folds:.4f}")


if __name__ == "__main__":
    main()
