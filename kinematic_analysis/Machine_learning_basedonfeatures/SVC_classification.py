#!/usr/bin/env python3
"""
Feature-based SVM surgical skill classification.

Loads pre-computed kinematic features (``.npy`` files), selects
task-specific feature subsets, and runs a grid-search SVC over LOSO folds.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from data_extract import DataExtract


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_LIST: list[str] = ["Suturing", "Knot_Tying", "Needle_Passing"]

# Number of features selected for each task
_N_FEATURES: dict[int, int] = {0: 28, 1: 20, 2: 24}

# SVM hyper-parameter search grid
_GAMMA_VALUES: list[float] = [0.001, 0.01, 0.1, 1, 10, 100]
_C_VALUES: list[float] = [0.001, 0.01, 0.1, 1, 10, 100]


# ---------------------------------------------------------------------------
# Feature loading and selection
# ---------------------------------------------------------------------------

def load_data(
    task_symbol: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load kinematic feature arrays and return task-specific subsets.

    The ``.npy`` files are expected to live alongside this script.

    Args:
        task_symbol: Task index (0 = Suturing, 1 = Knot_Tying, 2 = Needle_Passing).

    Returns:
        A 5-tuple ``(feature_su, feature_kt, feature_np, feature_raw, label)``.
    """
    task = TASK_LIST[task_symbol]
    feature = np.load(f"{task}_feature.npy")
    label = np.load(f"{task}_grs.npy")

    # Reshape to (n_trials, n_instruments * n_feature_types)
    feature_re = np.reshape(feature, (feature.shape[0], feature.shape[1] * feature.shape[2]))

    # Feature block indices (10 features per instrument × 4 instruments)
    # t (time), d (displacement), v (velocity x2), ta (turning angle x2),
    # s (smoothness x2), curvity (x2)
    def _block(col: int) -> np.ndarray:
        return feature_re[:, col: col + 1]

    def _block2(col: int) -> np.ndarray:
        return feature_re[:, col: col + 2]

    t   = np.hstack([_block(i * 10 + 0) for i in range(4)])   # time
    v   = np.hstack([_block2(i * 10 + 2) for i in range(4)])  # velocity mean+var
    s   = np.hstack([_block2(i * 10 + 6) for i in range(4)])  # smoothness mean+var
    cur = np.hstack([_block2(i * 10 + 4) for i in range(4)])  # curvature mean+var
    ta  = np.hstack([_block2(i * 10 + 8) for i in range(4)])  # turning angle mean+var

    feature_su = np.hstack((t, v, s, ta))           # Suturing: 28 features
    feature_kt = np.hstack((t, v, s))               # Knot_Tying: 20 features
    feature_np = np.hstack((v, cur, ta))             # Needle_Passing: 24 features

    return feature_su, feature_kt, feature_np, feature_re, label


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run feature-based SVC skill classification with grid search."""
    # Task selection: 0=Suturing, 1=Knot_Tying, 2=Needle_Passing
    task_symbol = 0
    select_symbol = 0  # which feature set to use (same task_symbol for matching task)

    data_su, data_kt, data_np, _, label = load_data(task_symbol)
    data_f = [data_su, data_kt, data_np][select_symbol]
    n_features = _N_FEATURES[select_symbol]

    extractor = DataExtract()
    extractor.get_category()
    extractor.get_txt_index()
    extractor.get_score()

    train_list = extractor.train_sum()
    test_list = extractor.test_sum()

    train_indices = [
        sorted(extractor.metaData_index[name] for name in fold)
        for fold in train_list
    ]
    test_indices = [
        sorted(extractor.metaData_index[name] for name in fold)
        for fold in test_list
    ]

    n_folds = len(train_indices)

    # Build per-fold feature matrices
    train_vectors: list[np.ndarray] = []
    test_vectors: list[np.ndarray] = []
    for fold_idx in range(n_folds):
        # Training set
        train_feat = np.zeros((1, n_features))
        for j in train_indices[fold_idx]:
            train_feat = np.vstack((train_feat, data_f[j]))
        train_vectors.append(np.delete(train_feat, 0, axis=0))

        # Test set
        test_feat = np.zeros((1, n_features))
        for j in test_indices[fold_idx]:
            test_feat = np.vstack((test_feat, data_f[j]))
        test_vectors.append(np.delete(test_feat, 0, axis=0))

    # Build per-fold label vectors
    idx_to_name = {v: k for k, v in extractor.metaData_index.items()}
    train_targets = [
        [extractor.metaData_score[idx_to_name[j]][1] for j in fold if j in idx_to_name]
        for fold in train_indices
    ]
    test_targets = [
        [extractor.metaData_score[idx_to_name[j]][1] for j in fold if j in idx_to_name]
        for fold in test_indices
    ]

    # Grid search
    best_score: float = 0.0
    best_parameters: dict = {}
    for gamma in _GAMMA_VALUES:
        for C in _C_VALUES:
            fold_score_sum = sum(
                SVC(gamma=gamma, C=C).fit(train_vectors[i], train_targets[i]).score(
                    test_vectors[i], test_targets[i]
                )
                for i in range(n_folds)
            )
            if fold_score_sum > best_score:
                best_score = fold_score_sum
                best_parameters = {"C": C, "gamma": gamma}

    print(f"best score: {best_score:.4f}")
    print(f"best parameters: {best_parameters}")
    print(f"Test set average score for {n_folds} sets: {best_score / n_folds:.4f}")


if __name__ == "__main__":
    main()
