#!/usr/bin/env python3
"""
SVM classification from CSV kinematic features.

Loads ``data_for_SVM.csv`` (produced by ``load_and_import_new_r.py``),
performs a train/test split, scales features, and fits an SVC.

Update DATA_PATH if the CSV lives outside the current working directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Path to the CSV produced by load_and_import_new_r.py
DATA_PATH: Path = Path(".")


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load kinematic features from the CSV file.

    Args:
        data_path: Directory containing ``data_for_SVM.csv``.

    Returns:
        A :class:`pandas.DataFrame` with feature columns and a ``"class"`` column.
    """
    return pd.read_csv(data_path / "data_for_SVM.csv")


def main() -> None:
    data = load_data()

    train_set, test_set = train_test_split(data, test_size=0.5, random_state=42)

    train_X = train_set.drop("class", axis=1)
    train_y = train_set["class"].copy()
    test_X = test_set.drop("class", axis=1)
    test_y = test_set["class"].copy()

    scaler = MinMaxScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    svm = SVC(C=100)
    svm.fit(train_X_scaled, train_y)

    print(f"Train accuracy: {svm.score(train_X_scaled, train_y):.4f}")
    print(f"Test  accuracy: {svm.score(test_X_scaled, test_y):.4f}")


if __name__ == "__main__":
    main()
