#!/usr/bin/env python3
"""
STIP + BoF Step 1 – Clustering.

Trains K-means vocabularies (HOG and HOF) for each LOSO fold and then
predicts BoF histogram vectors for every video, saving the results to disk.

Pipeline overview
-----------------
Step 1 (this script): BoF vocabulary training and histogram prediction.
Step 2 (feature_processing.py): Feature extraction and normalisation.
Step 3 (classification.py): SVM / kNN / Bayes classification.
Step 4: Sliding-window evaluation.

Note: This method is post-hoc analysis, not real-time.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from glob import glob
from sklearn.cluster import KMeans

import Global_Var
from data_extract_stip import DataExtract


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_LIST: list[str] = ["Suturing", "Knot_Tying", "Needle_Passing"]
TASK_SYMBOL: int = Global_Var.TASK_SYMBOL
N_CLUSTERS: int = Global_Var.CLUSTERS
SELECTED_FEATURES_NO: int = Global_Var.SELECTED_FEATURES_NO

# HOG feature dimensionality (columns 9–80 in STIP output)
_HOG_DIM: int = 72
# HOF feature dimensionality (columns 81–170 in STIP output)
_HOF_DIM: int = 90


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_last_frame(videos_features_location: str | Path, last_frame_map: dict) -> None:
    """Populate *last_frame_map* with the maximum frame number per surgery.

    Args:
        videos_features_location: Directory containing STIP ``.txt`` files.
        last_frame_map: Dict to update in place: ``{surgery_name: last_frame_no}``.
    """
    for txt_file in Path(videos_features_location).glob("*.txt"):
        data = np.loadtxt(txt_file, dtype=float)
        last_frame = float(np.max(data[:, 6]))
        with open(txt_file) as f:
            for count, line in enumerate(f):
                if count == 1:
                    # Line format: "# <path/to/surgery_name_start_end.txt>"
                    stem = line.strip().lstrip("# ").rsplit(".", 1)[0]
                    parts = stem.split("_")
                    # Drop the last two tokens (frame indices) to get the surgery name
                    surgery_name = "_".join(parts[:-2])
                    last_frame_map[surgery_name] = last_frame
                    break


def kmeans_predict(
    video_feature_path: str | Path,
    hog_kmeans,
    hof_kmeans,
    n_clusters: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Predict BoF histogram vectors for a single video feature file.

    Args:
        video_feature_path: Path to a STIP ``.txt`` feature file.
        hog_kmeans: Fitted K-means model for HOG features.
        hof_kmeans: Fitted K-means model for HOF features.
        n_clusters: Vocabulary size (number of clusters).

    Returns:
        A 3-tuple ``(surgery_name, hog_histogram, hof_histogram)``.
    """
    surgery_name = Path(video_feature_path).stem
    result_hog = np.zeros(n_clusters)
    result_hof = np.zeros(n_clusters)

    raw_data = np.loadtxt(video_feature_path, dtype=float)
    for row in raw_data:
        hog_data = np.reshape(row[9:81], (1, _HOG_DIM))
        hof_data = np.reshape(row[81:], (1, _HOF_DIM))
        result_hog[hog_kmeans.predict(hog_data)[0]] += 1
        result_hof[hof_kmeans.predict(hof_data)[0]] += 1

    cv2.normalize(result_hog, result_hog, norm_type=cv2.NORM_L2)
    cv2.normalize(result_hof, result_hof, norm_type=cv2.NORM_L2)
    return surgery_name, result_hog, result_hof


def load_hog_hof(
    videos_features_location: str | Path,
    surgery_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw HOG and HOF feature arrays for a named surgery.

    Args:
        videos_features_location: Directory containing STIP ``.txt`` files.
        surgery_name: Stem of the desired file (without extension).

    Returns:
        A 2-tuple ``(hog_array, hof_array)`` each shaped ``(n_frames, dim)``.
    """
    for txt_file in Path(videos_features_location).glob("*.txt"):
        if txt_file.stem == surgery_name:
            raw_data = np.loadtxt(txt_file, dtype=float)
            data_hog = np.reshape(raw_data[:, 9:81], (-1, _HOG_DIM))
            data_hof = np.reshape(raw_data[:, 81:], (-1, _HOF_DIM))
            return data_hog, data_hof
    raise FileNotFoundError(
        f"No feature file found for surgery '{surgery_name}' in {videos_features_location}"
    )


def random_subsample(data: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Return up to *n* randomly sampled rows from *data*.

    Args:
        data: Array of shape ``(total_samples, n_features)``.
        n: Maximum number of samples to return.
        seed: Random seed for reproducibility.

    Returns:
        Subsampled array of shape ``(min(n, total_samples), n_features)``.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(data.shape[0])
    return data[indices[:n]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Train BoF vocabularies and predict histogram vectors for all folds."""
    task = TASK_LIST[TASK_SYMBOL]

    extractor = DataExtract()
    extractor.get_category()
    extractor.get_txt_index()
    extractor.get_score()

    train_list = extractor.train_sum()

    # Path to STIP raw feature files (adjust as needed for your environment)
    videos_features_location = Path("..") / "stip+dct_code" / "raw_data" / task

    n_folds = len(train_list)
    clusters_dir = Path(".") / "clusters" / task
    time_series_dir = Path(".") / "time_series_data" / task

    # ------------------------------------------------------------------
    # Training: fit one HOG and one HOF K-means per fold
    # ------------------------------------------------------------------
    hog_repo = [clusters_dir / f"clusters_{N_CLUSTERS}_set{i + 1}_hog.pkl" for i in range(n_folds)]
    hof_repo = [clusters_dir / f"clusters_{N_CLUSTERS}_set{i + 1}_hof.pkl" for i in range(n_folds)]

    all_video_features = list(Path(videos_features_location).glob("*.txt"))

    for fold_idx, fold_train in enumerate(train_list):
        hog_vector = np.zeros((0, _HOG_DIM))
        hof_vector = np.zeros((0, _HOF_DIM))

        for surgery_name in fold_train:
            hog_data, hof_data = load_hog_hof(videos_features_location, surgery_name)
            hog_vector = np.vstack((hog_vector, hog_data))
            hof_vector = np.vstack((hof_vector, hof_data))

        hog_vector = np.float64(random_subsample(hog_vector, SELECTED_FEATURES_NO))
        hog_kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(hog_vector)
        joblib.dump(hog_kmeans, hog_repo[fold_idx])

        hof_vector = np.float64(random_subsample(hof_vector, SELECTED_FEATURES_NO))
        hof_kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(hof_vector)
        joblib.dump(hof_kmeans, hof_repo[fold_idx])

        print(f"\n####################")
        print(
            f"Done. Dumping HOG and HOF clusters of train set {fold_idx + 1} to "
            f"{hog_repo[fold_idx].name} {hof_repo[fold_idx].name}"
        )

    print("\n####################")
    print("Done. HOG and HOF clusters of all train sets have been dumped!")

    # ------------------------------------------------------------------
    # Prediction: generate BoF histogram vectors for every video
    # ------------------------------------------------------------------
    ts_name_repo = [time_series_dir / f"time_series_surgery_name_set{i + 1}_{N_CLUSTERS}.pkl" for i in range(n_folds)]
    ts_hog_repo = [time_series_dir / f"time_series_hog_set{i + 1}_{N_CLUSTERS}.pkl" for i in range(n_folds)]
    ts_hof_repo = [time_series_dir / f"time_series_hof_set{i + 1}_{N_CLUSTERS}.pkl" for i in range(n_folds)]

    # Find the maximum frame number for each surgery (used downstream)
    last_frame_map: dict = {}
    find_last_frame(videos_features_location, last_frame_map)

    for fold_idx in range(n_folds):
        hog_kmeans = joblib.load(hog_repo[fold_idx])
        hof_kmeans = joblib.load(hof_repo[fold_idx])

        video_feature_names: list[str] = []
        time_series_hog: list[np.ndarray] = []
        time_series_hof: list[np.ndarray] = []

        for feature_file in all_video_features:
            surgery_name, hog_hist, hof_hist = kmeans_predict(
                feature_file, hog_kmeans, hof_kmeans, N_CLUSTERS
            )
            video_feature_names.append(surgery_name)
            time_series_hog.append(hog_hist)
            time_series_hof.append(hof_hist)

        joblib.dump(video_feature_names, ts_name_repo[fold_idx])
        joblib.dump(time_series_hog, ts_hog_repo[fold_idx])
        joblib.dump(time_series_hof, ts_hof_repo[fold_idx])

        print(f"\n####################")
        print(
            f"Done. Dumping time series of set {fold_idx + 1} to "
            f"{ts_name_repo[fold_idx].parent.name}/{ts_name_repo[fold_idx].name}"
        )

    print("\n####################")
    print("Done. time series of all train sets have been dumped!")


if __name__ == "__main__":
    main()
