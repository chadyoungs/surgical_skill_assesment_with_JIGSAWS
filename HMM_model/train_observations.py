#!/usr/bin/env python3
"""
K-means observation clustering for the JIGSAWS HMM pipeline.

For each surgeme label, load the kinematic clip data, apply PCA, then fit a
K-means model and save it to ``observation_clusters/<task>/<surgeme>.pkl``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import Global_Var


# ---------------------------------------------------------------------------
# Task/surgeme configuration (mirrors hmm_model_o.py)
# ---------------------------------------------------------------------------

TASK_LIST: list[str] = ["Suturing", "Knot_Tying", "Needle_Passing"]

_SURGEMES_SUTURING_NP: list[str] = ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"]
_SURGEMES_KNOT_TYING: list[str] = ["G1", "G11", "G12", "G13", "G14", "G15"]


def _get_surgeme_labels(task_symbol: int) -> list[str]:
    """Return the ordered surgeme label list for *task_symbol*."""
    return _SURGEMES_KNOT_TYING if task_symbol == 1 else _SURGEMES_SUTURING_NP


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    clips_dir: str | Path,
    surgeme_pattern: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack kinematic data from all clip files that match *surgeme_pattern*.

    Args:
        clips_dir: Directory containing per-surgeme-clip ``.txt`` files.
        surgeme_pattern: Substring used to filter relevant files (e.g. ``"_G1_"``).

    Returns:
        A 4-tuple ``(X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2)`` where each array
        has shape ``(total_frames, 19)``.

    Raises:
        ValueError: If no matching files are found in *clips_dir*.
    """
    X_MTM_L_list: list[np.ndarray] = []
    X_MTM_R_list: list[np.ndarray] = []
    X_PSM_1_list: list[np.ndarray] = []
    X_PSM_2_list: list[np.ndarray] = []

    for txt_file in sorted(Path(clips_dir).glob("*.txt")):
        if surgeme_pattern in txt_file.stem:
            data = np.loadtxt(txt_file, dtype=np.float64)
            X_MTM_L_list.append(data[:, 0:19])
            X_MTM_R_list.append(data[:, 19:38])
            X_PSM_1_list.append(data[:, 38:57])
            X_PSM_2_list.append(data[:, 57:])

    if not X_MTM_L_list:
        raise ValueError(
            f"No data files found matching surgeme pattern '{surgeme_pattern}' in {clips_dir}"
        )
    return (
        np.vstack(X_MTM_L_list),
        np.vstack(X_MTM_R_list),
        np.vstack(X_PSM_1_list),
        np.vstack(X_PSM_2_list),
    )


# ---------------------------------------------------------------------------
# PCA helper
# ---------------------------------------------------------------------------

def pca_transform(data: np.ndarray) -> np.ndarray:
    """Fit a whitened PCA on *data* and return the transformed result."""
    pca = PCA(n_components=Global_Var.PCA_COMPONENTS, whiten=True, random_state=0).fit(data)
    return pca.transform(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Cluster kinematic observations and save K-means models to disk."""
    n_clusters = Global_Var.CLUSTERS
    task_symbol = Global_Var.TASK_SYMBOL
    task = TASK_LIST[task_symbol]

    # Root directory for per-clip kinematic files.
    # Update this path to point at your local Data_clips directory.
    split_files_path = (
        Path(".")
        / "da_vici_data_with_iDT_features"
        / "Data_clips"
        / f"{task}_clips"
        / "kinetic"
    )

    gestures = _get_surgeme_labels(task_symbol)
    # Patterns include leading/trailing underscores to avoid partial matches (e.g. G1 vs G11)
    surgeme_patterns = [f"_{g}_" for g in gestures]
    save_dir = Path(".") / "observation_clusters" / task

    for surgeme_pattern, gesture in zip(surgeme_patterns, gestures):
        X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2 = load_data(split_files_path, surgeme_pattern)
        data_result = np.float64(np.hstack((
            pca_transform(X_MTM_L),
            pca_transform(X_MTM_R),
            pca_transform(X_PSM_1),
            pca_transform(X_PSM_2),
        )))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_result)

        save_path = save_dir / f"{gesture}.pkl"
        joblib.dump(kmeans, save_path)
        print(f"Done. Dumping to {save_path.name}")
        print("####################\n")


if __name__ == "__main__":
    main()
