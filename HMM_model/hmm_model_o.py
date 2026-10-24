#!/usr/bin/env python3
"""
HMM model training for surgical skill assessment on the JIGSAWS dataset.

Builds separate expert and novice Hidden Markov Models from kinematic data,
then saves both models and the observation sequences to disk.
"""
from __future__ import annotations

from pathlib import Path

import glob
import joblib
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA

import Global_Var
from data_extract import DataExtract

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_LIST: list[str] = ["Suturing", "Knot_Tying", "Needle_Passing"]

# Surgeme labels per task
_SURGEMES_SUTURING_NP: list[str] = ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"]
_SURGEMES_KNOT_TYING: list[str] = ["G1", "G11", "G12", "G13", "G14", "G15"]


def _get_surgeme_labels(task_symbol: int) -> list[str]:
    """Return the ordered list of surgeme labels for *task_symbol*."""
    if task_symbol == 1:  # Knot_Tying
        return _SURGEMES_KNOT_TYING
    return _SURGEMES_SUTURING_NP  # Suturing (0) or Needle_Passing (2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load kinematic data from *path* and split into the four instrument streams.

    Args:
        path: Path to a whitespace-delimited kinematics ``.txt`` file.

    Returns:
        A 5-tuple ``(X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2, surgery_name)``.
    """
    surgery_name = Path(path).stem
    data = np.loadtxt(path, dtype=np.float64)

    X_MTM_L = data[:, 0:19]
    X_MTM_R = data[:, 19:38]
    X_PSM_1 = data[:, 38:57]
    X_PSM_2 = data[:, 57:]

    return X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2, surgery_name


# ---------------------------------------------------------------------------
# Feature transformation
# ---------------------------------------------------------------------------

def pca_transform(data: np.ndarray) -> np.ndarray:
    """Fit a whitened PCA on *data* and return the transformed result.

    Args:
        data: Input array of shape ``(n_samples, n_features)``.

    Returns:
        Transformed array of shape ``(n_samples, PCA_COMPONENTS)``.
    """
    pca = PCA(n_components=Global_Var.PCA_COMPONENTS, whiten=True, random_state=0).fit(data)
    return pca.transform(data)


# ---------------------------------------------------------------------------
# K-means cluster loader
# ---------------------------------------------------------------------------

def load_kmeans(surgeme: str, task_symbol: int) -> object:
    """Load the pre-trained K-means model for *surgeme* and *task_symbol*.

    Args:
        surgeme: Surgeme label, e.g. ``"G5"``.
        task_symbol: Task index (0 = Suturing, 1 = Knot_Tying, 2 = Needle_Passing).

    Returns:
        A fitted ``sklearn.cluster.KMeans`` instance.
    """
    task_dir = Path(".") / "observation_clusters" / TASK_LIST[task_symbol]
    surgeme_labels = _get_surgeme_labels(task_symbol)
    kmeans_path = task_dir / f"{surgeme}.pkl"
    return joblib.load(kmeans_path)


# ---------------------------------------------------------------------------
# HMM building
# ---------------------------------------------------------------------------

def build_hmm(
    files_path: str | Path,
    metadata: DataExtract,
    task_symbol: int,
) -> tuple[hmm.MultinomialHMM, hmm.MultinomialHMM, list]:
    """Build expert and novice HMMs from the kinematic files in *files_path*.

    Args:
        files_path: Directory containing per-trial kinematics ``.txt`` files.
        metadata: A fully populated :class:`DataExtract` instance.
        task_symbol: Task index (0 = Suturing, 1 = Knot_Tying, 2 = Needle_Passing).

    Returns:
        A 3-tuple ``(model_expert, model_novice, observations)``.
    """
    clusters = Global_Var.CLUSTERS
    states = _get_surgeme_labels(task_symbol)
    n_states = len(states)

    # Accumulation matrices
    A_expert = np.zeros((n_states, n_states))
    A_novice = np.zeros((n_states, n_states))
    B_expert = np.zeros((n_states, clusters))
    B_novice = np.zeros((n_states, clusters))
    pi_expert = np.zeros(n_states)
    pi_novice = np.zeros(n_states)

    observations: list = []

    for txt_file in sorted(Path(files_path).glob("*.txt")):
        X_MTM_L, X_MTM_R, X_PSM_1, X_PSM_2, surgery_name = load_data(txt_file)
        data_trans = np.hstack((
            pca_transform(X_MTM_L),
            pca_transform(X_MTM_R),
            pca_transform(X_PSM_1),
            pca_transform(X_PSM_2),
        ))

        is_novice = metadata.metaData_score[surgery_name][1] == 0

        # Initial state distribution (pi)
        first_surgeme = metadata.metaData_surgeme[surgery_name][2][0]
        for i, state in enumerate(states):
            if first_surgeme == state:
                if is_novice:
                    pi_novice[i] += 1
                else:
                    pi_expert[i] += 1

        # Transition (A) and emission (B) matrices
        surgeme_seq = metadata.metaData_surgeme[surgery_name][2]
        observation: list = []
        for seg_idx, surgeme in enumerate(surgeme_seq):
            start = int(metadata.metaData_surgeme[surgery_name][0][seg_idx])
            end = int(metadata.metaData_surgeme[surgery_name][1][seg_idx])
            kmeans = load_kmeans(surgeme, task_symbol)

            result = kmeans.predict(np.float64(data_trans[start:end, :]))

            state_idx = states.index(surgeme)
            for obs in result:
                if is_novice:
                    B_novice[state_idx, obs] += 1
                else:
                    B_expert[state_idx, obs] += 1

            observation.append(result.tolist())

            if seg_idx < len(surgeme_seq) - 1:
                next_state_idx = states.index(surgeme_seq[seg_idx + 1])
                if is_novice:
                    A_novice[state_idx, next_state_idx] += 1
                else:
                    A_expert[state_idx, next_state_idx] += 1

        observations.append(observation)

    # Normalise pi
    pi_novice /= np.sum(pi_novice)
    pi_expert /= np.sum(pi_expert)

    # Normalise A row-wise
    for i in range(n_states):
        if np.sum(A_novice[i]) != 0:
            A_novice[i] /= np.sum(A_novice[i])
        if np.sum(A_expert[i]) != 0:
            A_expert[i] /= np.sum(A_expert[i])

    # Normalise B row-wise
    for i in range(n_states):
        if np.sum(B_novice[i]) != 0:
            B_novice[i] /= np.sum(B_novice[i])
        if np.sum(B_expert[i]) != 0:
            B_expert[i] /= np.sum(B_expert[i])

    model_expert = hmm.MultinomialHMM(n_components=n_states)
    model_expert.startprob_ = pi_expert
    model_expert.emissionprob_ = B_expert
    model_expert.transmat_ = A_expert

    model_novice = hmm.MultinomialHMM(n_components=n_states)
    model_novice.startprob_ = pi_novice
    model_novice.emissionprob_ = B_novice
    model_novice.transmat_ = A_novice

    return model_expert, model_novice, observations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Train HMMs and persist models + observations to disk."""
    task_symbol = Global_Var.TASK_SYMBOL
    task = TASK_LIST[task_symbol]

    metadata = DataExtract()
    metadata.get_category()
    metadata.get_txt_index()
    metadata.get_score()
    metadata.get_frame_surgeme()
    metadata.get_frame_No()
    metadata.get_hmm_states()

    files_path = Path(".") / "da_vici_data_with_iDT_features" / task / "kinematics" / "AllGestures"
    model_dir = Path(".") / "models"
    obs_dir = Path(".") / "observations"

    model_expert, model_novice, observations = build_hmm(files_path, metadata, task_symbol)

    save_expert = model_dir / f"{task}_expert_model.pkl"
    save_novice = model_dir / f"{task}_novice_model.pkl"
    save_obs = obs_dir / f"{task}_observations.pkl"

    joblib.dump(model_expert, save_expert)
    joblib.dump(model_novice, save_novice)
    joblib.dump(observations, save_obs)

    print(f"Done. Dumping expert models to {save_expert.name}")
    print(f"Done. Dumping novice models to {save_novice.name}")
    print(f"Done. Dumping observations to {save_obs.name}")
    print("####################\n")


if __name__ == "__main__":
    main()
