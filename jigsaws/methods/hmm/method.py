"""HMM skill assessment: observation clustering, model building, likelihood test.

Pipeline (unchanged from the original):

1. ``hmm clusters`` quantises per-surgeme kinematics with PCA + K-means.
2. ``hmm train`` counts state transitions/emissions per skill class and builds
   one MultinomialHMM for experts and one for novices.
3. ``hmm evaluate`` scores every trial's observation sequence under both HMMs
   with the forward algorithm and reports novice/expert accuracy.

Artifacts live under ``artifacts/hmm/<task>/``; existing models in the legacy
``HMM_model/`` folder are still discovered as a fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...dataset import Dataset
from ...deps import require
from ...errors import ArtifactNotFoundError, DataError
from ...evaluation import BinaryMetrics
from ..base import Command, Context, Method

LOGGER = logging.getLogger(__name__)

DEFAULT_CLUSTERS = 64
DEFAULT_PCA_COMPONENTS = 4
_STREAMS: tuple[slice, ...] = (slice(0, 19), slice(19, 38), slice(38, 57), slice(57, 76))


# ---------------------------------------------------------------------------
# Observation quantisation
# ---------------------------------------------------------------------------


def _pca(data: np.ndarray, components: int) -> np.ndarray:
    pca_cls = require("sklearn.decomposition", extra="ml").PCA
    return pca_cls(n_components=components, whiten=True, random_state=0).fit(
        data
    ).transform(data)


def _observation_features(matrix: np.ndarray, components: int) -> np.ndarray:
    """Whitened PCA per instrument stream, concatenated (legacy recipe)."""
    return np.hstack([_pca(matrix[:, stream], components) for stream in _STREAMS])


def _load_clip_streams(clips_dir: Path, label: str) -> tuple[np.ndarray, int]:
    files = sorted(path for path in clips_dir.glob("*.txt") if f"_{label}_" in path.stem)
    if not files:
        raise DataError(f"No clip files matching '_{label}_' in {clips_dir}")
    matrices = []
    for path in files:
        matrix = np.loadtxt(path, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < 76:
            raise DataError(f"Expected >=76 columns in {path}, got {matrix.shape}")
        matrices.append(matrix[:, :76])
    return np.vstack(matrices), len(files)


def fit_clusters(
    dataset: Dataset,
    clips_dir: Path,
    output_dir: Path,
    *,
    n_clusters: int,
    pca_components: int,
    seed: int,
) -> tuple[Path, ...]:
    joblib = require("joblib", extra="ml")
    kmeans_cls = require("sklearn.cluster", extra="ml").KMeans
    if not clips_dir.is_dir():
        raise DataError(f"Kinematic clip directory not found: {clips_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, object]] = []
    paths: list[Path] = []
    for label in dataset.surgeme_labels:
        frames, n_clips = _load_clip_streams(clips_dir, label)
        if frames.shape[0] < pca_components:
            raise DataError(
                f"Surgeme {label} has {frames.shape[0]} frames; need >= {pca_components}"
            )
        features = _observation_features(frames, pca_components)
        model = kmeans_cls(n_clusters=n_clusters, random_state=seed).fit(features)
        path = output_dir / f"{label}.pkl"
        joblib.dump(model, path)
        paths.append(path)
        report.append(
            {"label": label, "clips": n_clips, "frames": int(frames.shape[0]), "clusters": n_clusters}
        )
        LOGGER.info("Clustered %s: %d clips, %d frames", label, n_clips, frames.shape[0])

    (output_dir / "clusters_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return tuple(paths)


# ---------------------------------------------------------------------------
# HMM building
# ---------------------------------------------------------------------------


def _normalize_rows(matrix: np.ndarray, *, uniform_when_empty: bool = True) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    if uniform_when_empty and np.any(sums == 0):
        matrix = matrix.copy()
        matrix[sums[:, 0] == 0] = 1.0
        sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)


@dataclass(frozen=True, slots=True)
class TrainedModels:
    expert: object
    novice: object
    names: tuple[str, ...]
    sequences: tuple[tuple[np.ndarray, ...], ...]
    n_expert: int
    n_novice: int


def build_models(
    dataset: Dataset,
    cluster_dir: Path,
    *,
    n_clusters: int,
    pca_components: int,
) -> TrainedModels:
    joblib = require("joblib", extra="ml")
    hmm_mod = require("hmmlearn.hmm", extra="ml")

    states = dataset.surgeme_labels
    state_index = {label: index for index, label in enumerate(states)}
    n_states = len(states)

    clusters = {}
    for label in states:
        path = cluster_dir / f"{label}.pkl"
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"Observation clusters missing for {label}: {path}\n"
                "Run 'hmm clusters' first or pass --clusters-dir."
            )
        clusters[label] = joblib.load(path)

    A_expert = np.zeros((n_states, n_states))
    A_novice = np.zeros((n_states, n_states))
    B_expert = np.zeros((n_states, n_clusters))
    B_novice = np.zeros((n_states, n_clusters))
    pi_expert = np.zeros(n_states)
    pi_novice = np.zeros(n_states)

    names: list[str] = []
    sequences: list[tuple[np.ndarray, ...]] = []
    n_expert = n_novice = 0

    for trial in dataset.trial_names:
        if not dataset.has_kinematics(trial):
            LOGGER.warning("No kinematics for %s; excluded from HMM training", trial)
            continue
        record = dataset[trial]
        matrix = dataset.load_kinematics(trial)
        features = _observation_features(matrix, pca_components)
        is_novice = record.grade == 0
        if is_novice:
            n_novice += 1
        else:
            n_expert += 1

        first_label = record.segments[0].label
        if first_label in state_index:
            target = pi_novice if is_novice else pi_expert
            target[state_index[first_label]] += 1

        observation: list[np.ndarray] = []
        for segment_index, segment in enumerate(record.segments):
            start, end = segment.start, segment.end
            rows = features[start:end]
            if rows.shape[0] == 0:
                raise DataError(
                    f"Empty observation window for {trial} segment {segment_index} "
                    f"({segment.label}, frames {start}:{end})"
                )
            if segment.label not in clusters:
                raise DataError(f"No K-means model for surgeme {segment.label}")
            predicted = clusters[segment.label].predict(np.asarray(rows, dtype=np.float64))
            state = state_index[segment.label]
            counts = np.bincount(predicted, minlength=n_clusters)[:n_clusters]
            if is_novice:
                B_novice[state] += counts
            else:
                B_expert[state] += counts
            observation.append(np.asarray(predicted))

            if segment_index + 1 < len(record.segments):
                next_state = state_index[record.segments[segment_index + 1].label]
                if is_novice:
                    A_novice[state, next_state] += 1
                else:
                    A_expert[state, next_state] += 1

        names.append(trial)
        sequences.append(tuple(observation))

    if n_expert == 0 or n_novice == 0:
        raise DataError("Both skill classes need at least one trial to train HMMs")
    if pi_expert.sum() == 0 or pi_novice.sum() == 0:
        LOGGER.warning("A start-state distribution was empty; using uniform start probabilities")
        pi_expert = np.full(n_states, 1.0 / n_states)
        pi_novice = np.full(n_states, 1.0 / n_states)
    else:
        pi_expert = _normalize_rows(pi_expert.reshape(1, -1))[0]
        pi_novice = _normalize_rows(pi_novice.reshape(1, -1))[0]

    A_expert = _normalize_rows(A_expert)
    A_novice = _normalize_rows(A_novice)
    B_expert = _normalize_rows(B_expert)
    B_novice = _normalize_rows(B_novice)

    expert = hmm_mod.MultinomialHMM(n_components=n_states)
    expert.startprob_ = pi_expert
    expert.transmat_ = A_expert
    expert.emissionprob_ = B_expert

    novice = hmm_mod.MultinomialHMM(n_components=n_states)
    novice.startprob_ = pi_novice
    novice.transmat_ = A_novice
    novice.emissionprob_ = B_novice

    return TrainedModels(
        expert=expert,
        novice=novice,
        names=tuple(names),
        sequences=tuple(sequences),
        n_expert=n_expert,
        n_novice=n_novice,
    )


# ---------------------------------------------------------------------------
# Forward likelihood
# ---------------------------------------------------------------------------


def forward_probability(
    transmat: np.ndarray,
    emissionprob: np.ndarray,
    startprob: np.ndarray,
    sequence: tuple[np.ndarray, ...] | list[np.ndarray],
) -> float:
    """Segment-level forward probability (legacy algorithm, vectorised).

    Each observation is a segment of cluster ids; its emission is the mean
    emission probability over the frames in that segment.
    """
    n_states = len(startprob)
    alpha = np.zeros((n_states, len(sequence)))
    first = np.asarray(sequence[0], dtype=int)
    alpha[:, 0] = startprob * np.mean(emissionprob[:, first], axis=1)
    for step in range(1, len(sequence)):
        symbols = np.asarray(sequence[step], dtype=int)
        emissions = np.mean(emissionprob[:, symbols], axis=1)
        alpha[:, step] = (alpha[:, step - 1] @ transmat) * emissions
    return float(np.sum(alpha[:, -1]))


# ---------------------------------------------------------------------------
# Artifact resolution
# ---------------------------------------------------------------------------


def _task_artifact(ctx: Context, *parts: str) -> Path:
    return ctx.settings.artifacts_dir / "hmm" / ctx.task.value / Path(*parts)


def _legacy(ctx: Context, *parts: str) -> Path:
    return ctx.settings.repo_root.joinpath("HMM_model", *parts)


def _resolve_clusters_dir(ctx: Context, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    artifact = _task_artifact(ctx, "clusters")
    if any(artifact.glob("*.pkl")):
        return artifact
    legacy = _legacy(ctx, "observation_clusters", ctx.task.value)
    if any(legacy.glob("*.pkl")):
        LOGGER.info("Using legacy observation clusters: %s", legacy)
        return legacy
    return artifact


def _resolve_models_dir(ctx: Context, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    artifact = _task_artifact(ctx, "models")
    if (artifact / f"{ctx.task.value}_expert_model.pkl").is_file():
        return artifact
    legacy = _legacy(ctx, "models")
    if (legacy / f"{ctx.task.value}_expert_model.pkl").is_file():
        LOGGER.info("Using legacy HMM models: %s", legacy)
        return legacy
    return artifact


def _resolve_observations(ctx: Context, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    artifact = _task_artifact(ctx, "observations", f"{ctx.task.value}_observations.pkl")
    if artifact.is_file():
        return artifact
    legacy = _legacy(ctx, "observations", f"{ctx.task.value}_observations.pkl")
    if legacy.is_file():
        LOGGER.info("Using legacy observations: %s", legacy)
        return legacy
    return artifact


def _load_observations(path: Path, dataset: Dataset) -> tuple[tuple[str, ...], tuple[tuple[np.ndarray, ...], ...]]:
    joblib = require("joblib", extra="ml")
    payload = joblib.load(path)
    if isinstance(payload, dict) and "names" in payload:
        return tuple(payload["names"]), tuple(tuple(seq) for seq in payload["sequences"])
    if isinstance(payload, list):
        names = tuple(dataset.kinematics_trials[: len(payload)])
        return names, tuple(tuple(seq) for seq in payload)
    raise DataError(f"Unrecognised observations file format: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class HmmMethod(Method):
    name = "hmm"
    summary = "Hidden Markov Model skill assessment (clusters -> train -> evaluate)"

    def commands(self) -> tuple[Command, ...]:
        return (
            Command("clusters", "Fit per-surgeme PCA + K-means observation vocabularies",
                    self._clusters, self._configure_clusters),
            Command("train", "Build expert and novice MultinomialHMMs", self._train,
                    self._configure_train),
            Command("evaluate", "Classify trials by forward likelihood", self._evaluate,
                    self._configure_evaluate),
        )

    @staticmethod
    def _configure_clusters(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--clips-dir", type=Path, default=None,
                            help="Directory of per-surgeme kinematic clip .txt files")
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
        parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
        parser.add_argument("--seed", type=int, default=0)

    @staticmethod
    def _configure_train(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--clusters-dir", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
        parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)

    @staticmethod
    def _configure_evaluate(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--models-dir", type=Path, default=None)
        parser.add_argument("--observations", type=Path, default=None)
        parser.add_argument("--output", type=Path, default=None)

    def _clusters(self, ctx: Context, args: argparse.Namespace) -> None:
        clips = args.clips_dir or (
            ctx.settings.data_root / "Data_clips" / f"{ctx.task.value}_clips" / "kinetic"
        )
        output = args.output_dir or _task_artifact(ctx, "clusters")
        paths = fit_clusters(
            ctx.dataset,
            clips,
            output,
            n_clusters=args.clusters,
            pca_components=args.pca_components,
            seed=args.seed,
        )
        LOGGER.info("Wrote %d cluster models to %s", len(paths), output)

    def _train(self, ctx: Context, args: argparse.Namespace) -> None:
        joblib = require("joblib", extra="ml")
        clusters_dir = _resolve_clusters_dir(ctx, args.clusters_dir)
        models = build_models(
            ctx.dataset,
            clusters_dir,
            n_clusters=args.clusters,
            pca_components=args.pca_components,
        )
        output = args.output_dir or _task_artifact(ctx, "models")
        output.mkdir(parents=True, exist_ok=True)
        task = ctx.task.value
        expert_path = output / f"{task}_expert_model.pkl"
        novice_path = output / f"{task}_novice_model.pkl"
        joblib.dump(models.expert, expert_path)
        joblib.dump(models.novice, novice_path)

        observations_dir = _task_artifact(ctx, "observations")
        observations_dir.mkdir(parents=True, exist_ok=True)
        observations_path = observations_dir / f"{task}_observations.pkl"
        joblib.dump(
            {"names": list(models.names), "sequences": [list(map(list, seq)) for seq in models.sequences]},
            observations_path,
        )

        report = {
            "task": task,
            "states": list(ctx.dataset.surgeme_labels),
            "n_expert_trials": models.n_expert,
            "n_novice_trials": models.n_novice,
            "clusters": args.clusters,
            "pca_components": args.pca_components,
            "clusters_dir": str(clusters_dir),
            "models": [str(expert_path), str(novice_path)],
            "observations": str(observations_path),
        }
        (output / "training_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info(
            "Trained HMMs on %d trials (expert=%d, novice=%d)",
            len(models.names),
            models.n_expert,
            models.n_novice,
        )

    def _evaluate(self, ctx: Context, args: argparse.Namespace) -> None:
        joblib = require("joblib", extra="ml")
        dataset = ctx.dataset
        models_dir = _resolve_models_dir(ctx, args.models_dir)
        task = ctx.task.value
        expert_path = models_dir / f"{task}_expert_model.pkl"
        novice_path = models_dir / f"{task}_novice_model.pkl"
        if not (expert_path.is_file() and novice_path.is_file()):
            raise ArtifactNotFoundError(
                f"HMM models not found in {models_dir}. Run 'hmm train' first."
            )
        expert = joblib.load(expert_path)
        novice = joblib.load(novice_path)

        observations_path = _resolve_observations(ctx, args.observations)
        if not observations_path.is_file():
            raise ArtifactNotFoundError(
                f"Observations not found: {observations_path}. Run 'hmm train' first."
            )
        names, sequences = _load_observations(observations_path, dataset)

        probs_expert = np.array([
            forward_probability(expert.transmat_, expert.emissionprob_, expert.startprob_, seq)
            for seq in sequences
        ])
        probs_novice = np.array([
            forward_probability(novice.transmat_, novice.emissionprob_, novice.startprob_, seq)
            for seq in sequences
        ])
        predictions = np.where(probs_expert > probs_novice, 2, 0)
        truths = np.array([dataset.grade(name) for name in names])

        metrics = BinaryMetrics.compute(truths, predictions)
        LOGGER.info(
            "accuracy = %.4f | expert_accuracy = %.4f | novice_accuracy = %.4f",
            metrics.accuracy,
            metrics.expert_accuracy,
            metrics.novice_accuracy,
        )

        output = args.output or _task_artifact(ctx, "evaluation.json")
        payload = {
            "task": task,
            "n_trials": len(names),
            "models_dir": str(models_dir),
            "observations": str(observations_path),
            "metrics": metrics.as_dict(),
            "trials": [
                {
                    "trial": name,
                    "label": int(truth),
                    "prediction": int(pred),
                    "p_expert": float(pe),
                    "p_novice": float(pn),
                }
                for name, truth, pred, pe, pn in zip(names, truths, predictions, probs_expert, probs_novice)
            ],
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote evaluation report: %s", output)


METHOD = HmmMethod()
