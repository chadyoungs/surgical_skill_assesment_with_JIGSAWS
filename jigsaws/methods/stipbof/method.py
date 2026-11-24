"""STIP + Bag-of-Features skill assessment.

``stipbof cluster`` learns per-fold HOG/HOF K-means vocabularies and converts
every video into two normalised histograms. ``stipbof classify`` aggregates
those histograms per trial and runs the standard LOSO grid search over an RBF
SVM. Fold construction, metrics and reporting come from the shared layers, so
only the method-specific feature logic lives here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ...dataset import Dataset
from ...deps import require
from ...errors import ArtifactNotFoundError, DataError
from ...evaluation import SearchResult, grid_search_loso_per_fold
from ..base import Command, Context, Method

LOGGER = logging.getLogger(__name__)

DEFAULT_CLUSTERS = 400
DEFAULT_MAX_SAMPLES = 100_000
_HOG_SLICE = slice(9, 81)   # 72 dims
_HOF_SLICE = slice(81, 171) # 90 dims
_DEFAULT_GAMMAS = (0.001, 0.01, 0.1, 1, 10, 100)
_DEFAULT_CS = (0.001, 0.01, 0.1, 1, 10, 100)


def _trial_of(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) > 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return stem


def _load_features(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 171:
        raise DataError(f"Expected >=171 STIP columns in {path}, got {matrix.shape}")
    return matrix


def _hog_hof(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[:, _HOG_SLICE], matrix[:, _HOF_SLICE]


def _l2_normalized(counts: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(counts))
    return counts / norm if norm else counts


def _histogram(
    matrix: np.ndarray, hog_model, hof_model, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    hog, hof = _hog_hof(matrix)
    hog_counts = np.bincount(hog_model.predict(hog), minlength=n_clusters)[:n_clusters]
    hof_counts = np.bincount(hof_model.predict(hof), minlength=n_clusters)[:n_clusters]
    return _l2_normalized(hog_counts.astype(np.float64)), _l2_normalized(hof_counts.astype(np.float64))


def _random_subsample(data: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if data.shape[0] <= limit:
        return data
    rng = np.random.default_rng(seed)
    return data[rng.permutation(data.shape[0])[:limit]]


def _feature_files(feature_dir: Path) -> list[Path]:
    files = sorted(feature_dir.glob("*.txt"))
    if not files:
        raise DataError(f"No STIP feature files (*.txt) in {feature_dir}")
    return files


def _stack_by_trials(
    files: list[Path], trials: set[str], *, hog_slice=_HOG_SLICE, hof_slice=_HOF_SLICE
) -> tuple[np.ndarray, np.ndarray, int]:
    hog_parts, hof_parts = [], []
    used = 0
    for path in files:
        if _trial_of(path.stem) not in trials:
            continue
        matrix = _load_features(path)
        hog, hof = matrix[:, hog_slice], matrix[:, hof_slice]
        hog_parts.append(hog)
        hof_parts.append(hof)
        used += 1
    if not hog_parts:
        raise DataError("No STIP feature files matched the requested trials")
    return np.vstack(hog_parts), np.vstack(hof_parts), used


def _aggregate_by_trial(
    names: list[str], hog: np.ndarray, hof: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Average histograms per trial so each trial contributes one SVM sample."""
    grouped: dict[str, list[np.ndarray]] = {}
    for name, hog_row, hof_row in zip(names, hog, hof):
        trial = _trial_of(Path(name).stem)
        grouped.setdefault(trial, []).append(np.concatenate([hog_row, hof_row]))
    trials = sorted(grouped)
    return trials, np.vstack([np.mean(grouped[trial], axis=0) for trial in trials])


class StipBofMethod(Method):
    name = "stipbof"
    aliases = ("stip",)
    summary = "STIP + bag-of-features skill classification (HOG/HOF + SVM)"

    def commands(self) -> tuple[Command, ...]:
        return (
            Command("cluster", "Learn per-fold HOG/HOF vocabularies and histograms",
                    self._cluster, self._configure_cluster),
            Command("classify", "LOSO SVM classification from BoF histograms",
                    self._classify, self._configure_classify),
        )

    @staticmethod
    def _configure_cluster(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--features-dir", type=Path, default=None,
                            help="Directory of per-video STIP .txt features")
        parser.add_argument("--kind", default="GestureClassification")
        parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
        parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--output-dir", type=Path, default=None)

    @staticmethod
    def _configure_classify(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--time-series-dir", type=Path, default=None)
        parser.add_argument("--kind", default="GestureClassification")
        parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
        parser.add_argument("--gammas", type=float, nargs="+", default=list(_DEFAULT_GAMMAS))
        parser.add_argument("--cs", type=float, nargs="+", default=list(_DEFAULT_CS))
        parser.add_argument("--output", type=Path, default=None)

    def _cluster(self, ctx: Context, args: argparse.Namespace) -> None:
        joblib = require("joblib", extra="ml")
        kmeans_cls = require("sklearn.cluster", extra="ml").KMeans

        feature_dir = args.features_dir or (ctx.settings.stip_features_dir / ctx.task.value)
        if not feature_dir.is_dir():
            raise DataError(
                f"STIP feature directory not found: {feature_dir}\n"
                "Pass --stip-features or --features-dir."
            )
        files = _feature_files(feature_dir)
        output = args.output_dir or (
            ctx.settings.artifacts_dir / "stipbof" / ctx.task.value / "time_series"
        )
        clusters_dir = output / "clusters"
        output.mkdir(parents=True, exist_ok=True)
        clusters_dir.mkdir(parents=True, exist_ok=True)

        folds = ctx.dataset.folds(kind=args.kind)
        report: list[dict[str, object]] = []
        for fold in folds:
            hog_matrix, hof_matrix, n_train_files = _stack_by_trials(files, set(fold.train))
            hog_sample = _random_subsample(hog_matrix, args.max_samples, args.seed)
            hof_sample = _random_subsample(hof_matrix, args.max_samples, args.seed)
            hog_model = kmeans_cls(n_clusters=args.clusters, random_state=args.seed).fit(hog_sample)
            hof_model = kmeans_cls(n_clusters=args.clusters, random_state=args.seed).fit(hof_sample)
            joblib.dump(hog_model, clusters_dir / f"{fold.scheme}_{fold.name}_hog.pkl")
            joblib.dump(hof_model, clusters_dir / f"{fold.scheme}_{fold.name}_hof.pkl")

            names: list[str] = []
            hog_hist: list[np.ndarray] = []
            hof_hist: list[np.ndarray] = []
            for path in files:
                matrix = _load_features(path)
                hog_row, hof_row = _histogram(matrix, hog_model, hof_model, args.clusters)
                names.append(path.name)
                hog_hist.append(hog_row)
                hof_hist.append(hof_row)
            joblib.dump(
                {"names": names, "hog": np.vstack(hog_hist), "hof": np.vstack(hof_hist)},
                output / f"{fold.scheme}_{fold.name}_histograms.pkl",
            )
            report.append(
                {
                    "fold": fold.name,
                    "n_train_files": n_train_files,
                    "n_videos": len(names),
                    "clusters": args.clusters,
                }
            )
            LOGGER.info(
                "Fold %s: vocabulary from %d training files, histograms for %d videos",
                fold.name,
                n_train_files,
                len(names),
            )

        (output / "clustering_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote BoF vocabularies and histograms to %s", output)

    def _classify(self, ctx: Context, args: argparse.Namespace) -> None:
        joblib = require("joblib", extra="ml")
        svc_cls = require("sklearn.svm", extra="ml").SVC

        dataset: Dataset = ctx.dataset
        time_series_dir = args.time_series_dir or (
            ctx.settings.artifacts_dir / "stipbof" / ctx.task.value / "time_series"
        )
        folds = dataset.folds(kind=args.kind)
        if not time_series_dir.is_dir():
            raise ArtifactNotFoundError(
                f"BoF time series not found: {time_series_dir}. Run 'stipbof cluster' first."
            )

        names: list[str] = []
        fold_features: dict[str, tuple[list[str], np.ndarray]] = {}
        for fold in folds:
            fold_file = time_series_dir / f"{fold.scheme}_{fold.name}_histograms.pkl"
            if not fold_file.is_file():
                raise ArtifactNotFoundError(
                    f"Missing histograms for fold {fold.name}: {fold_file}"
                )
            payload = joblib.load(fold_file)
            trial_names, matrix = _aggregate_by_trial(
                payload["names"], payload["hog"], payload["hof"]
            )
            fold_features[fold.name] = (trial_names, matrix)
            if not names:
                names = trial_names

        # All folds share the same video set; assert consistency rather than guess.
        for fold_name, (fold_names, _) in fold_features.items():
            if fold_names != names:
                raise DataError(
                    f"Fold {fold_name} covers different trials than fold {folds[0].name}; "
                    "re-run 'stipbof cluster' for a consistent artifact set."
                )
        n_features = fold_features[folds[0].name][1].shape[1]
        labels_by_name = {trial: dataset.grade(trial) for trial in names}

        LOGGER.info("Start to classify %d trials with %d BoF dimensions", len(names), n_features)
        result: SearchResult = grid_search_loso_per_fold(
            folds,
            fold_features,
            labels_by_name,
            lambda gamma, C: svc_cls(gamma=gamma, C=C),
            {"gamma": args.gammas, "C": args.cs},
        )
        result.log()
        output = args.output or (
            ctx.settings.artifacts_dir / "stipbof" / ctx.task.value / "classification.json"
        )
        result.write(
            output,
            extra={
                "task": ctx.task.value,
                "method": "stipbof",
                "n_trials": len(names),
                "n_features": int(n_features),
                "time_series_dir": str(time_series_dir),
            },
        )
        LOGGER.info("Wrote classification report: %s", output)


METHOD = StipBofMethod()
