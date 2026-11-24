"""Kinematic feature methods: feature export, SVM assessment, box plots.

All three commands operate on the same shared :mod:`jigsaws.kinematics`
feature extractor and the same :class:`Dataset` folds, so a new feature set
or classifier only needs to change the representation or the estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...dataset import Dataset
from ...deps import require
from ...errors import ArtifactNotFoundError, DataError
from ...evaluation import SearchResult, grid_search_loso
from ...kinematics import Instrument, compute_features
from ..base import Command, Context, Method

LOGGER = logging.getLogger(__name__)

_DEFAULT_GAMMAS = (0.001, 0.01, 0.1, 1, 10, 100)
_DEFAULT_CS = (0.001, 0.01, 0.1, 1, 10, 100)

# Metric axis labels used by the box-plot grid (legacy layout).
_PANEL_TITLES = (
    "Total time",
    "Total displacement",
    "Velocity mean",
    "Velocity variance",
    "Curvature mean",
    "Curvature variance",
    "Smoothness mean",
    "Smoothness variance",
    "Turning angle mean",
    "Turning angle variance",
)
_PANEL_FIELDS = (
    "time_sum",
    "displacement_sum",
    "velocity_mean",
    "velocity_variance",
    "curvature_mean",
    "curvature_variance",
    "smoothness_mean",
    "smoothness_variance",
    "turning_angle_mean",
    "turning_angle_variance",
)

# Task-specific feature subsets (columns are per instrument, 10 features each).
_TASK_BLOCKS = {
    "Suturing": (slice(0, 1), slice(2, 4), slice(6, 8), slice(8, 10)),
    "Knot_Tying": (slice(0, 1), slice(2, 4), slice(6, 8)),
    "Needle_Passing": (slice(2, 4), slice(4, 6), slice(8, 10)),
}

_CSV_HEADERS = (
    "time_sum",
    "displacement_sum_mtfl", "velocity_ave_mtfl", "velocity_var_mtfl",
    "curvity_ave_mtfl", "curvity_var_mtfl", "smooth_ave_mtfl", "smooth_var_mtfl",
    "turning_ave_mtfl", "turning_var_mtfl",
    "displacement_sum_mtfr", "velocity_ave_mtfr", "velocity_var_mtfr",
    "curvity_ave_mtfr", "curvity_var_mtfr", "smooth_ave_mtfr", "smooth_var_mtfr",
    "turning_ave_mtfr", "turning_var_mtfr",
    "displacement_sum_psm1", "velocity_ave_psm1", "velocity_var_psm1",
    "curvity_ave_psm1", "curvity_var_psm1", "smooth_ave_psm1", "smooth_var_psm1",
    "turning_ave_psm1", "turning_var_psm1",
    "displacement_sum_psm2", "velocity_ave_psm2", "velocity_var_psm2",
    "curvity_ave_psm2", "curvity_var_psm2", "smooth_ave_psm2", "smooth_var_psm2",
    "turning_ave_psm2", "turning_var_psm2",
    "class",
)


@dataclass(frozen=True, slots=True)
class FeatureTable:
    names: tuple[str, ...]
    features: np.ndarray  # (n_trials, 4, 10)
    grades: np.ndarray
    tiers: np.ndarray


def collect_feature_table(dataset: Dataset) -> FeatureTable:
    """Compute the ten features for every trial that has kinematics."""
    names: list[str] = []
    cubes: list[np.ndarray] = []
    grades: list[int] = []
    tiers: list[int] = []
    for record in dataset.records:
        if not dataset.has_kinematics(record.name):
            LOGGER.warning("No kinematics for %s; skipped", record.name)
            continue
        matrix = dataset.load_kinematics(record.name)
        cubes.append(np.array([compute_features(matrix, inst).as_tuple() for inst in Instrument]))
        names.append(record.name)
        grades.append(record.grade)
        tiers.append(record.tier)
    if not names:
        raise DataError("No trial has both metadata and a kinematics file")
    return FeatureTable(
        names=tuple(names),
        features=np.stack(cubes),
        grades=np.array(grades),
        tiers=np.array(tiers),
    )


def select_task_features(cube: np.ndarray, task: str) -> np.ndarray:
    """Flatten the task-specific blocks of ``(n, 4, 10)`` into ``(n, k)``."""
    if task not in _TASK_BLOCKS:
        raise DataError(f"No feature subset defined for task '{task}'")
    parts = []
    for block in _TASK_BLOCKS[task]:
        for instrument_index in range(cube.shape[1]):
            parts.append(cube[:, instrument_index, block])
    return np.hstack(parts)


def _resolve_feature_dir(ctx: Context, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    artifact = ctx.settings.artifacts_dir / "kinematic" / ctx.task.value / "features"
    legacy = ctx.settings.kinematic_features_dir
    task = ctx.task.value
    if (artifact / f"{task}_feature.npy").is_file():
        return artifact
    if (legacy / f"{task}_feature.npy").is_file():
        LOGGER.info("Using legacy feature directory: %s", legacy)
        return legacy
    return artifact


def load_feature_table(ctx: Context, explicit: Path | None) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Load ``<task>_feature.npy`` plus names/labels with legacy fallbacks."""
    directory = _resolve_feature_dir(ctx, explicit)
    task = ctx.task.value
    feature_path = directory / f"{task}_feature.npy"
    if not feature_path.is_file():
        raise ArtifactNotFoundError(
            f"Feature file not found: {feature_path}\n"
            "Run 'kinematic features' first or pass --features-dir."
        )
    features = np.load(feature_path)
    if features.ndim != 3 or features.shape[1:] != (len(Instrument), 10):
        raise DataError(
            f"Expected feature shape (n, {len(Instrument)}, 10), got {features.shape}"
        )

    names_path = directory / f"{task}_trials.npy"
    names = tuple(np.load(names_path).tolist()) if names_path.is_file() else ctx.dataset.trial_names
    if len(names) != features.shape[0]:
        raise DataError(
            f"{feature_path.name} has {features.shape[0]} rows but {len(names)} trial names; "
            "regenerate the feature table or supply --features-dir with matching files."
        )

    labels_path = directory / f"{task}_grs.npy"
    if labels_path.is_file():
        labels = np.load(labels_path).reshape(-1)
        if labels.shape[0] != len(names):
            raise DataError(
                f"{labels_path.name} has {labels.shape[0]} labels for {len(names)} rows"
            )
    else:
        labels = np.array([ctx.dataset.grade(name) for name in names])
    return names, features, labels


class KinematicMethod(Method):
    name = "kinematic"
    summary = "Kinematic trajectory features: export, SVM assessment, box plots"

    def commands(self) -> tuple[Command, ...]:
        return (
            Command("features", "Compute and export per-trial trajectory features",
                    self._features, self._configure_features),
            Command("svc", "LOSO RBF-SVM skill classification from feature tensors",
                    self._svc, self._configure_svc),
            Command("svc-csv", "Single train/test SVC from data_for_SVM.csv",
                    self._svc_csv, self._configure_svc_csv),
            Command("boxplot", "Expert vs novice box-plot grid for left/right instruments",
                    self._boxplot, self._configure_boxplot),
        )

    # ------------------------------------------------------------------
    # Feature export
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_features(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output-dir", type=Path, default=None)

    def _features(self, ctx: Context, args: argparse.Namespace) -> None:
        table = collect_feature_table(ctx.dataset)
        output = args.output_dir or (
            ctx.settings.artifacts_dir / "kinematic" / ctx.task.value / "features"
        )
        output.mkdir(parents=True, exist_ok=True)
        task = ctx.task.value

        np.save(output / f"{task}_feature.npy", table.features)
        np.save(output / f"{task}_grs.npy", table.grades.reshape(-1, 1))
        np.save(output / f"{task}_trials.npy", np.array(table.names))
        # The legacy CSV used the three-tier label for Needle_Passing only.
        csv_labels = table.tiers if task == "Needle_Passing" else table.grades
        _write_feature_csv(output / "data_for_SVM.csv", table.features, csv_labels)

        report = {
            "task": task,
            "n_trials": len(table.names),
            "feature_shape": list(table.features.shape),
            "outputs": [
                str(output / f"{task}_feature.npy"),
                str(output / f"{task}_grs.npy"),
                str(output / f"{task}_trials.npy"),
                str(output / "data_for_SVM.csv"),
            ],
        }
        (output / "features_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote features for %d trials to %s", len(table.names), output)

    # ------------------------------------------------------------------
    # SVC on feature tensors
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_svc(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--features-dir", type=Path, default=None)
        parser.add_argument("--kind", default="auto",
                            help="Split kind, or 'auto' (SkillDetection then gesture folds)")
        parser.add_argument("--gammas", type=float, nargs="+", default=list(_DEFAULT_GAMMAS))
        parser.add_argument("--cs", type=float, nargs="+", default=list(_DEFAULT_CS))
        parser.add_argument("--output", type=Path, default=None)

    def _svc(self, ctx: Context, args: argparse.Namespace) -> None:
        svc_cls = require("sklearn.svm", extra="ml")
        names, cube, labels = load_feature_table(ctx, args.features_dir)
        matrix = select_task_features(cube, ctx.task.value)
        kind, scheme = ctx.dataset.resolve_split_kind(args.kind)
        folds = ctx.dataset.folds(kind=kind, scheme=scheme)

        LOGGER.info(
            "SVC on %d trials, %d features, %d folds (%s/%s)",
            len(names),
            matrix.shape[1],
            len(folds),
            kind,
            scheme,
        )
        result: SearchResult = grid_search_loso(
            folds,
            names,
            matrix,
            labels,
            lambda gamma, C: svc_cls.SVC(gamma=gamma, C=C),
            {"gamma": args.gammas, "C": args.cs},
        )
        result.log()
        output = args.output or (
            ctx.settings.artifacts_dir / "kinematic" / ctx.task.value / "svc_report.json"
        )
        result.write(
            output,
            extra={
                "task": ctx.task.value,
                "method": "kinematic-svc",
                "n_trials": len(names),
                "n_features": int(matrix.shape[1]),
                "split_kind": kind,
                "split_scheme": scheme,
            },
        )
        LOGGER.info("Wrote SVC report: %s", output)

    # ------------------------------------------------------------------
    # CSV classifier (legacy data_for_SVM.csv workflow)
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_svc_csv(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--csv", type=Path, default=None)
        parser.add_argument("--test-size", type=float, default=0.5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--output", type=Path, default=None)

    def _svc_csv(self, ctx: Context, args: argparse.Namespace) -> None:
        sklearn_model_selection = require("sklearn.model_selection", extra="ml")
        sklearn_preprocessing = require("sklearn.preprocessing", extra="ml")
        sklearn_svm = require("sklearn.svm", extra="ml")

        path = args.csv or (_resolve_feature_dir(ctx, None) / "data_for_SVM.csv")
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"CSV not found: {path}. Run 'kinematic features' or pass --csv."
            )
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2 or rows[0][-1] != "class":
            raise DataError(f"'class' column missing from {path}")
        try:
            data = np.array([[float(value) for value in row] for row in rows[1:]])
        except ValueError as exc:
            raise DataError(f"Non-numeric value in {path}: {exc}") from exc

        features, labels = data[:, :-1], data[:, -1]
        train_x, test_x, train_y, test_y = sklearn_model_selection.train_test_split(
            features, labels, test_size=args.test_size, random_state=args.seed
        )

        scaler = sklearn_preprocessing.MinMaxScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)

        model = sklearn_svm.SVC(C=100).fit(train_scaled, train_y)
        train_accuracy = float(model.score(train_scaled, train_y))
        test_accuracy = float(model.score(test_scaled, test_y))
        LOGGER.info("Train accuracy: %.4f | Test accuracy: %.4f", train_accuracy, test_accuracy)

        output = args.output or (
            ctx.settings.artifacts_dir / "kinematic" / ctx.task.value / "svc_csv_report.json"
        )
        payload = {
            "task": ctx.task.value,
            "method": "kinematic-svc-csv",
            "csv": str(path),
            "n_samples": int(data.shape[0]),
            "test_size": args.test_size,
            "seed": args.seed,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote CSV SVC report: %s", output)

    # ------------------------------------------------------------------
    # Box plots
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_boxplot(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", type=Path, default=None)

    def _boxplot(self, ctx: Context, args: argparse.Namespace) -> None:
        table = collect_feature_table(ctx.dataset)
        expert = table.tiers == 2
        novice = table.tiers == 0
        if not expert.any() or not novice.any():
            raise DataError("Both expert and novice trials are required for the box plot")

        left_exp = table.features[expert, list(Instrument).index(Instrument.MTF_L), :]
        left_nov = table.features[novice, list(Instrument).index(Instrument.MTF_L), :]
        right_exp = table.features[expert, list(Instrument).index(Instrument.MTF_R), :]
        right_nov = table.features[novice, list(Instrument).index(Instrument.MTF_R), :]

        output = args.output or (
            ctx.settings.artifacts_dir / "figures" / ctx.task.value / "boxplot_mtf.png"
        )
        _plot_box_grid(
            left_exp, left_nov, right_exp, right_nov,
            title="MTF", output=output,
        )

        # PSM pair uses the same layout, as in the original report.
        psm1_exp = table.features[expert, list(Instrument).index(Instrument.PSM_1), :]
        psm1_nov = table.features[novice, list(Instrument).index(Instrument.PSM_1), :]
        psm2_exp = table.features[expert, list(Instrument).index(Instrument.PSM_2), :]
        psm2_nov = table.features[novice, list(Instrument).index(Instrument.PSM_2), :]
        psm_output = output.with_name("boxplot_psm.png")
        _plot_box_grid(
            psm1_exp, psm1_nov, psm2_exp, psm2_nov,
            title="PSM", output=psm_output,
        )
        LOGGER.info("Wrote box plots: %s, %s", output, psm_output)


def _write_feature_csv(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_HEADERS)
        for cube_row, label in zip(features, labels):
            mtf_l, mtf_r, psm_1, psm_2 = cube_row
            writer.writerow(
                [
                    str(mtf_l[0]),
                    *[str(value) for value in mtf_l[1:]],
                    *[str(value) for value in mtf_r[1:]],
                    *[str(value) for value in psm_1[1:]],
                    *[str(value) for value in psm_2[1:]],
                    str(int(label)),
                ]
            )


def _plot_box_grid(
    left_expert: np.ndarray,
    left_novice: np.ndarray,
    right_expert: np.ndarray,
    right_novice: np.ndarray,
    *,
    title: str,
    output: Path,
) -> Path:
    matplotlib = require("matplotlib", extra="plot")
    matplotlib.use("Agg")
    pyplot = require("matplotlib.pyplot", extra="plot")
    pyplot.rc("font", family="Times New Roman")

    fig, axes = pyplot.subplots(nrows=2, ncols=5, figsize=(12, 10))
    pyplot.subplots_adjust(hspace=0.5, wspace=0.5)

    for index, (axis, field, panel_title) in enumerate(
        zip(axes.ravel(), _PANEL_FIELDS, _PANEL_TITLES)
    ):
        if index == 0:
            series = [left_expert[:, index], left_novice[:, index]]
            labels = ["Exp", "Nov"]
        else:
            series = [
                left_expert[:, index],
                left_novice[:, index],
                right_expert[:, index],
                right_novice[:, index],
            ]
            labels = ["Exp", "Nov", "Exp", "Nov"]
        axis.boxplot(series, tick_labels=labels)
        axis.set_title(panel_title)
        if index != 0:
            axis.set_xlabel("L               R")
        axis.yaxis.grid(True)
        axis.yaxis.get_major_formatter().set_powerlimits((0, 2))
        axis.set_ylabel("values")

    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    pyplot.close(fig)
    return output


METHOD = KinematicMethod()
