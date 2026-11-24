"""Single source of truth for paths, tasks and runtime options.

All paths are resolved once in :class:`Settings` and passed down explicitly;
no module reads the working directory or mutates ``os.environ`` at import
time (the previous implementation did both, which made every script
position-dependent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .errors import ConfigError

TASKS: tuple[str, ...] = ("Suturing", "Knot_Tying", "Needle_Passing")
DEFAULT_TASK = "Suturing"


class Task(StrEnum):
    """JIGSAWS task, with the skill-grading thresholds used by the dataset."""

    SUTURING = "Suturing"
    KNOT_TYING = "Knot_Tying"
    NEEDLE_PASSING = "Needle_Passing"

    @classmethod
    def parse(cls, value: str | Task | None) -> Task:
        if isinstance(value, Task):
            return value
        if value is None:
            return cls(DEFAULT_TASK)
        try:
            return cls(value)
        except ValueError as exc:
            raise ConfigError(
                f"Unknown task '{value}'. Expected one of: {', '.join(TASKS)}"
            ) from exc

    @property
    def grade_threshold(self) -> int:
        """Score above this value is labelled expert (label ``2``)."""
        return 19 if self is Task.SUTURING else 15

    @property
    def tier_thresholds(self) -> tuple[int, int]:
        """``(low, high)`` cut-offs for the legacy three-tier skill label.

        ``score <= low`` -> novice (0), ``low < score < high`` -> intermediate
        (1), ``score >= high`` -> expert (2).
        """
        return (15, 20) if self is Task.NEEDLE_PASSING else (15, 19)

    def grade(self, score: int) -> int:
        """Binary skill grade: ``0`` novice, ``2`` expert."""
        return 0 if score <= self.grade_threshold else 2

    def tier(self, score: int) -> int:
        """Three-tier skill label retained for the box-plot comparison."""
        low, high = self.tier_thresholds
        if score <= low:
            return 0
        if score < high:
            return 1
        return 2


class Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_DATA_ROOT_CANDIDATES: tuple[Path, ...] = (
    Path("da_vici_data_with_iDT_features"),
    Path("kinematic_analysis") / "trajectory_plot" / "da_vici_data_with_iDT_features",
)


def _as_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_data_root(repo_root: Path, explicit: str | Path | None) -> Path:
    raw = explicit if explicit is not None else os.environ.get("JIGSAWS_DATA_ROOT")
    if raw:
        path = _as_path(raw, repo_root)
        if not path.is_dir():
            raise ConfigError(f"Dataset root does not exist: {path}")
        return path
    found = _first_existing(tuple(repo_root / item for item in _DATA_ROOT_CANDIDATES))
    return found if found is not None else repo_root / "da_vici_data_with_iDT_features"


def _resolve_experimental_setup(
    data_root: Path, explicit: str | Path | None, repo_root: Path
) -> Path:
    raw = (
        explicit
        if explicit is not None
        else os.environ.get("JIGSAWS_EXPERIMENTAL_SETUP")
    )
    if raw:
        path = _as_path(raw, repo_root)
        if not path.is_dir():
            raise ConfigError(f"Experimental_setup root does not exist: {path}")
        return path
    found = _first_existing(
        (data_root / "Experimental_setup", data_root.parent / "Experimental_setup")
    )
    return found if found is not None else data_root / "Experimental_setup"


def _resolve_optional_path(
    repo_root: Path,
    explicit: str | Path | None,
    env_var: str,
    default: Path,
) -> Path:
    raw = explicit if explicit is not None else os.environ.get(env_var)
    return _as_path(raw, repo_root) if raw else (repo_root / default).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, fully resolved runtime configuration."""

    repo_root: Path
    data_root: Path
    experimental_setup_root: Path
    artifacts_dir: Path
    stip_features_dir: Path
    kinematic_features_dir: Path
    trajectory_hand: Hand = Hand.LEFT

    @classmethod
    def resolve(
        cls,
        *,
        repo_root: str | Path | None = None,
        data_root: str | Path | None = None,
        experimental_setup_root: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        stip_features_dir: str | Path | None = None,
        kinematic_features_dir: str | Path | None = None,
        trajectory_hand: str | Hand | None = None,
    ) -> Settings:
        root = (
            _as_path(repo_root, Path.cwd()) if repo_root is not None else default_repo_root()
        )
        data = _resolve_data_root(root, data_root)
        setup = _resolve_experimental_setup(data, experimental_setup_root, root)
        artifacts = _resolve_optional_path(
            root,
            artifacts_dir,
            "JIGSAWS_ARTIFACTS",
            Path("artifacts"),
        )
        stip = _resolve_optional_path(
            root,
            stip_features_dir,
            "JIGSAWS_STIP_FEATURES",
            Path("stip+dct_code") / "raw_data",
        )
        features = _resolve_optional_path(
            root,
            kinematic_features_dir,
            "JIGSAWS_KINEMATIC_FEATURES",
            Path("kinematic_analysis") / "Machine_learning_basedonfeatures",
        )
        hand_value = (
            trajectory_hand
            if trajectory_hand is not None
            else os.environ.get("JIGSAWS_TRAJECTORY_HAND", Hand.LEFT.value)
        )
        try:
            hand = hand_value if isinstance(hand_value, Hand) else Hand(hand_value.lower())
        except ValueError as exc:
            raise ConfigError(
                f"Unsupported trajectory hand '{hand_value}'. Expected 'left' or 'right'."
            ) from exc
        return cls(
            repo_root=root,
            data_root=data,
            experimental_setup_root=setup,
            artifacts_dir=artifacts,
            stip_features_dir=stip,
            kinematic_features_dir=features,
            trajectory_hand=hand,
        )

    # ------------------------------------------------------------------
    # Dataset layout
    # ------------------------------------------------------------------

    def task_dir(self, task: Task) -> Path:
        return self.data_root / task.value

    def meta_file(self, task: Task) -> Path:
        return self.task_dir(task) / f"meta_file_{task.value}.txt"

    def transcriptions_dir(self, task: Task) -> Path:
        return self.task_dir(task) / "transcriptions"

    def kinematics_dir(self, task: Task) -> Path:
        return self.task_dir(task) / "kinematics" / "AllGestures"

    def videos_dir(self, task: Task) -> Path:
        return self.task_dir(task) / "video"

    def surgeme_video_dir(self, task: Task) -> Path:
        return self.task_dir(task) / "surgeme_video"

    def surgeme_image_dir(self, task: Task) -> Path:
        return self.task_dir(task) / "surgeme_img"

    def split_root(self, task: Task, kind: str, scheme: str) -> Path:
        return self.experimental_setup_root / task.value / "unBalanced" / kind / scheme

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def artifact_dir(self, *parts: str | Task) -> Path:
        path = self.artifacts_dir.joinpath(*(str(part) for part in parts))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def first_existing(*candidates: Path) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
