"""Dataset access: scores, skill labels, surgeme transcripts and folds.

This replaces four near-identical ``DataExtract``/``MetaData`` classes. Every
method now consumes the same immutable :class:`Dataset` object, so new
methods never re-implement parsing, grading or split handling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..config import Settings, Task
from ..errors import DataError
from ..io import read_whitespace_rows
from .splits import Fold, available_kinds, available_schemes, load_folds

LOGGER = logging.getLogger(__name__)

_SURGEME_SUFFIX = re.compile(r"\d+$")


def surgeme_sort_key(label: str) -> tuple[int, str]:
    match = _SURGEME_SUFFIX.search(label)
    return (int(match.group()) if match else -1, label)


@dataclass(frozen=True, slots=True)
class SurgemeSegment:
    """Frame interval (1-based, inclusive) annotated with a surgeme label."""

    start: int
    end: int
    label: str

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One trial: GRS score plus its surgeme timeline."""

    name: str
    score: int
    grade: int
    tier: int
    segments: tuple[SurgemeSegment, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(segment.label for segment in self.segments)

    @property
    def frame_range(self) -> tuple[int, int]:
        if not self.segments:
            return (0, 0)
        return (self.segments[0].start, self.segments[-1].end)


class Dataset:
    """Read-only view of one JIGSAWS task."""

    def __init__(self, settings: Settings, task: Task) -> None:
        self.settings = settings
        self.task = task
        self._records = self._load_records()
        self._by_name = {record.name: record for record in self._records}
        self._trials = tuple(record.name for record in self._records)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_records(self) -> tuple[TrialRecord, ...]:
        meta_file = self.settings.meta_file(self.task)
        transcriptions = self.settings.transcriptions_dir(self.task)
        if not meta_file.is_file():
            raise DataError(
                f"Metadata file not found: {meta_file}\n"
                "Point --data-root at the directory that contains "
                f"'{self.task.value}/meta_file_{self.task.value}.txt'."
            )
        if not transcriptions.is_dir():
            raise DataError(f"Transcription directory not found: {transcriptions}")

        scores = self._read_scores(meta_file)
        segments = {
            path.stem: self._read_transcription(path)
            for path in sorted(transcriptions.glob("*.txt"))
        }
        if not segments:
            raise DataError(f"No transcription files found in {transcriptions}")

        missing = sorted(set(scores) - set(segments))
        if missing:
            raise DataError(
                "Missing transcription files for scored trials: "
                + ", ".join(missing[:5])
                + (" ..." if len(missing) > 5 else "")
            )

        records = tuple(
            TrialRecord(
                name=name,
                score=score,
                grade=self.task.grade(score),
                tier=self.task.tier(score),
                segments=segments[name],
            )
            for name, score in scores.items()
        )
        LOGGER.debug("Loaded %d trials for %s", len(records), self.task.value)
        return records

    @staticmethod
    def _read_scores(path: Path) -> dict[str, int]:
        scores: dict[str, int] = {}
        for row in read_whitespace_rows(path):
            if len(row) < 3:
                raise DataError(f"Malformed metadata row in {path}: {' '.join(row)}")
            try:
                scores[row[0]] = int(row[2])
            except ValueError as exc:
                raise DataError(f"Non-numeric GRS score in {path}: {' '.join(row)}") from exc
        if not scores:
            raise DataError(f"No scores parsed from {path}")
        return scores

    @staticmethod
    def _read_transcription(path: Path) -> tuple[SurgemeSegment, ...]:
        segments: list[SurgemeSegment] = []
        for row in read_whitespace_rows(path):
            if len(row) < 3:
                raise DataError(f"Malformed transcription row in {path}: {' '.join(row)}")
            try:
                start, end = int(row[0]), int(row[1])
            except ValueError as exc:
                raise DataError(f"Bad frame index in {path}: {' '.join(row)}") from exc
            if end < start:
                raise DataError(f"Segment ends before it starts in {path}: {' '.join(row)}")
            segments.append(SurgemeSegment(start=start, end=end, label=row[2]))
        if not segments:
            raise DataError(f"Empty transcription file: {path}")
        return tuple(segments)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @cached_property
    def trial_names(self) -> tuple[str, ...]:
        return self._trials

    @cached_property
    def records(self) -> tuple[TrialRecord, ...]:
        return self._records

    def __getitem__(self, trial: str) -> TrialRecord:
        try:
            return self._by_name[trial]
        except KeyError as exc:
            raise DataError(f"Unknown trial '{trial}' for task {self.task.value}") from exc

    def __contains__(self, trial: object) -> bool:
        return trial in self._by_name

    def __len__(self) -> int:
        return len(self._records)

    @cached_property
    def grades(self) -> dict[str, int]:
        return {record.name: record.grade for record in self._records}

    def grade(self, trial: str) -> int:
        return self[trial].grade

    @cached_property
    def surgeme_labels(self) -> tuple[str, ...]:
        labels = {segment.label for record in self._records for segment in record.segments}
        return tuple(sorted(labels, key=surgeme_sort_key))

    @cached_property
    def label_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.surgeme_labels)}

    def ranked_trials(self, *, descending: bool = True) -> tuple[str, ...]:
        """Trials ordered by GRS score (ties keep metadata order)."""
        return tuple(
            record.name
            for record in sorted(
                self._records, key=lambda record: record.score, reverse=descending
            )
        )

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def kinematics_path(self, trial: str) -> Path:
        return self.settings.kinematics_dir(self.task) / f"{trial}.txt"

    def has_kinematics(self, trial: str) -> bool:
        return self.kinematics_path(trial).is_file()

    def load_kinematics(self, trial: str) -> np.ndarray:
        path = self.kinematics_path(trial)
        if not path.is_file():
            raise DataError(f"Kinematic file not found: {path}")
        return np.loadtxt(path, dtype=np.float64)

    @cached_property
    def kinematics_trials(self) -> tuple[str, ...]:
        return tuple(
            path.stem
            for path in sorted(self.settings.kinematics_dir(self.task).glob("*.txt"))
        )

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def split_kinds(self) -> tuple[str, ...]:
        return available_kinds(self.settings.experimental_setup_root, self.task.value)

    def split_schemes(self, kind: str) -> tuple[str, ...]:
        return available_schemes(self.settings.experimental_setup_root, self.task.value, kind)

    def folds(
        self,
        kind: str = "SkillDetection",
        scheme: str = "SuperTrialOut",
        *,
        iteration: str = "itr_1",
    ) -> tuple[Fold, ...]:
        root = self.settings.split_root(self.task, kind, scheme)
        if not root.is_dir():
            kinds = self.split_kinds()
            raise DataError(
                f"Splits not available for task {self.task.value}: {root}\n"
                f"Available split kinds: {', '.join(kinds) if kinds else 'none'}"
            )
        folds = load_folds(root, kind=kind, scheme=scheme, iteration=iteration)
        cleaned: list[Fold] = []
        for fold in folds:
            unknown = [name for name in fold.all_trials if name not in self]
            if unknown:
                LOGGER.warning(
                    "Fold %s references %d trial(s) absent from metadata; ignoring them",
                    fold.name,
                    len(unknown),
                )
            cleaned.append(
                Fold(
                    name=fold.name,
                    kind=kind,
                    scheme=scheme,
                    train=tuple(name for name in fold.train if name in self),
                    test=tuple(name for name in fold.test if name in self),
                )
            )
        return tuple(cleaned)

    def resolve_split_kind(self, requested: str = "auto") -> tuple[str, str]:
        """Return ``(kind, scheme)``, falling back to gesture folds for skill CV.

        Some tasks in the official release only ship ``GestureClassification``
        folds. Those folds partition trials the same way, so they are a valid
        trial-level LOSO fallback for skill assessment.
        """
        kinds = self.split_kinds()
        if requested != "auto":
            if requested not in kinds:
                raise DataError(
                    f"Split kind '{requested}' not available for {self.task.value}. "
                    f"Choose from: {', '.join(kinds) if kinds else 'none'}"
                )
            return requested, "SuperTrialOut"
        if "SkillDetection" in kinds:
            return "SkillDetection", "SuperTrialOut"
        if "GestureClassification" in kinds:
            LOGGER.warning(
                "SkillDetection splits missing for %s; using GestureClassification "
                "folds for trial-level LOSO",
                self.task.value,
            )
            return "GestureClassification", "SuperTrialOut"
        raise DataError(
            f"No usable LOSO splits for {self.task.value}. "
            f"Available: {', '.join(kinds) if kinds else 'none'}"
        )
