"""UCF-101-style split files and annotation JSON for gesture clips.

Both writers iterate the official gesture-classification folds, resolve each
split row back to its surgeme segment, and emit labels using the dataset-wide
label index. This is the shared implementation behind
``python main.py dataset splits|annotations``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..errors import DataError
from ..io import read_whitespace_rows, write_lines
from .metadata import Dataset, surgeme_sort_key
from .splits import Fold

_OPTIONS = ("Train", "Test")
_SUBSET = {"Train": "training", "Test": "validation"}


@dataclass(frozen=True, slots=True)
class ClipAnnotation:
    trial: str
    start: int
    end: int
    index: int
    label: str
    subset: str

    @property
    def clip_name(self) -> str:
        return f"{self.trial}_{self.label}_{self.index}.avi"

    @property
    def relative_path(self) -> str:
        return f"{self.label}/{self.clip_name}"

    @property
    def duration(self) -> int:
        return self.end - self.start


def read_clip_annotations(dataset: Dataset, fold: Fold, option: str) -> tuple[ClipAnnotation, ...]:
    """Resolve every clip listed in ``<fold>/itr_1/<option>.txt``."""
    path = (
        dataset.settings.split_root(dataset.task, fold.kind, fold.scheme)
        / fold.name
        / "itr_1"
        / f"{option}.txt"
    )
    if not path.is_file():
        raise DataError(f"Split file not found: {path}")

    annotations: list[ClipAnnotation] = []
    for row in read_whitespace_rows(path):
        stem = row[0].split(".")[0]
        parts = stem.split("_")
        if len(parts) < 3:
            raise DataError(f"Malformed clip identifier in {path}: {row[0]}")
        trial = "_".join(parts[:-2])
        try:
            start, end = int(parts[-2]), int(parts[-1])
        except ValueError as exc:
            raise DataError(f"Malformed frame range in {path}: {row[0]}") from exc
        record = dataset[trial]
        for index, segment in enumerate(record.segments):
            if segment.start == start:
                annotations.append(
                    ClipAnnotation(
                        trial=trial,
                        start=start,
                        end=end,
                        index=index,
                        label=segment.label,
                        subset=_SUBSET[option],
                    )
                )
                break
        else:
            raise DataError(
                f"No surgeme starting at frame {start} for {trial} "
                f"(referenced by {path.name})"
            )
    return tuple(annotations)


def write_split_files(
    dataset: Dataset,
    *,
    kind: str = "GestureClassification",
    scheme: str = "SuperTrialOut",
    output_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Write ``<scheme>_<fold>_{Train,Test}.txt`` UCF-style split files."""
    settings: Settings = dataset.settings
    target = Path(output_dir) if output_dir else settings.task_dir(dataset.task) / "surgeme_classifi_traintestsplit"
    target.mkdir(parents=True, exist_ok=True)
    label_index = dataset.label_index

    written: list[Path] = []
    for fold in dataset.folds(kind=kind, scheme=scheme):
        for option in _OPTIONS:
            lines = [
                f"{item.relative_path} {item.duration} {label_index[item.label]}"
                for item in read_clip_annotations(dataset, fold, option)
            ]
            written.append(
                write_lines(target / f"{scheme}_{fold.name}_{option}.txt", lines)
            )
    return tuple(written)


def write_annotations(
    dataset: Dataset,
    *,
    kind: str = "GestureClassification",
    scheme: str = "SuperTrialOut",
    output_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Write one PyTorchCon3D-compatible annotation JSON per fold."""
    target = (
        Path(output_dir)
        if output_dir
        else dataset.settings.task_dir(dataset.task) / "surgeme_annotation"
    )
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fold in dataset.folds(kind=kind, scheme=scheme):
        database: dict[str, dict] = {}
        labels: set[str] = set()
        for option in _OPTIONS:
            for item in read_clip_annotations(dataset, fold, option):
                database[item.clip_name] = {
                    "subset": item.subset,
                    "annotations": {"label": item.label},
                }
                labels.add(item.label)

        payload = {
            "database": database,
            "labels": sorted(labels, key=surgeme_sort_key),
        }
        path = target / f"{scheme}_{fold.name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return tuple(written)


def summarize(dataset: Dataset) -> dict[str, object]:
    """Dataset-level facts used by ``dataset info``."""
    return {
        "task": dataset.task.value,
        "data_root": str(dataset.settings.data_root),
        "metadata_file": str(dataset.settings.meta_file(dataset.task)),
        "trials_in_metadata": len(dataset),
        "trials_with_kinematics": len(dataset.kinematics_trials),
        "novices": sum(1 for record in dataset.records if record.grade == 0),
        "experts": sum(1 for record in dataset.records if record.grade == 2),
        "surgeme_labels": list(dataset.surgeme_labels),
        "split_kinds": list(dataset.split_kinds()),
        "score_range": (
            min(record.score for record in dataset.records),
            max(record.score for record in dataset.records),
        ),
    }
