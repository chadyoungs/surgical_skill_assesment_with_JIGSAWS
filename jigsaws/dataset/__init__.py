"""Shared dataset layer."""

from __future__ import annotations

from .annotations import (
    ClipAnnotation,
    read_clip_annotations,
    summarize,
    write_annotations,
    write_split_files,
)
from .metadata import Dataset, SurgemeSegment, TrialRecord, surgeme_sort_key
from .splits import Fold, available_kinds, available_schemes, read_split_file, trial_from_clip_stem

__all__ = [
    "ClipAnnotation",
    "Dataset",
    "Fold",
    "SurgemeSegment",
    "TrialRecord",
    "available_kinds",
    "available_schemes",
    "read_clip_annotations",
    "read_split_file",
    "summarize",
    "surgeme_sort_key",
    "trial_from_clip_stem",
    "write_annotations",
    "write_split_files",
]
