"""LOSO fold parsing from the official JIGSAWS ``Experimental_setup`` tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import DataError
from ..io import read_whitespace_rows


@dataclass(frozen=True, slots=True)
class Fold:
    """One leave-one-subject/super-trial-out fold, expressed in trial names."""

    name: str
    kind: str
    scheme: str
    train: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def all_trials(self) -> tuple[str, ...]:
        return self.train + self.test


def trial_from_clip_stem(stem: str) -> str:
    """``Suturing_B001_000080_005635`` -> ``Suturing_B001``."""
    parts = stem.split("_")
    return "_".join(parts[:-2]) if len(parts) > 2 else stem


def read_split_file(path: str | Path) -> tuple[str, ...]:
    """Return unique trial names in file order."""
    seen: dict[str, None] = {}
    for row in read_whitespace_rows(path):
        name = trial_from_clip_stem(row[0].split(".")[0])
        seen.setdefault(name, None)
    return tuple(seen)


def _natural_key(name: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for chunk in name.replace("-", "_").split("_"):
        parts.append((1, int(chunk)) if chunk.isdigit() else (0, chunk.lower()))
    return tuple(parts)


def load_folds(root: str | Path, *, kind: str, scheme: str, iteration: str = "itr_1") -> tuple[Fold, ...]:
    """Read ``<root>/<fold>/<iteration>/{Train,Test}.txt`` into :class:`Fold` objects."""
    directory = Path(root)
    if not directory.is_dir():
        raise DataError(f"Split directory not found: {directory}")

    folds: list[Fold] = []
    for fold_dir in sorted(directory.iterdir(), key=lambda p: _natural_key(p.name)):
        if not fold_dir.is_dir():
            continue
        train_file = fold_dir / iteration / "Train.txt"
        test_file = fold_dir / iteration / "Test.txt"
        if not (train_file.is_file() and test_file.is_file()):
            raise DataError(
                f"Fold '{fold_dir.name}' is missing {iteration}/Train.txt or Test.txt"
            )
        folds.append(
            Fold(
                name=fold_dir.name,
                kind=kind,
                scheme=scheme,
                train=read_split_file(train_file),
                test=read_split_file(test_file),
            )
        )
    if not folds:
        raise DataError(f"No folds found under {directory}")
    return tuple(folds)


def available_kinds(experimental_setup_root: Path, task: str) -> tuple[str, ...]:
    unbal = experimental_setup_root / task / "unBalanced"
    if not unbal.is_dir():
        return ()
    return tuple(sorted(p.name for p in unbal.iterdir() if p.is_dir()))


def available_schemes(experimental_setup_root: Path, task: str, kind: str) -> tuple[str, ...]:
    kind_dir = experimental_setup_root / task / "unBalanced" / kind
    if not kind_dir.is_dir():
        return ()
    return tuple(sorted(p.name for p in kind_dir.iterdir() if p.is_dir()))
