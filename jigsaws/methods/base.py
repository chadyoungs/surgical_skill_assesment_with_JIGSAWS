"""Method plugin protocol and execution context."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Sequence

from ..config import Settings, Task
from ..dataset import Dataset


@dataclass(frozen=True, slots=True)
class Command:
    """One CLI sub-command exposed by a method."""

    name: str
    help: str
    run: Callable[["Context", argparse.Namespace], None]
    configure: Callable[[argparse.ArgumentParser], None] | None = None
    aliases: tuple[str, ...] = ()


class Method(ABC):
    """A pluggable pipeline; subclasses live in ``jigsaws/methods/<name>/``."""

    name: ClassVar[str]
    summary: ClassVar[str] = ""
    aliases: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def commands(self) -> Sequence[Command]:
        """Return the sub-commands this method exposes."""


@dataclass
class Context:
    """Everything a command needs: resolved settings plus the current task."""

    settings: Settings
    task: Task
    _dataset: Dataset | None = field(default=None, init=False, repr=False)

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self._dataset = Dataset(self.settings, self.task)
        return self._dataset

    def artifact_dir(self, method: str, *parts: str) -> Path:
        return self.settings.artifact_dir(method, self.task, *parts)
