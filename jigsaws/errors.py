"""Exception types with user-facing messages."""

from __future__ import annotations


class JigsawsError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(JigsawsError):
    """Invalid configuration (bad path, unknown task, conflicting options)."""


class DataError(JigsawsError):
    """Dataset is missing or does not have the expected layout."""


class ArtifactNotFoundError(JigsawsError):
    """A required intermediate artifact has not been produced yet."""


class MissingDependencyError(JigsawsError):
    """An optional third-party package is not installed."""

    def __init__(self, module: str, extra: str) -> None:
        super().__init__(
            f"Missing optional dependency '{module}'. "
            f"Install it with: pip install -e '.[{extra}]'"
        )
        self.module = module
        self.extra = extra
