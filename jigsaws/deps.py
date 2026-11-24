"""Lazy imports for optional heavy dependencies.

Every method imports its numerical stack inside the command that needs it,
so ``python main.py --help`` works on a bare Python installation and users
get an actionable message instead of a traceback.
"""

from __future__ import annotations

from importlib import import_module

from .errors import MissingDependencyError


def require(module: str, *, extra: str = "ml"):
    """Import *module* or raise :class:`MissingDependencyError`."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        if exc.name != module.split(".")[0]:
            raise
        raise MissingDependencyError(module, extra) from exc
