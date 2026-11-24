"""Automatic discovery of method plugins.

A new method is added by creating ``jigsaws/methods/<name>/method.py`` with a
``METHOD`` object. Nothing else in the codebase needs editing: the CLI, help
text and ``python main.py methods`` pick it up automatically.
"""

from __future__ import annotations

import importlib
import pkgutil

from ..errors import JigsawsError
from .base import Command, Context, Method


def discover_methods() -> dict[str, Method]:
    registry: dict[str, Method] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.ispkg:
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}.method")
        method = getattr(module, "METHOD", None)
        if method is None:
            raise JigsawsError(
                f"Method package '{module_info.name}' does not export METHOD"
            )
        if method.name in registry:
            raise JigsawsError(f"Duplicate method name '{method.name}'")
        registry[method.name] = method
    return dict(sorted(registry.items()))


def get_method(name: str) -> Method:
    methods = discover_methods()
    for key, method in methods.items():
        if name == key or name in method.aliases:
            return method
    raise JigsawsError(
        "Unknown method '{}'. Available: {}".format(name, ", ".join(methods))
    )


__all__ = ["Command", "Context", "Method", "discover_methods", "get_method"]
