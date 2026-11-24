"""Shared evaluation primitives."""

from __future__ import annotations

from .cross_validation import (
    FoldScore,
    SearchResult,
    grid_search_loso,
    grid_search_loso_per_fold,
)
from .metrics import BinaryMetrics

__all__ = [
    "BinaryMetrics",
    "FoldScore",
    "SearchResult",
    "grid_search_loso",
    "grid_search_loso_per_fold",
]
