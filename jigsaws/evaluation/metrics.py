"""Skill-assessment metrics shared by every method."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Accuracy plus per-class recall and the 2x2 confusion matrix.

    Class order is ``(novice=0, expert=2)``; the matrix is
    ``[[true novice, novice->expert], [expert->novice, true expert]]``.
    """

    accuracy: float
    expert_accuracy: float
    novice_accuracy: float
    n_experts: int
    n_novices: int
    true_novice: int
    true_expert: int

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
        truth = np.asarray(y_true).reshape(-1)
        pred = np.asarray(y_pred).reshape(-1)
        if truth.shape != pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        if truth.size == 0:
            return cls(math.nan, math.nan, math.nan, 0, 0, 0, 0)

        unknown = set(np.unique(truth)) - {0, 2}
        if unknown:
            raise ValueError(f"Unsupported skill labels: {sorted(unknown)}")

        is_expert_true = truth == 2
        is_expert_pred = pred == 2
        n_experts = int(is_expert_true.sum())
        n_novices = int((~is_expert_true).sum())
        true_expert = int((is_expert_true & is_expert_pred).sum())
        true_novice = int((~is_expert_true & ~is_expert_pred).sum())
        correct = true_expert + true_novice
        return cls(
            accuracy=correct / truth.size,
            expert_accuracy=true_expert / n_experts if n_experts else math.nan,
            novice_accuracy=true_novice / n_novices if n_novices else math.nan,
            n_experts=n_experts,
            n_novices=n_novices,
            true_novice=true_novice,
            true_expert=true_expert,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "expert_accuracy": self.expert_accuracy,
            "novice_accuracy": self.novice_accuracy,
            "n_experts": self.n_experts,
            "n_novices": self.n_novices,
            "true_novice": self.true_novice,
            "true_expert": self.true_expert,
        }

    def format(self) -> str:
        return (
            f"accuracy={self.accuracy:.4f} "
            f"expert={self.expert_accuracy:.4f} "
            f"novice={self.novice_accuracy:.4f}"
        )
