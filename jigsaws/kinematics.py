"""Trajectory feature extraction shared by the kinematic pipelines.

The ten features per instrument are unchanged from the original
``calculation_new.py`` implementation (duration, path length, velocity,
curvature, smoothness, turning angle), but computed with NumPy arrays instead
of per-frame Python loops.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

FREQUENCY_HZ: float = 30.0
TIMESTAMP: float = 1.0 / FREQUENCY_HZ
MOTION_THRESHOLD: float = 1e-5


class Instrument(StrEnum):
    MTF_L = "MTF_L"
    MTF_R = "MTF_R"
    PSM_1 = "PSM_1"
    PSM_2 = "PSM_2"

    @property
    def position_columns(self) -> tuple[int, int, int]:
        return INSTRUMENT_COLUMNS[self]


INSTRUMENT_COLUMNS: dict[Instrument, tuple[int, int, int]] = {
    Instrument.MTF_L: (0, 1, 2),
    Instrument.MTF_R: (19, 20, 21),
    Instrument.PSM_1: (38, 39, 40),
    Instrument.PSM_2: (57, 58, 59),
}

FEATURE_NAMES: tuple[str, ...] = (
    "time_sum",
    "displacement_sum",
    "velocity_mean",
    "velocity_variance",
    "curvature_mean",
    "curvature_variance",
    "smoothness_mean",
    "smoothness_variance",
    "turning_angle_mean",
    "turning_angle_variance",
)


@dataclass(frozen=True, slots=True)
class InstrumentFeatures:
    time_sum: float
    displacement_sum: float
    velocity_mean: float
    velocity_variance: float
    curvature_mean: float
    curvature_variance: float
    smoothness_mean: float
    smoothness_variance: float
    turning_angle_mean: float
    turning_angle_variance: float

    def as_tuple(self) -> tuple[float, ...]:
        return astuple(self)

    def as_dict(self) -> dict[str, float]:
        return {name: value for name, value in zip(FEATURE_NAMES, astuple(self))}


def load_kinematics(path: str | Path) -> np.ndarray:
    """Load a whitespace-delimited 76-column kinematics file."""
    return np.loadtxt(path, dtype=np.float64)


def compute_features(
    kinematics: np.ndarray,
    instrument: Instrument,
    *,
    frequency: float = FREQUENCY_HZ,
    motion_threshold: float = MOTION_THRESHOLD,
) -> InstrumentFeatures:
    """Compute the ten trajectory features for one instrument."""
    matrix = np.asarray(kinematics, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Kinematics must be a 2-D array with at least two frames")

    dt = 1.0 / frequency
    positions = matrix[:, INSTRUMENT_COLUMNS[instrument]]
    deltas = np.diff(positions, axis=0)
    step_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    active = np.all(np.abs(deltas) > motion_threshold, axis=1)

    if active.any():
        start = int(np.argmax(active)) + 1
        stop = int(len(deltas) - 1 - np.argmax(active[::-1]))
    else:
        start = stop = 0

    moments = stop - start
    active_deltas = deltas[start:stop]
    active_lengths = step_lengths[start:stop]
    time_sum = moments * dt
    displacement_sum = float(np.sum(active_lengths))

    velocities = active_deltas / dt
    speeds = active_lengths / dt
    velocity_mean = float(np.mean(speeds)) if speeds.size else float("nan")
    velocity_variance = float(np.var(speeds)) if speeds.size else float("nan")

    accelerations = (
        np.gradient(velocities, dt, axis=0)
        if moments > 1
        else np.zeros_like(velocities)
    )
    acceleration_magnitude = np.sqrt(np.sum(accelerations**2, axis=1))

    if moments:
        speed_norm = np.sqrt(np.sum(velocities**2, axis=1))
        cross = np.cross(velocities, accelerations)
        numerator = np.sqrt(np.sum(cross**2, axis=1))
        denominator = speed_norm**3
        curvature = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        curvature_mean = float(np.mean(curvature))
        curvature_variance = float(np.var(curvature))
    else:
        curvature_mean = curvature_variance = float("nan")

    if moments > 1:
        jerk = np.gradient(accelerations, dt, axis=0)
        smoothness = np.sqrt(np.sum(jerk**2, axis=1))
        smoothness_mean = float(np.mean(smoothness))
        smoothness_variance = float(np.var(smoothness))
    else:
        smoothness_mean = smoothness_variance = float("nan")

    if moments >= 2:
        direction = active_deltas
        norms = np.sqrt(np.sum(direction**2, axis=1))
        denominator = norms[:-1] * norms[1:]
        dot = np.sum(direction[:-1] * direction[1:], axis=1)
        cos = np.divide(
            dot,
            denominator,
            out=np.ones_like(dot),
            where=denominator != 0,
        )
        angles = np.arccos(np.clip(cos, -1.0, 1.0))
        turning_mean = float(np.mean(angles))
        turning_variance = float(np.var(angles))
    else:
        turning_mean = turning_variance = float("nan")

    return InstrumentFeatures(
        time_sum=time_sum,
        displacement_sum=displacement_sum,
        velocity_mean=velocity_mean,
        velocity_variance=velocity_variance,
        curvature_mean=curvature_mean,
        curvature_variance=curvature_variance,
        smoothness_mean=smoothness_mean,
        smoothness_variance=smoothness_variance,
        turning_angle_mean=turning_mean,
        turning_angle_variance=turning_variance,
    )


def compute_all(
    kinematics: np.ndarray, **kwargs: float
) -> dict[Instrument, InstrumentFeatures]:
    """Compute features for all four instruments in one call."""
    return {
        instrument: compute_features(kinematics, instrument, **kwargs)
        for instrument in Instrument
    }
