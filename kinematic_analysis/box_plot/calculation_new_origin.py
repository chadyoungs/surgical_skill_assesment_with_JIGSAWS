#!/usr/bin/env python3
"""
Kinematic feature calculation for JIGSAWS trial data.

Computes motion features (duration, displacement, velocity, curvature,
smoothness, turning angle) for each robotic arm from raw kinematic data.
"""
from __future__ import annotations

import math

import numpy as np


# Column offsets for each instrument in the 76-column kinematic file
_INSTRUMENT_COLS: dict[str, tuple[int, int, int]] = {
    "MTF_L":  (0,  1,  2),
    "MTF_R":  (19, 20, 21),
    "PSM_1":  (38, 39, 40),
    "PSM_2":  (57, 58, 59),
}

# Data acquisition frequency (Hz)
_FREQUENCY: float = 30.0
_TIMESTAMP: float = 1.0 / _FREQUENCY

# Threshold for detecting motion onset/offset
_MOTION_THRESHOLD: float = 1e-5


class DataCal:
    """Calculate trajectory-based motion features for one robotic instrument."""

    def getFile(self, x: np.ndarray, choose: str) -> None:  # noqa: N802 – kept for backward compat
        """Set the raw kinematic data and selected instrument.

        Args:
            x: Full kinematic data array of shape ``(n_frames, 76)``.
            choose: Instrument name, one of ``"MTF_L"``, ``"MTF_R"``,
                ``"PSM_1"``, or ``"PSM_2"``.
        """
        self.data = x
        self.description_choose = choose

    def mov_determine(self) -> None:
        """Extract displacement series and locate the motion start/stop frames."""
        cols = _INSTRUMENT_COLS[self.description_choose]
        des_x = self.data[:, cols[0]]
        des_y = self.data[:, cols[1]]
        des_z = self.data[:, cols[2]]

        n_steps = self.data.shape[0] - 1

        # Per-frame displacement components
        p_var_x = np.diff(des_x)
        p_var_y = np.diff(des_y)
        p_var_z = np.diff(des_z)
        p_var = np.sqrt(p_var_x**2 + p_var_y**2 + p_var_z**2)

        self.p_var_x = p_var_x.tolist()
        self.p_var_y = p_var_y.tolist()
        self.p_var_z = p_var_z.tolist()
        self.p_var = p_var.tolist()

        # Motion start: first frame where all three axes exceed the threshold
        self.move_start_moment = 0
        for i in range(n_steps):
            if (abs(p_var_x[i]) > _MOTION_THRESHOLD
                    and abs(p_var_y[i]) > _MOTION_THRESHOLD
                    and abs(p_var_z[i]) > _MOTION_THRESHOLD):
                self.move_start_moment = i + 1
                self.move_start_time = self.move_start_moment * _TIMESTAMP
                break

        # Motion stop: last frame where all three axes exceed the threshold
        self.move_stop_moment = 0
        for i in range(-1, -(n_steps + 1), -1):
            if (abs(p_var_x[i]) > _MOTION_THRESHOLD
                    and abs(p_var_y[i]) > _MOTION_THRESHOLD
                    and abs(p_var_z[i]) > _MOTION_THRESHOLD):
                self.move_stop_moment = (n_steps + 1) + i - 1
                self.move_stop_time = self.move_stop_moment * _TIMESTAMP
                break

        start, stop = self.move_start_moment, self.move_stop_moment
        self.move_data_x_sum = self.p_var_x[start:stop]
        self.move_data_y_sum = self.p_var_y[start:stop]
        self.move_data_z_sum = self.p_var_z[start:stop]
        self.move_data_sum = self.p_var[start:stop]

    def time_cal(self) -> None:
        """Compute total motion duration."""
        self.time_moments_sum: int = self.move_stop_moment - self.move_start_moment
        self.time_sum: float = self.time_moments_sum * _TIMESTAMP

    def p_displacement_sum_cal(self) -> None:
        """Compute total path length (sum of displacement magnitudes)."""
        self.p_sum: float = sum(self.move_data_sum)

    def vel_average_cal(self) -> None:
        """Compute velocity time series and summary statistics."""
        self.v_var_x = [x / _TIMESTAMP for x in self.move_data_x_sum]
        self.v_var_y = [y / _TIMESTAMP for y in self.move_data_y_sum]
        self.v_var_z = [z / _TIMESTAMP for z in self.move_data_z_sum]
        self.v_var = [s / _TIMESTAMP for s in self.move_data_sum]

        self.v_average: float = float(np.mean(self.v_var))
        self.v_variance: float = float(np.var(self.v_var))

    def acc_cal(self) -> None:
        """Compute acceleration time series."""
        self.acc_data_x = np.gradient(self.v_var_x, _TIMESTAMP)
        self.acc_data_y = np.gradient(self.v_var_y, _TIMESTAMP)
        self.acc_data_z = np.gradient(self.v_var_z, _TIMESTAMP)
        self.acc_data = [
            math.sqrt(self.acc_data_x[s]**2 + self.acc_data_y[s]**2 + self.acc_data_z[s]**2)
            for s in range(self.time_moments_sum)
        ]

    def curvity_cal(self) -> None:
        """Compute curvature time series and summary statistics."""
        curvity: list[float] = []
        for i in range(self.time_moments_sum):
            v_vec = np.array([self.v_var_x[i], self.v_var_y[i], self.v_var_z[i]])
            a_vec = np.array([self.acc_data_x[i], self.acc_data_y[i], self.acc_data_z[i]])
            up_normal = np.linalg.norm(np.cross(v_vec, a_vec))
            down = np.linalg.norm(v_vec) ** 3
            curvity.append(0.0 if down == 0 else float(up_normal / down))

        self.curvity_average: float = float(np.mean(curvity))
        self.curvity_variance: float = float(np.var(curvity))

    def smoothness_cal(self) -> None:
        """Compute smoothness (jerk) time series and summary statistics."""
        smoothness_x = np.gradient(self.acc_data_x, _TIMESTAMP)
        smoothness_y = np.gradient(self.acc_data_y, _TIMESTAMP)
        smoothness_z = np.gradient(self.acc_data_z, _TIMESTAMP)
        self.smoothness = [
            math.sqrt(smoothness_x[s]**2 + smoothness_y[s]**2 + smoothness_z[s]**2)
            for s in range(self.time_moments_sum)
        ]
        self.smoothness_average: float = float(np.mean(self.smoothness))
        self.smoothness_variance: float = float(np.var(self.smoothness))

    def Turning_angle_cal(self) -> None:  # noqa: N802 – kept for backward compat
        """Compute turning-angle time series and summary statistics."""
        u_vectors = [
            (self.move_data_x_sum[i], self.move_data_y_sum[i], self.move_data_z_sum[i])
            for i in range(self.time_moments_sum)
        ]
        turning_angles: list[float] = []
        for i in range(self.time_moments_sum - 1):
            u_i = np.array(u_vectors[i])
            u_j = np.array(u_vectors[i + 1])
            denom = np.linalg.norm(u_i) * np.linalg.norm(u_j)
            cos_val = 1.0 if denom == 0 else float(np.dot(u_i, u_j) / denom)
            cos_val = max(-1.0, min(1.0, cos_val))
            turning_angles.append(math.acos(cos_val))

        self.Turning_angle_average: float = float(np.mean(turning_angles))
        self.Turning_angle_variance: float = float(np.var(turning_angles))

    def cal_processing(self) -> tuple[float, ...]:
        """Run the full feature-extraction pipeline and return a feature tuple.

        Returns:
            A 10-tuple:
            ``(time_sum, p_sum, v_average, v_variance,
               curvity_average, curvity_variance,
               smoothness_average, smoothness_variance,
               Turning_angle_average, Turning_angle_variance)``
        """
        self.mov_determine()
        self.time_cal()
        self.p_displacement_sum_cal()
        self.vel_average_cal()
        self.acc_cal()
        self.curvity_cal()
        self.smoothness_cal()
        self.Turning_angle_cal()

        return (
            self.time_sum,
            self.p_sum,
            self.v_average,
            self.v_variance,
            self.curvity_average,
            self.curvity_variance,
            self.smoothness_average,
            self.smoothness_variance,
            self.Turning_angle_average,
            self.Turning_angle_variance,
        )
