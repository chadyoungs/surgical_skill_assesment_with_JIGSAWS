#!/usr/bin/env python3
"""
Aggregate trajectory features and produce comparative box-plot visualisations.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="Times New Roman")


class TotalData:
    """Aggregate per-trial kinematic features for one robotic instrument."""

    def getFile(self, x: np.ndarray, choose: str) -> None:  # noqa: N802 – kept for backward compat
        """Select the instrument slice from the feature array.

        Args:
            x: Feature array of shape ``(n_trials, 4, 10)`` where the second
               axis indexes instruments in order MTF_L, MTF_R, PSM_1, PSM_2.
            choose: Instrument name: ``"MTF_L"``, ``"MTF_R"``,
                ``"PSM_1"``, or ``"PSM_2"``.
        """
        _instrument_index: dict[str, int] = {
            "MTF_L": 0, "MTF_R": 1, "PSM_1": 2, "PSM_2": 3,
        }
        self.description_choose = choose
        self.feature = x[:, _instrument_index[choose], :]

    def total_analysis(self) -> None:
        """Unpack the ten feature columns into named attributes."""
        self.total_time_sum = self.feature[:, 0]
        self.total_displacement_sum = self.feature[:, 1]
        self.total_v_average = self.feature[:, 2]
        self.total_v_variance = self.feature[:, 3]
        self.total_curvity_average = self.feature[:, 4]
        self.total_curvity_variance = self.feature[:, 5]
        self.total_smoothness_average = self.feature[:, 6]
        self.total_smoothness_variance = self.feature[:, 7]
        self.total_Turning_angle_average = self.feature[:, 8]
        self.total_Turning_angle_variance = self.feature[:, 9]

    def visual_comparison(
        self,
        other_data: "TotalData",
        data: "TotalData",
        data_other_data: "TotalData",
    ) -> None:
        """Plot a 2×5 grid of box plots comparing expert and novice groups.

        Args:
            other_data: Expert data for the same instrument as ``self`` (left arm).
            data: Expert data for the paired instrument (right arm).
            data_other_data: Novice data for the paired instrument (right arm).
        """
        plot_data = [
            [other_data.total_time_sum, self.total_time_sum],
            [other_data.total_displacement_sum, self.total_displacement_sum,
             data.total_displacement_sum, data_other_data.total_displacement_sum],
            [other_data.total_v_average, self.total_v_average,
             data.total_v_average, data_other_data.total_v_average],
            [other_data.total_v_variance, self.total_v_variance,
             data.total_v_variance, data_other_data.total_v_variance],
            [other_data.total_curvity_average, self.total_curvity_average,
             data.total_curvity_average, data_other_data.total_curvity_average],
            [other_data.total_curvity_variance, self.total_curvity_variance,
             data.total_curvity_variance, data_other_data.total_curvity_variance],
            [other_data.total_smoothness_average, self.total_smoothness_average,
             data.total_smoothness_average, data_other_data.total_smoothness_average],
            [other_data.total_smoothness_variance, self.total_smoothness_variance,
             data.total_smoothness_variance, data_other_data.total_smoothness_variance],
            [other_data.total_Turning_angle_average, self.total_Turning_angle_average,
             data.total_Turning_angle_average, data_other_data.total_Turning_angle_average],
            [other_data.total_Turning_angle_variance, self.total_Turning_angle_variance,
             data.total_Turning_angle_variance, data_other_data.total_Turning_angle_variance],
        ]

        plot_title = [
            "Total time", "Total displacement", "Velocity mean", "Velocity variance",
            "Curvature mean", "Curvature variance", "Smoothness mean", "Smoothness variance",
            "Turning angle mean", "Turning angle variance",
        ]

        labels_two = ["Exp", "Nov"]
        labels_four = ["Exp", "Nov", "Exp", "Nov"]

        font_dict = {"fontsize": 10, "fontweight": 20}
        pad_setting = 11.5

        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        for idx, ax in enumerate(axs.ravel()):
            labels = labels_two if idx == 0 else labels_four
            ax.boxplot(plot_data[idx], vert=True, labels=labels)
            ax.set_title(plot_title[idx], fontdict=font_dict, pad=pad_setting)
            if idx != 0:
                ax.set_xlabel("L               R")
            ax.yaxis.grid(True)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
            ax.set_ylabel("values")

        fig.suptitle(self.description_choose.split("_")[0])
        plt.show()
