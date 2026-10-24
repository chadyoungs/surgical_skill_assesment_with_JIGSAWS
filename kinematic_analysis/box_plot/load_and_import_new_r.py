#!/usr/bin/env python3
"""
Data loading, feature extraction, and CSV export for kinematic analysis.

Extends :mod:`load_and_import_new` by additionally writing the computed
features to a ``data_for_SVM.csv`` file alongside the ``.npy`` arrays.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from calculation_new import DataCal

# CSV column headers
_CSV_HEADERS = [
    "time_sum",
    "displacement_sum_mtfl", "velocity_ave_mtfl", "velocity_var_mtfl",
    "curvity_ave_mtfl", "curvity_var_mtfl",
    "smooth_ave_mtfl", "smooth_var_mtfl",
    "turning_ave_mtfl", "turning_var_mtfl",
    "displacement_sum_mtfr", "velocity_ave_mtfr", "velocity_var_mtfr",
    "curvity_ave_mtfr", "curvity_var_mtfr",
    "smooth_ave_mtfr", "smooth_var_mtfr",
    "turning_ave_mtfr", "turning_var_mtfr",
    "displacement_sum_psm1", "velocity_ave_psm1", "velocity_var_psm1",
    "curvity_ave_psm1", "curvity_var_psm1",
    "smooth_ave_psm1", "smooth_var_psm1",
    "turning_ave_psm1", "turning_var_psm1",
    "displacement_sum_psm2", "velocity_ave_psm2", "velocity_var_psm2",
    "curvity_ave_psm2", "curvity_var_psm2",
    "smooth_ave_psm2", "smooth_var_psm2",
    "turning_ave_psm2", "turning_var_psm2",
    "class",
]


class TimeSeriesData:
    """Load kinematic data, compute features, and export results to CSV."""

    def __init__(self) -> None:
        self.classmode: list[str] = ["GRS"]
        self.metaData: dict = {}
        self.file_name: str = ""

    def choose_file(self, file_name: str | Path) -> None:
        """Set the root directory containing the meta-data file."""
        self.file_name = str(file_name)
        print(f"loading data name is: {self.file_name}")

    def getMetaData(self) -> None:  # noqa: N802 – kept for backward compat
        """Parse the task meta-data file and populate ``self.metaData``."""
        for meta_file in Path(self.file_name).glob("meta_file_*.txt"):
            for line in meta_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    break
                parts = line.split()
                surgery_name = parts[0]
                skill_level = parts[1]
                scores = [int(e) for e in parts[2:]]
                self.metaData[surgery_name] = (skill_level, scores)

    def getSkillLevel(self, surgery_name: str) -> int | None:  # noqa: N802 – kept for backward compat
        """Return the skill label or ``None`` if the surgery is not in metadata."""
        if surgery_name not in self.metaData:
            return None

        if self.classmode[0] != "GRS":
            return None

        score_grs = self.metaData[surgery_name][1][0]

        if "Knot_Tying" in surgery_name:
            return 0 if score_grs <= 15 else 2
        if "Suturing" in surgery_name:
            return 0 if score_grs <= 19 else 2
        if "Needle_Passing" in surgery_name:
            if score_grs <= 15:
                return 0
            elif score_grs < 20:
                return 1
            return 2
        return None

    def getKinematicData(  # noqa: N802 – kept for backward compat
        self,
        url: str | Path,
        csv_path: str | Path = "data_for_SVM.csv",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load kinematic files, compute features, write CSV, and return arrays.

        Args:
            url: Directory containing per-trial ``.txt`` kinematic files.
            csv_path: Destination CSV file path.

        Returns:
            A 2-tuple ``(dataX, dataY)`` where *dataX* has shape
            ``(n_trials, 4, 10)`` and *dataY* has shape ``(n_trials, 1)``.
        """
        data_x: list = []
        data_y = np.zeros((0, 1))

        print(f"loading data from url:\t{url}")
        with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(_CSV_HEADERS)

            for file in sorted(Path(url).glob("*.txt")):
                surgery_name = file.stem
                y = self.getSkillLevel(surgery_name)
                if y is None:
                    continue
                print(y)

                x = np.genfromtxt(str(file), delimiter="", dtype=np.float32)

                calculator = DataCal()
                calculator.getFile(x, "MTF_L")
                feature_mtf_l = calculator.cal_processing()
                calculator.getFile(x, "MTF_R")
                feature_mtf_r = calculator.cal_processing()
                calculator.getFile(x, "PSM_1")
                feature_psm_1 = calculator.cal_processing()
                calculator.getFile(x, "PSM_2")
                feature_psm_2 = calculator.cal_processing()

                writer.writerow([
                    str(feature_mtf_l[0]),
                    str(feature_mtf_l[1]), str(feature_mtf_l[2]), str(feature_mtf_l[3]),
                    str(feature_mtf_l[4]), str(feature_mtf_l[5]),
                    str(feature_mtf_l[6]), str(feature_mtf_l[7]),
                    str(feature_mtf_l[8]), str(feature_mtf_l[9]),
                    str(feature_mtf_r[1]), str(feature_mtf_r[2]), str(feature_mtf_r[3]),
                    str(feature_mtf_r[4]), str(feature_mtf_r[5]),
                    str(feature_mtf_r[6]), str(feature_mtf_r[7]),
                    str(feature_mtf_r[8]), str(feature_mtf_r[9]),
                    str(feature_psm_1[1]), str(feature_psm_1[2]), str(feature_psm_1[3]),
                    str(feature_psm_1[4]), str(feature_psm_1[5]),
                    str(feature_psm_1[6]), str(feature_psm_1[7]),
                    str(feature_psm_1[8]), str(feature_psm_1[9]),
                    str(feature_psm_2[1]), str(feature_psm_2[2]), str(feature_psm_2[3]),
                    str(feature_psm_2[4]), str(feature_psm_2[5]),
                    str(feature_psm_2[6]), str(feature_psm_2[7]),
                    str(feature_psm_2[8]), str(feature_psm_2[9]),
                    str(y),
                ])

                feature = np.vstack((feature_mtf_l, feature_mtf_r, feature_psm_1, feature_psm_2))
                data_x.append(feature.tolist())
                data_y = np.vstack((data_y, y))

        return np.array(data_x), data_y
