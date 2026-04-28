#!/usr/bin/env python3
"""
Data loading and feature extraction for kinematic analysis.

Reads JIGSAWS meta-data and per-trial kinematic files, computes trajectory
features via :class:`DataCal`, and returns feature/label arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from calculation_new import DataCal


class TimeSeriesData:
    """Load kinematic data and compute per-trial trajectory features."""

    def __init__(self) -> None:
        self.classmode: list[str] = ["GRS"]
        self.metaData: dict = {}
        self.file_name: str = ""

    def choose_file(self, file_name: str | Path) -> None:
        """Set the root directory containing the meta-data file.

        Args:
            file_name: Path to the task directory (e.g. ``".../Suturing"``).
        """
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
        """Return the skill label for *surgery_name*, or ``None`` if unknown.

        Args:
            surgery_name: Trial identifier, e.g. ``"Suturing_D001"``.

        Returns:
            ``0`` for novice, ``1`` for intermediate, ``2`` for expert,
            or ``None`` if the surgery is not in the metadata.
        """
        if surgery_name not in self.metaData:
            return None

        if self.classmode[0] != "GRS":
            return None

        score_grs = self.metaData[surgery_name][1][0]

        if "Knot_Tying" in surgery_name:
            if score_grs <= 15:
                return 0
            elif score_grs < 19:
                return 1
            return 2
        if "Suturing" in surgery_name:
            if score_grs <= 15:
                return 0
            elif score_grs < 19:
                return 1
            return 2
        if "Needle_Passing" in surgery_name:
            if score_grs <= 15:
                return 0
            elif score_grs < 20:
                return 1
            return 2
        return None

    def getKinematicData(  # noqa: N802 – kept for backward compat
        self, url: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load kinematic files, compute features, and return data/label arrays.

        Args:
            url: Directory containing per-trial ``.txt`` kinematic files.

        Returns:
            A 2-tuple ``(dataX, dataY)`` where *dataX* has shape
            ``(n_trials, 4, 10)`` and *dataY* has shape ``(n_trials, 1)``.
        """
        data_x: list = []
        data_y = np.zeros((0, 1))

        print(f"loading data from url:\t{url}")
        for file in sorted(Path(url).glob("*.txt")):
            surgery_name = file.stem
            y = self.getSkillLevel(surgery_name)
            if y is None:
                continue

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

            feature = np.vstack((feature_mtf_l, feature_mtf_r, feature_psm_1, feature_psm_2))
            data_x.append(feature.tolist())
            data_y = np.vstack((data_y, y))

        return np.array(data_x), data_y
