#!/usr/bin/env python3
"""
Box-plot comparison of expert vs novice kinematic features.

Update DATA_DIR and KINEMATICS_DIR below to point at your local dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from load_and_import_new import TimeSeriesData
from analysis_new import TotalData

# ---------------------------------------------------------------------------
# Paths – update these to match your local dataset location
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(".") / "da_vici_data_with_iDT_features" / "Suturing"
KINEMATICS_DIR: Path = DATA_DIR / "kinematics" / "AllGestures"


def main() -> None:
    timeseriesdata = TimeSeriesData()
    timeseriesdata.choose_file(DATA_DIR)
    timeseriesdata.getMetaData()
    feature, label = timeseriesdata.getKinematicData(KINEMATICS_DIR)

    feature_novice = feature[np.where(label == 0)[0]]
    feature_expert = feature[np.where(label == 2)[0]]

    # Left-arm (MTF_L) comparison
    expert_mtf_l = TotalData()
    expert_mtf_l.getFile(feature_expert, "MTF_L")
    expert_mtf_l.total_analysis()

    novice_mtf_l = TotalData()
    novice_mtf_l.getFile(feature_novice, "MTF_L")
    novice_mtf_l.total_analysis()

    # Right-arm (MTF_R) comparison
    expert_mtf_r = TotalData()
    expert_mtf_r.getFile(feature_expert, "MTF_R")
    expert_mtf_r.total_analysis()

    novice_mtf_r = TotalData()
    novice_mtf_r.getFile(feature_novice, "MTF_R")
    novice_mtf_r.total_analysis()

    novice_mtf_l.visual_comparison(expert_mtf_l, expert_mtf_r, novice_mtf_r)

    # PSM1 comparison
    expert_psm1 = TotalData()
    expert_psm1.getFile(feature_expert, "PSM_1")
    expert_psm1.total_analysis()

    novice_psm1 = TotalData()
    novice_psm1.getFile(feature_novice, "PSM_1")
    novice_psm1.total_analysis()

    # PSM2 comparison
    expert_psm2 = TotalData()
    expert_psm2.getFile(feature_expert, "PSM_2")
    expert_psm2.total_analysis()

    novice_psm2 = TotalData()
    novice_psm2.getFile(feature_novice, "PSM_2")
    novice_psm2.total_analysis()

    novice_psm1.visual_comparison(expert_psm1, expert_psm2, novice_psm2)


if __name__ == "__main__":
    main()

