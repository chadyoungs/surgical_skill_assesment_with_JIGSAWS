# -*- coding: utf-8 -*-
from pathlib import Path
import sys

from load_and_import_new import TimeSeriesData
from analysis_new import TotalData
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import get_task_name, get_task_root


def main() -> None:
    task_name = get_task_name()
    task_root = get_task_root()
    all_gestures_dir = task_root / "kinematics" / "AllGestures"

    timeseriesdata = TimeSeriesData()
    timeseriesdata.choose_file(str(task_root))
    timeseriesdata.getMetaData()
    feature, label = timeseriesdata.getKinematicData(str(all_gestures_dir))

    feature_novice = feature[np.where(label == 0)[0]]
    feature_expert = feature[np.where(label == 2)[0]]

    analysisexpert_mtfL = TotalData()
    analysisexpert_mtfL.getFile(feature_expert, "MTF_L")
    analysisexpert_mtfL.total_analysis()

    analysisnovice_mtfL = TotalData()
    analysisnovice_mtfL.getFile(feature_novice, "MTF_L")
    analysisnovice_mtfL.total_analysis()

    analysisexpert_mtfR = TotalData()
    analysisexpert_mtfR.getFile(feature_expert, "MTF_R")
    analysisexpert_mtfR.total_analysis()

    analysisnovice_mtfR = TotalData()
    analysisnovice_mtfR.getFile(feature_novice, "MTF_R")
    analysisnovice_mtfR.total_analysis()

    analysisnovice_mtfL.visual_comparison(
        analysisexpert_mtfL,
        analysisexpert_mtfR,
        analysisnovice_mtfR,
    )

    analysisexpert_psm1 = TotalData()
    analysisexpert_psm1.getFile(feature_expert, "PSM_1")
    analysisexpert_psm1.total_analysis()

    analysisnovice_psm1 = TotalData()
    analysisnovice_psm1.getFile(feature_novice, "PSM_1")
    analysisnovice_psm1.total_analysis()

    analysisexpert_psm2 = TotalData()
    analysisexpert_psm2.getFile(feature_expert, "PSM_2")
    analysisexpert_psm2.total_analysis()

    analysisnovice_psm2 = TotalData()
    analysisnovice_psm2.getFile(feature_novice, "PSM_2")
    analysisnovice_psm2.total_analysis()

    analysisnovice_psm1.visual_comparison(
        analysisexpert_psm1,
        analysisexpert_psm2,
        analysisnovice_psm2,
    )
    print("Box-plot analysis completed for {}.".format(task_name))


if __name__ == "__main__":
    main()
