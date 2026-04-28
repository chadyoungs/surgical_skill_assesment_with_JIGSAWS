#!/usr/bin/env python3
"""
Plot 3D trajectories of expert vs novice surgical tool tips.

Loads the highest- and lowest-scoring trials and plots their 3-D position
curves for the left or right instrument tip side by side.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d  # noqa: F401 – registers the '3d' projection

from data_extract_stip import DataExtract, TASK_LIST, TASK_SYMBOL

plt.rc("font", family="Times New Roman")

# Column offsets for (x, y, z) of each instrument tip
_POSITION_COLS: dict[str, tuple[int, int, int]] = {
    "left_MTM":  (0,  1,  2),
    "right_MTM": (19, 20, 21),
    "PSM1":      (38, 39, 40),
    "PSM2":      (57, 58, 59),
}


def main() -> None:
    extractor = DataExtract()
    data_expert_name, data_novice_name = extractor.get_score()

    task = TASK_LIST[TASK_SYMBOL]
    kinematics_dir = Path(".") / "da_vici_data_with_iDT_features" / task / "kinematics" / "AllGestures"

    data_expert = np.loadtxt(kinematics_dir / f"{data_expert_name}.txt")
    data_novice = np.loadtxt(kinematics_dir / f"{data_novice_name}.txt")

    # Select left or right tip: 'l' for left_MTM/PSM1, 'r' for right_MTM/PSM2
    mark = "l"

    if mark == "l":
        tip = "left_MTM"
        save_name = "3D trajectories of left hand of expert and novice surgeon"
    else:
        tip = "right_MTM"
        save_name = "3D trajectories of right hand of expert and novice surgeon"

    cols = _POSITION_COLS[tip]
    ax = plt.axes(projection="3d")
    ax.set_title("3D_trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.plot(data_expert[:, cols[0]], data_expert[:, cols[1]], data_expert[:, cols[2]],
            c="b", label="expert")
    ax.plot(data_novice[:, cols[0]], data_novice[:, cols[1]], data_novice[:, cols[2]],
            c="r", label="novice")

    plt.legend(loc="best")
    plt.savefig(save_name, dpi=300)


if __name__ == "__main__":
    main()
