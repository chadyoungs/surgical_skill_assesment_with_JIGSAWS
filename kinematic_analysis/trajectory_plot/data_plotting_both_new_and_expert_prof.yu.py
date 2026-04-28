#!/usr/bin/env python3
"""
Plot 3D PSM trajectories comparing a specific expert and novice trial.

Update EXPERT_FILE and NOVICE_FILE below to point at the desired trials.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d  # noqa: F401 – registers the '3d' projection

# ---------------------------------------------------------------------------
# Paths – update to match your local dataset location
# ---------------------------------------------------------------------------
_KINEMATICS = Path(".") / "da_vici_data_with_iDT_features" / "Suturing" / "kinematics" / "AllGestures"
EXPERT_FILE: Path = _KINEMATICS / "Suturing_D001.txt"
NOVICE_FILE: Path = _KINEMATICS / "Suturing_B001.txt"


def main() -> None:
    data_expert = np.loadtxt(EXPERT_FILE)
    data_novice = np.loadtxt(NOVICE_FILE)

    # PSM1 columns (x=38, y=39, z=40)
    x_expert_left = data_expert[:, 38]
    y_expert_left = data_expert[:, 39]
    z_expert_left = data_expert[:, 40]

    x_novice_left = data_novice[:, 38]
    y_novice_left = data_novice[:, 39]
    z_novice_left = data_novice[:, 40]

    ax = plt.axes(projection="3d")
    ax.set_title("3D_Position_Curve_Suturing")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.plot(x_expert_left, y_expert_left, z_expert_left, c="b", label="expert_PSM1")
    ax.plot(x_novice_left, y_novice_left, z_novice_left, c="m", label="newhand_PSM1")

    plt.legend()
    plt.savefig("fig_expert_and_novice", dpi=300)


if __name__ == "__main__":
    main()
