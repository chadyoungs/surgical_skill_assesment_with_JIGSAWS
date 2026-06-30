from pathlib import Path
import sys

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import get_task_name, get_task_root
from data_extract_stip import DataExtract

plt.rc("font", family="Times New Roman")

POSITION_DICT = {
    "left_MTM": (0, 1, 2),
    "right_MTM": (19, 20, 21),
    "PSM1": (38, 39, 40),
    "PSM2": (57, 58, 59),
}


def main() -> None:
    extractor = DataExtract()
    data_expert_name, data_novice_name = extractor.get_score()

    task_root = get_task_root()
    all_gestures_dir = task_root / "kinematics" / "AllGestures"
    data_expert = np.loadtxt(all_gestures_dir / f"{data_expert_name}.txt")
    data_novice = np.loadtxt(all_gestures_dir / f"{data_novice_name}.txt")

    left_key = "left_MTM"
    right_key = "right_MTM"

    x_expert_left = data_expert[:, POSITION_DICT[left_key][0]]
    y_expert_left = data_expert[:, POSITION_DICT[left_key][1]]
    z_expert_left = data_expert[:, POSITION_DICT[left_key][2]]
    x_novice_left = data_novice[:, POSITION_DICT[left_key][0]]
    y_novice_left = data_novice[:, POSITION_DICT[left_key][1]]
    z_novice_left = data_novice[:, POSITION_DICT[left_key][2]]

    x_expert_right = data_expert[:, POSITION_DICT[right_key][0]]
    y_expert_right = data_expert[:, POSITION_DICT[right_key][1]]
    z_expert_right = data_expert[:, POSITION_DICT[right_key][2]]
    x_novice_right = data_novice[:, POSITION_DICT[right_key][0]]
    y_novice_right = data_novice[:, POSITION_DICT[right_key][1]]
    z_novice_right = data_novice[:, POSITION_DICT[right_key][2]]

    ax = plt.axes(projection="3d")
    ax.set_title(f"3D trajectories - {get_task_name()}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plot_left = True
    if plot_left:
        ax.plot(x_expert_left, y_expert_left, z_expert_left, c="b", label="expert")
        ax.plot(x_novice_left, y_novice_left, z_novice_left, c="r", label="novice")
        output_name = "3D_trajectories_left.png"
    else:
        ax.plot(x_expert_right, y_expert_right, z_expert_right, c="b", label="expert")
        ax.plot(x_novice_right, y_novice_right, z_novice_right, c="r", label="novice")
        output_name = "3D_trajectories_right.png"

    plt.legend(loc="best")
    plt.savefig(output_name, dpi=300)


if __name__ == "__main__":
    main()
