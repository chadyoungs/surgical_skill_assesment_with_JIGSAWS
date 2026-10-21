from __future__ import annotations

import os
from pathlib import Path

TASKS = ("Suturing", "Knot_Tying", "Needle_Passing")
DEFAULT_TASK = TASKS[0]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent


def get_data_root() -> Path:
    raw_value = os.environ.get("JIGSAWS_DATA_ROOT")
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return (get_repo_root() / "da_vici_data_with_iDT_features").resolve()


def get_box_plot_data_path() -> Path:
    raw_value = os.environ.get("JIGSAWS_BOX_PLOT_DATA")
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return (get_repo_root() / "kinematic_analysis" / "box_plot").resolve()


def get_stip_features_root() -> Path:
    raw_value = os.environ.get("JIGSAWS_STIP_FEATURE_ROOT")
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return (get_repo_root() / "stip+dct_code" / "raw_data").resolve()


def get_task_name(task_name: str | None = None) -> str:
    selected_task = task_name or os.environ.get("JIGSAWS_TASK", DEFAULT_TASK)
    if selected_task not in TASKS:
        raise ValueError(
            "Unsupported task '{}'. Expected one of: {}".format(
                selected_task, ", ".join(TASKS)
            )
        )
    return selected_task


def get_task_index(task_name: str | None = None) -> int:
    return TASKS.index(get_task_name(task_name))


def get_task_root(task_name: str | None = None) -> Path:
    return get_data_root() / get_task_name(task_name)


def get_experimental_setup_root(task_name: str | None = None) -> Path:
    return (
        get_data_root()
        / "Experimental_setup"
        / get_task_name(task_name)
        / "unBalanced"
    )


def get_trajectory_hand() -> str:
    hand = os.environ.get("JIGSAWS_TRAJECTORY_HAND", "left").lower()
    if hand not in {"left", "right"}:
        raise ValueError("Unsupported trajectory hand '{}'.".format(hand))
    return hand
