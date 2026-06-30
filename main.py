from __future__ import annotations

import argparse
import os
import runpy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from runtime_config import DEFAULT_TASK, TASKS, get_repo_root

REPO_ROOT = get_repo_root()

def build_helper_entry(method: str) -> dict[str, object]:
    return {
        "path": REPO_ROOT / "helper" / "main.py",
        "argv": lambda args: [
            "main.py",
            "--task",
            args.task,
            "--option",
            method,
        ],
    }


SCRIPT_REGISTRY = {
    ("helper", "generate_metadata"): build_helper_entry("generate_metadata"),
    ("helper", "generate_gesture_clips"): build_helper_entry("generate_gesture_clips"),
    ("helper", "image_stitch"): build_helper_entry("image_stitch"),
    ("hmm", "train_observations"): {
        "path": REPO_ROOT / "HMM_model" / "train_observations.py",
        "argv": lambda args: ["train_observations.py"],
    },
    ("hmm", "train_model"): {
        "path": REPO_ROOT / "HMM_model" / "hmm_model_o.py",
        "argv": lambda args: ["hmm_model_o.py"],
    },
    ("hmm", "test"): {
        "path": REPO_ROOT / "HMM_model" / "test.py",
        "argv": lambda args: ["test.py"],
    },
    ("stip", "cluster"): {
        "path": REPO_ROOT / "stip+bof" / "clustering.py",
        "argv": lambda args: ["clustering.py"],
    },
    ("stip", "classify"): {
        "path": REPO_ROOT / "stip+bof" / "classification.py",
        "argv": lambda args: ["classification.py"],
    },
    ("kinematic", "feature_classify"): {
        "path": REPO_ROOT
        / "kinematic_analysis"
        / "Machine_learning_basedonfeatures"
        / "SVC_classification.py",
        "argv": lambda args: ["SVC_classification.py"],
    },
    ("kinematic", "box_plot"): {
        "path": REPO_ROOT / "kinematic_analysis" / "box_plot" / "main_new.py",
        "argv": lambda args: ["main_new.py"],
    },
    ("kinematic", "box_classify"): {
        "path": REPO_ROOT
        / "kinematic_analysis"
        / "box_plot"
        / "SVC_classification.py",
        "argv": lambda args: ["SVC_classification.py"],
    },
    ("kinematic", "trajectory_plot"): {
        "path": REPO_ROOT
        / "kinematic_analysis"
        / "trajectory_plot"
        / "plotting_expert_novice.py",
        "argv": lambda args: ["plotting_expert_novice.py"],
    },
}


@contextmanager
def temporary_environment(
    task: str,
    data_root: Path,
    box_plot_data: Path,
    stip_feature_root: Path,
    trajectory_hand: str,
) -> Iterator[None]:
    original_values = {
        "JIGSAWS_TASK": os.environ.get("JIGSAWS_TASK"),
        "JIGSAWS_DATA_ROOT": os.environ.get("JIGSAWS_DATA_ROOT"),
        "JIGSAWS_BOX_PLOT_DATA": os.environ.get("JIGSAWS_BOX_PLOT_DATA"),
        "JIGSAWS_STIP_FEATURE_ROOT": os.environ.get("JIGSAWS_STIP_FEATURE_ROOT"),
        "JIGSAWS_TRAJECTORY_HAND": os.environ.get("JIGSAWS_TRAJECTORY_HAND"),
    }
    os.environ["JIGSAWS_TASK"] = task
    os.environ["JIGSAWS_DATA_ROOT"] = str(data_root)
    os.environ["JIGSAWS_BOX_PLOT_DATA"] = str(box_plot_data)
    os.environ["JIGSAWS_STIP_FEATURE_ROOT"] = str(stip_feature_root)
    os.environ["JIGSAWS_TRAJECTORY_HAND"] = trajectory_hand
    try:
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def temporary_sys_path(script_dir: Path) -> Iterator[None]:
    inserted_paths = []
    for path in (script_dir, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted_paths.append(path_str)
    try:
        yield
    finally:
        for path_str in inserted_paths:
            if path_str in sys.path:
                sys.path.remove(path_str)


@contextmanager
def temporary_working_directory(target_dir: Path) -> Iterator[None]:
    original_dir = Path.cwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(original_dir)


@contextmanager
def temporary_argv(argv: list[str]) -> Iterator[None]:
    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for JIGSAWS repository methods."
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default=DEFAULT_TASK,
        help="Task to run.",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "da_vici_data_with_iDT_features"),
        help="Absolute or relative path to the dataset root.",
    )
    parser.add_argument(
        "--box-plot-data",
        default=str(REPO_ROOT / "kinematic_analysis" / "box_plot"),
        help="Path used by the box-plot classifier script.",
    )
    parser.add_argument(
        "--stip-feature-root",
        default=str(REPO_ROOT / "stip+dct_code" / "raw_data"),
        help="Path to the STIP raw feature root.",
    )
    parser.add_argument(
        "--trajectory-hand",
        choices=["left", "right"],
        default="left",
        help="Hand to render for the trajectory plot.",
    )

    subparsers = parser.add_subparsers(dest="family", required=True)

    helper_parser = subparsers.add_parser("helper")
    helper_parser.add_argument(
        "method",
        choices=["generate_metadata", "generate_gesture_clips", "image_stitch"],
    )

    hmm_parser = subparsers.add_parser("hmm")
    hmm_parser.add_argument(
        "method",
        choices=["train_observations", "train_model", "test"],
    )

    stip_parser = subparsers.add_parser("stip")
    stip_parser.add_argument("method", choices=["cluster", "classify"])

    kinematic_parser = subparsers.add_parser("kinematic")
    kinematic_parser.add_argument(
        "method",
        choices=[
            "feature_classify",
            "box_plot",
            "box_classify",
            "trajectory_plot",
        ],
    )
    return parser


def run_selected_script(args: argparse.Namespace) -> None:
    spec = SCRIPT_REGISTRY[(args.family, args.method)]
    script_path = spec["path"]
    argv = spec["argv"](args)
    data_root = Path(args.data_root).expanduser().resolve()
    box_plot_data = Path(args.box_plot_data).expanduser().resolve()
    stip_feature_root = Path(args.stip_feature_root).expanduser().resolve()

    with temporary_environment(
        args.task,
        data_root,
        box_plot_data,
        stip_feature_root,
        args.trajectory_hand,
    ):
        with temporary_sys_path(script_path.parent):
            with temporary_working_directory(script_path.parent):
                with temporary_argv(argv):
                    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_selected_script(args)


if __name__ == "__main__":
    main()
