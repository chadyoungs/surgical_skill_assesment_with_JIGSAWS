#! /usr/bin/env python3
"""
Metadata generation utilities for the JIGSAWS dataset.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import pickle

import numpy as np

from constant import ROOT, TASK


class MetaData:
    def __init__(
        self,
        score_grades: int = 2,
        surgeme_simplified: bool = False,
        res_file_loc: str | Path | None = None,
    ) -> None:
        self.task = TASK
        self.meta_file_loc = ROOT / TASK / f"meta_file_{TASK}.txt"
        self.trans_file_dir = ROOT / TASK / "transcriptions"
        self.traintestsplit_dir = ROOT / "Experimental_setup" / TASK / "unBalanced"
        self.traintestsplit_skill_dir = self.traintestsplit_dir / "SkillDetection"
        self.traintestsplit_classifi_dir = self.traintestsplit_dir / "GestureClassification"

        self.score_grades = score_grades
        self.surgeme_simplified = surgeme_simplified

        self.metadata_res: dict = defaultdict(dict)
        self.train_test_split_res: dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        self.surgeme_dict: dict[str, int] = {}

        self.res_file_loc = Path(res_file_loc) if res_file_loc else None

        self.get_surgeme()

    def _skill_grade(self, score: int) -> int:
        """Return a skill grade (0 = novice, 2 = expert) for the given score.

        Grades are based on task-specific score thresholds.
        """
        if self.score_grades == 2:
            if self.task == "Knot_Tying":
                return 0 if score <= 15 else 2
            elif self.task == "Suturing":
                return 0 if score <= 19 else 2
            elif self.task == "Needle_Passing":
                return 0 if score <= 15 else 2
        return 0

    def _surgeme_simplified(self, raw_surgeme: str) -> None:
        pass

    def _find_surgeme_video(
        self, trial_name: str, surgeme_start_frame_idx: int
    ) -> tuple[str, str, str]:
        search_data = self.metadata_res[trial_name]["surgeme_start_end"]
        for idx, (start, end, surgeme) in enumerate(
            zip(
                search_data["start_frame_idx"],
                search_data["end_frame_idx"],
                search_data["surgeme"],
            )
        ):
            if start == surgeme_start_frame_idx:
                surgeme_video_name = "_".join([trial_name, surgeme, str(idx)]) + ".avi"
                label = self.metadata_res["metadata"]["surgeme_label_mapping"][surgeme]
                return (
                    f"{surgeme}/{surgeme_video_name} {end - start} {label}",
                    surgeme_video_name,
                    surgeme,
                )
        raise ValueError(
            f"No surgeme found for trial '{trial_name}' at start frame {surgeme_start_frame_idx}"
        )

    def get_score(self) -> None:
        content = np.loadtxt(self.meta_file_loc, dtype=str)
        for row in content:
            trial_name, trial_score = row[0], int(row[2])
            self.metadata_res[trial_name]["score"] = trial_score
            self.metadata_res[trial_name]["grade"] = self._skill_grade(trial_score)

    def train_test_split_skill_detection(self) -> None:
        """Populate ``train_test_split_res`` with skill-detection splits."""
        outmethod_list = [
            p.name
            for p in self.traintestsplit_skill_dir.iterdir()
            if p.name != "OneTrialOut"  # OneTrialOut not included
        ]

        for outmethod in outmethod_list:
            outmethod_abs = self.traintestsplit_skill_dir / outmethod
            for out_dir in outmethod_abs.iterdir():
                for option in ("Train", "Test"):
                    out_traintest_abs = out_dir / "itr_1" / f"{option}.txt"
                    content = np.loadtxt(out_traintest_abs, dtype=str)

                    content_list = []
                    for row in content:
                        trial_name = "_".join(row[0].split("_")[:2])
                        try:
                            trial_label = self._skill_grade(int(row[1]))
                        except ValueError:
                            print(out_dir)
                            continue
                        content_list.append((trial_name, trial_label))

                    self.train_test_split_res["SkillDetection"][outmethod][out_dir.name][option] = content_list

    def train_test_split_converter(self) -> None:
        """Write UCF-101-style train/test split files (label index from zero)."""
        save_traintest_split_dir = ROOT / TASK / "surgeme_classifi_traintestsplit"
        save_traintest_split_dir.mkdir(exist_ok=True)

        outmethod_list = [
            p.name
            for p in self.traintestsplit_classifi_dir.iterdir()
            if p.name != "OneTrialOut"  # OneTrialOut not included
        ]

        for outmethod in outmethod_list:
            outmethod_abs = self.traintestsplit_classifi_dir / outmethod
            for out_dir in outmethod_abs.iterdir():
                for option in ("Train", "Test"):
                    out_traintest_abs = out_dir / "itr_1" / f"{option}.txt"
                    content = np.loadtxt(out_traintest_abs, dtype=str)

                    save_file_loc = save_traintest_split_dir / f"{outmethod}_{out_dir.name}_{option}.txt"
                    with open(save_file_loc, "w") as f:
                        for row in content:
                            trial_name = "_".join(row[0].split("_")[:2])
                            surgeme_start_frame_idx = int(row[0].split("_")[2])
                            res_str, _, _ = self._find_surgeme_video(trial_name, surgeme_start_frame_idx)
                            f.write(res_str + "\n")

    def generate_annotation_json(self) -> None:
        """Write UCF-101-style annotation JSON files for PyTorchCon3D.

        Label indices start from zero.
        """
        train_test_mapping = {"Train": "training", "Test": "validation"}
        save_annotation_dir = ROOT / TASK / "surgeme_annotation"
        save_annotation_dir.mkdir(exist_ok=True)

        outmethod_list = [
            p.name
            for p in self.traintestsplit_classifi_dir.iterdir()
            if p.name != "OneTrialOut"  # OneTrialOut not included
        ]

        for outmethod in outmethod_list:
            outmethod_abs = self.traintestsplit_classifi_dir / outmethod
            for out_dir in outmethod_abs.iterdir():
                annotations: dict = {"database": {}}
                labels_dict: dict[str, int] = {}
                save_file_loc = save_annotation_dir / f"{outmethod}_{out_dir.name}.json"

                for option in ("Train", "Test"):
                    out_traintest_abs = out_dir / "itr_1" / f"{option}.txt"
                    content = np.loadtxt(out_traintest_abs, dtype=str)
                    for row in content:
                        trial_name = "_".join(row[0].split("_")[:2])
                        surgeme_start_frame_idx = int(row[0].split("_")[2])
                        # surgeme_video_name is the clip filename; real_label is the gesture name (e.g. "G1")
                        _, surgeme_video_name, real_label = self._find_surgeme_video(
                            trial_name, surgeme_start_frame_idx
                        )
                        annotations["database"][surgeme_video_name] = {
                            "subset": train_test_mapping[option],
                            "annotations": {"label": real_label},
                        }
                        labels_dict[real_label] = 0

                annotations["labels"] = sorted(labels_dict, key=lambda x: int(x[1:]))
                with open(save_file_loc, "w") as f:
                    json.dump(annotations, f, indent=4)

    def get_surgeme(self) -> None:
        """Load surgeme transcriptions and populate ``metadata_res``."""
        for txt_file in self.trans_file_dir.iterdir():
            trial_name = txt_file.stem
            start_frame_idx_list: list[int] = []
            end_frame_idx_list: list[int] = []
            surgeme_list: list[str] = []

            content = np.loadtxt(txt_file, dtype=str)
            for row in content:
                start_frame_idx, end_frame_idx, surgeme = int(row[0]), int(row[1]), row[2]
                start_frame_idx_list.append(start_frame_idx)
                end_frame_idx_list.append(end_frame_idx)
                surgeme_list.append(surgeme)

                if surgeme not in self.surgeme_dict:
                    self.surgeme_dict[surgeme] = 1

            self.metadata_res[trial_name]["surgery_start_end"] = {
                "start_frame": int(content[0][0]),
                "end_frame": int(content[-1][1]),
            }
            self.metadata_res[trial_name]["surgeme_start_end"] = {
                "start_frame_idx": start_frame_idx_list,
                "end_frame_idx": end_frame_idx_list,
                "surgeme": surgeme_list,
            }

        sorted_surgemes = sorted(self.surgeme_dict, key=lambda x: int(x[1:]))
        self.metadata_res["metadata"] = {
            "surgeme_list": sorted_surgemes,
            "surgeme_label_mapping": {s: idx for idx, s in enumerate(sorted_surgemes)},
        }

    def generate_metadata(self) -> None:
        self.get_score()
        self.train_test_split_skill_detection()

    def generate_train_test_files(self) -> None:
        self.train_test_split_converter()

    def save_to_pkl(self) -> None:
        with open(self.res_file_loc, "wb") as f:
            pickle.dump(self.metadata_res, f)


if __name__ == "__main__":
    trigger = MetaData()
    trigger.generate_annotation_json()
