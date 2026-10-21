#! /usr/bin/env python3
"""
Data extraction utilities for the JIGSAWS dataset.

Originally created 2020-07-20 by xiaoxiaoyang.
Extended to include metadata consolidation, train/test split parsing,
annotation JSON generation, and pickle serialisation — based on the
JIGSAWS_dataset_helper project (https://github.com/chadyoungs/JIGSAWS_dataset_helper).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Global_Var
from runtime_config import get_data_root

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

task_list = ['Suturing', 'Knot_Tying', 'Needle_Passing']

# 0 for Suturing, 1 for Knot_Tying, 2 for Needle_Passing
TASK_SYMBOL = Global_Var.TASK_SYMBOL
TASK = task_list[TASK_SYMBOL]

_DATA_ROOT = get_data_root()

# remind that root_path must point to SuperTrialOut folder
root_path = str(_DATA_ROOT / 'Experimental_setup' / TASK / 'unBalanced' / 'SkillDetection' / 'SuperTrialOut')
root_path_score = str(_DATA_ROOT / TASK)
root_path_trans = str(_DATA_ROOT / TASK / 'transcriptions')

_traintestsplit_dir = _DATA_ROOT / 'Experimental_setup' / TASK / 'unBalanced'
_traintestsplit_skill_dir = _traintestsplit_dir / 'SkillDetection'
_traintestsplit_classifi_dir = _traintestsplit_dir / 'GestureClassification'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _surgery_name_from_stem(stem: str) -> str:
    """Extract surgery name by dropping the last two '_'-separated tokens (frame numbers)."""
    return '_'.join(stem.split('_')[:-2])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataExtract:
    """Extract and consolidate JIGSAWS metadata.

    Legacy per-type dicts (``metaData``, ``metaData_surgeme``, etc.) are kept
    for backward compatibility with the existing HMM code.  All data is *also*
    mirrored into ``metadata_res`` — a single nested dict matching the layout
    used by the JIGSAWS_dataset_helper project — which can be persisted to disk
    via :meth:`save_to_pkl`.
    """

    def __init__(self, res_file_loc: str | Path | None = None) -> None:
        # ---- legacy dicts (kept for backward compat) ----------------------
        self.metaData: dict = {}
        self.metaData_surgeme: dict = {}
        self.metaData_score: dict = {}
        self.metaData_index: dict = {}
        self.metaData_hmmstates: dict = {}
        self.metaData_hmmindex: dict = {}

        # ---- consolidated metadata (matches helper layout) ----------------
        self.metadata_res: dict = defaultdict(dict)
        self.train_test_split_res: dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        self.surgeme_dict: dict[str, int] = {}

        self.res_file_loc = Path(res_file_loc) if res_file_loc else None

    # ------------------------------------------------------------------
    # Category / directory helpers
    # ------------------------------------------------------------------

    def get_category(self) -> None:
        """Retrieve subdirectories inside *root_path* (one per LOSO fold)."""
        self.category = glob(root_path + "/" + "*")
        self.category_abs = [os.path.basename(i) for i in self.category]

    # ------------------------------------------------------------------
    # Surgeme / transcription parsing
    # ------------------------------------------------------------------

    def get_frame_surgeme(self) -> None:
        """Parse transcription files and populate legacy ``metaData_surgeme``
        as well as ``metadata_res`` with surgeme start/end information.
        """
        txt_files = glob(os.path.join(root_path_trans, '*.txt'))
        surgeme_dict: dict[str, int] = {}

        for file in txt_files:
            list_surgeme: list[str] = []
            list_frameStartNo: list[str] = []
            list_frameEndNo: list[str] = []

            surgery_name = os.path.splitext(os.path.basename(file))[0]
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        break
                    parts = line.split()
                    list_frameStartNo.append(parts[0])
                    list_frameEndNo.append(parts[1])
                    list_surgeme.append(parts[2])
                    if parts[2] not in surgeme_dict:
                        surgeme_dict[parts[2]] = 1

            # legacy tuple
            self.metaData_surgeme[surgery_name] = (list_frameStartNo, list_frameEndNo, list_surgeme)

            # consolidated dict
            self.metadata_res[surgery_name]['surgery_start_end'] = {
                'start_frame': int(list_frameStartNo[0]),
                'end_frame': int(list_frameEndNo[-1]),
            }
            self.metadata_res[surgery_name]['surgeme_start_end'] = {
                'start_frame_idx': [int(x) for x in list_frameStartNo],
                'end_frame_idx': [int(x) for x in list_frameEndNo],
                'surgeme': list_surgeme,
            }

        def _surgeme_sort_key(label: str) -> int:
            try:
                return int(label[1:])
            except (ValueError, IndexError):
                return 0

        sorted_surgemes = sorted(surgeme_dict, key=_surgeme_sort_key)
        self.surgeme_dict = surgeme_dict
        self.metadata_res['metadata'] = {
            'surgeme_list': sorted_surgemes,
            'surgeme_label_mapping': {s: idx for idx, s in enumerate(sorted_surgemes)},
        }

    # ------------------------------------------------------------------
    # HMM state sequence
    # ------------------------------------------------------------------

    def get_hmm_states(self) -> None:
        """Build a per-frame HMM state map into ``metaData_hmmstates``."""
        txt_files = glob(os.path.join(root_path_trans, '*.txt'))
        for file in txt_files:
            surgery_name = os.path.splitext(os.path.basename(file))[0]
            starts = self.metaData_surgeme[surgery_name][0]
            ends = self.metaData_surgeme[surgery_name][1]
            surgemes = self.metaData_surgeme[surgery_name][2]

            for i in range(int(ends[-1])):
                frame_No = i + 1
                for j in range(len(starts)):
                    if int(starts[j]) <= frame_No <= int(ends[j]):
                        self.metaData_hmmstates[frame_No] = surgemes[j]

    # ------------------------------------------------------------------
    # Frame number range per trial
    # ------------------------------------------------------------------

    def get_frame_No(self) -> None:
        """Populate ``metaData`` with (start_frame, end_frame) per trial."""
        for count, c in enumerate(self.category):
            txt_files = glob(os.path.join(c, 'itr_1', '*.txt'))

            if count >= 1:
                break

            for file in txt_files:
                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0:
                            break

                        stem = line.split('.')[0]
                        parts = stem.split('_')
                        start_frame_No = int(parts[-2])
                        end_frame_No = int(parts[-1])
                        surgery_name = _surgery_name_from_stem(stem)

                        self.metaData[surgery_name] = (start_frame_No, end_frame_No)

    # ------------------------------------------------------------------
    # Score / skill level
    # ------------------------------------------------------------------

    def get_score(self) -> None:
        """Parse the meta-file and populate ``metaData_score`` and
        the ``score``/``grade`` fields of ``metadata_res``.
        """
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, 'meta_file_' + category + '.txt')

        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    break
                b = line.split()
                surgery_name = b[0]
                surgery_score_sum = int(b[2])

                scores_grade = self.skill_level(surgery_name, surgery_score_sum)
                # legacy tuple
                self.metaData_score[surgery_name] = (surgery_score_sum, scores_grade)
                # consolidated dict
                self.metadata_res[surgery_name]['score'] = surgery_score_sum
                self.metadata_res[surgery_name]['grade'] = scores_grade

    def skill_level(self, surgery_name: str, score: int) -> int:
        """Return 0 (novice) or 2 (expert) based on task-specific thresholds."""
        if 'Knot_Tying' in surgery_name:
            return 0 if score <= 15 else 2
        elif 'Suturing' in surgery_name:
            return 0 if score <= 19 else 2
        elif 'Needle_Passing' in surgery_name:
            return 0 if score <= 15 else 2
        else:
            raise ValueError("Unrecognised task in surgery name: {}".format(surgery_name))

    # ------------------------------------------------------------------
    # Index helpers (HMM / ML)
    # ------------------------------------------------------------------

    def get_index(self) -> None:
        """Populate ``metaData_hmmindex`` with novice/expert index lists
        (used during HMM training/testing).
        """
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, 'meta_file_' + category + '.txt')

        novice_list: list[int] = []
        expert_list: list[int] = []
        with open(txt_file, 'r') as f:
            for count, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    break
                surgery_name = line.split()[0]

                if self.metaData_score[surgery_name][1] == 0:
                    novice_list.append(count)
                else:
                    expert_list.append(count)

        self.metaData_hmmindex["novice"] = novice_list
        self.metaData_hmmindex["expert"] = expert_list

    def get_txt_index(self) -> None:
        """Populate ``metaData_index`` mapping surgery name → row index in the
        meta-file (used during ML train/test).
        """
        category = os.path.basename(root_path_score)
        txt_file = os.path.join(root_path_score, 'meta_file_' + category + '.txt')

        with open(txt_file, 'r') as f:
            for count, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    break
                surgery_name = line.split()[0]
                self.metaData_index[surgery_name] = count

    # ------------------------------------------------------------------
    # Train / test split helpers
    # ------------------------------------------------------------------

    def _read_split_file(self, split_file: str) -> list[str]:
        """Parse a Train.txt or Test.txt split file and return surgery names."""
        names: list[str] = []
        with open(split_file, 'r') as f:
            for line in f:
                stem = line.split()[0]
                names.append(_surgery_name_from_stem(stem))
        return names

    def train_sum(self) -> list[list[str]]:
        """Return per-fold lists of training surgery names (LOSO)."""
        return [self._read_split_file(os.path.join(c, 'itr_1', 'Train.txt'))
                for c in self.category]

    def test_sum(self) -> list[list[str]]:
        """Return per-fold lists of test surgery names (LOSO)."""
        return [self._read_split_file(os.path.join(c, 'itr_1', 'Test.txt'))
                for c in self.category]

    def train_test_split_skill_detection(self) -> None:
        """Populate ``train_test_split_res`` with skill-detection splits
        (mirrors the helper's ``MetaData.train_test_split_skill_detection``).
        """
        if not _traintestsplit_skill_dir.exists():
            return

        outmethod_list = [
            p.name
            for p in _traintestsplit_skill_dir.iterdir()
            if p.name != 'OneTrialOut'
        ]

        for outmethod in outmethod_list:
            outmethod_abs = _traintestsplit_skill_dir / outmethod
            for out_dir in outmethod_abs.iterdir():
                for option in ('Train', 'Test'):
                    out_traintest_abs = out_dir / 'itr_1' / f'{option}.txt'
                    content = np.loadtxt(str(out_traintest_abs), dtype=str)

                    content_list: list[tuple[str, int]] = []
                    for row in content:
                        trial_name = '_'.join(row[0].split('_')[:2])
                        try:
                            score = int(row[1])
                        except (ValueError, IndexError):
                            # skip rows where the score column is missing or non-numeric
                            continue
                        try:
                            trial_label = self.skill_level(trial_name, score)
                        except ValueError:
                            # skip rows with an unrecognised task name
                            continue
                        content_list.append((trial_name, trial_label))

                    self.train_test_split_res['SkillDetection'][outmethod][out_dir.name][option] = content_list

    # ------------------------------------------------------------------
    # Annotation / split file generation (from helper)
    # ------------------------------------------------------------------

    def _find_surgeme_video(
        self, trial_name: str, surgeme_start_frame_idx: int
    ) -> tuple[str, str, str]:
        """Locate the clip filename and label for a given trial and start frame."""
        search_data = self.metadata_res[trial_name]['surgeme_start_end']
        for idx, (start, end, surgeme) in enumerate(
            zip(
                search_data['start_frame_idx'],
                search_data['end_frame_idx'],
                search_data['surgeme'],
            )
        ):
            if start == surgeme_start_frame_idx:
                surgeme_video_name = '_'.join([trial_name, surgeme, str(idx)]) + '.avi'
                label = self.metadata_res['metadata']['surgeme_label_mapping'][surgeme]
                return (
                    f"{surgeme}/{surgeme_video_name} {end - start} {label}",
                    surgeme_video_name,
                    surgeme,
                )
        raise ValueError(
            f"No surgeme found for trial '{trial_name}' at start frame {surgeme_start_frame_idx}"
        )

    def train_test_split_converter(self) -> None:
        """Write UCF-101-style train/test split text files for gesture
        classification (label index from zero).
        """
        if not _traintestsplit_classifi_dir.exists():
            return

        save_traintest_split_dir = _DATA_ROOT / TASK / 'surgeme_classifi_traintestsplit'
        save_traintest_split_dir.mkdir(exist_ok=True)

        outmethod_list = [
            p.name
            for p in _traintestsplit_classifi_dir.iterdir()
            if p.name != 'OneTrialOut'
        ]

        for outmethod in outmethod_list:
            outmethod_abs = _traintestsplit_classifi_dir / outmethod
            for out_dir in outmethod_abs.iterdir():
                for option in ('Train', 'Test'):
                    out_traintest_abs = out_dir / 'itr_1' / f'{option}.txt'
                    content = np.loadtxt(str(out_traintest_abs), dtype=str)

                    save_file_loc = save_traintest_split_dir / f'{outmethod}_{out_dir.name}_{option}.txt'
                    with open(save_file_loc, 'w') as f:
                        for row in content:
                            trial_name = '_'.join(row[0].split('_')[:2])
                            surgeme_start_frame_idx = int(row[0].split('_')[2])
                            res_str, _, _ = self._find_surgeme_video(trial_name, surgeme_start_frame_idx)
                            f.write(res_str + '\n')

    def generate_annotation_json(self) -> None:
        """Write UCF-101-style annotation JSON files compatible with
        PyTorchCon3D.  Label indices start from zero.
        """
        if not _traintestsplit_classifi_dir.exists():
            return

        train_test_mapping = {'Train': 'training', 'Test': 'validation'}
        save_annotation_dir = _DATA_ROOT / TASK / 'surgeme_annotation'
        save_annotation_dir.mkdir(exist_ok=True)

        outmethod_list = [
            p.name
            for p in _traintestsplit_classifi_dir.iterdir()
            if p.name != 'OneTrialOut'
        ]

        for outmethod in outmethod_list:
            outmethod_abs = _traintestsplit_classifi_dir / outmethod
            for out_dir in outmethod_abs.iterdir():
                annotations: dict = {'database': {}}
                labels_dict: dict[str, int] = {}
                save_file_loc = save_annotation_dir / f'{outmethod}_{out_dir.name}.json'

                for option in ('Train', 'Test'):
                    out_traintest_abs = out_dir / 'itr_1' / f'{option}.txt'
                    content = np.loadtxt(str(out_traintest_abs), dtype=str)
                    for row in content:
                        trial_name = '_'.join(row[0].split('_')[:2])
                        surgeme_start_frame_idx = int(row[0].split('_')[2])
                        _, surgeme_video_name, real_label = self._find_surgeme_video(
                            trial_name, surgeme_start_frame_idx
                        )
                        annotations['database'][surgeme_video_name] = {
                            'subset': train_test_mapping[option],
                            'annotations': {'label': real_label},
                        }
                        labels_dict[real_label] = 0

                annotations['labels'] = sorted(
                    labels_dict,
                    key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 0,
                )
                with open(save_file_loc, 'w') as f:
                    json.dump(annotations, f, indent=4)

    # ------------------------------------------------------------------
    # Convenience: run all extraction steps at once
    # ------------------------------------------------------------------

    def generate_metadata(self) -> None:
        """Run the full extraction pipeline and populate all metadata dicts."""
        self.get_frame_surgeme()
        self.get_score()
        self.train_test_split_skill_detection()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_pkl(self, path: str | Path | None = None) -> None:
        """Serialise ``metadata_res`` to a pickle file.

        Args:
            path: Destination path.  Falls back to ``self.res_file_loc`` if
                  not provided.  Raises ``ValueError`` if neither is set.
        """
        target = Path(path) if path else self.res_file_loc
        if target is None:
            raise ValueError(
                "No output path specified.  Pass a path to save_to_pkl() "
                "or set res_file_loc in the constructor."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'wb') as f:
            pickle.dump(dict(self.metadata_res), f)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def print_test(self) -> None:
        """Print the start frame of Suturing_E004 (for quick sanity-checks)."""
        print(self.metaData["Suturing_E004"][0])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

'''
if __name__ == "__main__":
    test = DataExtract()
    test.get_category()
    test.get_frame_No()
    test.get_score()
    test.test_sum()
    test.get_frame_surgeme()
    test.get_hmm_states()
    test.get_index()
'''
