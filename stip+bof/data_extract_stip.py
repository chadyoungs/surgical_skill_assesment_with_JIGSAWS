#! /usr/bin/env python3
"""
Data extraction utilities for the STIP + BoF surgical skill pipeline.

Reads JIGSAWS metadata (scores, transcriptions, train/test splits) and
exposes per-trial index lookups used by the clustering and classification
scripts.
"""
from __future__ import annotations

from pathlib import Path
from glob import glob

import Global_Var

# ---------------------------------------------------------------------------
# Module-level path configuration
# ---------------------------------------------------------------------------

TASK_LIST: list[str] = ["Suturing", "Knot_Tying", "Needle_Passing"]
TASK_SYMBOL: int = Global_Var.TASK_SYMBOL
TASK: str = TASK_LIST[TASK_SYMBOL]

# Paths are relative to the working directory (one level above this file).
# root_path must point to the SuperTrialOut folder.
_BASE = Path(".")
root_path: str = str(
    _BASE / "Experimental_setup" / TASK / "unBalanced" / "GestureClassification" / "SuperTrialOut"
)
root_path_score: str = str(_BASE / "da_vici_data_with_iDT_features" / TASK)
root_path_trans: str = str(_BASE / "da_vici_data_with_iDT_features" / TASK / "transcriptions")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _surgery_name_from_stem(stem: str) -> str:
    """Extract surgery name by dropping the last two ``_``-separated tokens."""
    return "_".join(stem.split("_")[:-2])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataExtract:
    """Extract and consolidate JIGSAWS metadata for the STIP + BoF pipeline."""

    def __init__(self) -> None:
        self.metaData: dict = {}
        self.metaData_surgeme: dict = {}
        self.metaData_score: dict = {}
        self.metaData_index: dict = {}
        self.metaData_hmmstates: dict = {}
        self.metaData_hmmindex: dict = {}

    # ------------------------------------------------------------------
    # Category / directory helpers
    # ------------------------------------------------------------------

    def get_category(self) -> None:
        """Retrieve subdirectories inside *root_path* (one per LOSO fold)."""
        self.category: list[str] = glob(root_path + "/*")
        self.category_abs: list[str] = [Path(p).name for p in self.category]

    # ------------------------------------------------------------------
    # Surgeme / transcription parsing
    # ------------------------------------------------------------------

    def get_frame_surgeme(self) -> None:
        """Parse transcription files and populate ``metaData_surgeme``."""
        for txt_file in Path(root_path_trans).glob("*.txt"):
            surgery_name = txt_file.stem
            list_surgeme: list[str] = []
            list_frame_start: list[str] = []
            list_frame_end: list[str] = []

            for line in txt_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    break
                parts = line.split()
                list_frame_start.append(parts[0])
                list_frame_end.append(parts[1])
                list_surgeme.append(parts[2])

            self.metaData_surgeme[surgery_name] = (list_frame_start, list_frame_end, list_surgeme)

    # ------------------------------------------------------------------
    # HMM state sequence
    # ------------------------------------------------------------------

    def get_hmm_states(self) -> None:
        """Build a per-frame HMM state map into ``metaData_hmmstates``."""
        for txt_file in Path(root_path_trans).glob("*.txt"):
            surgery_name = txt_file.stem
            starts, ends, surgemes = self.metaData_surgeme[surgery_name]
            for i in range(int(ends[-1])):
                frame_no = i + 1
                for j in range(len(starts)):
                    if int(starts[j]) <= frame_no <= int(ends[j]):
                        self.metaData_hmmstates[frame_no] = surgemes[j]

    # ------------------------------------------------------------------
    # Frame number range per trial
    # ------------------------------------------------------------------

    def get_frame_No(self) -> None:  # noqa: N802 – kept for backward compat
        """Populate ``metaData`` with ``(start_frame, end_frame)`` per trial."""
        for count, category_dir in enumerate(self.category):
            txt_files = glob(str(Path(category_dir) / "itr_1" / "*.txt"))
            if count >= 1:
                break
            for file in txt_files:
                with open(file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            break
                        stem = line.split(".")[0]
                        parts = stem.split("_")
                        start_frame = int(parts[-2])
                        end_frame = int(parts[-1])
                        surgery_name = _surgery_name_from_stem(stem)
                        self.metaData[surgery_name] = (start_frame, end_frame)

    # ------------------------------------------------------------------
    # Score / skill level
    # ------------------------------------------------------------------

    def get_score(self) -> tuple[str, str]:
        """Parse the meta-file, populate ``metaData_score``, and return the
        names of the two highest-scoring trials (used for BoF training).
        """
        meta_file = Path(root_path_score) / f"meta_file_{TASK}.txt"
        score_list: list[int] = []
        surgery_list: list[str] = []

        for line in meta_file.read_text().splitlines():
            line = line.strip()
            if not line:
                break
            parts = line.split()
            surgery_name = parts[0]
            surgery_score = int(parts[2])
            surgery_list.append(surgery_name)
            score_list.append(surgery_score)
            self.metaData_score[surgery_name] = (surgery_score, self.skill_level(surgery_name, surgery_score))

        score_list_sorted = sorted(score_list)
        max_1 = score_list_sorted[-1]
        max_2 = score_list_sorted[-2]
        return surgery_list[score_list.index(max_1)], surgery_list[score_list.index(max_2)]

    def skill_level(self, surgery_name: str, score: int) -> int:
        """Return ``0`` (novice) or ``2`` (expert) for *surgery_name* and *score*."""
        if "Knot_Tying" in surgery_name or "Needle_Passing" in surgery_name:
            return 0 if score <= 15 else 2
        # Suturing
        return 0 if score <= 19 else 2

    # ------------------------------------------------------------------
    # Index helpers (HMM / ML)
    # ------------------------------------------------------------------

    def get_index(self) -> None:
        """Populate ``metaData_hmmindex`` with novice/expert index lists."""
        meta_file = Path(root_path_score) / f"meta_file_{TASK}.txt"
        novice_list: list[int] = []
        expert_list: list[int] = []
        for count, line in enumerate(meta_file.read_text().splitlines()):
            line = line.strip()
            if not line:
                break
            surgery_name = line.split()[0]
            if self.metaData_score[surgery_name][1] == 0:
                novice_list.append(count)
            else:
                expert_list.append(count)
        self.metaData_hmmindex["novice"] = novice_list
        self.metaData_hmmindex["expert"] = expert_list

    def get_txt_index(self) -> None:
        """Populate ``metaData_index`` mapping surgery name → row index."""
        meta_file = Path(root_path_score) / f"meta_file_{TASK}.txt"
        for count, line in enumerate(meta_file.read_text().splitlines()):
            line = line.strip()
            if not line:
                break
            self.metaData_index[line.split()[0]] = count

    # ------------------------------------------------------------------
    # Train / test split helpers (LOSO)
    # ------------------------------------------------------------------

    def _read_split_file(self, split_file: str) -> list[str]:
        """Parse a ``Train.txt`` or ``Test.txt`` split file and return
        unique surgery names, preserving original order.
        """
        seen: set[str] = set()
        names: list[str] = []
        with open(split_file) as f:
            for line in f:
                stem = line.split()[0]
                name = _surgery_name_from_stem(stem)
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def train_sum(self) -> list[list[str]]:
        """Return per-fold lists of training surgery names (LOSO)."""
        return [
            self._read_split_file(str(Path(c) / "itr_1" / "Train.txt"))
            for c in self.category
        ]

    def test_sum(self) -> list[list[str]]:
        """Return per-fold lists of test surgery names (LOSO)."""
        return [
            self._read_split_file(str(Path(c) / "itr_1" / "Test.txt"))
            for c in self.category
        ]
