# JIGSAWS Dataset Helper

A collection of Python utilities for working with the [JIGSAWS surgical activity dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/).

## Supported Tasks

- `Suturing`
- `Knot_Tying`
- `Needle_Passing`

## File Overview

| File | Purpose |
|---|---|
| `constant.py` | Global constants (dataset root path, task name, image dimensions) |
| `exception.py` | Custom exception types |
| `main.py` | CLI entry-point |
| `tools.py` | Frame-capture and image-stitching utilities |
| `metadata_generation.py` | `MetaData` class – loads transcriptions, scores, and train/test splits |
| `surgeme_generation.py` | Splits full-trial videos into per-surgeme clip files |

## Usage

Run from the repository root (the dataset directory must be a sibling of this folder):

```bash
# Generate metadata (scores + train/test splits)
python main.py --task Suturing --option generate_metadata

# Split trial videos into per-surgeme clips
python main.py --task Suturing --option generate_gesture_clips

# Stitch images side-by-side
python main.py --task Suturing --option image_stitch
```

### Selecting a task

The active task is set via the `TASK_CHOICE` constant in `constant.py`
(index into `TASK_LIST`).  The `--task` CLI argument is accepted for
forward-compatibility but is not yet wired into the runtime constants.

## Expected Directory Structure

```
<dataset_root>/
├── Suturing/
│   ├── meta_file_Suturing.txt
│   ├── transcriptions/
│   ├── video/
│   ├── surgeme_video/        # created by generate_gesture_clips
│   └── ...
├── Experimental_setup/
│   └── Suturing/
│       └── unBalanced/
│           ├── SkillDetection/
│           └── GestureClassification/
└── JIGSAWS_dataset_helper/   # this repo
```
