# JIGSAWS Surgical Skill Assessment (Refactored)

A unified JIGSAWS surgical-skill assessment repository. The only entry point is the top-level `main.py`; every method shares the same dataset loading, fold handling, metrics and reporting layers. Adding a new method does not require re-implementing metadata parsing, train/test splits or evaluation.

> The original algorithm semantics are preserved: task-specific skill thresholds, HMM forward likelihood, the STIP BoF + SVM grid and the ten kinematic features match the previous implementation. Only the code organisation and runtime interface changed.

## 1. Environment Setup (venv)

Python **3.11 / 3.12** is recommended. `hmmlearn` may not have a wheel for Python 3.14, and source builds can fail with a missing `Python.h`. Python 3.14 still works for dataset inspection, kinematic features, SVC and plotting.

```bash
cd surgical_skill_assesment_with_JIGSAWS

python3.11 -m venv .venv          # or: python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[all]"
```

If only Python 3.14 is available, install extras separately. `pip install -e ".[ml]"` may fail while building `hmmlearn`; start with `.[plot]` and `.[media]`, or use a 3.11/3.12 venv.

Install only what you need:

```bash
pip install -e .            # dataset info/export/splits/annotations, kinematic features
pip install -e ".[media]"   # OpenCV + Pillow: clips / frames / capture / figures
pip install -e ".[ml]"      # scikit-learn + joblib + hmmlearn: SVC / HMM / STIP
pip install -e ".[plot]"    # matplotlib: boxplot / 3D trajectory
pip install -e ".[all]"     # everything
```

`.venv/` is already ignored by Git. All commands below assume the venv is active.

## 2. Verify the Entry Point

> Run every command from the repository root.

```bash
python main.py --help
python main.py methods
```

`methods` lists all registered methods and their sub-commands. `--task` may appear before the method name or after the sub-command:

```bash
python main.py --task Knot_Tying dataset info
python main.py dataset info --task Knot_Tying
```

The default task is `Suturing`. Run `dataset info` first to confirm that the dataset and labels are resolved correctly:

```text
$ python main.py dataset info
Task: Suturing
Data root: /.../da_vici_data_with_iDT_features
Trials: 39 (novice=23, expert=16, kinematics=39)
GRS score range: 8..30
Surgemes (10): G1, G2, G3, G4, G5, G6, G8, G9, G10, G11
Split kinds: GestureClassification, GestureRecognition, SkillDetection
```

## 3. Command and Output Reference

### 3.1 Dataset and Annotations

| Command | Result / output | Requires |
| --- | --- | --- |
| `python main.py dataset info` | Prints trial counts, novice/expert counts, GRS range, surgeme labels and available split kinds | numpy |
| `python main.py dataset info --json` | Same information as JSON on stdout | numpy |
| `python main.py dataset export` | `artifacts/dataset/<Task>/<Task>_metadata.json`: score/grade/tier/segments per trial | numpy |
| `python main.py dataset splits` | `data_root/<Task>/surgeme_classifi_traintestsplit/<scheme>_<fold>_Train.txt` and `..._Test.txt` | numpy |
| `python main.py dataset annotations` | `data_root/<Task>/surgeme_annotation/<scheme>_<fold>.json` in PyTorchCon3D format | numpy |
| `python main.py dataset clips` | `data_root/<Task>/surgeme_video/<Gx>/<trial>_<Gx>_<idx>.avi` | OpenCV |
| `python main.py dataset frames` | `data_root/<Task>/surgeme_img/<Gx>/<clip_stem>/img_00000.jpg ...` | OpenCV |
| `python main.py dataset capture --gesture G5` | `artifacts/dataset/<Task>/captures/G5.jpg` | OpenCV |

Notes:

- `splits`, `annotations`, `clips` and `frames` write **inside the dataset directory** because external PyTorchCon3D/C3D training code reads those relative paths. `export` and `capture` write under `artifacts/`.
- Every command accepts `--output-dir` / `--output` to override the default location.
- `splits` defaults to `--kind GestureClassification --scheme SuperTrialOut`. `clips` keeps existing clips unless `--overwrite` is passed.

### 3.2 Kinematic Features, SVM and Box Plots

| Command | Result / output | Requires |
| --- | --- | --- |
| `python main.py kinematic features` | npy/csv/json files under `artifacts/kinematic/<Task>/features/` | numpy |
| `python main.py kinematic svc` | `artifacts/kinematic/<Task>/svc_report.json`; prints per-fold accuracy and the best parameters | scikit-learn |
| `python main.py kinematic svc-csv` | `artifacts/kinematic/<Task>/svc_csv_report.json`; prints train/test accuracy | scikit-learn |
| `python main.py kinematic boxplot` | `artifacts/figures/<Task>/boxplot_mtf.png` and `boxplot_psm.png` | matplotlib |

Full `kinematic features` output:

```text
artifacts/kinematic/<Task>/features/
├── <Task>_feature.npy       # (n_trials, 4, 10): four instruments x ten features
├── <Task>_grs.npy           # (n_trials, 1): 0 = novice, 2 = expert
├── <Task>_trials.npy        # trial names aligned with feature rows
├── data_for_SVM.csv         # flattened feature table, class in the last column
└── features_report.json     # trial count, shape, output file list
```

Instrument order is fixed as `MTF_L, MTF_R, PSM_1, PSM_2`. Each instrument has ten values:

```text
time_sum, displacement_sum, velocity_mean, velocity_variance,
curvature_mean, curvature_variance, smoothness_mean, smoothness_variance,
turning_angle_mean, turning_angle_variance
```

Task-specific feature subsets:

| Task | Features used | Dimension |
| --- | --- | --- |
| Suturing | time + velocity + smoothness + turning angle | 28 |
| Knot_Tying | time + velocity + smoothness | 20 |
| Needle_Passing | velocity + curvature + turning angle | 24 |

`svc` behaviour:

- It first looks for the five `SkillDetection/SuperTrialOut` folds. If unavailable, it falls back to the five `GestureClassification` folds, which is the case for Knot_Tying and Needle_Passing.
- It runs LOSO over the `gamma` / `C` grid `0.001,0.01,0.1,1,10,100` and selects parameters by mean test accuracy.
- If `artifacts/kinematic/<Task>/features` is absent, it falls back to the legacy `kinematic_analysis/Machine_learning_basedonfeatures/<Task>_feature.npy`. Use `--features-dir` to override.

Typical sequence:

```bash
python main.py kinematic features
python main.py kinematic svc
python main.py kinematic boxplot
```

### 3.3 HMM

```bash
python main.py hmm clusters --clips-dir /path/to/Data_clips/<Task>_clips/kinetic
python main.py hmm train
python main.py hmm evaluate
```

| Command | Result / output |
| --- | --- |
| `hmm clusters` | `artifacts/hmm/<Task>/clusters/<Gx>.pkl` (one PCA + KMeans vocabulary per surgeme) plus `clusters_report.json` |
| `hmm train` | `artifacts/hmm/<Task>/models/<task>_expert_model.pkl`, `<task>_novice_model.pkl`, `models/training_report.json`; `artifacts/hmm/<Task>/observations/<task>_observations.pkl` |
| `hmm evaluate` | `artifacts/hmm/<Task>/evaluation.json`: overall accuracy, expert/novice accuracy, per-trial predictions and likelihoods |

`hmm clusters --clips-dir` must point to per-surgeme kinematic text clips whose filenames contain a label such as `_G1_`. If legacy `HMM_model/observation_clusters/<Task>` or `HMM_model/models` exist, `train` / `evaluate` use them as a fallback automatically. `--clusters-dir` / `--models-dir` override this.

Typical output:

```text
INFO: Trained HMMs on 39 trials (expert=16, novice=23)
INFO: accuracy = 0.7949 | expert_accuracy = 0.8750 | novice_accuracy = 0.7391
INFO: Wrote evaluation report: artifacts/hmm/Suturing/evaluation.json
```

### 3.4 STIP + BoF

```bash
python main.py stipbof cluster --features-dir /path/to/stip/features/<Task>
python main.py stipbof classify
```

| Command | Result / output |
| --- | --- |
| `stipbof cluster` | `artifacts/stipbof/<Task>/time_series/clusters/<scheme>_<fold>_{hog,hof}.pkl`; `time_series/<scheme>_<fold>_histograms.pkl`; `time_series/clustering_report.json` |
| `stipbof classify` | `artifacts/stipbof/<Task>/classification.json`: per-fold accuracy, best gamma/C and grid history |

- Each fold fits HOG/HOF vocabularies using only the training trials of that fold, then builds histograms for every trial, preventing test-fold leakage.
- Histograms are aggregated per trial so each SVM sample is one trial (HOG + HOF concatenated).
- Defaults are `--kind GestureClassification`, `--clusters 400`, `--max-samples 100000`.
- STIP filenames must be resolvable by `_trial_of()`. Both `Suturing_B001.txt` and `Suturing_B001_000080_005635.txt` are supported.

### 3.5 3D Trajectories and Figures

| Command | Result / output | Requires |
| --- | --- | --- |
| `python main.py trajectory plot` | `artifacts/figures/<Task>/3d_trajectory_left_hand.png` plus a JSON sidecar | matplotlib |
| `python main.py trajectory plot --hand right` | `artifacts/figures/<Task>/3d_trajectory_right_hand.png` | matplotlib |
| `python main.py trajectory plot --tip psm1` | `artifacts/figures/<Task>/3d_trajectory_psm1.png` | matplotlib |
| `python main.py trajectory plot --tip psm2` | `artifacts/figures/<Task>/3d_trajectory_psm2.png` | matplotlib |
| `python main.py figures stitch a.png b.png --output out.jpg` | Horizontal composite; without `--output`, writes `artifacts/figures/<Task>/image_stitch.jpg` | Pillow |
| `python main.py figures resize in.jpg --width 1280 --height 720 --output out.jpg` | Resized image; without `--output`, writes `<stem>_<width>x<height>.jpg` next to the input | Pillow |

`trajectory plot` uses the highest-GRS trial as the expert and the lowest-GRS trial as the novice. Both can be overridden:

```bash
python main.py trajectory plot --tip psm1 \
  --expert Suturing_D001 \
  --novice Suturing_B001
```

The JSON sidecar records the actual expert/novice trials, their scores, the instrument and the figure path.

## 4. Paths and Configuration

All paths are resolved once at startup. They are auto-detected when possible and can be overridden by CLI options or environment variables:

| Option | Environment variable | Default / detection rule |
| --- | --- | --- |
| `--task` | — | `Suturing` |
| `--data-root` | `JIGSAWS_DATA_ROOT` | `da_vici_data_with_iDT_features`; falls back to `kinematic_analysis/trajectory_plot/da_vici_data_with_iDT_features` |
| `--experimental-setup` | `JIGSAWS_EXPERIMENTAL_SETUP` | `data_root/Experimental_setup` or `data_root.parent/Experimental_setup` |
| `--artifacts-dir` | `JIGSAWS_ARTIFACTS` | `artifacts/` |
| `--stip-features` | `JIGSAWS_STIP_FEATURES` | `stip+dct_code/raw_data` |
| `--kinematic-features` | `JIGSAWS_KINEMATIC_FEATURES` | `kinematic_analysis/Machine_learning_basedonfeatures` |
| `--trajectory-hand` | `JIGSAWS_TRAJECTORY_HAND` | `left` |

Example:

```bash
python main.py \
  --data-root /data/JIGSAWS/da_vici_data_with_iDT_features \
  --artifacts-dir /tmp/jigsaws-artifacts \
  --task Needle_Passing \
  kinematic features
```

## 5. Typical Workflows

### 5.1 Kinematic Assessment and Figures Only

```bash
python main.py dataset info
python main.py kinematic features
python main.py kinematic svc
python main.py kinematic boxplot
python main.py trajectory plot --tip mtm --hand left
python main.py trajectory plot --tip mtm --hand right
python main.py trajectory plot --tip psm1
```

Outputs:

```text
artifacts/kinematic/Suturing/features/*.npy
artifacts/kinematic/Suturing/features/data_for_SVM.csv
artifacts/kinematic/Suturing/svc_report.json
artifacts/figures/Suturing/boxplot_mtf.png
artifacts/figures/Suturing/boxplot_psm.png
artifacts/figures/Suturing/3d_trajectory_left_hand.png
artifacts/figures/Suturing/3d_trajectory_right_hand.png
artifacts/figures/Suturing/3d_trajectory_psm1.png
```

### 5.2 HMM

```bash
python main.py hmm clusters --clips-dir /path/to/Data_clips/Suturing_clips/kinetic
python main.py hmm train
python main.py hmm evaluate
```

### 5.3 STIP + BoF

```bash
python main.py stipbof cluster --features-dir /path/to/stip/raw_data/Suturing
python main.py stipbof classify
```

### 5.4 Dataset Annotations and Video Clips

```bash
python main.py dataset export
python main.py dataset splits
python main.py dataset annotations
python main.py dataset clips
python main.py dataset frames
```

## 6. Adding a New Method

Create `method.py` and an empty `__init__.py` under `jigsaws/methods/<name>/`; the CLI discovers it automatically:

```python
# jigsaws/methods/my_method/method.py
from ..base import Command, Context, Method

class MyMethod(Method):
    name = "mymethod"
    summary = "One-line description shown by 'main.py methods'"

    def commands(self):
        return (Command("run", "Run my method", self._run, self._configure),)

    @staticmethod
    def _configure(parser):
        parser.add_argument("--output", default=None)

    def _run(self, ctx: Context, args) -> None:
        # Metadata, labels and folds are already resolved by ctx.dataset.
        for fold in ctx.dataset.folds(kind="SkillDetection"):
            ...
        ctx.artifact_dir(self.name)  # -> artifacts/mymethod/<Task>/

METHOD = MyMethod()
```

```python
# jigsaws/methods/my_method/__init__.py
from __future__ import annotations
```

Reusable pieces:

- `ctx.dataset.records` / `grade()` / `tier()` / `folds()` / `resolve_split_kind()`
- `jigsaws.evaluation.grid_search_loso` for a global feature matrix
- `jigsaws.evaluation.grid_search_loso_per_fold` for per-fold features such as BoF vocabularies
- `jigsaws.evaluation.BinaryMetrics` / `SearchResult`
- `ctx.artifact_dir(...)` for a consistent output directory

Optional dependencies are imported only when a command actually runs:

```python
from ...deps import require
svc_cls = require("sklearn.svm", extra="ml").SVC
```

## 7. Legacy Command Compatibility

Existing shell scripts keep working through automatic command translation:

| Legacy command | New command |
| --- | --- |
| `helper generate_metadata` | `dataset info` |
| `helper generate_gesture_clips` | `dataset clips` |
| `hmm train_observations` | `hmm clusters` |
| `hmm train_model` | `hmm train` |
| `hmm test` | `hmm evaluate` |
| `stip cluster` / `stip classify` | `stipbof cluster` / `stipbof classify` |
| `kinematic feature_classify` | `kinematic svc` |
| `kinematic box_plot` | `kinematic boxplot` |
| `kinematic box_classify` | `kinematic svc-csv` |
| `kinematic trajectory_plot` | `trajectory plot` |

`helper image_stitch` is intentionally not mapped: the old implementation produced an empty canvas. Use `figures stitch` instead.

## 8. Repository Layout

```text
main.py                     # thin top-level entry point
jigsaws/
├── config.py               # Task / Hand / Settings, paths and thresholds
├── dataset/                # metadata, folds, annotations, video clipping
├── evaluation/             # metrics, LOSO grid search, reports
├── kinematics.py           # ten kinematic features (vectorised)
├── images.py               # stitching and resizing
├── methods/                # auto-discovered plugins
│   ├── dataset/ figures/
│   ├── hmm/ stipbof/
│   └── kinematic/ trajectory/
└── cli.py                  # argument parsing, legacy mapping, error output
tests/
└── test_core.py            # threshold, metrics, dataset and feature regressions
```

## 9. Tests

```bash
python -m unittest discover -s tests -v
```

The tests require only numpy and the local dataset; scikit-learn, hmmlearn and matplotlib are not needed. Coverage includes:

- task-specific binary thresholds and the legacy three-tier label
- confusion-matrix metrics
- Suturing metadata counts and labels
- SkillDetection fold integrity
- feature tensor shape
- numerical regression for `Suturing_B001` MTF_L features
