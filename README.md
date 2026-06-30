# surgical_skill_assesment_with_JIGSAWS

This repository contains several JIGSAWS skill-assessment pipelines:

- HMM-based assessment (`HMM_model`)
- STIP + bag-of-features classification (`stip+bof`)
- Kinematic analysis (`kinematic_analysis`)
- Dataset preparation helpers (`helper`)

## Unified entrypoint

Use the repository-level CLI instead of running scripts by hand from different folders.

```bash
python main.py --task Suturing <family> <method>
```

### Families and methods

```bash
# helper utilities
python main.py helper generate_metadata
python main.py helper generate_gesture_clips
python main.py helper image_stitch

# HMM pipeline
python main.py hmm train_observations
python main.py hmm train_model
python main.py hmm test

# STIP + BoF pipeline
python main.py stip cluster
python main.py stip classify

# kinematic analysis
python main.py kinematic feature_classify
python main.py kinematic box_plot
python main.py kinematic box_classify
python main.py kinematic trajectory_plot
```

## Shared runtime configuration

The unified CLI sets these runtime values for all supported methods:

- `JIGSAWS_TASK`: `Suturing`, `Knot_Tying`, or `Needle_Passing`
- `JIGSAWS_DATA_ROOT`: dataset root path
- `JIGSAWS_BOX_PLOT_DATA`: box-plot CSV directory

Optional overrides:

```bash
python main.py \
  --task Knot_Tying \
  --data-root /absolute/path/to/da_vici_data_with_iDT_features \
  helper generate_metadata
```

## Notes

- The refactor keeps the existing method scripts in place.
- Task selection is now shared instead of being hardcoded separately in each method.
- Path handling is now based on runtime configuration instead of manual source edits.
