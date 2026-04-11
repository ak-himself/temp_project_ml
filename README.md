# ASL Static Gesture Interpreter

This repository is developed in staged execution.

Current approved implementation focus:
- Step 1 (capture and dataset tooling)

## Step 1 Final Design

Each captured sample writes two artifacts with strict 1:1 sample mapping:

1. RGB image (for CNN)
- Source: strict ROI crop only
- Resize: configurable (default 128x128)
- Path: data/raw/rgb/<CLASS>/imgXXXXXX.png

2. Feature vector (for tabular models)
- 117D order:
  - 63 landmarks
  - 15 joint angles
  - 30 pairwise distances
  - 5 finger states (0/1/2)
  - 4 palm direction one-hot (camera/face/up/down)
- Path: data/raw/landmarks/<CLASS>/lmXXXXXX.npy

Sample identity:
- Shared numeric sample_id across both files:
  - img000123.png <-> lm000123.npy

## Removed From Step 1

- subject_id usage
- subject-based logging
- skeleton image saving to dataset

Note:
- Skeleton rendering is used for live preview/debug overlay only.

## Capture Log

Path:
- logs/capture_log.csv

Columns:
- timestamp
- sample_id
- label
- rgb_path
- feature_path
- finger_states
- palm_direction

## Finger Analysis Mode

Command:

```bash
python capture.py --analyze-fingers
```

Purpose:
- inspect geometric finger parameters for threshold tuning later
- no calibration logic
- no class labeling

Overlay includes:
- per-finger angle
- per-finger tip-to-base ratio
- palm orientation (camera/face/up/down/left/right)
- FPS
- ROI status

Logging in analysis mode:
- press k to append one row to logs/finger_analysis.csv
- press q to quit

finger_analysis.csv columns:
- timestamp
- palm_orientation
- 5 angles
- 5 ratios

Note:
- Analysis mode does not classify finger states and does not apply threshold decisions.
- It is only for collecting statistics before threshold design.

## Palm Direction Feature

4-class feature in 117D vector:
- camera
- face
- up
- down

Computed from palm normal:
- v1 = index_mcp - wrist
- v2 = pinky_mcp - wrist
- normal = cross(v1, v2)

Extended 6-class orientation is used in analysis overlay/logging:
- camera, face, up, down, left, right

## Commands

Capture one class:

```bash
python capture.py --gesture A --target 1200
```

Run validation checks only:

```bash
python capture.py --audit --target 1200
```

Run preview + validation:

```bash
python capture.py --preview --target 1200
python capture.py --preview --target 1200 --no-preview-window
```

Run live stacked prediction (Step 3):

```bash
python capture.py --predict-stacked
```

Run live hierarchical prediction (Step 4):

```bash
python capture.py --predict-hierarchy
```

Run live final stabilized prediction (Step 5):

```bash
python capture.py --predict-final
```

Run final operator UI (Step 6):

```bash
python capture.py --ui-final
```

Step 6 controls:
- `space`: pause/resume live inference
- `c`: clear stable output state
- `q`: quit

Step 5 output behavior:
- `RAW` is direct Step 4 hierarchy output
- `SMOOTH` is temporal-window aggregation
- `FINAL` is debounced stable output for UI/use

Step 4 output behavior:
- `FINAL` line is produced by hierarchy policy
- `Reason` explains which rule accepted/rejected prediction
- `UNCERTAIN` means confidence/support criteria were not met

## Validation Tool Coverage

Validation verifies:
- RGB <-> feature pairing integrity
- per-class counts
- missing sample pairs
- finger-state distribution
- palm-direction distribution
- feature dimension integrity (expects 117)

Outputs:
- logs/dataset_validation.json
- logs/dataset_validation.md
- logs/dataset_preview.jpg

## Keys In Capture Mode

- s: start/pause auto capture
- q: quit

## Configuration

Step 1 tunables are in config.py, including:
- camera/ROI settings
- image size
- quality thresholds
- feature dimensions
- placeholder finger-state thresholds for future manual tuning
- Step 4 hierarchy thresholds:
  - stack high-confidence acceptance
  - stack margin acceptance
  - base-model agreement minimum
  - base agreement confidence minimum
  - stack-support fallback thresholds
- Step 5 temporal stabilization thresholds:
  - window size and minimum frames
  - smoothing acceptance confidence
  - debounce frame count
  - stable-label timeout frames

## Active Class Set

Current classes are 26 alphabets only:
- A-Z
