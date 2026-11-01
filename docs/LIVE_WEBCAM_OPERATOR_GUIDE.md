# Live Webcam Interface — Operator Guide

Related Jira work item: ASGI2-14

## Scope of the current implementation

The repository exposes multiple live camera modes through `app.py`:

- `--predict-models`: displays the separate XGBoost, Random Forest, MLP and CNN predictions.
- `--predict-stacked`: displays the stacking meta-model result together with the base-model outputs.
- `--predict-hierarchy`: displays the hierarchical decision result and its fallback reason.
- `--predict-final`: displays the temporal-stabilized prediction with the debug overlay.
- `--ui-final`: displays the reduced final operator view.

All modes accept `--camera <index>`; the default camera index is configured as `0`.

## Launch

From the repository root:

```bash
python app.py --ui-final
```

For the detailed diagnostic overlay:

```bash
python app.py --predict-final
```

Before launch, install the dependencies and confirm the model artifacts expected by the selected mode are present under `models/`. The final/stacked/hierarchical modes require the XGBoost, Random Forest, MLP, CNN and stacking artifacts.

## What the final operator view displays

The current `run_live_final_prediction` implementation:

1. Opens the selected camera and reads frames continuously.
2. Uses MediaPipe Hands to identify hand landmarks.
3. Selects a hand within the configured region of interest.
4. Builds the engineered feature vector and obtains base-model predictions.
5. Combines the base-model outputs through the stacking meta-model.
6. Applies hierarchical confidence decisions and temporal stabilization.
7. Displays the stable prediction label, confidence value, confidence bar, live/paused state, hand status and FPS.

The implementation draws the detected hand landmarks and ROI over the camera frame. The reduced final view suppresses the detailed raw/smoothed/debug text shown by `--predict-final`.

## Keyboard controls in the final UI

| Key | Current behaviour |
|---|---|
| `q` | Quit the live window and release the camera |
| `space` | Pause/resume inference |
| `c` | Clear the stable prediction/history state |

The current final UI code does not implement a persistent notepad buffer, text concatenation, Space-character insertion or Backspace editing. Those acceptance criteria in ASGI2-14 remain open until an actual text-entry interface is implemented and verified. The `space` key currently pauses/resumes inference; it does not insert a space character.

## Operational limitations and verification

- A usable camera must be available at the selected index.
- Required model files must exist and be loadable; the code raises an error when required artifacts are missing.
- No accuracy, CPU latency or real-time frame-rate guarantee is asserted by this guide.
- Verify camera start/stop, missing-hand handling, prediction stability, pause/resume and manual clear on the target machine.
- Record the machine, camera, Python/library versions and actual test observations when preparing release evidence.

This document describes the checked-in implementation; it does not claim the remaining text-entry acceptance criteria are complete.
