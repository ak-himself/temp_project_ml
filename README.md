# ASL Static Gesture Interpreter

This README is a strict step-by-step runbook for training and testing the full pipeline.

## Project Status

- Class set: 26 classes (`A-Z`)
- Feature vector: 119D (`63 + 15 + 30 + 5 + 6`)
- Single-hand only pipeline (no two-hand mode)

## Prerequisites

From project root:

```bash
pyenv shell 3.12.7
pip install -r requirements.txt
```

Quick sanity check:

```bash
./app --help
python training/train_all.py --dry-run --model all
```

## Latest Run Commands (15 Apr 2026)

Use these from project root with pyenv Python 3.12.7:

```bash
pyenv shell 3.12.7
./app --help
./app --ui-final
python -m py_compile src/asl_app/capture_app.py src/asl_app/live_ui.py
```

Single launcher command for all runtime modes:

```bash
./app --ui-final
```

## Command Index (Updated)

Use these exact commands from project root.

```bash
# 1) Capture one class (repeat for A-Z)
./app --gesture A --target 1200

# 2) Audit dataset
./app --audit --target 1200

# 3) Build dataset preview image/report
./app --preview --target 1200

# 4) Finger analysis mode
./app --analyze-fingers

# 5) Train base models
python training/train_xgboost.py
python training/train_rf.py
python training/train_mlp.py
python training/train_cnn.py --epochs 40 --image-size 224 --batch-size 128

# 6) Train all base models in one command
python training/train_all.py --model all --epochs 40 --image-size 224 --batch-size 128

# 7) Live base-model prediction view
./app --predict-models

# 8) Train stacking model
python training/train_stacking.py

# 9) Live stacked prediction view
./app --predict-stacked

# 10) Live hierarchical decision view
./app --predict-hierarchy

# 11) Live temporal-stabilized view
./app --predict-final

# 12) Final clean UI
./app --ui-final

# 13) Evaluate final stacked model (matrices + F1 + MSE + more)
python training/evaluate_final.py --out-prefix final_model_eval

```

## Step 1: Data Capture and Validation

1. Capture class data (repeat per class)

```bash
./app --gesture A --target 1200
./app --gesture B --target 1200
# ... continue for all classes up to Z
```

2. Validate dataset integrity

```bash
./app --audit --target 1200
```

3. Optional visual preview report

```bash
./app --preview --target 1200
```

Step 1 outputs:
- `data/rgb/<CLASS>/imgXXXXXX.png`
- `data/landmarks/<CLASS>/lmXXXXXX.npy`
- `logs/capture_log.csv`
- `logs/dataset_validation.json`
- `logs/dataset_validation.md`

## Step 2: Train Base Models

1. Train tabular models

```bash
python training/train_xgboost.py
python training/train_rf.py
python training/train_mlp.py
```

2. Train CNN (powerful machine recommended)

```bash
python training/train_cnn.py --epochs 40 --image-size 224 --batch-size 128
```

3. Optional one-command trainer for all Step 2 models

```bash
python training/train_all.py --model all --epochs 40 --image-size 224 --batch-size 128
```

4. Test Step 2 live model view

```bash
./app --predict-models
```

## Step 3: Train and Test Stacking Model

1. Train stacking meta-model

```bash
python training/train_stacking.py
```

2. Test stacked live prediction

```bash
./app --predict-stacked
```

## Step 4: Hierarchy Layer Test

1. Run hierarchical live prediction

```bash
./app --predict-hierarchy
```

Step 4 behavior:
- `FINAL`: hierarchy decision output
- `Reason`: acceptance/rejection rule used
- `UNCERTAIN`: no rule accepted the frame

## Step 5: Temporal Stabilization Test

1. Run temporal-stabilized live prediction

```bash
./app --predict-final
```

Step 5 behavior:
- `RAW`: direct Step 4 output
- `SMOOTH`: windowed temporal smoothing
- `FINAL`: debounced stable output

## Step 6: Final Operator UI

1. Run final clean UI mode

```bash
./app --ui-final
```

Controls:
- `space`: pause/resume inference
- `c`: clear stable state
- `q`: quit

## Re-Training After Replacing CNN

If you retrain CNN on another machine and replace `models/cnn.h5`, re-run Step 3:

```bash
python training/train_stacking.py
```

Then re-test:

```bash
./app --predict-stacked
./app --predict-hierarchy
./app --predict-final
./app --ui-final
```

## Step 7: Final Model Evaluation

Run full evaluation for the final stacked model:

```bash
python training/evaluate_final.py --out-prefix final_model_eval
# optional faster smoke run on large datasets:
python training/evaluate_final.py --out-prefix final_model_eval_quick --max-per-class 100
```

This generates:
- `logs/final_model_eval.json` (accuracy, precision/recall, F1, balanced accuracy, log loss, MSE/RMSE/MAE, report)
- `logs/final_model_eval_confusion_train.png`
- `logs/final_model_eval_confusion_val.png`
- `logs/final_model_eval_confusion_test.png`

## Key Artifacts

- Base models: `models/xgboost.pkl`, `models/rf.pkl`, `models/mlp.pkl`, `models/cnn.h5`
- Stacking model: `models/stack_meta.pkl`
- Stacking config: `models/stacking_config.json`
- Metrics: `logs/*_metrics.json`
- Final evaluation: `logs/final_model_eval.json` + `logs/final_model_eval_confusion_*.png`

## How It Works

1. Capture phase:
- A hand inside ROI is detected from webcam frames.
- For each accepted frame, two artifacts are saved using the same sample id:
- `data/rgb/<CLASS>/imgXXXXXX.png`
- `data/landmarks/<CLASS>/lmXXXXXX.npy`
- Capture metadata is appended to `logs/capture_log.csv`.

2. Dataset validation phase:
- Audit checks pairing integrity, class coverage, and feature-shape consistency.
- Outputs are written to `logs/dataset_validation.json`, `logs/dataset_validation.md`, and optional `logs/dataset_preview.jpg`.

3. Base model training phase:
- XGBoost, Random Forest, and MLP train on 119D tabular features.
- CNN trains on RGB crops.
- Models are saved under `models/` and metrics JSON files under `logs/`.

4. Stacking phase:
- Base model probabilities are combined and used to train a meta-model.
- Meta-model and config are saved as `models/stack_meta.pkl` and `models/stacking_config.json`.

5. Live inference phase:
- Runtime computes features from current hand frame.
- Depending on mode, it shows base outputs, stacked output, hierarchy decision, temporal smoothing, or final clean UI.

## File Roles

- `app`: single executable root launcher for runtime capture/inference modes.
- `src/asl_app/capture_app.py`: capture-focused runtime implementation (capture, preview, audit, shared helpers, CLI dispatch).
- `src/asl_app/live_ui.py`: live analysis and prediction UI runtime modes.
- `src/asl_app/config.py`: central runtime/training path and threshold configuration.
- `src/asl_app/classes.json`: canonical class list (`A-Z`) and class_count.
- `src/asl_app/skeleton_canvas.py`: skeleton rendering utilities for visualization windows.
- `src/asl_app/training_common.py`: shared training utilities (manifest load, split, evaluation, save helpers).
- `training/train_xgboost.py`: XGBoost training entrypoint.
- `training/train_rf.py`: Random Forest training entrypoint.
- `training/train_mlp.py`: MLP training entrypoint.
- `training/train_cnn.py`: CNN training entrypoint.
- `training/train_stacking.py`: stacking meta-model training entrypoint.
- `training/train_all.py`: orchestrator to train one or all base models.
- `data/rgb/`: captured RGB crops by class.
- `data/landmarks/`: captured 119D feature vectors by class.
- `models/`: serialized trained models and stacking config.
- `logs/`: capture logs, validation reports, and model metrics JSON.
- `requirements.txt`: Python dependencies.
