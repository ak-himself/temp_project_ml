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
```

## Step 1: Data Capture and Validation

1. Capture class data (repeat per class)

```bash
python capture.py --gesture A --target 1200
python capture.py --gesture B --target 1200
# ... continue for all classes up to Z
```

2. Validate dataset integrity

```bash
python capture.py --audit --target 1200
```

3. Optional visual preview report

```bash
python capture.py --preview --target 1200
```

Step 1 outputs:
- `data/raw/rgb/<CLASS>/imgXXXXXX.png`
- `data/raw/landmarks/<CLASS>/lmXXXXXX.npy`
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
python training/step2_train.py --model all --epochs 40 --image-size 224 --batch-size 128
```

4. Test Step 2 live model view

```bash
python capture.py --predict-models
```

## Step 3: Train and Test Stacking Model

1. Train stacking meta-model

```bash
python training/train_stacking.py
```

2. Test stacked live prediction

```bash
python capture.py --predict-stacked
```

## Step 4: Hierarchy Layer Test

1. Run hierarchical live prediction

```bash
python capture.py --predict-hierarchy
```

Step 4 behavior:
- `FINAL`: hierarchy decision output
- `Reason`: acceptance/rejection rule used
- `UNCERTAIN`: no rule accepted the frame

## Step 5: Temporal Stabilization Test

1. Run temporal-stabilized live prediction

```bash
python capture.py --predict-final
```

Step 5 behavior:
- `RAW`: direct Step 4 output
- `SMOOTH`: windowed temporal smoothing
- `FINAL`: debounced stable output

## Step 6: Final Operator UI

1. Run final clean UI mode

```bash
python capture.py --ui-final
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
python capture.py --predict-stacked
python capture.py --predict-hierarchy
python capture.py --predict-final
python capture.py --ui-final
```

## Key Artifacts

- Base models: `models/xgboost.pkl`, `models/rf.pkl`, `models/mlp.pkl`, `models/cnn.h5`
- Stacking model: `models/stack_meta.pkl`
- Stacking config: `models/stacking_config.json`
- Metrics: `logs/step2/*_metrics.json`
