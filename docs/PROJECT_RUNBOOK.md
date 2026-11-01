# Project Runbook — ASL Static Gesture Interpreter

Related Jira work item: ASGI2-17

## Project scope

This repository contains an end-to-end static ASL hand-gesture classification workflow for 26 alphabet classes (A–Z). The documented pipeline covers dataset capture and validation, feature extraction, base-model training, stacking, live inference modes, and evaluation.

This runbook records the commands and interfaces currently documented in the repository. It is not a test report; run the checks in the relevant environment before treating a stage as verified.

## Environment

- Python: the repository README specifies the pyenv-managed Python 3.12.8 interpreter.
- Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

- Basic CLI checks:

```bash
python app.py --help
python training/train_all.py --dry-run --model all
```

The README notes that TensorFlow may print a CUDA-driver warning on systems without a matching CUDA runtime; CPU execution may still be used. No inference-latency figure is asserted here.

## Data and feature pipeline

1. Capture one gesture class at a time:

```bash
python app.py --gesture A --target 1200
```

Repeat for the required classes. The documented class set is A–Z (26 classes).

2. Audit and preview the captured dataset:

```bash
python app.py --audit --target 1200
python app.py --preview --target 1200
```

3. The README documents MediaPipe Hands with 21 keypoints per hand and (x, y, z) coordinates, normalized relative to the wrist and hand scale.

4. The documented engineered vector is 119 dimensions:
   - Landmark coordinates: 63
   - Joint angles: 15
   - Pairwise distances: 30
   - Finger states: 5
   - Palm-orientation one-hot features: 6

Dataset locations documented by the repository:
- RGB crops: `data/rgb/<CLASS>/`
- Landmark vectors: `data/landmarks/<CLASS>/`

## Model training and inference

Train the documented base models:

```bash
python training/train_xgboost.py
python training/train_rf.py
python training/train_mlp.py
python training/train_cnn.py --epochs 40 --image-size 224 --batch-size 128
```

Or invoke the combined base-model training script:

```bash
python training/train_all.py --model all --epochs 40 --image-size 224 --batch-size 128
```

Train the stacking meta-model:

```bash
python training/train_stacking.py
```

The README describes a logistic-regression meta-classifier over outputs from XGBoost, Random Forest, MLP and CNN.

Available live modes documented by the CLI:
- `--predict-models`
- `--predict-stacked`
- `--predict-hierarchy`
- `--predict-final`
- `--ui-final`

Example final UI launch:

```bash
python app.py --ui-final
```

## Evaluation

Run final model evaluation:

```bash
python training/evaluate_final.py --out-prefix final_model_eval
```

The README lists these expected output paths:
- `logs/final_model_eval.json`
- `logs/final_model_eval_confusion_train.png`
- `logs/final_model_eval_confusion_val.png`
- `logs/final_model_eval_confusion_test.png`

The evaluator is documented as producing accuracy, balanced accuracy, macro/weighted precision, recall and F1, top-3 accuracy, log loss, error metrics, a classification report and confusion matrices. Actual values must be taken from a completed evaluation run; this document does not invent them.

## CNN input contract — alignment note

The current repository README describes the CNN branch as consuming resized RGB hand crops. The ASGI2 Jira backlog separately contains ASGI2-10, titled “Train skeleton-image CNN branch.” These are not equivalent input contracts. Before declaring ASGI2-10 implemented, reconcile the Jira design with the actual training code and model artifacts, then update the relevant documentation with the verified implementation. This runbook intentionally preserves the README's current description instead of claiming the skeleton-image branch already exists.

## Final demonstration sequence

A reproducible demonstration can follow this order, provided the required data and model artifacts are present:

1. Show the project CLI help.
2. Show dataset audit/preview output.
3. Start the final live UI with `python app.py --ui-final`.
4. Demonstrate supported A–Z gestures using a working camera.
5. Run final evaluation and show the generated metrics/artifacts.

Record the machine, Python/library versions, dataset revision, model artifact identifiers and actual command output alongside any reported measurements.
