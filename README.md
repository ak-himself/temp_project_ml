# ASL Static Gesture Interpreter

End-to-end ASL (A-Z) static hand-gesture project with:
- Dataset capture + validation
- Multiple base classifiers (XGBoost, Random Forest, MLP, CNN)
- Stacking meta-model
- Live inference modes (base, stacked, hierarchy, temporal, final UI)
- Final evaluation with confusion-matrix diagrams and full metrics

## 1. What This Project Does

Input:
- Webcam stream (single hand in ROI)

Outputs:
- Captured RGB hand crops (`data/rgb/<CLASS>/imgXXXXXX.png`)
- Captured feature vectors (`data/landmarks/<CLASS>/lmXXXXXX.npy`)
- Trained models in `models/`
- Validation reports and metrics in `logs/`
- Confusion matrix diagrams (`.png`) for final model evaluation

Class set:
- 26 classes: `A` to `Z`

## 2. Environment Setup

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

## 3. Main Entrypoints

Primary runtime launcher:

```bash
./app
```

Python launcher equivalent:

```bash
python app.py
```

Example:

```bash
./app --ui-final
```

## 4. End-to-End Workflow (All Steps)

### Step 1: Capture Dataset

Capture one class (repeat for all A-Z):

```bash
./app --gesture A --target 1200
```

Optional capture tuning:

```bash
./app --gesture A --target 1200 --camera 0 --interval 0.01
```

### Step 1.1: Audit and Preview

```bash
./app --audit --target 1200
./app --preview --target 1200
```

### Step 1.2: Finger Analysis Mode

```bash
./app --analyze-fingers
```

### Step 2: Train Base Models

```bash
python training/train_xgboost.py
python training/train_rf.py
python training/train_mlp.py
python training/train_cnn.py --epochs 40 --image-size 224 --batch-size 128
```

Or run all base models with one command:

```bash
python training/train_all.py --model all --epochs 40 --image-size 224 --batch-size 128
```

### Step 3: Train Stacking Meta-Model

```bash
python training/train_stacking.py
```

### Step 4-6: Live Inference Modes

Base-model live predictions:

```bash
./app --predict-models
```

Stacked-model live predictions:

```bash
./app --predict-stacked
```

Hierarchical decision live mode:

```bash
./app --predict-hierarchy
```

Temporal stabilized live mode:

```bash
./app --predict-final
```

Final clean UI mode:

```bash
./app --ui-final
```

### Step 7: Final Model Evaluation (Metrics + Diagrams)

Full evaluation:

```bash
python training/evaluate_final.py --out-prefix final_model_eval
```

Faster subset evaluation (large datasets):

```bash
python training/evaluate_final.py --out-prefix final_model_eval_quick --max-per-class 100
```

Generated artifacts:
- `logs/final_model_eval.json`
- `logs/final_model_eval_confusion_train.png`
- `logs/final_model_eval_confusion_val.png`
- `logs/final_model_eval_confusion_test.png`

## 5. Runtime CLI Reference (`./app`)

Modes (mutually exclusive):
- `--gesture <CLASS>` capture mode
- `--audit` dataset validation
- `--preview` dataset preview + validation
- `--analyze-fingers` finger-analysis UI
- `--predict-models` base model live UI
- `--predict-stacked` stacked model live UI
- `--predict-hierarchy` hierarchical live UI
- `--predict-final` temporal stabilized debug UI
- `--ui-final` final clean operator UI

Common options:
- `--target` target samples per class (default from config)
- `--camera` camera index
- `--interval` capture interval (seconds)
- `--no-preview-window` preview without interactive window

## 6. Feature Engineering (Landmarks and Vector Layout)

### 6.1 Input Landmark Source

- MediaPipe Hands
- 21 keypoints per hand
- Each keypoint has `(x, y, z)`
- Landmarks are normalized relative to wrist and scale-normalized

### 6.2 Landmark Index Map (21 points)

- `0` wrist
- `1` thumb_cmc
- `2` thumb_mcp
- `3` thumb_ip
- `4` thumb_tip
- `5` index_mcp
- `6` index_pip
- `7` index_dip
- `8` index_tip
- `9` middle_mcp
- `10` middle_pip
- `11` middle_dip
- `12` middle_tip
- `13` ring_mcp
- `14` ring_pip
- `15` ring_dip
- `16` ring_tip
- `17` pinky_mcp
- `18` pinky_pip
- `19` pinky_dip
- `20` pinky_tip

Finger chains used by code:
- thumb: `[0,1,2,3,4]`
- index: `[0,5,6,7,8]`
- middle: `[0,9,10,11,12]`
- ring: `[0,13,14,15,16]`
- pinky: `[0,17,18,19,20]`

### 6.3 Final Feature Vector (`119D`)

- Landmark coordinates: `63D` (`21 x 3`)
- Joint angles: `15D`
- Pairwise distances: `30D`
- Finger states: `5D` (closed/half-open/open per finger)
- Palm orientation one-hot: `6D` (`camera, face, up, down, left, right`)

Total: `63 + 15 + 30 + 5 + 6 = 119D`

## 7. Models Used and How They Work

## 7.1 Base Models (Step 2)

### XGBoost (`models/xgboost.pkl`)

Task:
- Multiclass classification on `119D` tabular features

Key parameters:
- `objective=multi:softprob`
- `num_class=26`
- `n_estimators=500`
- `max_depth=8`
- `learning_rate=0.05`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `reg_lambda=1.0`
- `tree_method=hist`
- `n_jobs=-1`
- `random_state=42` (default)
- `eval_metric=mlogloss`

Notes:
- Uses balanced class weights converted to sample weights.

### Random Forest (`models/rf.pkl`)

Task:
- Multiclass classification on `119D` tabular features

Key parameters:
- `n_estimators=400`
- `class_weight=balanced`
- `n_jobs=-1`
- `random_state=42`
- `max_features=sqrt`
- `min_samples_leaf=1`

### MLP (`models/mlp.pkl`)

Task:
- Multiclass classification on `119D` tabular features

Pipeline:
- `StandardScaler` + `MLPClassifier`

MLP parameters:
- `hidden_layer_sizes=(256, 128, 64)`
- `activation=relu`
- `solver=adam`
- `alpha=1e-4`
- `learning_rate_init=1e-3`
- `max_iter=250`
- `early_stopping=True`
- `validation_fraction=0.1`
- `n_iter_no_change=12`
- `random_state=42`

### CNN (`models/cnn.h5`)

Task:
- Multiclass classification on RGB hand crops

Input:
- Resized RGB image (`--image-size`, default training script uses `224`)

Architecture (high-level):
- Rescaling `1/255`
- Conv2D blocks: `32 -> 64 -> 128 -> 192`
- BatchNorm + ReLU
- MaxPooling after first 3 conv stages
- GlobalAveragePooling
- Dropout `0.35`
- Dense `192` + ReLU + BatchNorm
- Dropout `0.25`
- Final Dense softmax (`26` classes)

Training setup:
- Optimizer: `Adam(1e-3)`
- Loss: `categorical_crossentropy`
- Class weighting: balanced
- Early stopping: patience `5`
- ReduceLROnPlateau: factor `0.5`, patience `2`
- Checkpoint monitor: `val_accuracy` (best model saved)

## 7.2 Stacking Model (Step 3)

Output file:
- `models/stack_meta.pkl`

How it works:
- Takes base-model probabilities from: XGB, RF, MLP, CNN
- Concatenates them into meta-features
- Trains a logistic-regression meta-classifier

Meta-model parameters:
- `LogisticRegression`
- `solver=lbfgs`
- `C=1.0`
- `max_iter=400`
- `class_weight=balanced`
- `random_state=42`

Config file:
- `models/stacking_config.json`
- Stores class list, base model names, feature order, and image size

## 8. Decision Policies and Thresholds

All values from `src/asl_app/config.py`.

### Capture and detection

- ROI size ratio: `0.60`
- ROI right margin ratio: `0.04`
- Min brightness: `45.0`
- Min sharpness: `20.0`
- Min detection confidence: `0.60`
- Min tracking confidence: `0.60`
- Max hands: `1`
- Hand model complexity: `0`
- Mirror left hand to right: `True`

### Finger state rules

Angle thresholds:
- `< 110` => closed (`0`)
- `110..160` => half-open (`1`)
- `> 160` => open (`2`)

Ratio rule used in fusion:
- `< 1.35` => closed
- `> 1.75` => open
- `1.10..1.90` => half-open

### Dataset imbalance checks

- Warning ratio: `1.5`
- Fail ratio: `2.5`

### Hierarchy mode thresholds

- Stack accept confidence: `0.62`
- Stack accept margin: `0.08`
- Base-model min agreement: `2`
- Base-model mean confidence: `0.55`
- Stack support confidence: `0.42`
- Stack top-2 support minimum: `2`

### Temporal stabilization thresholds

- Window size: `6`
- Min frames: `3`
- Accept confidence: `0.48`
- Debounce frames: `2`
- Stable TTL frames: `10`

## 9. Data Splits and Evaluation Policy

Default split strategy (`stratified_random_split`):
- Train: `70%`
- Validation: `15%`
- Test: `15%`
- Stratified by class labels
- Default `random_state=42`

Final evaluator computes (train/val/test):
- Accuracy
- Balanced accuracy
- Precision macro/weighted
- Recall macro/weighted
- F1 macro/weighted
- Top-3 accuracy
- Log loss
- MSE, RMSE, MAE (from one-hot targets vs predicted probabilities)
- Full classification report
- Confusion matrix diagrams (PNG)

## 10. Folder and File Roles

## Root-level files

- `app`: executable launcher (`./app ...`)
- `app.py`: Python launcher (`python app.py ...`)
- `requirements.txt`: dependencies
- `README.md`: project runbook
- `context.md`: internal project notes

## Source package (`src/asl_app`)

- `src/asl_app/config.py`
: Global configuration (paths, thresholds, camera settings, feature dimensions, policy thresholds)

- `src/asl_app/classes.json`
: Canonical class definition (`A-Z`, count 26)

- `src/asl_app/capture_app.py`
: Capture/audit/preview pipeline + feature extraction + parser/dispatch to live UI

- `src/asl_app/live_ui.py`
: Live inference modes (base, stacked, hierarchy, temporal/final UI)

- `src/asl_app/skeleton_canvas.py`
: Skeleton rendering helpers on white canvas with finger color coding

- `src/asl_app/training_common.py`
: Shared training/evaluation utilities (manifest loading, splits, metrics, model save/load helpers)

- `src/asl_app/__init__.py`
: Package initializer

## Training scripts (`training`)

- `training/train_xgboost.py`
- `training/train_rf.py`
- `training/train_mlp.py`
- `training/train_cnn.py`
- `training/train_all.py`
- `training/train_stacking.py`
- `training/evaluate_final.py`

## Data and artifacts

- `data/rgb/`
: Captured RGB images by class

- `data/landmarks/`
: Captured `119D` feature vectors by class

- `models/`
: Trained model artifacts (`xgboost.pkl`, `rf.pkl`, `mlp.pkl`, `cnn.h5`, `stack_meta.pkl`, `stacking_config.json`)

- `logs/`
: Capture logs, dataset validation, model metrics JSON, final evaluation JSON + confusion matrix PNGs

## 11. Controls in Live Modes

Common keyboard controls:
- `q`: quit

Capture mode:
- `s`: start/pause auto capture

Finger analysis mode:
- `k`: append current row to finger analysis CSV

Final UI mode:
- `space`: pause/resume
- `c`: clear stable prediction state

## 12. Recommended Command Order (Clean Run)

```bash
pyenv shell 3.12.7
pip install -r requirements.txt

# capture + validate
./app --gesture A --target 1200
./app --audit --target 1200
./app --preview --target 1200

# train base + stack
python training/train_all.py --model all --epochs 40 --image-size 224 --batch-size 128
python training/train_stacking.py

# live final UI
./app --ui-final

# final evaluation + diagrams
python training/evaluate_final.py --out-prefix final_model_eval
```

## 13. Notes

- If `capture_log.csv` contains stale paths, final evaluation automatically falls back to scanning paired files in `data/rgb` and `data/landmarks`.
- TensorFlow CUDA warnings may appear on systems without a matching CUDA runtime; CPU inference/evaluation can still complete.
