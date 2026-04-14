from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from asl_app.training_common import (
    MODELS_DIR,
    build_tabular_arrays,
    ensure_dirs,
    evaluate_predictions,
    filter_valid_rows,
    load_capture_manifest,
    load_classes,
    save_metrics,
    save_model_pickle,
    select_by_idx,
    split_rows,
    stratified_random_split,
    summarize_split_class_balance,
    summarize_split_subjects,
)


def _build_cnn_dataset(
    rows: list[dict[str, str]],
    y_map: dict[str, int],
    image_size: tuple[int, int],
    batch_size: int,
) -> tuple[tf.data.Dataset, np.ndarray]:
    paths = [r["image_path"] for r in rows]
    y_int = np.array([y_map[r["class_name"]] for r in rows], dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path: tf.Tensor) -> tf.Tensor:
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, image_size)
        return tf.cast(img, tf.float32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, y_int


def _get_tabular_probs(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    prob = np.asarray(model.predict_proba(x), dtype=np.float32)
    if prob.ndim != 2 or prob.shape[1] != num_classes:
        raise ValueError(f"Tabular model probability shape mismatch: expected (N, {num_classes}), got {tuple(prob.shape)}")
    return prob


def _get_cnn_probs(model: tf.keras.Model, ds: tf.data.Dataset, num_classes: int) -> np.ndarray:
    prob = np.asarray(model.predict(ds, verbose=0), dtype=np.float32)
    if prob.ndim != 2 or prob.shape[1] != num_classes:
        raise ValueError(f"CNN probability shape mismatch: expected (N, {num_classes}), got {tuple(prob.shape)}")
    return prob


def _concat_meta_features(probs: list[np.ndarray]) -> np.ndarray:
    if not probs:
        raise ValueError("No probability arrays provided for stacking")
    n = probs[0].shape[0]
    for p in probs:
        if p.shape[0] != n:
            raise ValueError("All probability arrays must have same number of rows")
    return np.concatenate(probs, axis=1).astype(np.float32)


def _load_required_models() -> tuple[object, object, object, tf.keras.Model]:
    xgb_path = MODELS_DIR / "xgboost.pkl"
    rf_path = MODELS_DIR / "rf.pkl"
    mlp_path = MODELS_DIR / "mlp.pkl"
    cnn_path = MODELS_DIR / "cnn.h5"

    required = [xgb_path, rf_path, mlp_path, cnn_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Step 2 model files:\n" + "\n".join(missing))

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
    cnn = tf.keras.models.load_model(str(cnn_path))
    return xgb, rf, mlp, cnn


def train_stacking(
    random_state: int = 42,
    image_size: tuple[int, int] | None = None,
    batch_size: int = 128,
) -> dict:
    ensure_dirs()
    classes = load_classes()
    rows = filter_valid_rows(load_capture_manifest(), classes)
    splits = stratified_random_split(rows, random_state=random_state)

    x, y, _ = build_tabular_arrays(rows, classes)
    y_map = {name: i for i, name in enumerate(classes)}

    x_train, y_train = select_by_idx(x, y, splits.train_idx)
    x_val, y_val = select_by_idx(x, y, splits.val_idx)
    x_test, y_test = select_by_idx(x, y, splits.test_idx)

    val_rows = split_rows(rows, splits.val_idx)
    test_rows = split_rows(rows, splits.test_idx)

    xgb, rf, mlp, cnn = _load_required_models()
    if image_size is None:
        # Use the trained CNN's expected input size to avoid preprocessing mismatch.
        input_h = int(cnn.input_shape[1])
        input_w = int(cnn.input_shape[2])
        image_size = (input_h, input_w)

    p_xgb_train = _get_tabular_probs(xgb, x_train, len(classes))
    p_rf_train = _get_tabular_probs(rf, x_train, len(classes))
    p_mlp_train = _get_tabular_probs(mlp, x_train, len(classes))

    p_xgb_val = _get_tabular_probs(xgb, x_val, len(classes))
    p_rf_val = _get_tabular_probs(rf, x_val, len(classes))
    p_mlp_val = _get_tabular_probs(mlp, x_val, len(classes))

    p_xgb_test = _get_tabular_probs(xgb, x_test, len(classes))
    p_rf_test = _get_tabular_probs(rf, x_test, len(classes))
    p_mlp_test = _get_tabular_probs(mlp, x_test, len(classes))

    val_ds, _ = _build_cnn_dataset(val_rows, y_map, image_size=image_size, batch_size=batch_size)
    test_ds, _ = _build_cnn_dataset(test_rows, y_map, image_size=image_size, batch_size=batch_size)

    # For train split reporting only, we use a deterministic non-shuffled subset pipeline.
    train_rows = split_rows(rows, splits.train_idx)
    train_ds, _ = _build_cnn_dataset(train_rows, y_map, image_size=image_size, batch_size=batch_size)

    p_cnn_train = _get_cnn_probs(cnn, train_ds, len(classes))
    p_cnn_val = _get_cnn_probs(cnn, val_ds, len(classes))
    p_cnn_test = _get_cnn_probs(cnn, test_ds, len(classes))

    x_meta_train = _concat_meta_features([p_xgb_val, p_rf_val, p_mlp_val, p_cnn_val])
    y_meta_train = y_val

    meta_model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=400,
        class_weight="balanced",
        random_state=random_state,
    )
    meta_model.fit(x_meta_train, y_meta_train)

    x_meta_eval_train = _concat_meta_features([p_xgb_train, p_rf_train, p_mlp_train, p_cnn_train])
    x_meta_eval_val = _concat_meta_features([p_xgb_val, p_rf_val, p_mlp_val, p_cnn_val])
    x_meta_eval_test = _concat_meta_features([p_xgb_test, p_rf_test, p_mlp_test, p_cnn_test])

    pred_train = meta_model.predict(x_meta_eval_train)
    prob_train = meta_model.predict_proba(x_meta_eval_train)
    pred_val = meta_model.predict(x_meta_eval_val)
    prob_val = meta_model.predict_proba(x_meta_eval_val)
    pred_test = meta_model.predict(x_meta_eval_test)
    prob_test = meta_model.predict_proba(x_meta_eval_test)

    metrics = {
        "model": "stacking",
        "base_models": ["xgboost", "rf", "mlp", "cnn"],
        "meta_model": "logistic_regression_multinomial",
        "meta_train_source": "validation_split",
        "split_strategy": "stratified_random",
        "subject_split": summarize_split_subjects(rows, splits),
        "class_balance": summarize_split_class_balance(y, splits, classes),
        "splits": {
            "train": evaluate_predictions(y_train, pred_train, prob_train, classes),
            "val": evaluate_predictions(y_val, pred_val, prob_val, classes),
            "test": evaluate_predictions(y_test, pred_test, prob_test, classes),
        },
    }

    save_model_pickle(meta_model, MODELS_DIR / "stack_meta.pkl")
    with (MODELS_DIR / "stacking_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "base_models": [
                    "xgboost.pkl",
                    "rf.pkl",
                    "mlp.pkl",
                    "cnn.h5",
                ],
                "meta_model": "stack_meta.pkl",
                "meta_feature_order": ["xgboost", "rf", "mlp", "cnn"],
                "image_size": [int(image_size[0]), int(image_size[1])],
            },
            f,
            indent=2,
        )

    save_metrics("stacking", metrics, classes)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 3 - Train Stacking Meta-Model")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=0, help="Image size used for CNN preprocessing (0=auto from cnn.h5)")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_size = None if args.image_size <= 0 else (args.image_size, args.image_size)
    metrics = train_stacking(
        random_state=args.random_state,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    print("Stacking training complete")
    print("Meta-train source: validation split")
    print(f"Train acc: {metrics['splits']['train']['accuracy']:.4f}")
    print(f"Val acc:   {metrics['splits']['val']['accuracy']:.4f}")
    print(f"Test acc:  {metrics['splits']['test']['accuracy']:.4f}")
    print(f"Saved model: {MODELS_DIR / 'stack_meta.pkl'}")
    print(f"Saved config: {MODELS_DIR / 'stacking_config.json'}")


if __name__ == "__main__":
    main()
