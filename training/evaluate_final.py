from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from asl_app.training_common import (
    LOG_DIR,
    MODELS_DIR,
    build_tabular_arrays,
    ensure_dirs,
    filter_valid_rows,
    load_capture_manifest,
    load_classes,
    select_by_idx,
    split_rows,
    stratified_random_split,
)


def _extract_sample_id(stem: str) -> int:
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def _rows_from_data_dirs(class_names: list[str]) -> list[dict[str, str]]:
    data_dir = ROOT / "data"
    rgb_root = data_dir / "rgb"
    lm_root = data_dir / "landmarks"

    rows: list[dict[str, str]] = []
    for cls in class_names:
        rgb_dir = rgb_root / cls
        lm_dir = lm_root / cls
        if not rgb_dir.exists() or not lm_dir.exists():
            continue

        rgb_by_id = {}
        for p in sorted(rgb_dir.glob("img*")):
            sid = _extract_sample_id(p.stem)
            if sid >= 0:
                rgb_by_id[sid] = p

        lm_by_id = {}
        for p in sorted(lm_dir.glob("lm*.npy")):
            sid = _extract_sample_id(p.stem)
            if sid >= 0:
                lm_by_id[sid] = p

        for sid in sorted(set(rgb_by_id) & set(lm_by_id)):
            rows.append(
                {
                    "class_name": cls,
                    "image_path": str(rgb_by_id[sid]),
                    "landmark_path": str(lm_by_id[sid]),
                    "subject_id": "S00",
                }
            )

    return rows


def _select_head_per_class(rows: list[dict[str, str]], max_per_class: int) -> list[dict[str, str]]:
    if max_per_class <= 0:
        return rows

    out: list[dict[str, str]] = []
    counters: dict[str, int] = {}
    for row in rows:
        cls = row["class_name"]
        cnt = counters.get(cls, 0)
        if cnt >= max_per_class:
            continue
        counters[cls] = cnt + 1
        out.append(row)
    return out


def _load_eval_rows(class_names: list[str]) -> list[dict[str, str]]:
    try:
        return filter_valid_rows(load_capture_manifest(), class_names)
    except (FileNotFoundError, ValueError):
        rows = _rows_from_data_dirs(class_names)
        if not rows:
            raise ValueError(
                "No valid samples found. Capture log paths are stale and fallback scan found no paired files in data/rgb and data/landmarks."
            )
        return rows


def _build_cnn_dataset(
    rows: list[dict[str, str]],
    image_size: tuple[int, int],
    batch_size: int,
) -> tf.data.Dataset:
    paths = [r["image_path"] for r in rows]
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path: tf.Tensor) -> tf.Tensor:
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, image_size)
        return tf.cast(img, tf.float32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    num_classes = len(class_names)
    labels = list(range(num_classes))

    if y_prob.ndim != 2 or y_prob.shape[1] != num_classes:
        raise ValueError(
            f"Expected probability shape (N, {num_classes}), got {tuple(y_prob.shape)}"
        )

    y_true_oh = np.eye(num_classes, dtype=np.float32)[y_true]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    mse = float(mean_squared_error(y_true_oh, y_prob))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_oh, y_prob))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=min(3, num_classes), labels=labels)),
        "log_loss": float(log_loss(y_true, y_prob, labels=labels)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def _write_confusion_png(
    out_path: Path,
    confusion: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    max_val = float(confusion.max()) if confusion.size else 0.0
    thresh = max_val / 2.0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            val = int(confusion[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_final_model(
    random_state: int = 42,
    batch_size: int = 128,
    out_prefix: str = "final_model_eval",
    max_per_class: int = 0,
) -> dict:
    ensure_dirs()

    classes = load_classes()
    rows = _select_head_per_class(_load_eval_rows(classes), max_per_class=max_per_class)
    if not rows:
        raise ValueError("No rows available for evaluation after max_per_class filtering")
    splits = stratified_random_split(rows, random_state=random_state)

    x, y, _ = build_tabular_arrays(rows, classes)

    x_train, y_train = select_by_idx(x, y, splits.train_idx)
    x_val, y_val = select_by_idx(x, y, splits.val_idx)
    x_test, y_test = select_by_idx(x, y, splits.test_idx)

    train_rows = split_rows(rows, splits.train_idx)
    val_rows = split_rows(rows, splits.val_idx)
    test_rows = split_rows(rows, splits.test_idx)

    xgb_path = MODELS_DIR / "xgboost.pkl"
    rf_path = MODELS_DIR / "rf.pkl"
    mlp_path = MODELS_DIR / "mlp.pkl"
    cnn_path = MODELS_DIR / "cnn.h5"
    stack_meta_path = MODELS_DIR / "stack_meta.pkl"

    required = [xgb_path, rf_path, mlp_path, cnn_path, stack_meta_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
    cnn = tf.keras.models.load_model(str(cnn_path))
    stack_meta = joblib.load(stack_meta_path)

    image_h = int(cnn.input_shape[1])
    image_w = int(cnn.input_shape[2])

    train_ds = _build_cnn_dataset(train_rows, image_size=(image_h, image_w), batch_size=batch_size)
    val_ds = _build_cnn_dataset(val_rows, image_size=(image_h, image_w), batch_size=batch_size)
    test_ds = _build_cnn_dataset(test_rows, image_size=(image_h, image_w), batch_size=batch_size)

    def _stack_prob(x_part: np.ndarray, ds_part: tf.data.Dataset) -> np.ndarray:
        p_xgb = np.asarray(xgb.predict_proba(x_part), dtype=np.float32)
        p_rf = np.asarray(rf.predict_proba(x_part), dtype=np.float32)
        p_mlp = np.asarray(mlp.predict_proba(x_part), dtype=np.float32)
        p_cnn = np.asarray(cnn.predict(ds_part, verbose=0), dtype=np.float32)
        meta_in = np.concatenate([p_xgb, p_rf, p_mlp, p_cnn], axis=1).astype(np.float32)
        p_stack = np.asarray(stack_meta.predict_proba(meta_in), dtype=np.float32)
        return p_stack

    prob_train = _stack_prob(x_train, train_ds)
    prob_val = _stack_prob(x_val, val_ds)
    prob_test = _stack_prob(x_test, test_ds)

    pred_train = np.argmax(prob_train, axis=1)
    pred_val = np.argmax(prob_val, axis=1)
    pred_test = np.argmax(prob_test, axis=1)

    train_metrics = _evaluate_split(y_train, pred_train, prob_train, classes)
    val_metrics = _evaluate_split(y_val, pred_val, prob_val, classes)
    test_metrics = _evaluate_split(y_test, pred_test, prob_test, classes)

    out = {
        "model": "final_stacking",
        "base_models": ["xgboost", "rf", "mlp", "cnn"],
        "meta_model": "stack_meta",
        "random_state": random_state,
        "batch_size": batch_size,
        "splits": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
    }

    out_json = LOG_DIR / f"{out_prefix}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    _write_confusion_png(
        LOG_DIR / f"{out_prefix}_confusion_train.png",
        np.asarray(train_metrics["confusion_matrix"], dtype=np.int64),
        classes,
        title="Train Confusion Matrix",
    )
    _write_confusion_png(
        LOG_DIR / f"{out_prefix}_confusion_val.png",
        np.asarray(val_metrics["confusion_matrix"], dtype=np.int64),
        classes,
        title="Validation Confusion Matrix",
    )
    _write_confusion_png(
        LOG_DIR / f"{out_prefix}_confusion_test.png",
        np.asarray(test_metrics["confusion_matrix"], dtype=np.int64),
        classes,
        title="Test Confusion Matrix",
    )

    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate final stacked model with full metrics")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-prefix", type=str, default="final_model_eval")
    parser.add_argument("--max-per-class", type=int, default=0, help="Evaluate using first K samples per class (0=all)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_final_model(
        random_state=args.random_state,
        batch_size=args.batch_size,
        out_prefix=args.out_prefix,
        max_per_class=args.max_per_class,
    )

    test = metrics["splits"]["test"]
    print("Final model evaluation complete")
    print(f"Test accuracy:       {test['accuracy']:.4f}")
    print(f"Test f1_macro:      {test['f1_macro']:.4f}")
    print(f"Test f1_weighted:   {test['f1_weighted']:.4f}")
    print(f"Test precision_mac: {test['precision_macro']:.4f}")
    print(f"Test recall_macro:  {test['recall_macro']:.4f}")
    print(f"Test mse:           {test['mse']:.6f}")
    print(f"Test rmse:          {test['rmse']:.6f}")
    print(f"Test mae:           {test['mae']:.6f}")
    print(f"Saved: {LOG_DIR / (args.out_prefix + '.json')}")
    print(f"Saved: {LOG_DIR / (args.out_prefix + '_confusion_test.png')}")


if __name__ == "__main__":
    main()
