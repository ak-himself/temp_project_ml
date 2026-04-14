from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from asl_app.training_common import (
    LOG_DIR,
    MODELS_DIR,
    ensure_dirs,
    evaluate_predictions,
    filter_valid_rows,
    keras_label_onehot,
    load_capture_manifest,
    load_classes,
    save_metrics,
    split_rows,
    stratified_random_split,
    summarize_split_class_balance,
    summarize_split_subjects,
)


def build_cnn_model(num_classes: int, input_shape: tuple[int, int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(192, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(192, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_tf_dataset(rows: list[dict[str, str]], y_map: dict[str, int], image_size: tuple[int, int], batch_size: int, shuffle: bool) -> tuple[tf.data.Dataset, np.ndarray]:
    paths = [r["image_path"] for r in rows]
    y_int = np.array([y_map[r["class_name"]] for r in rows], dtype=np.int64)
    y_oh = keras_label_onehot(y_int, num_classes=len(y_map))

    ds = tf.data.Dataset.from_tensor_slices((paths, y_oh))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(50000, len(paths)), reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, y_int


def _predict_with_probs(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict(ds, verbose=0)
    pred = np.argmax(prob, axis=1)
    return pred, prob


def _select_head_per_class(rows: list[dict[str, str]], max_per_class: int) -> list[dict[str, str]]:
    if max_per_class <= 0:
        return rows

    counters: dict[str, int] = {}
    out: list[dict[str, str]] = []
    # Manifest rows are chronological/sample-id ordered; keep the earliest K per class.
    for row in rows:
        cls = row["class_name"]
        count = counters.get(cls, 0)
        if count >= max_per_class:
            continue
        counters[cls] = count + 1
        out.append(row)

    return out


def train_cnn(
    random_state: int = 42,
    epochs: int = 24,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 64,
    max_per_class: int = 0,
) -> dict:
    ensure_dirs()
    classes = load_classes()
    rows = filter_valid_rows(load_capture_manifest(), classes)
    rows = _select_head_per_class(rows, max_per_class=max_per_class)
    if not rows:
        raise ValueError("No rows available for CNN training after max_per_class filtering")
    splits = stratified_random_split(rows, random_state=random_state)

    y_map = {name: i for i, name in enumerate(classes)}
    train_rows = split_rows(rows, splits.train_idx)
    val_rows = split_rows(rows, splits.val_idx)
    test_rows = split_rows(rows, splits.test_idx)

    train_ds, y_train = _build_tf_dataset(train_rows, y_map, image_size=image_size, batch_size=batch_size, shuffle=True)
    train_eval_ds, _ = _build_tf_dataset(train_rows, y_map, image_size=image_size, batch_size=batch_size, shuffle=False)
    val_ds, y_val = _build_tf_dataset(val_rows, y_map, image_size=image_size, batch_size=batch_size, shuffle=False)
    test_ds, y_test = _build_tf_dataset(test_rows, y_map, image_size=image_size, batch_size=batch_size, shuffle=False)

    class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=y_train)
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr.tolist())}

    model = build_cnn_model(num_classes=len(classes), input_shape=(image_size[0], image_size[1], 3))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "cnn.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(str(MODELS_DIR / "cnn.h5"))

    pred_train, prob_train = _predict_with_probs(best_model, train_eval_ds)
    pred_val, prob_val = _predict_with_probs(best_model, val_ds)
    pred_test, prob_test = _predict_with_probs(best_model, test_ds)

    metrics = {
        "model": "cnn",
        "split_strategy": "stratified_random",
        "subject_split": summarize_split_subjects(rows, splits),
        "class_balance": summarize_split_class_balance(np.array([y_map[r["class_name"]] for r in rows], dtype=np.int64), splits, classes),
        "splits": {
            "train": evaluate_predictions(y_train, pred_train, prob_train, classes),
            "val": evaluate_predictions(y_val, pred_val, prob_val, classes),
            "test": evaluate_predictions(y_test, pred_test, prob_test, classes),
        },
    }

    save_metrics("cnn", metrics, classes)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2 - Train CNN on skeleton images")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=0, help="Use only first K samples per class (0=all)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_cnn(
        random_state=args.random_state,
        epochs=args.epochs,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
    )
    print("CNN training complete")
    print(f"Train acc: {metrics['splits']['train']['accuracy']:.4f}")
    print(f"Val acc:   {metrics['splits']['val']['accuracy']:.4f}")
    print(f"Test acc:  {metrics['splits']['test']['accuracy']:.4f}")
    print(f"Saved model: {Path(MODELS_DIR / 'cnn.h5')}")
    print(f"Logs dir: {LOG_DIR}")


if __name__ == "__main__":
    main()
