from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_FILE = Path(__file__).resolve().parent / "classes.json"
CAPTURE_LOG_FILE = PROJECT_ROOT / "logs" / "capture_log.csv"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

FEATURE_TOTAL_DIM = 119
DEFAULT_RANDOM_STATE = 42


@dataclass
class DatasetSplits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_classes() -> list[str]:
    with CLASS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    classes = list(data["classes"])
    expected = int(data.get("class_count", len(classes)))
    if len(classes) != expected:
        raise ValueError(f"class_count mismatch in classes.json: class_count={expected}, actual={len(classes)}")
    return classes


def load_capture_manifest() -> list[dict[str, str]]:
    if not CAPTURE_LOG_FILE.exists():
        raise FileNotFoundError(f"Capture log not found: {CAPTURE_LOG_FILE}")

    rows: list[dict[str, str]] = []
    seen = set()
    with CAPTURE_LOG_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "class_name" not in row and "label" in row:
                row["class_name"] = row.get("label", "")
            if "image_path" not in row and "rgb_path" in row:
                row["image_path"] = row.get("rgb_path", "")
            if "landmark_path" not in row and "feature_path" in row:
                row["landmark_path"] = row.get("feature_path", "")

            key = row.get("image_path", "")
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(row)

    if not rows:
        raise ValueError("Capture log is empty after deduplication")
    return rows


def filter_valid_rows(rows: list[dict[str, str]], class_names: list[str]) -> list[dict[str, str]]:
    valid: list[dict[str, str]] = []
    class_set = set(class_names)
    for row in rows:
        label = row.get("class_name", "")
        image_path = Path(row.get("image_path", ""))
        landmark_path = Path(row.get("landmark_path", ""))
        if label not in class_set:
            continue
        if not image_path.exists():
            continue
        if not landmark_path.exists():
            continue
        valid.append(row)

    if not valid:
        raise ValueError("No valid samples found (missing files/labels/subject_id)")
    return valid


def stratified_random_split(
    rows: list[dict[str, str]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplits:
    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1.0:
        raise ValueError("Split ratios must satisfy: val_size > 0, test_size > 0, val_size + test_size < 1")

    n = len(rows)
    idx = np.arange(n)
    y = np.array([r["class_name"] for r in rows], dtype=object)

    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_val_y = y[train_val_idx]
    val_ratio_in_train_val = val_size / (1.0 - test_size)
    tr_rel, val_rel = train_test_split(
        np.arange(train_val_idx.shape[0]),
        test_size=val_ratio_in_train_val,
        random_state=random_state,
        stratify=train_val_y,
    )

    train_idx = train_val_idx[tr_rel]
    val_idx = train_val_idx[val_rel]

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Data leakage detected: split indices overlap")

    return DatasetSplits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def label_to_index(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def build_tabular_arrays(rows: list[dict[str, str]], class_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_map = label_to_index(class_names)
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups: list[str] = []

    for row in rows:
        vec = np.load(row["landmark_path"]).astype(np.float32).reshape(-1)
        if vec.shape[0] != FEATURE_TOTAL_DIM:
            raise ValueError(
                f"Feature vector mismatch at {row['landmark_path']}: expected {FEATURE_TOTAL_DIM}, got {vec.shape[0]}"
            )
        x_list.append(vec)
        y_list.append(y_map[row["class_name"]])
        groups.append(row.get("subject_id", "S00"))

    x = np.vstack(x_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(groups)
    return x, y, g


def select_by_idx(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x[idx], y[idx]


def split_rows(rows: list[dict[str, str]], idx: np.ndarray) -> list[dict[str, str]]:
    return [rows[i] for i in idx.tolist()]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    if y_prob.ndim != 2 or y_prob.shape[1] != len(class_names):
        raise ValueError(
            f"Probability output must be (N, {len(class_names)}), got {tuple(y_prob.shape)}"
        )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "accuracy": acc,
        "classification_report": report,
    }


def save_metrics(
    model_name: str,
    metrics: dict[str, Any],
    _class_names: list[str],
) -> None:
    ensure_dirs()
    json_path = LOG_DIR / f"{model_name}_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_model_pickle(model: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def summarize_split_subjects(rows: list[dict[str, str]], splits: DatasetSplits) -> dict[str, list[str]]:
    return {
        "train_subjects": sorted({rows[i].get("subject_id", "S00") for i in splits.train_idx.tolist()}),
        "val_subjects": sorted({rows[i].get("subject_id", "S00") for i in splits.val_idx.tolist()}),
        "test_subjects": sorted({rows[i].get("subject_id", "S00") for i in splits.test_idx.tolist()}),
    }


def summarize_split_class_balance(y: np.ndarray, splits: DatasetSplits, class_names: list[str]) -> dict[str, dict[str, int]]:
    def _counts(idx: np.ndarray) -> dict[str, int]:
        labels, counts = np.unique(y[idx], return_counts=True)
        out = {name: 0 for name in class_names}
        for lab, cnt in zip(labels.tolist(), counts.tolist()):
            out[class_names[int(lab)]] = int(cnt)
        return out

    return {
        "train": _counts(splits.train_idx),
        "val": _counts(splits.val_idx),
        "test": _counts(splits.test_idx),
    }


def keras_label_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out