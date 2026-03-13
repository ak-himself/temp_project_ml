from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train landmark-based gesture classifier")
    parser.add_argument("--csv", type=Path, default=Path("data/processed/landmarks.csv"))
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "landmark_classifier.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must include 'label' column")

    y = df["label"].to_numpy()
    x = df.drop(columns=["label", "image_path"], errors="ignore").to_numpy(dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_val)
    labels = sorted(list(np.unique(y)))
    print(classification_report(y_val, y_pred, labels=labels))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_pred, labels=labels).tolist())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "classes": labels,
        "feature_count": int(x.shape[1]),
    }
    joblib.dump(payload, args.out)

    meta_path = args.out.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"classes": labels, "feature_count": int(x.shape[1])}, f, indent=2)

    print(f"Saved landmark model: {args.out}")
    print(f"Saved landmark metadata: {meta_path}")


if __name__ == "__main__":
    main()
