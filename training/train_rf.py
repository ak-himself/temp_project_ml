from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier

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
    stratified_random_split,
    summarize_split_class_balance,
    summarize_split_subjects,
)


def train_rf(random_state: int = 42) -> dict:
    ensure_dirs()
    classes = load_classes()
    rows = filter_valid_rows(load_capture_manifest(), classes)
    splits = stratified_random_split(rows, random_state=random_state)
    x, y, _ = build_tabular_arrays(rows, classes)

    x_train, y_train = select_by_idx(x, y, splits.train_idx)
    x_val, y_val = select_by_idx(x, y, splits.val_idx)
    x_test, y_test = select_by_idx(x, y, splits.test_idx)

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        max_features="sqrt",
        min_samples_leaf=1,
    )
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    prob_train = model.predict_proba(x_train)
    pred_val = model.predict(x_val)
    prob_val = model.predict_proba(x_val)
    pred_test = model.predict(x_test)
    prob_test = model.predict_proba(x_test)

    metrics = {
        "model": "rf",
        "split_strategy": "stratified_random",
        "subject_split": summarize_split_subjects(rows, splits),
        "class_balance": summarize_split_class_balance(y, splits, classes),
        "splits": {
            "train": evaluate_predictions(y_train, pred_train, prob_train, classes),
            "val": evaluate_predictions(y_val, pred_val, prob_val, classes),
            "test": evaluate_predictions(y_test, pred_test, prob_test, classes),
        },
    }

    save_model_pickle(model, MODELS_DIR / "rf.pkl")
    save_metrics("rf", metrics, classes)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2 - Train Random Forest on 113D tabular features")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_rf(random_state=args.random_state)
    print("Random Forest training complete")
    print(f"Train acc: {metrics['splits']['train']['accuracy']:.4f}")
    print(f"Val acc:   {metrics['splits']['val']['accuracy']:.4f}")
    print(f"Test acc:  {metrics['splits']['test']['accuracy']:.4f}")
    print(f"Saved model: {Path(MODELS_DIR / 'rf.pkl')}")


if __name__ == "__main__":
    main()
