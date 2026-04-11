from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.train_cnn import train_cnn
from training.train_mlp import train_mlp
from training.train_rf import train_rf
from training.train_xgboost import train_xgboost


MODEL_ORDER = ["xgboost", "rf", "mlp", "cnn"]


def _progress_bar(done: int, total: int, width: int = 36) -> str:
    ratio = 0.0 if total <= 0 else done / total
    ratio = max(0.0, min(1.0, ratio))
    fill = int(round(ratio * width))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + f"] {done}/{total} ({ratio * 100:5.1f}%)"


def _run_with_spinner(task_name: str, fn: Callable[[], Any]) -> Any:
    done = threading.Event()
    result_holder: dict[str, Any] = {}
    err_holder: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result_holder["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            err_holder["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    frames = "|/-\\"
    idx = 0
    start = time.time()
    while not done.is_set():
        elapsed = time.time() - start
        frame = frames[idx % len(frames)]
        idx += 1
        print(f"\r{frame} Training {task_name}... elapsed {elapsed:6.1f}s", end="", flush=True)
        time.sleep(0.2)

    thread.join()
    elapsed = time.time() - start
    print(f"\rOK Training {task_name}... elapsed {elapsed:6.1f}s")

    if "error" in err_holder:
        raise err_holder["error"]
    return result_holder["result"]


def choose_model_interactive() -> str:
    print("Select Step 2 training target:")
    print("1) xgboost")
    print("2) rf")
    print("3) mlp")
    print("4) cnn")
    print("5) all")
    choice = input("Enter choice [1-5]: ").strip()
    mapping = {
        "1": "xgboost",
        "2": "rf",
        "3": "mlp",
        "4": "cnn",
        "5": "all",
    }
    return mapping.get(choice, "all")


def run_target(model_name: str, random_state: int, epochs: int, image_size: int, batch_size: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    selected = MODEL_ORDER if model_name == "all" else [model_name]
    total = len(selected)

    print("Training progress:")
    print(_progress_bar(0, total))

    for i, model in enumerate(selected, start=1):
        if model == "xgboost":
            results["xgboost"] = _run_with_spinner(
                "xgboost",
                lambda: train_xgboost(random_state=random_state),
            )
        elif model == "rf":
            results["rf"] = _run_with_spinner(
                "rf",
                lambda: train_rf(random_state=random_state),
            )
        elif model == "mlp":
            results["mlp"] = _run_with_spinner(
                "mlp",
                lambda: train_mlp(random_state=random_state),
            )
        elif model == "cnn":
            results["cnn"] = _run_with_spinner(
                "cnn",
                lambda: train_cnn(
                    random_state=random_state,
                    epochs=epochs,
                    image_size=(image_size, image_size),
                    batch_size=batch_size,
                ),
            )

        print(_progress_bar(i, total))

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2 Master Trainer (XGBoost, RF, MLP, CNN)")
    parser.add_argument(
        "--model",
        choices=["xgboost", "rf", "mlp", "cnn", "all"],
        default="",
        help="Model to train. If omitted, prompt will ask.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24, help="CNN epochs")
    parser.add_argument("--image-size", type=int, default=224, help="CNN image size")
    parser.add_argument("--batch-size", type=int, default=64, help="CNN batch size")
    parser.add_argument("--dry-run", action="store_true", help="Show selected plan without training")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_name = args.model if args.model else choose_model_interactive()

    print(f"Selected: {model_name}")
    print(f"random_state={args.random_state}, epochs={args.epochs}, image_size={args.image_size}, batch_size={args.batch_size}")

    if args.dry_run:
        print("Dry run complete. No training executed.")
        return

    results = run_target(
        model_name=model_name,
        random_state=args.random_state,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    print("\nStep 2 training summary:")
    for name, metrics in results.items():
        tr = metrics["splits"]["train"]["accuracy"]
        va = metrics["splits"]["val"]["accuracy"]
        te = metrics["splits"]["test"]["accuracy"]
        print(f"- {name}: train={tr:.4f}, val={va:.4f}, test={te:.4f}")


if __name__ == "__main__":
    main()
