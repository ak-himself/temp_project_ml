from __future__ import annotations

from pathlib import Path

from src.config import RAW_GESTURE_DIR


def main() -> None:
    root = Path(RAW_GESTURE_DIR)
    if not root.exists():
        print("No dataset found yet.")
        return

    print("Class counts:")
    total = 0
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        count = len(list(class_dir.glob("img_*.jpg")))
        total += count
        print(f"- {class_dir.name}: {count}")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()
