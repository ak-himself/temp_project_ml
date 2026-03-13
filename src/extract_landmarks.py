from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.config import MAX_NUM_HANDS, MIN_DET_CONFIDENCE, RAW_GESTURE_DIR


mp_hands = mp.solutions.hands


def normalize_hand(hand_xyz: np.ndarray) -> np.ndarray:
    # Normalize around wrist and scale by max radius for translation/scale robustness.
    wrist = hand_xyz[0:1, :]
    rel = hand_xyz - wrist
    radius = np.max(np.linalg.norm(rel[:, :2], axis=1))
    if radius < 1e-6:
        radius = 1.0
    rel = rel / radius
    return rel.reshape(-1)


def hand_to_array(hand_landmarks) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    return normalize_hand(pts)


def build_two_hand_feature(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None

    # Fixed 2-hand representation: [Right hand feature, Left hand feature].
    # Missing hand is zero-filled.
    feature_len = 63
    right = np.zeros(feature_len, dtype=np.float32)
    left = np.zeros(feature_len, dtype=np.float32)

    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
        feat = hand_to_array(hand_lm)
        label = None
        if results.multi_handedness and idx < len(results.multi_handedness):
            label = results.multi_handedness[idx].classification[0].label

        if label == "Right":
            right = feat
        elif label == "Left":
            left = feat
        else:
            # Fallback when handedness is missing.
            if np.all(right == 0):
                right = feat
            elif np.all(left == 0):
                left = feat

    return np.concatenate([right, left], axis=0)


def iter_images(root: Path):
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img_path in sorted(class_dir.glob("img_*.jpg")):
            yield class_dir.name, img_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract hand landmark features from captured gesture images")
    parser.add_argument("--data-dir", type=Path, default=RAW_GESTURE_DIR)
    parser.add_argument("--out", type=Path, default=Path("data/processed/landmarks.csv"))
    parser.add_argument("--max-hands", type=int, default=MAX_NUM_HANDS)
    parser.add_argument("--min-det", type=float, default=MIN_DET_CONFIDENCE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str, np.ndarray]] = []
    skipped = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=0.5,
    ) as hands:
        for class_name, img_path in iter_images(args.data_dir):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            feat = build_two_hand_feature(results)
            if feat is None:
                skipped += 1
                continue

            records.append((class_name, str(img_path), feat))

    if not records:
        raise RuntimeError("No valid landmark records extracted")

    feature_size = records[0][2].shape[0]
    header = ["label", "image_path"] + [f"f_{i}" for i in range(feature_size)]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, img_path, feat in records:
            writer.writerow([label, img_path, *feat.tolist()])

    print(f"Saved landmark dataset: {args.out}")
    print(f"Records: {len(records)}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
