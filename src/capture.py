from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp

from src.config import (
    ALLOW_TWO_HAND_GESTURES,
    CAPTURE_INTERVAL_SEC,
    CAPTURE_LOG_FILE,
    CAMERA_INDEX,
    CLASS_FILE,
    DEFAULT_TARGET_IMAGES_PER_CLASS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMAGE_SIZE,
    MAX_NUM_HANDS,
    MIN_BRIGHTNESS,
    MIN_DET_CONFIDENCE,
    MIN_SHARPNESS,
    MIN_TRACK_CONFIDENCE,
    RAW_GESTURE_DIR,
    REQUIRED_HANDEDNESS,
    ROI_RIGHT_MARGIN_RATIO,
    ROI_SIDE_RATIO,
)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def load_classes(class_file: Path) -> list[str]:
    with class_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["classes"])


def compute_right_square_roi(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    side = int(min(frame_w, frame_h) * ROI_SIDE_RATIO)
    x2 = int(frame_w * (1.0 - ROI_RIGHT_MARGIN_RATIO))
    x1 = max(0, x2 - side)
    y1 = max(0, (frame_h - side) // 2)
    y2 = min(frame_h, y1 + side)
    return x1, y1, x2, y2


def measure_quality(roi_bgr) -> tuple[float, float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


def landmarks_inside_roi(hand_landmarks, frame_w: int, frame_h: int, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    for lm in hand_landmarks.landmark:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        if px < x1 or px > x2 or py < y1 or py > y2:
            return False
    return True


def next_image_index(class_dir: Path) -> int:
    max_idx = 0
    for p in class_dir.glob("img_*.jpg"):
        stem = p.stem
        try:
            idx = int(stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


def append_capture_log(
    csv_path: Path,
    class_name: str,
    image_path: Path,
    brightness: float,
    sharpness: float,
    handedness: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp",
                "class_name",
                "image_path",
                "brightness",
                "sharpness",
                "handedness",
            ])
        writer.writerow([
            int(time.time()),
            class_name,
            str(image_path),
            f"{brightness:.2f}",
            f"{sharpness:.2f}",
            handedness,
        ])


def run_capture(
    gesture: str,
    target_images: int,
    camera_index: int,
    allow_two_hands: bool,
    capture_interval_sec: float,
) -> None:
    classes = load_classes(CLASS_FILE)
    if gesture not in classes:
        raise ValueError(f"Gesture '{gesture}' is not in classes.json")

    class_dir = RAW_GESTURE_DIR / gesture
    class_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    img_index = next_image_index(class_dir)
    saved_count = len(list(class_dir.glob("img_*.jpg")))
    last_capture_ts = 0.0
    auto_capture = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Mirror view: make camera behavior match a real mirror.
            frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            roi = compute_right_square_roi(frame_w, frame_h)
            x1, y1, x2, y2 = roi
            roi_bgr = frame[y1:y2, x1:x2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            can_capture = False
            status = "No hand"
            handedness_name = "None"

            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                for h_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                selected_idx = 0
                if results.multi_handedness and len(results.multi_handedness) == hand_count:
                    if REQUIRED_HANDEDNESS:
                        idx_match = None
                        for i, handed in enumerate(results.multi_handedness):
                            label = handed.classification[0].label
                            if label == REQUIRED_HANDEDNESS:
                                idx_match = i
                                break
                        if idx_match is None:
                            status = f"{REQUIRED_HANDEDNESS} hand not found"
                        else:
                            selected_idx = idx_match
                    handedness_name = results.multi_handedness[selected_idx].classification[0].label

                if hand_count > 1 and not allow_two_hands:
                    status = "Multiple hands detected"
                else:
                    hand_lm = results.multi_hand_landmarks[selected_idx]
                    in_roi = landmarks_inside_roi(hand_lm, frame_w, frame_h, roi)
                    brightness, sharpness = measure_quality(roi_bgr)

                    if not in_roi:
                        status = "Move hand inside square"
                    elif brightness < MIN_BRIGHTNESS:
                        status = "Too dark"
                    elif sharpness < MIN_SHARPNESS:
                        status = "Too blurry"
                    else:
                        can_capture = True
                        status = "Ready"

            now = time.time()
            if auto_capture and can_capture and now - last_capture_ts >= capture_interval_sec and saved_count < target_images:
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                filename = f"img_{img_index:06d}.jpg"
                out_path = class_dir / filename
                cv2.imwrite(str(out_path), crop)

                b, s = measure_quality(crop)
                append_capture_log(CAPTURE_LOG_FILE, gesture, out_path, b, s, handedness_name)

                img_index += 1
                saved_count += 1
                last_capture_ts = now

            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.putText(frame, f"Class: {gesture}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Saved: {saved_count}/{target_images}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Keys: s=start/pause q=quit | interval={capture_interval_sec:.3f}s",
                (20, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
            )

            if saved_count >= target_images:
                cv2.putText(frame, "Target reached", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                auto_capture = False

            cv2.imshow("Gesture Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                auto_capture = not auto_capture

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture gesture images with ROI and hand landmarks")
    parser.add_argument("--gesture", required=True, help="Gesture class name (must exist in classes.json)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_IMAGES_PER_CLASS, help="Target images for this class")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument(
        "--interval",
        type=float,
        default=CAPTURE_INTERVAL_SEC,
        help="Minimum seconds between saved images (lower is faster)",
    )
    parser.add_argument(
        "--allow-two-hands",
        action="store_true",
        default=ALLOW_TWO_HAND_GESTURES,
        help="Allow two-hand gestures in capture",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    run_capture(
        gesture=args.gesture,
        target_images=args.target,
        camera_index=args.camera,
        allow_two_hands=args.allow_two_hands,
        capture_interval_sec=args.interval,
    )


if __name__ == "__main__":
    main()
