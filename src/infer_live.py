from __future__ import annotations

import argparse
import textwrap
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import models, transforms

from src.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_NUM_HANDS,
    MIN_BRIGHTNESS,
    MIN_DET_CONFIDENCE,
    MIN_SHARPNESS,
    MIN_TRACK_CONFIDENCE,
    MODELS_DIR,
    REQUIRED_HANDEDNESS,
    ROI_RIGHT_MARGIN_RATIO,
    ROI_SIDE_RATIO,
)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


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


def build_model(num_classes: int, checkpoint: dict) -> torch.nn.Module:
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    if model_name != "efficientnet_b0":
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def apply_action(buffer_text: str, label: str) -> str:
    if label == "Space":
        return buffer_text + " "
    if label == "Backspace":
        return buffer_text[:-1]
    if label == "Clear":
        return ""
    if len(label) == 1:
        return buffer_text + label
    return buffer_text


def draw_notepad(canvas, text_value: str, status_lines: list[str]) -> None:
    canvas[:] = (245, 247, 250)
    h, w = canvas.shape[:2]

    cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), (210, 210, 210), 2)
    cv2.putText(canvas, "Live Notepad", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)

    wrapped = textwrap.wrap(text_value, width=42) if text_value else [""]
    y = 90
    for line in wrapped[-16:]:
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        y += 30

    y = h - 90
    for line in status_lines:
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (70, 70, 70), 2)
        y += 24


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live gesture inference with notepad actions")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "gesture_classifier.pt",
        help="Model checkpoint path (e.g., models/gesture_classifier.pt)",
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=0.60, help="Min confidence to accept prediction")
    parser.add_argument("--stability", type=int, default=6, help="Consecutive frames required")
    parser.add_argument("--cooldown", type=float, default=0.45, help="Seconds between accepted actions")
    parser.add_argument(
        "--allow-two-hands",
        action="store_true",
        default=True,
        help="Allow two-hand gestures (recommended for Space/Backspace/Clear)",
    )
    parser.add_argument(
        "--single-hand-only",
        action="store_false",
        dest="allow_two_hands",
        help="Reject frames with multiple hands",
    )
    return parser


def resolve_model_path(model_arg: Path) -> Path:
    candidates = [model_arg]
    if not model_arg.is_absolute():
        candidates.append(Path.cwd() / model_arg)
    if model_arg.parent == Path("."):
        candidates.append(MODELS_DIR / model_arg.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    pretty_candidates = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Model checkpoint not found. Tried:\n"
        f"{pretty_candidates}\n"
        "Tip: run with --model models/gesture_classifier.pt"
    )


def main() -> None:
    args = build_parser().parse_args()

    model_path = resolve_model_path(args.model)
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names: list[str] = checkpoint["class_names"]
    mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])
    image_size = checkpoint.get("image_size", [224, 224])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), checkpoint=checkpoint).to(device)

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((int(image_size[0]), int(image_size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    buffer_text = ""
    pred_history: deque[str] = deque(maxlen=args.stability)
    last_action_ts = 0.0
    last_committed = "None"

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

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            x1, y1, x2, y2 = compute_right_square_roi(frame_w, frame_h)
            roi = (x1, y1, x2, y2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            has_hand = bool(results.multi_hand_landmarks)

            status = "No hand"
            can_infer = False

            pred_label = "Unknown"
            pred_conf = 0.0
            if has_hand:
                hand_count = len(results.multi_hand_landmarks)
                for hand_lm in results.multi_hand_landmarks:
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

                if hand_count > 1 and not args.allow_two_hands:
                    status = "Multiple hands detected"
                else:
                    hand_lm = results.multi_hand_landmarks[selected_idx]
                    in_roi = landmarks_inside_roi(hand_lm, frame_w, frame_h, roi)
                    roi_bgr = frame[y1:y2, x1:x2]
                    brightness, sharpness = measure_quality(roi_bgr)

                    if not in_roi:
                        status = "Move hand inside square"
                    elif brightness < MIN_BRIGHTNESS:
                        status = "Too dark"
                    elif sharpness < MIN_SHARPNESS:
                        status = "Too blurry"
                    else:
                        can_infer = True
                        status = "Ready"

            if can_infer:
                roi_bgr = frame[y1:y2, x1:x2]
                inp = preprocess(roi_bgr).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_idx = int(torch.argmax(probs).item())
                    pred_conf = float(probs[pred_idx].item())
                    pred_label = class_names[pred_idx]

            accepted_for_history = pred_label
            if pred_conf < args.threshold:
                accepted_for_history = "Unknown"
            pred_history.append(accepted_for_history)

            now = time.time()
            can_commit = now - last_action_ts >= args.cooldown
            if (
                len(pred_history) == args.stability
                and len(set(pred_history)) == 1
                and pred_history[-1] != "Unknown"
                and can_commit
            ):
                committed = pred_history[-1]
                buffer_text = apply_action(buffer_text, committed)
                last_committed = committed
                last_action_ts = now

            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.putText(frame, f"Pred: {pred_label} ({pred_conf:.2f})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Last action: {last_committed}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Keys: q=quit c=clear", (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

            notepad = np.full((frame_h, 520, 3), 255, dtype=np.uint8)
            draw_notepad(
                notepad,
                text_value=buffer_text,
                status_lines=[
                    f"threshold={args.threshold:.2f} stability={args.stability} cooldown={args.cooldown:.2f}s",
                    f"allow_two_hands={args.allow_two_hands}",
                    f"model_classes={len(class_names)}",
                    f"model={model_path.name}",
                ],
            )

            combined = cv2.hconcat([frame, notepad])
            cv2.imshow("Gesture Notepad", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                buffer_text = ""
                last_committed = "ManualClear"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
