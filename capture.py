from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from config import (
    ANALYSIS_WINDOW_NAME,
    CAMERA_FOURCC,
    CAMERA_INDEX,
    CAPTURE_INTERVAL_SEC,
    CAPTURE_LOG_FILE,
    CAPTURE_WINDOW_NAME,
    CLASS_FILE,
    DATASET_PREVIEW_FILE,
    DATASET_VALIDATION_JSON_FILE,
    DATASET_VALIDATION_MD_FILE,
    DEFAULT_TARGET_IMAGES_PER_CLASS,
    FEATURE_FINGER_STATE_DIM,
    FEATURE_JOINT_ANGLE_DIM,
    FEATURE_LANDMARK_DIM,
    FEATURE_PAIRWISE_DISTANCE_DIM,
    FEATURE_PALM_DIRECTION_DIM,
    FEATURE_TOTAL_DIM,
    FINGER_ANALYSIS_LOG_FILE,
    FINGER_STATE_ANGLE_HALF_OPEN_MAX,
    FINGER_STATE_ANGLE_HALF_OPEN_MIN,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HIERARCHY_STACK_ACCEPT_CONF,
    HIERARCHY_STACK_ACCEPT_MARGIN,
    HIERARCHY_BASE_MEAN_CONF,
    HIERARCHY_BASE_MIN_AGREEMENT,
    HIERARCHY_STACK_SUPPORT_CONF,
    HIERARCHY_STACK_SUPPORT_TOP2_MIN,
    IMBALANCE_FAIL_RATIO,
    IMBALANCE_WARN_RATIO,
    LOG_DIR,
    HAND_MODEL_COMPLEXITY,
    MAX_NUM_HANDS,
    MODELS_DIR,
    MIRROR_LEFT_HAND_TO_RIGHT,
    MIN_BRIGHTNESS,
    MIN_DET_CONFIDENCE,
    MIN_SHARPNESS,
    MIN_TRACK_CONFIDENCE,
    PREFER_V4L2_BACKEND,
    PALM_ORIENTATION_6,
    PROCESSED_DIR,
    RAW_LANDMARK_DIR,
    RAW_RGB_DIR,
    REQUIRED_HANDEDNESS,
    RGB_IMAGE_SIZE,
    ROI_RIGHT_MARGIN_RATIO,
    ROI_SIDE_RATIO,
    TEMPORAL_ACCEPT_CONF,
    TEMPORAL_DEBOUNCE_FRAMES,
    TEMPORAL_MIN_FRAMES,
    TEMPORAL_STABLE_TTL_FRAMES,
    TEMPORAL_WINDOW_SIZE,
    ENABLE_AUTOFOCUS,
)
from skeleton_canvas import render_skeleton_canvas

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FINGER_NAME_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

# N2 = 30 selected pairwise distances.
PAIRWISE_DISTANCE_PAIRS = [
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
    (4, 0), (8, 0), (12, 0), (16, 0), (20, 0),
    (8, 6), (12, 10), (16, 14), (20, 18),
    (4, 2), (8, 5), (12, 9), (16, 13), (20, 17),
    (5, 9), (9, 13), (13, 17), (5, 17),
    (6, 10), (10, 14),
]


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


def open_camera(camera_index: int) -> cv2.VideoCapture:
    if PREFER_V4L2_BACKEND:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if CAMERA_FOURCC and len(CAMERA_FOURCC) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))

    if ENABLE_AUTOFOCUS:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    return cap


def print_camera_settings(cap: cv2.VideoCapture, camera_index: int) -> None:
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"Camera[{camera_index}] active: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")


def measure_quality(roi_bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


def measure_hand_sharpness(frame_bgr: np.ndarray, hand_landmarks, margin: int = 12) -> float:
    h, w = frame_bgr.shape[:2]
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(w, max(xs) + margin)
    y2 = min(h, max(ys) + margin)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0

    patch = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def landmarks_inside_roi(hand_landmarks, frame_w: int, frame_h: int, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    for lm in hand_landmarks.landmark:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        if px < x1 or px > x2 or py < y1 or py > y2:
            return False
    return True


def hand_bbox_area(hand_landmarks, frame_w: int, frame_h: int) -> float:
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]
    return float(max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys))))


def collect_hand_infos(results, frame_w: int, frame_h: int, roi: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    if not results.multi_hand_landmarks:
        return infos

    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
        hand_label = "Unknown"
        if results.multi_handedness and idx < len(results.multi_handedness):
            hand_label = results.multi_handedness[idx].classification[0].label
        infos.append(
            {
                "idx": idx,
                "handedness": hand_label,
                "in_roi": landmarks_inside_roi(hand_lm, frame_w, frame_h, roi),
                "area": hand_bbox_area(hand_lm, frame_w, frame_h),
                "landmarks": hand_lm,
            }
        )
    return infos


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_t = float(np.dot(v1, v2) / (n1 * n2))
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def normalize_landmarks_points(hand_landmarks, mirror_x: bool) -> np.ndarray:
    pts = np.array([[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_landmarks.landmark], dtype=np.float32)
    if mirror_x:
        pts[:, 0] = 1.0 - pts[:, 0]

    wrist = pts[0:1, :]
    rel = pts - wrist
    scale = float(np.max(np.linalg.norm(rel, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return rel / scale


def compute_joint_angles(points_21x3: np.ndarray) -> np.ndarray:
    vals: list[float] = []
    for finger in FINGER_NAME_ORDER:
        c = FINGER_CHAINS[finger]
        vals.append(_angle_deg(points_21x3[c[0]], points_21x3[c[1]], points_21x3[c[2]]))
        vals.append(_angle_deg(points_21x3[c[1]], points_21x3[c[2]], points_21x3[c[3]]))
        vals.append(_angle_deg(points_21x3[c[2]], points_21x3[c[3]], points_21x3[c[4]]))
    out = np.array(vals, dtype=np.float32)
    if out.shape[0] != FEATURE_JOINT_ANGLE_DIM:
        raise ValueError(f"Joint angle dim mismatch: expected {FEATURE_JOINT_ANGLE_DIM}, got {out.shape[0]}")
    return out


def compute_pairwise_distances(points_21x3: np.ndarray) -> np.ndarray:
    d = [float(np.linalg.norm(points_21x3[i] - points_21x3[j])) for i, j in PAIRWISE_DISTANCE_PAIRS]
    out = np.array(d, dtype=np.float32)
    if out.shape[0] != FEATURE_PAIRWISE_DISTANCE_DIM:
        raise ValueError(f"Pairwise dim mismatch: expected {FEATURE_PAIRWISE_DISTANCE_DIM}, got {out.shape[0]}")
    return out


def compute_finger_analysis_metrics(points_21x3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles: list[float] = []
    ratios: list[float] = []
    wrist = points_21x3[0]

    for finger in FINGER_NAME_ORDER:
        c = FINGER_CHAINS[finger]
        # PIP-equivalent metric on middle joints for each finger.
        angle = _angle_deg(points_21x3[c[1]], points_21x3[c[2]], points_21x3[c[3]])
        tip = points_21x3[c[4]]
        mcp = points_21x3[c[1]]
        ratio = float(np.linalg.norm(tip - wrist) / max(np.linalg.norm(mcp - wrist), 1e-8))
        angles.append(angle)
        ratios.append(ratio)

    return np.array(angles, dtype=np.float32), np.array(ratios, dtype=np.float32)


def compute_finger_states_three_level(angles5: np.ndarray, ratios5: np.ndarray) -> np.ndarray:
    states: list[int] = []
    for i in range(5):
        angle = float(angles5[i])
        ratio = float(ratios5[i])

        # Angle-only vote (global thresholds)
        if angle < 110.0:
            angle_state = 0
        elif angle > 160.0:
            angle_state = 2
        else:
            angle_state = 1

        # Ratio-only vote (global thresholds)
        if ratio < 1.35:
            ratio_state = 0
        elif ratio > 1.75:
            ratio_state = 2
        elif 1.10 <= ratio <= 1.90:
            ratio_state = 1
        else:
            ratio_state = 1

        # Conflict handling: prioritize ratio when disagreement occurs.
        states.append(ratio_state if ratio_state != angle_state else angle_state)
    out = np.array(states, dtype=np.float32)
    if out.shape[0] != FEATURE_FINGER_STATE_DIM:
        raise ValueError(f"Finger-state dim mismatch: expected {FEATURE_FINGER_STATE_DIM}, got {out.shape[0]}")
    return out


def compute_palm_normal(points_21x3: np.ndarray) -> np.ndarray:
    wrist = points_21x3[0]
    index_mcp = points_21x3[5]
    pinky_mcp = points_21x3[17]
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    # Use opposite normal direction to match camera-facing coordinate convention
    # used by the on-screen mirrored preview and expected labels.
    normal = np.cross(v1, v2)
    n = float(np.linalg.norm(normal))
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (-normal / n).astype(np.float32)


def classify_palm_orientation6(normal: np.ndarray) -> str:
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    abs_vals = [abs(nx), abs(ny), abs(nz)]
    k = int(np.argmax(abs_vals))
    if k == 0:
        return "right" if nx > 0 else "left"
    if k == 1:
        return "down" if ny > 0 else "up"
    return "face" if nz > 0 else "camera"


def palm_direction_one_hot6(label: str) -> np.ndarray:
    out = np.zeros((FEATURE_PALM_DIRECTION_DIM,), dtype=np.float32)
    mapping = {name: i for i, name in enumerate(PALM_ORIENTATION_6)}
    if label in mapping:
        out[mapping[label]] = 1.0
    return out


def build_feature_vector(points_21x3: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    landmarks = points_21x3.reshape(-1).astype(np.float32)
    if landmarks.shape[0] != FEATURE_LANDMARK_DIM:
        raise ValueError(f"Landmark dim mismatch: expected {FEATURE_LANDMARK_DIM}, got {landmarks.shape[0]}")

    joint_angles = compute_joint_angles(points_21x3)
    pairwise = compute_pairwise_distances(points_21x3)
    analysis_angles, analysis_ratios = compute_finger_analysis_metrics(points_21x3)
    finger_states = compute_finger_states_three_level(analysis_angles, analysis_ratios)
    normal = compute_palm_normal(points_21x3)
    palm_label6 = classify_palm_orientation6(normal)
    palm6 = palm_direction_one_hot6(palm_label6)

    feat = np.concatenate([landmarks, joint_angles, pairwise, finger_states, palm6], axis=0).astype(np.float32)
    if feat.shape[0] != FEATURE_TOTAL_DIM:
        raise ValueError(f"Feature dim mismatch: expected {FEATURE_TOTAL_DIM}, got {feat.shape[0]}")
    return feat, finger_states, palm_label6


def _extract_numeric_suffix(stem: str) -> int:
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return -1
    return int(digits)


def next_sample_index(rgb_dir: Path, lm_dir: Path) -> int:
    mx = 0
    for p in rgb_dir.glob("img*.png"):
        mx = max(mx, _extract_numeric_suffix(p.stem))
    for p in lm_dir.glob("lm*.npy"):
        mx = max(mx, _extract_numeric_suffix(p.stem))
    return mx + 1


def append_capture_log(
    sample_id: int,
    label: str,
    rgb_path: Path,
    feature_path: Path,
    finger_states: np.ndarray,
    palm_direction: str,
) -> None:
    CAPTURE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CAPTURE_LOG_FILE.exists()
    with CAPTURE_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "sample_id", "label", "rgb_path", "feature_path", "finger_states", "palm_direction"])
        writer.writerow(
            [
                int(time.time()),
                sample_id,
                label,
                str(rgb_path),
                str(feature_path),
                "".join(str(int(v)) for v in finger_states.tolist()),
                palm_direction,
            ]
        )


def append_analysis_log(angles5: np.ndarray, ratios5: np.ndarray, palm_orientation: str) -> None:
    FINGER_ANALYSIS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FINGER_ANALYSIS_LOG_FILE.exists()
    with FINGER_ANALYSIS_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "palm_orientation",
                    "thumb_angle", "index_angle", "middle_angle", "ring_angle", "pinky_angle",
                    "thumb_ratio", "index_ratio", "middle_ratio", "ring_ratio", "pinky_ratio",
                ]
            )
        writer.writerow(
            [
                int(time.time()),
                palm_orientation,
                *[f"{v:.3f}" for v in angles5.tolist()],
                *[f"{v:.4f}" for v in ratios5.tolist()],
            ]
        )


def _class_pairs(class_name: str) -> tuple[set[int], set[int]]:
    rgb_dir = RAW_RGB_DIR / class_name
    lm_dir = RAW_LANDMARK_DIR / class_name

    rgb_ids = {_extract_numeric_suffix(p.stem) for p in rgb_dir.glob("img*.png") if _extract_numeric_suffix(p.stem) >= 0}
    lm_ids = {_extract_numeric_suffix(p.stem) for p in lm_dir.glob("lm*.npy") if _extract_numeric_suffix(p.stem) >= 0}
    return rgb_ids, lm_ids


def build_dataset_validation(classes: list[str], target_images: int) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []

    fstate_dist = {finger: {"0": 0, "1": 0, "2": 0} for finger in FINGER_NAME_ORDER}
    palm_dist = {k: 0 for k in PALM_ORIENTATION_6}
    invalid_feature_dim = 0

    total_paired = 0
    total_rgb = 0
    total_lm = 0

    for class_name in classes:
        rgb_ids, lm_ids = _class_pairs(class_name)
        paired = sorted(rgb_ids & lm_ids)
        missing_rgb = sorted(lm_ids - rgb_ids)
        missing_lm = sorted(rgb_ids - lm_ids)

        total_paired += len(paired)
        total_rgb += len(rgb_ids)
        total_lm += len(lm_ids)

        for sid in paired:
            feature_path = RAW_LANDMARK_DIR / class_name / f"lm{sid:06d}.npy"
            vec = np.load(feature_path).reshape(-1)
            if vec.shape[0] != FEATURE_TOTAL_DIM:
                invalid_feature_dim += 1
                continue

            s0 = FEATURE_LANDMARK_DIM + FEATURE_JOINT_ANGLE_DIM + FEATURE_PAIRWISE_DISTANCE_DIM
            states = vec[s0:s0 + FEATURE_FINGER_STATE_DIM]
            p0 = s0 + FEATURE_FINGER_STATE_DIM
            palm6 = vec[p0:p0 + FEATURE_PALM_DIRECTION_DIM]

            for i, finger in enumerate(FINGER_NAME_ORDER):
                k = str(int(np.clip(round(float(states[i])), 0, 2)))
                fstate_dist[finger][k] += 1

            p_idx = int(np.argmax(palm6)) if np.sum(palm6) > 0 else -1
            if 0 <= p_idx < len(PALM_ORIENTATION_6):
                palm_dist[PALM_ORIENTATION_6[p_idx]] += 1

        per_class.append(
            {
                "class": class_name,
                "rgb_count": len(rgb_ids),
                "feature_count": len(lm_ids),
                "paired_count": len(paired),
                "target": target_images,
                "coverage_pct": round(100.0 * len(paired) / max(target_images, 1), 2),
                "missing_rgb_count": len(missing_rgb),
                "missing_feature_count": len(missing_lm),
            }
        )

    paired_vals = [x["paired_count"] for x in per_class]
    min_count = min(paired_vals) if paired_vals else 0
    max_count = max(paired_vals) if paired_vals else 0
    imbalance_ratio = float(max_count / max(min_count, 1)) if paired_vals else 0.0

    if imbalance_ratio <= IMBALANCE_WARN_RATIO:
        balance_state = "ok"
    elif imbalance_ratio <= IMBALANCE_FAIL_RATIO:
        balance_state = "warn"
    else:
        balance_state = "fail"

    return {
        "timestamp": int(time.time()),
        "target_images_per_class": target_images,
        "summary": {
            "total_rgb": total_rgb,
            "total_features": total_lm,
            "total_paired": total_paired,
            "invalid_feature_dim": invalid_feature_dim,
            "min_paired_count": min_count,
            "max_paired_count": max_count,
            "imbalance_ratio": round(imbalance_ratio, 4),
            "balance_state": balance_state,
        },
        "per_class": per_class,
        "finger_state_distribution": fstate_dist,
        "palm_direction_distribution": palm_dist,
    }


def write_dataset_validation_reports(validation: dict[str, Any]) -> None:
    DATASET_VALIDATION_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_VALIDATION_JSON_FILE.open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    s = validation["summary"]
    lines = [
        "# Dataset Validation Report",
        "",
        f"- Timestamp: {validation['timestamp']}",
        f"- Total RGB files: {s['total_rgb']}",
        f"- Total feature files: {s['total_features']}",
        f"- Total paired samples: {s['total_paired']}",
        f"- Invalid feature dimension count: {s['invalid_feature_dim']}",
        f"- Min paired class count: {s['min_paired_count']}",
        f"- Max paired class count: {s['max_paired_count']}",
        f"- Imbalance ratio: {s['imbalance_ratio']}",
        f"- Balance state: {s['balance_state']}",
        "",
        "## Per-class Pairing",
        "",
        "| Class | RGB | Features | Paired | Target | Coverage % | Missing RGB | Missing Feature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation["per_class"]:
        lines.append(
            f"| {row['class']} | {row['rgb_count']} | {row['feature_count']} | {row['paired_count']} | {row['target']} | {row['coverage_pct']} | {row['missing_rgb_count']} | {row['missing_feature_count']} |"
        )

    lines.extend([
        "",
        "## Finger-state Distribution",
        "",
        "| Finger | Closed(0) | Half-open(1) | Open(2) |",
        "|---|---:|---:|---:|",
    ])
    for finger in FINGER_NAME_ORDER:
        d = validation["finger_state_distribution"][finger]
        lines.append(f"| {finger} | {d['0']} | {d['1']} | {d['2']} |")

    lines.extend([
        "",
        "## Palm-direction Distribution",
        "",
        "| Direction | Count |",
        "|---|---:|",
    ])
    for k, v in validation["palm_direction_distribution"].items():
        lines.append(f"| {k} | {v} |")

    with DATASET_VALIDATION_MD_FILE.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_preview_mosaic(classes: list[str], target_images: int, cell_size: int = 220) -> np.ndarray:
    cols = 7
    rows = int(np.ceil(len(classes) / cols))
    header_h = 36
    canvas = np.full((rows * cell_size + header_h, cols * cell_size, 3), 250, dtype=np.uint8)

    for idx, class_name in enumerate(classes):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_size
        y0 = header_h + r * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 1)
        rgb_dir = RAW_RGB_DIR / class_name
        files = sorted(rgb_dir.glob("img*.png")) if rgb_dir.exists() else []
        count = len(files)
        coverage = 100.0 * count / max(target_images, 1)

        if files:
            sample = cv2.imread(str(files[-1]))
            if sample is not None:
                sample = cv2.resize(sample, (cell_size - 10, cell_size - 48), interpolation=cv2.INTER_AREA)
                canvas[y0 + 28:y0 + 28 + sample.shape[0], x0 + 5:x0 + 5 + sample.shape[1]] = sample
        else:
            cv2.putText(canvas, "MISSING", (x0 + 45, y0 + cell_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 220), 2)

        color = (40, 140, 40) if count >= target_images else (30, 30, 30)
        cv2.putText(canvas, class_name, (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas, f"n={count}", (x0 + 70, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
        cv2.putText(canvas, f"{coverage:.1f}%", (x0 + 140, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)

    cv2.putText(canvas, "Dataset Preview (RGB)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return canvas


def draw_progress_bar(frame: np.ndarray, saved_count: int, target_images: int, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    ratio = float(np.clip(saved_count / max(target_images, 1), 0.0, 1.0))
    fill_w = int(ratio * w)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), (80, 210, 120), -1)
    cv2.putText(frame, f"{ratio * 100:.1f}%", (x + w + 8, y + h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)


def draw_capture_feedback(
    frame: np.ndarray,
    gesture: str,
    status: str,
    handedness: str,
    saved_count: int,
    target_images: int,
    interval_sec: float,
    auto_capture: bool,
    fps: float,
    gate_checks: dict[str, bool],
    finger_states: np.ndarray | None,
    palm_orientation6: str,
) -> None:
    line_h = 24
    x0 = 20
    y0 = 30

    mode_text = "RUN" if auto_capture else "PAUSE"
    mode_color = (80, 220, 120) if auto_capture else (60, 160, 255)
    cv2.putText(frame, f"Mode: {mode_text}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2)
    cv2.putText(frame, f"Class: {gesture}", (x0, y0 + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(frame, f"Saved: {saved_count}/{target_images}", (x0, y0 + 2 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(frame, f"Status: {status}", (x0, y0 + 3 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 240, 120), 2)
    cv2.putText(frame, f"Hand: {handedness}", (x0, y0 + 4 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 240, 120), 2)
    cv2.putText(frame, f"Palm(6): {palm_orientation6}", (x0, y0 + 5 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 245, 220), 1)

    gate_y = y0 + 6 * line_h
    for idx, (name, passed) in enumerate(gate_checks.items()):
        color = (80, 220, 120) if passed else (70, 90, 250)
        mark = "OK" if passed else "NO"
        cv2.putText(frame, f"- {name}: {mark}", (x0, gate_y + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    if finger_states is not None and finger_states.shape[0] == 5:
        fs_text = " ".join(f"{n[0].upper()}:{int(v)}" for n, v in zip(FINGER_NAME_ORDER, finger_states.tolist()))
        cv2.putText(frame, f"Finger states: {fs_text}", (x0, gate_y + 5 * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 245, 220), 1)

    draw_progress_bar(frame, saved_count, target_images, x0, gate_y + 125, 280, 14)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | interval={interval_sec:.3f}s | keys: s=start/pause, q=quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (220, 220, 220),
        1,
    )


def draw_finger_analysis_overlay(
    frame: np.ndarray,
    angles5: np.ndarray | None,
    ratios5: np.ndarray | None,
    palm6: str,
    roi_status: str,
    fps: float,
) -> None:
    x0 = 20
    y0 = 30
    line_h = 24

    cv2.putText(frame, "Finger Analysis Mode", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 120), 2)
    cv2.putText(frame, f"ROI status: {roi_status}", (x0, y0 + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 240, 120), 2)
    cv2.putText(frame, f"Palm orientation: {palm6}", (x0, y0 + 2 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 245, 220), 1)

    if angles5 is not None and ratios5 is not None:
        for i, finger in enumerate(FINGER_NAME_ORDER):
            y = y0 + (4 + i) * line_h
            cv2.putText(
                frame,
                f"{finger.upper()}: angle={angles5[i]:6.1f} deg ratio={ratios5[i]:.3f}",
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (80, 220, 120),
                1,
            )

    cv2.putText(frame, f"FPS: {fps:.1f} | keys: k=log row, q=quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1)


def run_dataset_preview_and_validation(classes: list[str], target_images: int, no_window: bool) -> None:
    validation = build_dataset_validation(classes, target_images)
    write_dataset_validation_reports(validation)

    preview_img = build_preview_mosaic(classes=classes, target_images=target_images)
    DATASET_PREVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DATASET_PREVIEW_FILE), preview_img)

    print(f"Wrote {DATASET_VALIDATION_JSON_FILE}")
    print(f"Wrote {DATASET_VALIDATION_MD_FILE}")
    print(f"Wrote {DATASET_PREVIEW_FILE}")
    print(json.dumps(validation["summary"], indent=2))

    if not no_window:
        cv2.imshow("ASL Dataset Preview", preview_img)
        cv2.waitKey(0)
        cv2.destroyWindow("ASL Dataset Preview")


def run_capture(
    gesture: str,
    target_images: int,
    camera_index: int,
    capture_interval_sec: float,
) -> None:
    classes = load_classes(CLASS_FILE)
    if gesture not in classes:
        raise ValueError(f"Gesture '{gesture}' is not in classes.json")

    rgb_dir = RAW_RGB_DIR / gesture
    lm_dir = RAW_LANDMARK_DIR / gesture
    rgb_dir.mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    print_camera_settings(cap, camera_index)

    sample_idx = next_sample_index(rgb_dir, lm_dir)
    saved_count = len(list(rgb_dir.glob("img*.png")))
    last_capture_ts = 0.0
    auto_capture = False
    prev_frame_ts = time.time()
    skeleton_window_name = "ASL Skeleton White"
    overlay_window_name = "ASL Overlay Skeleton"
    simple_window_name = "ASL RGB ROI Clean"

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()
            frame_overlay = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            roi = compute_right_square_roi(frame_w, frame_h)
            x1, y1, x2, y2 = roi
            roi_bgr_clean = frame_clean[y1:y2, x1:x2].copy()

            results = hands.process(cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            hand_infos = collect_hand_infos(results, frame_w, frame_h, roi)

            selected = None
            status = "No hand"
            handedness = "None"
            can_capture = False
            finger_states_live = None
            palm_live6 = "camera"
            gate_checks = {
                "hand_detected": len(hand_infos) > 0,
                "inside_roi": False,
                "brightness": False,
                "sharpness": False,
            }

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_overlay,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            brightness, sharpness = measure_quality(roi_bgr_clean)
            inside = [h for h in hand_infos if h["in_roi"]]

            if len(hand_infos) > 1:
                status = "Multiple hands detected"
            elif len(inside) == 0:
                status = "Move hand fully inside ROI"
            else:
                candidate = inside[0]
                if REQUIRED_HANDEDNESS and candidate["handedness"] != REQUIRED_HANDEDNESS:
                    status = f"{REQUIRED_HANDEDNESS} hand not found"
                else:
                    selected = candidate
                    handedness = str(candidate["handedness"])

            if selected is not None:
                # For virtual cams, evaluate sharpness on hand crop (more stable than whole ROI).
                sharpness = measure_hand_sharpness(frame_clean, selected["landmarks"])
                gate_checks["inside_roi"] = True
                if brightness >= MIN_BRIGHTNESS:
                    gate_checks["brightness"] = True
                if sharpness >= MIN_SHARPNESS:
                    gate_checks["sharpness"] = True

                if brightness < MIN_BRIGHTNESS:
                    status = "Too dark"
                elif sharpness < MIN_SHARPNESS:
                    status = "Too blurry"
                else:
                    mirror = MIRROR_LEFT_HAND_TO_RIGHT and handedness.startswith("Left")
                    points = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                    _, finger_states_live, _ = build_feature_vector(points)
                    palm_live6 = classify_palm_orientation6(compute_palm_normal(points))
                    can_capture = True
                    status = "Ready"

            skeleton_canvas = render_skeleton_canvas(
                results=results,
                frame_shape=frame_clean.shape,
                roi=roi,
                canvas_size=RGB_IMAGE_SIZE,
                selected_hand_index=None,
                encode_depth=True,
            )

            now = time.time()
            if auto_capture and can_capture and now - last_capture_ts >= capture_interval_sec and saved_count < target_images:
                sample_id = sample_idx
                rgb_path = rgb_dir / f"img{sample_id:06d}.png"
                feature_path = lm_dir / f"lm{sample_id:06d}.npy"

                # Save strictly clean RGB ROI (no landmarks/skeleton overlays).
                rgb_crop = cv2.resize(roi_bgr_clean, RGB_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(rgb_path), rgb_crop)

                mirror = MIRROR_LEFT_HAND_TO_RIGHT and handedness.startswith("Left")
                points = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                feature_vec, finger_states, palm_direction = build_feature_vector(points)
                np.save(str(feature_path), feature_vec.astype(np.float32))

                append_capture_log(
                    sample_id=sample_id,
                    label=gesture,
                    rgb_path=rgb_path,
                    feature_path=feature_path,
                    finger_states=finger_states,
                    palm_direction=palm_direction,
                )

                sample_idx += 1
                saved_count += 1
                last_capture_ts = now

            now_fps = time.time()
            fps = 1.0 / max(now_fps - prev_frame_ts, 1e-6)
            prev_frame_ts = now_fps

            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), (40, 255, 40), 2)

            # Window 2: white-background skeleton.
            cv2.imshow(skeleton_window_name, cv2.resize(skeleton_canvas, (RGB_IMAGE_SIZE[0] * 2, RGB_IMAGE_SIZE[1] * 2), interpolation=cv2.INTER_NEAREST))

            draw_capture_feedback(
                frame=frame_overlay,
                gesture=gesture,
                status=status,
                handedness=handedness,
                saved_count=saved_count,
                target_images=target_images,
                interval_sec=capture_interval_sec,
                auto_capture=auto_capture,
                fps=fps,
                gate_checks=gate_checks,
                finger_states=finger_states_live,
                palm_orientation6=palm_live6,
            )

            if saved_count >= target_images:
                cv2.putText(frame_overlay, "Target reached", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                auto_capture = False

            # Window 1: simple clean RGB view with ROI only.
            cv2.imshow(simple_window_name, frame_clean)
            # Window 3: current style on RGB with landmarks/skeleton overlay.
            cv2.imshow(overlay_window_name, frame_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                auto_capture = not auto_capture

    cap.release()
    cv2.destroyAllWindows()


def run_finger_analysis(camera_index: int) -> None:
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    print_camera_settings(cap, camera_index)

    prev_ts = time.time()
    skeleton_window_name = "ASL Skeleton White"
    overlay_window_name = "ASL Analysis Overlay"
    simple_window_name = "ASL Analysis RGB ROI Clean"
    windows_positioned = False

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()
            frame_overlay = frame.copy()
            h, w = frame.shape[:2]
            roi = compute_right_square_roi(w, h)
            x1, y1, x2, y2 = roi
            results = hands.process(cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            hand_infos = collect_hand_infos(results, w, h, roi)
            inside = [x for x in hand_infos if x["in_roi"]]

            roi_status = "No hand"
            angles5 = None
            ratios5 = None
            palm6 = "camera"

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_overlay,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            if len(inside) == 0:
                roi_status = "Hand not fully inside ROI"
            else:
                selected = max(inside, key=lambda x: x["area"])
                mirror = MIRROR_LEFT_HAND_TO_RIGHT and str(selected["handedness"]) == "Left"
                points = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                angles5, ratios5 = compute_finger_analysis_metrics(points)
                palm6 = classify_palm_orientation6(compute_palm_normal(points))
                roi_status = "Ready"

            skeleton_canvas = render_skeleton_canvas(
                results=results,
                frame_shape=frame_clean.shape,
                roi=roi,
                canvas_size=RGB_IMAGE_SIZE,
                selected_hand_index=None,
                encode_depth=True,
            )

            now = time.time()
            fps = 1.0 / max(now - prev_ts, 1e-6)
            prev_ts = now

            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), (40, 255, 40), 2)

            # Window 2: white-background skeleton.
            cv2.imshow(skeleton_window_name, cv2.resize(skeleton_canvas, (RGB_IMAGE_SIZE[0] * 2, RGB_IMAGE_SIZE[1] * 2), interpolation=cv2.INTER_NEAREST))
            # Window 3: overlay/landmark view.
            draw_finger_analysis_overlay(frame_overlay, angles5, ratios5, palm6, roi_status, fps)

            # Window 1: simple clean RGB view with ROI only.
            cv2.imshow(simple_window_name, frame_clean)
            cv2.imshow(overlay_window_name, frame_overlay)

            if not windows_positioned:
                margin = 20
                overlay_x, overlay_y = margin, 80
                rgb_x = overlay_x + frame_overlay.shape[1] + margin
                rgb_y = overlay_y

                skeleton_w = RGB_IMAGE_SIZE[0] * 2
                skeleton_h = RGB_IMAGE_SIZE[1] * 2
                skeleton_x = rgb_x
                skeleton_y = max(margin, rgb_y - skeleton_h - margin)

                # Keep current window sizes; only adjust placement.
                cv2.moveWindow(overlay_window_name, overlay_x, overlay_y)
                cv2.moveWindow(simple_window_name, rgb_x, rgb_y)
                cv2.moveWindow(skeleton_window_name, skeleton_x, skeleton_y)
                windows_positioned = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("k") and angles5 is not None and ratios5 is not None:
                append_analysis_log(angles5, ratios5, palm6)

    cap.release()
    cv2.destroyAllWindows()


def _top_label_and_confidence(prob: np.ndarray, classes: list[str]) -> tuple[str, float]:
    idx = int(np.argmax(prob))
    return classes[idx], float(prob[idx])


def _top2_indices(prob: np.ndarray) -> tuple[int, int]:
    order = np.argsort(prob)
    top1 = int(order[-1])
    top2 = int(order[-2]) if prob.shape[0] > 1 else top1
    return top1, top2


def decide_hierarchical_label(
    p_stack: np.ndarray,
    p_xgb: np.ndarray,
    p_rf: np.ndarray,
    p_mlp: np.ndarray,
    p_cnn: np.ndarray,
    classes: list[str],
) -> tuple[str, float, str]:
    top_idx, second_idx = _top2_indices(p_stack)
    top_conf = float(p_stack[top_idx])
    second_conf = float(p_stack[second_idx])
    margin = top_conf - second_conf

    if top_conf >= HIERARCHY_STACK_ACCEPT_CONF and margin >= HIERARCHY_STACK_ACCEPT_MARGIN:
        return classes[top_idx], top_conf, "stack_high_conf"

    base_probs = [p_xgb, p_rf, p_mlp, p_cnn]
    base_top_idx = [int(np.argmax(p)) for p in base_probs]
    votes: dict[int, int] = {}
    for idx in base_top_idx:
        votes[idx] = votes.get(idx, 0) + 1

    agree_idx, agree_votes = max(votes.items(), key=lambda kv: kv[1])
    if agree_votes >= HIERARCHY_BASE_MIN_AGREEMENT:
        confs = [float(p[agree_idx]) for p in base_probs if int(np.argmax(p)) == agree_idx]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        if mean_conf >= HIERARCHY_BASE_MEAN_CONF:
            return classes[agree_idx], mean_conf, f"base_agreement_{agree_votes}"

    support_top2 = 0
    for p in base_probs:
        b1, b2 = _top2_indices(p)
        if top_idx in (b1, b2):
            support_top2 += 1
    if top_conf >= HIERARCHY_STACK_SUPPORT_CONF and support_top2 >= HIERARCHY_STACK_SUPPORT_TOP2_MIN:
        return classes[top_idx], top_conf, f"stack_supported_{support_top2}"

    return "UNCERTAIN", top_conf, "reject_low_conflict"


def smooth_temporal_label(
    history: deque[tuple[str, float]],
    min_frames: int,
) -> tuple[str, float, int]:
    if len(history) < max(1, min_frames):
        return "UNCERTAIN", 0.0, 0

    score_by_label: dict[str, float] = {}
    count_by_label: dict[str, int] = {}
    for label, conf in history:
        if label == "UNCERTAIN":
            continue
        score_by_label[label] = score_by_label.get(label, 0.0) + float(conf)
        count_by_label[label] = count_by_label.get(label, 0) + 1

    if not score_by_label:
        return "UNCERTAIN", 0.0, 0

    best_label = max(score_by_label.keys(), key=lambda k: (score_by_label[k], count_by_label[k]))
    best_score = float(score_by_label[best_label])
    support = int(count_by_label[best_label])
    norm_conf = best_score / float(len(history))
    return best_label, norm_conf, support


def run_live_model_predictions(camera_index: int) -> None:
    import joblib
    import tensorflow as tf

    classes = load_classes(CLASS_FILE)
    xgb_path = MODELS_DIR / "xgboost.pkl"
    rf_path = MODELS_DIR / "rf.pkl"
    mlp_path = MODELS_DIR / "mlp.pkl"
    cnn_path = MODELS_DIR / "cnn.h5"

    required = [xgb_path, rf_path, mlp_path, cnn_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
    cnn = tf.keras.models.load_model(str(cnn_path))

    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    print_camera_settings(cap, camera_index)

    predict_window = "ASL Live Predict Models"
    prev_ts = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()
            frame_pred = frame.copy()

            h, w = frame.shape[:2]
            roi = compute_right_square_roi(w, h)
            x1, y1, x2, y2 = roi
            roi_clean = frame_clean[y1:y2, x1:x2].copy()

            results = hands.process(cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            hand_infos = collect_hand_infos(results, w, h, roi)
            inside = [x for x in hand_infos if x["in_roi"]]

            selected = None
            status = "No hand"
            if len(hand_infos) > 1:
                status = "Multiple hands detected"
            elif len(inside) == 0:
                status = "Move hand inside ROI"
            else:
                selected = inside[0]
                status = "Ready"

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_pred,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.rectangle(frame_pred, (x1, y1), (x2, y2), (40, 255, 40), 2)

            now = time.time()
            fps = 1.0 / max(now - prev_ts, 1e-6)
            prev_ts = now

            cv2.putText(frame_pred, f"Status: {status}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 240, 120), 2)
            cv2.putText(frame_pred, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

            y_text = 98
            if selected is not None:
                mirror = MIRROR_LEFT_HAND_TO_RIGHT and str(selected["handedness"]) == "Left"
                pts = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                feat, _, _ = build_feature_vector(pts)

                p_xgb = np.asarray(xgb.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_rf = np.asarray(rf.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_mlp = np.asarray(mlp.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)

                rgb_in = cv2.resize(roi_clean, RGB_IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
                p_cnn = np.asarray(cnn.predict(np.expand_dims(rgb_in, axis=0), verbose=0)[0], dtype=np.float32)

                preds = {
                    "XGB": _top_label_and_confidence(p_xgb, classes),
                    "RF": _top_label_and_confidence(p_rf, classes),
                    "MLP": _top_label_and_confidence(p_mlp, classes),
                    "CNN": _top_label_and_confidence(p_cnn, classes),
                }

                for name in ["XGB", "RF", "MLP", "CNN"]:
                    lab, conf = preds[name]
                    cv2.putText(
                        frame_pred,
                        f"{name}: {lab} ({conf:.3f})",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (80, 220, 120),
                        2,
                    )
                    y_text += 30

            cv2.putText(frame_pred, "q=quit", (20, frame_pred.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow(predict_window, frame_pred)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_live_stacked_prediction(camera_index: int) -> None:
    import joblib
    import tensorflow as tf

    stack_cfg_path = MODELS_DIR / "stacking_config.json"
    stack_meta_path = MODELS_DIR / "stack_meta.pkl"
    if not stack_cfg_path.exists() or not stack_meta_path.exists():
        raise FileNotFoundError(
            "Missing Step 3 files. Expected models/stack_meta.pkl and models/stacking_config.json"
        )

    with stack_cfg_path.open("r", encoding="utf-8") as f:
        stack_cfg = json.load(f)

    classes_cfg = list(stack_cfg.get("classes", []))
    if not classes_cfg:
        raise ValueError("Invalid stacking_config.json: classes is empty")

    classes_project = load_classes(CLASS_FILE)
    if classes_project != classes_cfg:
        raise ValueError("Class mismatch between classes.json and stacking_config.json")
    classes = classes_project

    image_size_cfg = stack_cfg.get("image_size", [RGB_IMAGE_SIZE[0], RGB_IMAGE_SIZE[1]])
    if not isinstance(image_size_cfg, list) or len(image_size_cfg) != 2:
        raise ValueError("Invalid stacking_config.json: image_size must be [H, W]")
    stack_h_cfg, stack_w_cfg = int(image_size_cfg[0]), int(image_size_cfg[1])

    xgb_path = MODELS_DIR / "xgboost.pkl"
    rf_path = MODELS_DIR / "rf.pkl"
    mlp_path = MODELS_DIR / "mlp.pkl"
    cnn_path = MODELS_DIR / "cnn.h5"

    required = [xgb_path, rf_path, mlp_path, cnn_path, stack_meta_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
    cnn = tf.keras.models.load_model(str(cnn_path))
    stack_meta = joblib.load(stack_meta_path)

    # Prefer the real CNN input shape; stacking_config may be stale after model replacement.
    stack_h = int(cnn.input_shape[1])
    stack_w = int(cnn.input_shape[2])
    if (stack_h, stack_w) != (stack_h_cfg, stack_w_cfg):
        print(
            f"Warning: stacking_config image_size=({stack_h_cfg},{stack_w_cfg}) "
            f"but cnn.h5 expects ({stack_h},{stack_w}); using CNN input size."
        )

    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    print_camera_settings(cap, camera_index)

    predict_window = "ASL Live Predict Stacked"
    prev_ts = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()
            frame_pred = frame.copy()

            h, w = frame.shape[:2]
            roi = compute_right_square_roi(w, h)
            x1, y1, x2, y2 = roi
            roi_clean = frame_clean[y1:y2, x1:x2].copy()

            results = hands.process(cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            hand_infos = collect_hand_infos(results, w, h, roi)
            inside = [x for x in hand_infos if x["in_roi"]]

            selected = None
            status = "No hand"
            if len(hand_infos) > 1:
                status = "Multiple hands detected"
            elif len(inside) == 0:
                status = "Move hand inside ROI"
            else:
                selected = inside[0]
                status = "Ready"

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_pred,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.rectangle(frame_pred, (x1, y1), (x2, y2), (40, 255, 40), 2)

            now = time.time()
            fps = 1.0 / max(now - prev_ts, 1e-6)
            prev_ts = now

            cv2.putText(frame_pred, f"Status: {status}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 240, 120), 2)
            cv2.putText(frame_pred, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

            y_text = 98
            if selected is not None:
                mirror = MIRROR_LEFT_HAND_TO_RIGHT and str(selected["handedness"]) == "Left"
                pts = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                feat, _, _ = build_feature_vector(pts)

                p_xgb = np.asarray(xgb.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_rf = np.asarray(rf.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_mlp = np.asarray(mlp.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)

                rgb_in = cv2.resize(roi_clean, (stack_w, stack_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                p_cnn = np.asarray(cnn.predict(np.expand_dims(rgb_in, axis=0), verbose=0)[0], dtype=np.float32)

                p_meta_in = np.concatenate([p_xgb, p_rf, p_mlp, p_cnn], axis=0).reshape(1, -1)
                p_stack = np.asarray(stack_meta.predict_proba(p_meta_in)[0], dtype=np.float32)

                preds = {
                    "STACK": _top_label_and_confidence(p_stack, classes),
                    "XGB": _top_label_and_confidence(p_xgb, classes),
                    "RF": _top_label_and_confidence(p_rf, classes),
                    "MLP": _top_label_and_confidence(p_mlp, classes),
                    "CNN": _top_label_and_confidence(p_cnn, classes),
                }

                stack_lab, stack_conf = preds["STACK"]
                cv2.putText(
                    frame_pred,
                    f"STACK: {stack_lab} ({stack_conf:.3f})",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (40, 240, 140),
                    2,
                )
                y_text += 34

                for name in ["XGB", "RF", "MLP", "CNN"]:
                    lab, conf = preds[name]
                    cv2.putText(
                        frame_pred,
                        f"{name}: {lab} ({conf:.3f})",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.66,
                        (80, 220, 120),
                        2,
                    )
                    y_text += 30

            cv2.putText(frame_pred, "q=quit", (20, frame_pred.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow(predict_window, frame_pred)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_live_hierarchical_prediction(camera_index: int) -> None:
    import joblib
    import tensorflow as tf

    stack_cfg_path = MODELS_DIR / "stacking_config.json"
    stack_meta_path = MODELS_DIR / "stack_meta.pkl"
    if not stack_cfg_path.exists() or not stack_meta_path.exists():
        raise FileNotFoundError(
            "Missing Step 3 files. Expected models/stack_meta.pkl and models/stacking_config.json"
        )

    with stack_cfg_path.open("r", encoding="utf-8") as f:
        stack_cfg = json.load(f)

    classes_cfg = list(stack_cfg.get("classes", []))
    if not classes_cfg:
        raise ValueError("Invalid stacking_config.json: classes is empty")

    classes_project = load_classes(CLASS_FILE)
    if classes_project != classes_cfg:
        raise ValueError("Class mismatch between classes.json and stacking_config.json")
    classes = classes_project

    xgb_path = MODELS_DIR / "xgboost.pkl"
    rf_path = MODELS_DIR / "rf.pkl"
    mlp_path = MODELS_DIR / "mlp.pkl"
    cnn_path = MODELS_DIR / "cnn.h5"

    required = [xgb_path, rf_path, mlp_path, cnn_path, stack_meta_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
    cnn = tf.keras.models.load_model(str(cnn_path))
    stack_meta = joblib.load(stack_meta_path)

    stack_h = int(cnn.input_shape[1])
    stack_w = int(cnn.input_shape[2])

    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    print_camera_settings(cap, camera_index)

    predict_window = "ASL Live Predict Hierarchy"
    prev_ts = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()
            frame_pred = frame.copy()

            h, w = frame.shape[:2]
            roi = compute_right_square_roi(w, h)
            x1, y1, x2, y2 = roi
            roi_clean = frame_clean[y1:y2, x1:x2].copy()

            results = hands.process(cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            hand_infos = collect_hand_infos(results, w, h, roi)
            inside = [x for x in hand_infos if x["in_roi"]]

            selected = None
            status = "No hand"
            if len(hand_infos) > 1:
                status = "Multiple hands detected"
            elif len(inside) == 0:
                status = "Move hand inside ROI"
            else:
                selected = inside[0]
                status = "Ready"

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_pred,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.rectangle(frame_pred, (x1, y1), (x2, y2), (40, 255, 40), 2)

            now = time.time()
            fps = 1.0 / max(now - prev_ts, 1e-6)
            prev_ts = now

            cv2.putText(frame_pred, f"Status: {status}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 240, 120), 2)
            cv2.putText(frame_pred, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

            y_text = 98
            if selected is not None:
                mirror = MIRROR_LEFT_HAND_TO_RIGHT and str(selected["handedness"]) == "Left"
                pts = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                feat, _, _ = build_feature_vector(pts)

                p_xgb = np.asarray(xgb.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_rf = np.asarray(rf.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_mlp = np.asarray(mlp.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)

                rgb_in = cv2.resize(roi_clean, (stack_w, stack_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                p_cnn = np.asarray(cnn.predict(np.expand_dims(rgb_in, axis=0), verbose=0)[0], dtype=np.float32)

                p_meta_in = np.concatenate([p_xgb, p_rf, p_mlp, p_cnn], axis=0).reshape(1, -1)
                p_stack = np.asarray(stack_meta.predict_proba(p_meta_in)[0], dtype=np.float32)

                final_label, final_conf, reason = decide_hierarchical_label(
                    p_stack=p_stack,
                    p_xgb=p_xgb,
                    p_rf=p_rf,
                    p_mlp=p_mlp,
                    p_cnn=p_cnn,
                    classes=classes,
                )

                final_color = (40, 240, 140) if final_label != "UNCERTAIN" else (70, 90, 250)
                cv2.putText(
                    frame_pred,
                    f"FINAL: {final_label} ({final_conf:.3f})",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    final_color,
                    2,
                )
                y_text += 34
                cv2.putText(
                    frame_pred,
                    f"Reason: {reason}",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (210, 220, 250),
                    1,
                )
                y_text += 30

                preds = {
                    "STACK": _top_label_and_confidence(p_stack, classes),
                    "XGB": _top_label_and_confidence(p_xgb, classes),
                    "RF": _top_label_and_confidence(p_rf, classes),
                    "MLP": _top_label_and_confidence(p_mlp, classes),
                    "CNN": _top_label_and_confidence(p_cnn, classes),
                }

                for name in ["STACK", "XGB", "RF", "MLP", "CNN"]:
                    lab, conf = preds[name]
                    cv2.putText(
                        frame_pred,
                        f"{name}: {lab} ({conf:.3f})",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (80, 220, 120),
                        2,
                    )
                    y_text += 28

            cv2.putText(frame_pred, "q=quit", (20, frame_pred.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow(predict_window, frame_pred)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_live_final_prediction(camera_index: int, debug_overlay: bool = True) -> None:
    import joblib
    import tensorflow as tf

    stack_cfg_path = MODELS_DIR / "stacking_config.json"
    stack_meta_path = MODELS_DIR / "stack_meta.pkl"
    if not stack_cfg_path.exists() or not stack_meta_path.exists():
        raise FileNotFoundError(
            "Missing Step 3 files. Expected models/stack_meta.pkl and models/stacking_config.json"
        )

    with stack_cfg_path.open("r", encoding="utf-8") as f:
        stack_cfg = json.load(f)

    classes_cfg = list(stack_cfg.get("classes", []))
    if not classes_cfg:
        raise ValueError("Invalid stacking_config.json: classes is empty")

    classes_project = load_classes(CLASS_FILE)
    if classes_project != classes_cfg:
        raise ValueError("Class mismatch between classes.json and stacking_config.json")
    classes = classes_project

    xgb_path = MODELS_DIR / "xgboost.pkl"
    rf_path = MODELS_DIR / "rf.pkl"
    mlp_path = MODELS_DIR / "mlp.pkl"
    cnn_path = MODELS_DIR / "cnn.h5"

    required = [xgb_path, rf_path, mlp_path, cnn_path, stack_meta_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    mlp = joblib.load(mlp_path)
    cnn = tf.keras.models.load_model(str(cnn_path))
    stack_meta = joblib.load(stack_meta_path)

    stack_h = int(cnn.input_shape[1])
    stack_w = int(cnn.input_shape[2])

    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    print_camera_settings(cap, camera_index)

    predict_window = "ASL Live Predict Final" if debug_overlay else "ASL Final UI"
    prev_ts = time.time()

    history: deque[tuple[str, float]] = deque(maxlen=TEMPORAL_WINDOW_SIZE)
    stable_label = "UNCERTAIN"
    stable_conf = 0.0
    stable_reason = "init"
    pending_label = ""
    pending_count = 0
    stable_age = 0
    paused = False

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DET_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()
            frame_pred = frame.copy()

            h, w = frame.shape[:2]
            roi = compute_right_square_roi(w, h)
            x1, y1, x2, y2 = roi
            roi_clean = frame_clean[y1:y2, x1:x2].copy()

            results = hands.process(cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            hand_infos = collect_hand_infos(results, w, h, roi)
            inside = [x for x in hand_infos if x["in_roi"]]

            selected = None
            status = "No hand"
            if len(hand_infos) > 1:
                status = "Multiple hands detected"
            elif len(inside) == 0:
                status = "Move hand inside ROI"
            else:
                selected = inside[0]
                status = "Ready"

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_pred,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (40, 255, 40), 2)
            cv2.rectangle(frame_pred, (x1, y1), (x2, y2), (40, 255, 40), 2)

            now = time.time()
            fps = 1.0 / max(now - prev_ts, 1e-6)
            prev_ts = now

            cv2.putText(frame_pred, f"Status: {status}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 240, 120), 2)
            cv2.putText(frame_pred, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

            y_text = 98
            if selected is not None and not paused:
                mirror = MIRROR_LEFT_HAND_TO_RIGHT and str(selected["handedness"]) == "Left"
                pts = normalize_landmarks_points(selected["landmarks"], mirror_x=mirror)
                feat, _, _ = build_feature_vector(pts)

                p_xgb = np.asarray(xgb.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_rf = np.asarray(rf.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
                p_mlp = np.asarray(mlp.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)

                rgb_in = cv2.resize(roi_clean, (stack_w, stack_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                p_cnn = np.asarray(cnn.predict(np.expand_dims(rgb_in, axis=0), verbose=0)[0], dtype=np.float32)

                p_meta_in = np.concatenate([p_xgb, p_rf, p_mlp, p_cnn], axis=0).reshape(1, -1)
                p_stack = np.asarray(stack_meta.predict_proba(p_meta_in)[0], dtype=np.float32)

                raw_label, raw_conf, raw_reason = decide_hierarchical_label(
                    p_stack=p_stack,
                    p_xgb=p_xgb,
                    p_rf=p_rf,
                    p_mlp=p_mlp,
                    p_cnn=p_cnn,
                    classes=classes,
                )

                history.append((raw_label, raw_conf))
                smooth_label, smooth_conf, support = smooth_temporal_label(history, min_frames=TEMPORAL_MIN_FRAMES)

                if smooth_label != "UNCERTAIN" and smooth_conf >= TEMPORAL_ACCEPT_CONF:
                    if smooth_label == stable_label:
                        stable_conf = smooth_conf
                        stable_reason = f"stable_support_{support}"
                        stable_age = 0
                        pending_label = ""
                        pending_count = 0
                    else:
                        if pending_label == smooth_label:
                            pending_count += 1
                        else:
                            pending_label = smooth_label
                            pending_count = 1
                        if pending_count >= TEMPORAL_DEBOUNCE_FRAMES:
                            stable_label = smooth_label
                            stable_conf = smooth_conf
                            stable_reason = f"debounced_{pending_count}"
                            stable_age = 0
                            pending_label = ""
                            pending_count = 0
                else:
                    pending_label = ""
                    pending_count = 0
                    stable_age += 1

                if stable_age > TEMPORAL_STABLE_TTL_FRAMES and raw_label == "UNCERTAIN":
                    stable_label = "UNCERTAIN"
                    stable_conf = 0.0
                    stable_reason = "stable_timeout"

                raw_color = (40, 240, 140) if raw_label != "UNCERTAIN" else (70, 90, 250)
                smooth_color = (40, 240, 140) if smooth_label != "UNCERTAIN" else (70, 90, 250)
                stable_color = (40, 240, 140) if stable_label != "UNCERTAIN" else (70, 90, 250)

                if debug_overlay:
                    cv2.putText(frame_pred, f"RAW: {raw_label} ({raw_conf:.3f})", (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.66, raw_color, 2)
                    y_text += 30
                    cv2.putText(
                        frame_pred,
                        f"SMOOTH: {smooth_label} ({smooth_conf:.3f}) w={len(history)} s={support}",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        smooth_color,
                        2,
                    )
                    y_text += 30
                    cv2.putText(frame_pred, f"FINAL: {stable_label} ({stable_conf:.3f})", (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.78, stable_color, 2)
                    y_text += 34
                    cv2.putText(
                        frame_pred,
                        f"Reason: raw={raw_reason}, final={stable_reason}, pend={pending_count}",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (210, 220, 250),
                        1,
                    )

            if not debug_overlay:
                # Final UI view: present only stable output and confidence bar.
                final_color = (40, 240, 140) if stable_label != "UNCERTAIN" else (70, 90, 250)
                cv2.putText(frame_pred, "ASL FINAL OUTPUT", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
                cv2.putText(frame_pred, f"{stable_label}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.6, final_color, 3)
                cv2.putText(frame_pred, f"Confidence: {stable_conf:.3f}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

                bar_x, bar_y, bar_w, bar_h = 20, 235, 360, 18
                cv2.rectangle(frame_pred, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (190, 190, 190), 1)
                fill = int(max(0.0, min(1.0, stable_conf)) * bar_w)
                cv2.rectangle(frame_pred, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), final_color, -1)

                mode_txt = "PAUSED" if paused else "LIVE"
                mode_color = (60, 160, 255) if paused else (80, 220, 120)
                cv2.putText(frame_pred, f"Mode: {mode_txt}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                cv2.putText(frame_pred, f"Status: {status}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 240, 120), 2)
                cv2.putText(frame_pred, f"FPS: {fps:.1f}", (20, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

            cv2.putText(
                frame_pred,
                "q=quit | space=pause/resume | c=clear stable",
                (20, frame_pred.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 220, 220),
                1,
            )

            cv2.imshow(predict_window, frame_pred)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("c"):
                stable_label = "UNCERTAIN"
                stable_conf = 0.0
                stable_reason = "manual_clear"
                pending_label = ""
                pending_count = 0
                stable_age = 0
                history.clear()

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 1 capture and validation tools")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gesture", help="Gesture class name from classes.json")
    mode.add_argument("--audit", action="store_true", help="Run dataset validation checks")
    mode.add_argument("--preview", action="store_true", help="Build dataset preview + validation")
    mode.add_argument("--analyze-fingers", action="store_true", help="Run finger analysis mode")
    mode.add_argument("--predict-models", action="store_true", help="Live per-model predictions in two windows")
    mode.add_argument("--predict-hierarchy", action="store_true", help="Live hierarchical prediction window (Step 4)")
    mode.add_argument("--predict-final", action="store_true", help="Live temporal-stabilized prediction window (Step 5)")
    mode.add_argument("--ui-final", action="store_true", help="Final operator UI mode (Step 6)")
    mode.add_argument("--predict-stacked", action="store_true", help="Live stacked-model prediction window (Step 3)")

    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_IMAGES_PER_CLASS, help="Target samples per class")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--interval", type=float, default=CAPTURE_INTERVAL_SEC, help="Seconds between captures")
    parser.add_argument("--no-preview-window", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    classes = load_classes(CLASS_FILE)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.audit:
        validation = build_dataset_validation(classes, args.target)
        write_dataset_validation_reports(validation)
        print(json.dumps(validation["summary"], indent=2))
        print(f"Wrote {DATASET_VALIDATION_JSON_FILE}")
        print(f"Wrote {DATASET_VALIDATION_MD_FILE}")
        return

    if args.preview:
        run_dataset_preview_and_validation(classes=classes, target_images=args.target, no_window=args.no_preview_window)
        return

    if args.analyze_fingers:
        run_finger_analysis(camera_index=args.camera)
        return

    if args.predict_models:
        run_live_model_predictions(camera_index=args.camera)
        return

    if args.predict_stacked:
        run_live_stacked_prediction(camera_index=args.camera)
        return

    if args.predict_hierarchy:
        run_live_hierarchical_prediction(camera_index=args.camera)
        return

    if args.predict_final:
        run_live_final_prediction(camera_index=args.camera, debug_overlay=True)
        return

    if args.ui_final:
        run_live_final_prediction(camera_index=args.camera, debug_overlay=False)
        return

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if not args.gesture:
        raise ValueError("--gesture is required for capture mode")

    run_capture(
        gesture=args.gesture,
        target_images=args.target,
        camera_index=args.camera,
        capture_interval_sec=args.interval,
    )

    validation = build_dataset_validation(classes, args.target)
    write_dataset_validation_reports(validation)
    print(json.dumps(validation["summary"], indent=2))
    print(f"Wrote {DATASET_VALIDATION_JSON_FILE}")
    print(f"Wrote {DATASET_VALIDATION_MD_FILE}")


if __name__ == "__main__":
    main()
