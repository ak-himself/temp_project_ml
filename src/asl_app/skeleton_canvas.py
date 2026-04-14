from __future__ import annotations

import cv2
import numpy as np

# Landmark index groups for coloring each finger differently.
THUMB = [0, 1, 2, 3, 4]
INDEX = [0, 5, 6, 7, 8]
MIDDLE = [0, 9, 10, 11, 12]
RING = [0, 13, 14, 15, 16]
PINKY = [0, 17, 18, 19, 20]
PALM = [0, 5, 9, 13, 17, 0]

# BGR colors (OpenCV).
COLOR_PALM = (70, 70, 70)
COLOR_THUMB = (0, 140, 255)
COLOR_INDEX = (0, 220, 80)
COLOR_MIDDLE = (255, 180, 0)
COLOR_RING = (255, 80, 180)
COLOR_PINKY = (180, 0, 255)


def _draw_chain(canvas: np.ndarray, points: list[tuple[int, int]], chain: list[int], color: tuple[int, int, int]) -> None:
    for i in range(len(chain) - 1):
        p1 = points[chain[i]]
        p2 = points[chain[i + 1]]
        cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)


def _to_canvas_points(
    hand_landmarks,
    frame_w: int,
    frame_h: int,
    roi: tuple[int, int, int, int],
    canvas_w: int,
    canvas_h: int,
) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = roi
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)

    points: list[tuple[int, int]] = []
    for lm in hand_landmarks.landmark:
        px = lm.x * frame_w
        py = lm.y * frame_h
        nx = (px - x1) / roi_w
        ny = (py - y1) / roi_h
        cx = int(np.clip(nx, 0.0, 1.0) * (canvas_w - 1))
        cy = int(np.clip(ny, 0.0, 1.0) * (canvas_h - 1))
        points.append((cx, cy))
    return points


def _point_depths(hand_landmarks) -> list[float]:
    zs = [float(lm.z) for lm in hand_landmarks.landmark]
    z_min = min(zs)
    z_max = max(zs)
    span = z_max - z_min
    if span < 1e-8:
        return [0.5 for _ in zs]
    return [(z - z_min) / span for z in zs]


def _depth_tint(base_color: tuple[int, int, int], depth01: float) -> tuple[int, int, int]:
    # Near points are brighter for better visual separation in static skeleton images.
    gain = 0.65 + 0.55 * (1.0 - np.clip(depth01, 0.0, 1.0))
    return tuple(int(np.clip(c * gain, 0, 255)) for c in base_color)


def render_skeleton_canvas(
    results,
    frame_shape: tuple[int, int, int],
    roi: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
    selected_hand_index: int | None = None,
    encode_depth: bool = True,
) -> np.ndarray:
    canvas_h, canvas_w = canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if not results or not results.multi_hand_landmarks:
        return canvas

    frame_h, frame_w = frame_shape[:2]
    if selected_hand_index is None:
        hand_indices = list(range(len(results.multi_hand_landmarks)))
    else:
        if selected_hand_index < 0 or selected_hand_index >= len(results.multi_hand_landmarks):
            return canvas
        hand_indices = [selected_hand_index]

    for hand_idx in hand_indices:
        hand_lm = results.multi_hand_landmarks[hand_idx]
        points = _to_canvas_points(hand_lm, frame_w, frame_h, roi, canvas_w, canvas_h)
        depths = _point_depths(hand_lm)

        _draw_chain(canvas, points, PALM, COLOR_PALM)
        _draw_chain(canvas, points, THUMB, COLOR_THUMB)
        _draw_chain(canvas, points, INDEX, COLOR_INDEX)
        _draw_chain(canvas, points, MIDDLE, COLOR_MIDDLE)
        _draw_chain(canvas, points, RING, COLOR_RING)
        _draw_chain(canvas, points, PINKY, COLOR_PINKY)

        for idx, pt in enumerate(points):
            if idx in THUMB:
                color = COLOR_THUMB
            elif idx in INDEX:
                color = COLOR_INDEX
            elif idx in MIDDLE:
                color = COLOR_MIDDLE
            elif idx in RING:
                color = COLOR_RING
            elif idx in PINKY:
                color = COLOR_PINKY
            else:
                color = COLOR_PALM

            radius = 4
            draw_color = color
            if encode_depth:
                draw_color = _depth_tint(color, depths[idx])
                radius = int(np.clip(3 + 4 * (1.0 - depths[idx]), 3, 7))
            cv2.circle(canvas, pt, radius, draw_color, -1, cv2.LINE_AA)

    return canvas
