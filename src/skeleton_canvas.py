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


def render_skeleton_canvas(
    results,
    frame_shape: tuple[int, int, int],
    roi: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> np.ndarray:
    canvas_h, canvas_w = canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if not results or not results.multi_hand_landmarks:
        return canvas

    frame_h, frame_w = frame_shape[:2]
    for hand_lm in results.multi_hand_landmarks:
        points = _to_canvas_points(hand_lm, frame_w, frame_h, roi, canvas_w, canvas_h)

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
            cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)

    return canvas
