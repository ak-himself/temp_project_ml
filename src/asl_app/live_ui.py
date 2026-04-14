from __future__ import annotations

import json
import time
from collections import deque

import cv2
import numpy as np

from .capture_app import (
    CLASS_FILE,
    FINGER_NAME_ORDER,
    HAND_MODEL_COMPLEXITY,
    HIERARCHY_BASE_MEAN_CONF,
    HIERARCHY_BASE_MIN_AGREEMENT,
    HIERARCHY_STACK_ACCEPT_CONF,
    HIERARCHY_STACK_ACCEPT_MARGIN,
    HIERARCHY_STACK_SUPPORT_CONF,
    HIERARCHY_STACK_SUPPORT_TOP2_MIN,
    MAX_NUM_HANDS,
    MIN_DET_CONFIDENCE,
    MIN_TRACK_CONFIDENCE,
    MIRROR_LEFT_HAND_TO_RIGHT,
    MODELS_DIR,
    RGB_IMAGE_SIZE,
    TEMPORAL_ACCEPT_CONF,
    TEMPORAL_DEBOUNCE_FRAMES,
    TEMPORAL_MIN_FRAMES,
    TEMPORAL_STABLE_TTL_FRAMES,
    TEMPORAL_WINDOW_SIZE,
    append_analysis_log,
    build_feature_vector,
    classify_palm_orientation6,
    collect_hand_infos,
    compute_finger_analysis_metrics,
    compute_palm_normal,
    compute_right_square_roi,
    draw_finger_analysis_overlay,
    load_classes,
    mp_drawing,
    mp_hands,
    mp_styles,
    normalize_landmarks_points,
    open_camera,
    print_camera_settings,
    render_skeleton_canvas,
)


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

            if debug_overlay:
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
