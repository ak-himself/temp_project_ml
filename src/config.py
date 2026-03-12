from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_GESTURE_DIR = DATA_DIR / "raw" / "gestures"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
CLASS_FILE = PROJECT_ROOT / "classes.json"
CAPTURE_LOG_FILE = LOG_DIR / "capture_log.csv"

# Camera and frame
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SHOW_FPS = True

# ROI: square on the right side spanning top+bottom halves.
# Side is 60% of min(frame_w, frame_h), centered vertically and near right edge.
ROI_SIDE_RATIO = 0.60
ROI_RIGHT_MARGIN_RATIO = 0.04

# Capture policy
DEFAULT_TARGET_IMAGES_PER_CLASS = 1500
# 0.04s ~= 25 captures/sec, about 5x faster than 0.20s.
CAPTURE_INTERVAL_SEC = 0.04
MIN_BRIGHTNESS = 45.0
MIN_SHARPNESS = 60.0

# Detection policy
MIN_DET_CONFIDENCE = 0.60
MIN_TRACK_CONFIDENCE = 0.60
MAX_NUM_HANDS = 2
ALLOW_TWO_HAND_GESTURES = False
REQUIRED_HANDEDNESS = "Right"

# Model-ready image size
IMAGE_SIZE = (224, 224)
