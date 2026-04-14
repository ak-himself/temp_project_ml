from __future__ import annotations

from pathlib import Path

# Paths
# src/asl_app/config.py -> project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_RGB_DIR = DATA_DIR / "rgb"
RAW_LANDMARK_DIR = DATA_DIR / "landmarks"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
CLASS_FILE = Path(__file__).resolve().parent / "classes.json"
CAPTURE_LOG_FILE = LOG_DIR / "capture_log.csv"
FINGER_ANALYSIS_LOG_FILE = LOG_DIR / "finger_analysis.csv"
DATASET_VALIDATION_JSON_FILE = LOG_DIR / "dataset_validation.json"
DATASET_VALIDATION_MD_FILE = LOG_DIR / "dataset_validation.md"
DATASET_PREVIEW_FILE = LOG_DIR / "dataset_preview.jpg"

# Camera and frame
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SHOW_FPS = True
CAPTURE_WINDOW_NAME = "ASL Dataset Capture"
ANALYSIS_WINDOW_NAME = "ASL Finger Analysis"

# Camera quality tuning
PREFER_V4L2_BACKEND = True
CAMERA_FOURCC = "MJPG"
ENABLE_AUTOFOCUS = True

# ROI: square on the right side.
ROI_SIDE_RATIO = 0.60
ROI_RIGHT_MARGIN_RATIO = 0.04

# Capture policy
DEFAULT_TARGET_IMAGES_PER_CLASS = 1500
CAPTURE_INTERVAL_SEC = 0.01
MIN_BRIGHTNESS = 45.0
# Lower default for DroidCam/virtual-camera denoised feeds.
MIN_SHARPNESS = 20.0

# Detection policy
MIN_DET_CONFIDENCE = 0.60
MIN_TRACK_CONFIDENCE = 0.60
MAX_NUM_HANDS = 1
HAND_MODEL_COMPLEXITY = 0
REQUIRED_HANDEDNESS = None
MIRROR_LEFT_HAND_TO_RIGHT = True

# RGB image output for CNN
RGB_IMAGE_SIZE = (128, 128)

# Feature schema (strict contract)
FEATURE_LANDMARK_DIM = 63
FEATURE_JOINT_ANGLE_DIM = 15
FEATURE_PAIRWISE_DISTANCE_DIM = 30
FEATURE_FINGER_STATE_DIM = 5
FEATURE_PALM_DIRECTION_DIM = 6
FEATURE_TOTAL_DIM = (
    FEATURE_LANDMARK_DIM
    + FEATURE_JOINT_ANGLE_DIM
    + FEATURE_PAIRWISE_DISTANCE_DIM
    + FEATURE_FINGER_STATE_DIM
    + FEATURE_PALM_DIRECTION_DIM
)

# Finger state thresholds (global angle rule):
# angle < 110 -> closed (0)
# 110 <= angle <= 165 -> half-open (1)
# angle > 165 -> open (2)
FINGER_STATE_ANGLE_HALF_OPEN_MIN = 110.0
FINGER_STATE_ANGLE_HALF_OPEN_MAX = 165.0

PALM_ORIENTATION_6 = ("camera", "face", "up", "down", "left", "right")

# Validation and balance checks
IMBALANCE_WARN_RATIO = 1.5
IMBALANCE_FAIL_RATIO = 2.5

# Step 4 hierarchy policy thresholds
HIERARCHY_STACK_ACCEPT_CONF = 0.62
HIERARCHY_STACK_ACCEPT_MARGIN = 0.08
HIERARCHY_BASE_MIN_AGREEMENT = 2
HIERARCHY_BASE_MEAN_CONF = 0.55
HIERARCHY_STACK_SUPPORT_CONF = 0.42
HIERARCHY_STACK_SUPPORT_TOP2_MIN = 2

# Step 5 temporal stabilization policy
# Default preset: lower latency while keeping flicker controlled.
TEMPORAL_WINDOW_SIZE = 6
TEMPORAL_MIN_FRAMES = 3
TEMPORAL_ACCEPT_CONF = 0.48
TEMPORAL_DEBOUNCE_FRAMES = 2
TEMPORAL_STABLE_TTL_FRAMES = 10
