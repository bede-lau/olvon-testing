"""
Pose validation utilities for MediaPipe landmark data.
Checks full-body visibility, pose stability, and body orientation.

Compatible with mediapipe >= 0.10.30 (Tasks API).
Landmarks are a list of NormalizedLandmark objects with .x, .y, .z, .visibility.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe landmark indices we require for a full-body scan
REQUIRED_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Orientation detection thresholds (tunable)
SIDE_VIEW_THRESHOLD = 0.08       # Shoulder width below this → side view
FRONT_BACK_THRESHOLD = 0.10      # Shoulder width above this → front or back
ELEVATED_OFFSET = 0.20           # Nose above shoulder midpoint by this → elevated
NOSE_VISIBILITY_THRESHOLD = 0.5  # Nose visibility below this → likely back view

# Model download config (shared with capture_wizard.py)
MODEL_PATH = Path(__file__).parent.parent / "assets" / "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

REQUIRED_INDICES = list(REQUIRED_LANDMARKS.values())


def check_full_body_visible(landmarks, threshold: float = 0.5) -> bool:
    """
    Check that all required landmarks are visible above a confidence threshold.

    Args:
        landmarks: list of NormalizedLandmark (mediapipe Tasks API)
        threshold: minimum visibility score (0-1)

    Returns:
        True if all required landmarks are sufficiently visible
    """
    if landmarks is None:
        return False
    for idx in REQUIRED_INDICES:
        if idx >= len(landmarks):
            return False
        if landmarks[idx].visibility < threshold:
            return False
    return True


def compute_pose_variance(landmark_history: list, indices: list | None = None) -> float:
    """
    Compute the average positional variance across frames for specified landmarks.

    Args:
        landmark_history: list of (N, 3) arrays, one per frame
        indices: which landmark indices to check; defaults to REQUIRED_INDICES

    Returns:
        Mean variance across specified landmarks and xyz dimensions
    """
    if len(landmark_history) < 2:
        return float("inf")

    indices = indices or REQUIRED_INDICES
    # Stack into (frames, landmarks, 3)
    stacked = np.array([[frame[i] for i in indices] for frame in landmark_history])
    # Variance along frame axis, then mean across landmarks and dimensions
    return float(np.var(stacked, axis=0).mean())


class PoseBuffer:
    """
    Rolling buffer of landmark positions to assess pose stability.
    """

    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.frames: list = []

    def add_frame(self, landmarks) -> None:
        """
        Add a frame of landmarks to the buffer.

        Args:
            landmarks: list of NormalizedLandmark (mediapipe Tasks API)
        """
        if landmarks is None:
            return
        coords = []
        for lm in landmarks:
            coords.append([lm.x, lm.y, lm.z])
        self.frames.append(coords)
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)

    def is_stable(self, variance_threshold: float = 0.0005) -> bool:
        """
        Check if the pose has been stable over the buffer window.

        Args:
            variance_threshold: maximum acceptable mean variance

        Returns:
            True if enough frames collected and variance is low
        """
        if len(self.frames) < self.buffer_size // 2:
            return False
        variance = compute_pose_variance(self.frames)
        return variance < variance_threshold

    def clear(self) -> None:
        """Reset the buffer."""
        self.frames.clear()


def _ensure_model():
    """Download the pose landmarker model if not present."""
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pose landmarker model to %s ...", MODEL_PATH)
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    logger.info("Download complete.")


def detect_orientation(landmarks) -> str:
    """
    Detect which direction the person is facing based on landmark geometry.

    Args:
        landmarks: list of NormalizedLandmark (mediapipe Tasks API)

    Returns:
        One of: "front", "right", "left", "back", "elevated", "unknown"
    """
    if landmarks is None or len(landmarks) < 29:
        return "unknown"

    nose = landmarks[REQUIRED_LANDMARKS["nose"]]
    l_shoulder = landmarks[REQUIRED_LANDMARKS["left_shoulder"]]
    r_shoulder = landmarks[REQUIRED_LANDMARKS["right_shoulder"]]

    shoulder_width = abs(l_shoulder.x - r_shoulder.x)
    shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2.0
    nose_vis = nose.visibility

    # Back: nose not visible, shoulders wide enough
    if nose_vis < NOSE_VISIBILITY_THRESHOLD and shoulder_width > FRONT_BACK_THRESHOLD:
        return "back"

    # Side view: shoulders appear narrow (foreshortened)
    if shoulder_width < SIDE_VIEW_THRESHOLD:
        # If left_shoulder.x > right_shoulder.x, the person's right side faces camera
        if l_shoulder.x > r_shoulder.x:
            return "right"
        else:
            return "left"

    # Front-facing checks (shoulders wide, nose visible)
    # Elevated: nose significantly above shoulder midpoint
    nose_offset = shoulder_mid_y - nose.y  # positive = nose above shoulders
    if nose_offset > ELEVATED_OFFSET:
        return "elevated"

    # Default front
    if nose_vis >= NOSE_VISIBILITY_THRESHOLD and shoulder_width >= FRONT_BACK_THRESHOLD:
        return "front"

    return "unknown"


def validate_image_orientation(image_rgb: np.ndarray, expected: str) -> tuple[bool, str, str]:
    """
    Validate that a single image shows the expected body orientation.
    Uses MediaPipe in IMAGE mode (single-frame, no video tracking).

    Args:
        image_rgb: RGB image as numpy array (H, W, 3)
        expected: expected orientation ("front", "right", "left", "back", "elevated")

    Returns:
        (accepted, detected_orientation, message)
    """
    try:
        import mediapipe as mp
    except ImportError:
        return True, "unknown", "mediapipe not available, skipping validation"

    _ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )

    landmarker = PoseLandmarker.create_from_options(options)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return False, "none", "No pose detected in the image."

        landmarks = result.pose_landmarks[0]
        detected = detect_orientation(landmarks)

        if detected == expected:
            return True, detected, f"Orientation matches: {expected}"

        return False, detected, f"Expected {expected}, detected {detected}"
    finally:
        landmarker.close()
