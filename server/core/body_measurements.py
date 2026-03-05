"""
Body measurement extraction from photos using MediaPipe pose landmarks.

Replaces the old ANNY-based 3D mesh approach with a 2D landmark-based method.
Fallback chain: landmarks → height/weight empirical → population averages.
"""

import logging
import math
from pathlib import Path

import numpy as np

from server.core.diagnostics import PipelineLog, log_fallback
from server.core.sizing_logic import _estimate_from_height_weight

logger = logging.getLogger(__name__)

# Population average measurements (cm) — used as last-resort fallback
POPULATION_AVERAGES = {
    "chest": 96.0,
    "waist": 82.0,
    "hip": 96.0,
    "shoulder_width": 44.0,
}

# Landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Circumference conversion factors (width * pi * factor)
CHEST_FACTOR = 0.75        # biacromial → chest circumference (empirical, not circular)
HIP_FACTOR = 0.86          # MediaPipe hip joint landmark → hip circumference
WAIST_RATIO = 0.85         # waist ≈ 85% of chest circumference
NOSE_TO_ANKLE_RATIO = 0.92 # nose-to-ankle ≈ 92% of total body height


def _extract_from_landmarks(
    landmarks, image_height: int, image_width: int, height_cm: float | None,
) -> dict | None:
    """
    Extract body measurements from MediaPipe pose landmarks.

    Uses nose-to-ankle pixel distance + known height to compute pixel-to-cm ratio,
    then converts shoulder and hip pixel widths to circumferences.

    Returns dict with chest, waist, hip, shoulder_width or None on failure.
    """
    if landmarks is None or len(landmarks) < 29:
        return None

    nose = landmarks[NOSE]
    l_shoulder = landmarks[LEFT_SHOULDER]
    r_shoulder = landmarks[RIGHT_SHOULDER]
    l_hip = landmarks[LEFT_HIP]
    r_hip = landmarks[RIGHT_HIP]
    l_ankle = landmarks[LEFT_ANKLE]
    r_ankle = landmarks[RIGHT_ANKLE]

    # Check minimum visibility
    key_landmarks = [nose, l_shoulder, r_shoulder, l_hip, r_hip, l_ankle, r_ankle]
    if any(lm.visibility < 0.5 for lm in key_landmarks):
        return None

    # Pixel distances (normalized coordinates × image height for vertical)
    nose_y = nose.y * image_height
    ankle_y = ((l_ankle.y + r_ankle.y) / 2.0) * image_height
    nose_to_ankle_px = abs(ankle_y - nose_y)

    if nose_to_ankle_px < 50:  # too small to be reliable
        return None

    # Shoulder and hip widths in pixels (normalized × image_width ≈ image_height for aspect)
    shoulder_width_norm = abs(l_shoulder.x - r_shoulder.x)
    hip_width_norm = abs(l_hip.x - r_hip.x)

    if height_cm and height_cm > 0:
        # Nose-to-ankle ≈ NOSE_TO_ANKLE_RATIO of total height
        estimated_body_height_px = nose_to_ankle_px / NOSE_TO_ANKLE_RATIO
        px_to_cm = height_cm / estimated_body_height_px

        shoulder_width_cm = shoulder_width_norm * image_width * px_to_cm
        hip_width_cm = hip_width_norm * image_width * px_to_cm
    else:
        # Without height, use anthropometric ratios:
        # shoulder width ≈ 25% of height, so we estimate from proportions
        # nose-to-ankle ≈ NOSE_TO_ANKLE_RATIO height → height_estimate = nose_to_ankle / ratio
        # shoulder_ratio = shoulder_px / nose_to_ankle_px
        shoulder_ratio = shoulder_width_norm * image_width / nose_to_ankle_px
        hip_ratio = hip_width_norm * image_width / nose_to_ankle_px

        # Average height assumption: 170cm
        assumed_height = 170.0
        estimated_body_px = nose_to_ankle_px / NOSE_TO_ANKLE_RATIO
        px_to_cm = assumed_height / estimated_body_px

        shoulder_width_cm = shoulder_ratio * nose_to_ankle_px * px_to_cm
        hip_width_cm = hip_ratio * nose_to_ankle_px * px_to_cm

    # Convert widths to circumferences
    chest_circumference = shoulder_width_cm * math.pi * CHEST_FACTOR
    hip_circumference = hip_width_cm * math.pi * HIP_FACTOR
    waist_circumference = chest_circumference * WAIST_RATIO

    return {
        "chest": round(chest_circumference, 1),
        "waist": round(waist_circumference, 1),
        "hip": round(hip_circumference, 1),
        "shoulder_width": round(shoulder_width_cm, 1),
    }


def extract(
    front_photo: str | Path,
    side_photo: str | Path | None = None,
    height_cm: float | None = None,
    weight_kg: float | None = None,
    pipeline_log: PipelineLog | None = None,
) -> dict:
    """
    Extract body measurements from photos with fallback chain.

    Fallback order:
    1. MediaPipe landmarks from front photo + height → computed measurements
    2. Height/weight empirical formulas (sizing_logic._estimate_from_height_weight)
    3. Population averages

    Args:
        front_photo: path to front-facing full-body photo
        side_photo: optional side photo (reserved for future use)
        height_cm: user height in cm (improves accuracy)
        weight_kg: user weight in kg (improves accuracy)
        pipeline_log: optional structured log collector

    Returns:
        dict with chest, waist, hip, shoulder_width, and optional height_cm/weight_kg/bmi
    """
    front_photo = Path(front_photo)
    result = None

    # Try landmark extraction
    if front_photo.exists():
        try:
            import cv2
            import mediapipe as mp
            from client.utils.pose_validator import _ensure_model, MODEL_PATH

            _ensure_model()

            img = cv2.imread(str(front_photo))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(MODEL_PATH)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=0.5,
                )
                landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                try:
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=img_rgb
                    )
                    detection = landmarker.detect(mp_image)

                    if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                        landmarks = detection.pose_landmarks[0]
                        result = _extract_from_landmarks(landmarks, h, w, height_cm)
                        if result:
                            logger.info("Body measurements extracted from landmarks")
                finally:
                    landmarker.close()

        except Exception as e:
            log_fallback(logger, "body_measurements", e, pipeline_log)

    # Fallback: height/weight empirical
    if result is None and height_cm and height_cm > 0:
        logger.info("Falling back to height/weight empirical estimates")
        hw_est = _estimate_from_height_weight(height_cm, weight_kg)
        result = {
            "chest": round(hw_est["chest"], 1),
            "waist": round(hw_est["waist"], 1),
            "hip": round(hw_est["hip"], 1),
            "shoulder_width": round(hw_est["chest"] / math.pi / CHEST_FACTOR, 1),
        }

    # Last resort: population averages
    if result is None:
        logger.info("Using population average measurements")
        result = dict(POPULATION_AVERAGES)

    # Attach height/weight/bmi metadata
    if height_cm and height_cm > 0:
        result["height_cm"] = height_cm
    if weight_kg and weight_kg > 0:
        result["weight_kg"] = weight_kg
    if height_cm and height_cm > 0 and weight_kg and weight_kg > 0:
        result["bmi"] = round(weight_kg / ((height_cm / 100) ** 2), 1)

    return result
