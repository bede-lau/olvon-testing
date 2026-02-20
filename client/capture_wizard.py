"""
State-machine webcam capture wizard for 3D body scanning.
Guides user through 5 capture angles with orientation detection, audio + visual feedback.
"""

import os
import time
import logging
from pathlib import Path
from enum import Enum, auto

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Import project utilities
from client.utils.pose_validator import (
    check_full_body_visible,
    detect_orientation,
    PoseBuffer,
    _ensure_model,
    MODEL_PATH,
)
from client.utils.audio_feedback import AudioFeedback


class CaptureState(Enum):
    WAITING = auto()
    VALIDATING = auto()
    STABILIZING = auto()
    CAPTURING = auto()
    ROTATING = auto()
    COMPLETE = auto()


# 5 capture angles with expected orientation and audio cue keys
CAPTURE_ANGLES = [
    {"name": "front",    "orientation": "front",    "audio_key": "turn_front"},
    {"name": "right",    "orientation": "right",    "audio_key": "turn_right"},
    {"name": "back",     "orientation": "back",     "audio_key": "turn_back"},
    {"name": "left",     "orientation": "left",     "audio_key": "turn_left"},
    {"name": "elevated", "orientation": "elevated",  "audio_key": "turn_elevated"},
]

STABILITY_HOLD_SECONDS = 1.0
GUIDANCE_REPEAT_INTERVAL = 5.0  # seconds between re-speaking orientation guidance


class CaptureWizard:
    def __init__(self, output_dir: str = "client/output_captures", audio_enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio = AudioFeedback(enabled=audio_enabled)
        self.pose_buffer = PoseBuffer(buffer_size=30)

        self.state = CaptureState.WAITING
        self.current_angle_idx = 0
        self.stability_start_time = None
        self.captured_images: list[Path] = []
        self.last_guidance_time = 0.0
        self.last_detected_orientation = "unknown"

        # MediaPipe pose landmarker (new Tasks API for mediapipe >= 0.10.30)
        try:
            import mediapipe as mp
            _ensure_model()

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=VisionRunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            self.mp = mp
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
            raise

    @property
    def current_angle(self) -> dict:
        if self.current_angle_idx < len(CAPTURE_ANGLES):
            return CAPTURE_ANGLES[self.current_angle_idx]
        return CAPTURE_ANGLES[-1]

    def _detect_pose(self, frame_rgb: np.ndarray, timestamp_ms: int):
        """Run pose detection and return landmarks wrapper or None."""
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]  # list of NormalizedLandmark
        return None

    def _check_orientation(self, landmarks) -> bool:
        """Check if detected orientation matches the required angle."""
        expected = self.current_angle["orientation"]
        detected = detect_orientation(landmarks)
        self.last_detected_orientation = detected

        if detected == expected:
            return True

        # Repeat guidance every GUIDANCE_REPEAT_INTERVAL seconds
        now = time.time()
        if now - self.last_guidance_time >= GUIDANCE_REPEAT_INTERVAL:
            self.audio.speak("wrong_orientation")
            self.last_guidance_time = now

        return False

    def _draw_overlay(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose landmarks and status text on frame."""
        annotated = frame.copy()

        if landmarks:
            h, w = annotated.shape[:2]
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)

        # Status bar
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 80), (0, 0, 0), -1)

        angle_name = self.current_angle["name"] if self.current_angle_idx < len(CAPTURE_ANGLES) else "done"
        total = len(CAPTURE_ANGLES)
        status_text = f"State: {self.state.name} | Angle: {angle_name} | Captured: {len(self.captured_images)}/{total}"
        cv2.putText(annotated, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Orientation feedback
        expected = self.current_angle["orientation"] if self.current_angle_idx < len(CAPTURE_ANGLES) else ""
        orientation_text = f"Required: {expected.upper()} | Detected: {self.last_detected_orientation.upper()}"
        color = (0, 255, 0) if self.last_detected_orientation == expected else (0, 0, 255)
        cv2.putText(annotated, orientation_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if self.state == CaptureState.STABILIZING and self.stability_start_time:
            elapsed = time.time() - self.stability_start_time
            remaining = max(0, STABILITY_HOLD_SECONDS - elapsed)
            cv2.putText(annotated, f"Hold still: {remaining:.1f}s", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if self.state == CaptureState.VALIDATING:
            cv2.putText(annotated, "Checking pose...", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        return annotated

    def _save_capture(self, frame: np.ndarray) -> Path:
        """Save the current frame as a capture."""
        angle_name = self.current_angle["name"]
        filename = f"capture_{self.current_angle_idx:02d}_{angle_name}.png"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), frame)
        logger.info("Saved capture: %s", filepath)
        return filepath

    def run(self) -> list[Path]:
        """Main capture loop. Returns list of captured image paths."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return []

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.audio.speak("starting")
        self.state = CaptureState.WAITING
        self.audio.speak(self.current_angle["audio_key"])

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms == 0:
                    timestamp_ms = frame_count * 33  # ~30fps fallback
                frame_count += 1

                landmarks = self._detect_pose(frame_rgb, timestamp_ms)

                # Update detected orientation for overlay
                if landmarks:
                    self.last_detected_orientation = detect_orientation(landmarks)

                # State machine transitions
                if self.state == CaptureState.WAITING:
                    if landmarks and check_full_body_visible(landmarks):
                        if self._check_orientation(landmarks):
                            self.state = CaptureState.VALIDATING
                            self.pose_buffer.clear()

                elif self.state == CaptureState.VALIDATING:
                    if not landmarks or not check_full_body_visible(landmarks):
                        self.state = CaptureState.WAITING
                        self.audio.speak("validation_fail")
                    elif not self._check_orientation(landmarks):
                        self.state = CaptureState.WAITING
                    else:
                        self.pose_buffer.add_frame(landmarks)
                        if self.pose_buffer.is_stable():
                            self.state = CaptureState.STABILIZING
                            self.stability_start_time = time.time()
                            self.audio.speak("stabilizing")

                elif self.state == CaptureState.STABILIZING:
                    if not landmarks or not check_full_body_visible(landmarks):
                        self.state = CaptureState.WAITING
                        self.stability_start_time = None
                        self.pose_buffer.clear()
                    elif not self._check_orientation(landmarks):
                        self.state = CaptureState.WAITING
                        self.stability_start_time = None
                        self.pose_buffer.clear()
                    else:
                        self.pose_buffer.add_frame(landmarks)
                        if not self.pose_buffer.is_stable():
                            self.state = CaptureState.VALIDATING
                            self.stability_start_time = None
                        elif time.time() - self.stability_start_time >= STABILITY_HOLD_SECONDS:
                            self.state = CaptureState.CAPTURING

                elif self.state == CaptureState.CAPTURING:
                    path = self._save_capture(frame)
                    self.captured_images.append(path)
                    self.audio.speak("captured")
                    self.current_angle_idx += 1

                    if self.current_angle_idx >= len(CAPTURE_ANGLES):
                        self.state = CaptureState.COMPLETE
                        self.audio.speak("complete")
                    else:
                        self.state = CaptureState.ROTATING
                        self.audio.speak(self.current_angle["audio_key"])
                        self.pose_buffer.clear()
                        self.stability_start_time = None

                elif self.state == CaptureState.ROTATING:
                    # Wait for user to rotate, then re-validate
                    self.state = CaptureState.WAITING

                elif self.state == CaptureState.COMPLETE:
                    annotated = self._draw_overlay(frame, landmarks)
                    cv2.imshow("Olvon Body Scan", annotated)
                    cv2.waitKey(2000)
                    break

                # Display
                annotated = self._draw_overlay(frame, landmarks)
                cv2.imshow("Olvon Body Scan", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User quit capture")
                    break

        finally:
            self.landmarker.close()
            cap.release()
            cv2.destroyAllWindows()

        logger.info("Capture complete: %d images saved to %s", len(self.captured_images), self.output_dir)
        return self.captured_images


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    wizard = CaptureWizard()
    images = wizard.run()
    print(f"\nCaptured {len(images)} images:")
    for img in images:
        print(f"  {img}")
