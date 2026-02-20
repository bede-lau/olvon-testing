"""
State-machine webcam capture wizard for 3D body scanning.
Guides user through 12 capture angles with audio + visual feedback.
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
from client.utils.pose_validator import check_full_body_visible, PoseBuffer
from client.utils.audio_feedback import AudioFeedback


class CaptureState(Enum):
    WAITING = auto()
    VALIDATING = auto()
    STABILIZING = auto()
    CAPTURING = auto()
    ROTATING = auto()
    COMPLETE = auto()


# 12 capture angles with corresponding audio cue keys
CAPTURE_ANGLES = [
    {"name": "front",       "audio_key": "turn_front"},
    {"name": "front_right", "audio_key": "turn_front_right"},
    {"name": "right",       "audio_key": "turn_right"},
    {"name": "back_right",  "audio_key": "turn_back_right"},
    {"name": "back",        "audio_key": "turn_back"},
    {"name": "back_left",   "audio_key": "turn_back_left"},
    {"name": "left",        "audio_key": "turn_left"},
    {"name": "front_left",  "audio_key": "turn_front_left"},
    {"name": "elevated_1",  "audio_key": "turn_elevated"},
    {"name": "elevated_2",  "audio_key": "turn_elevated"},
    {"name": "elevated_3",  "audio_key": "turn_elevated"},
    {"name": "elevated_4",  "audio_key": "turn_elevated"},
]

STABILITY_HOLD_SECONDS = 1.0


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

        # MediaPipe pose
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
            raise

    @property
    def current_angle(self) -> dict:
        if self.current_angle_idx < len(CAPTURE_ANGLES):
            return CAPTURE_ANGLES[self.current_angle_idx]
        return CAPTURE_ANGLES[-1]

    def _draw_overlay(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose landmarks and status text on frame."""
        annotated = frame.copy()

        if landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        # Status bar
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 0), -1)

        angle_name = self.current_angle["name"] if self.current_angle_idx < len(CAPTURE_ANGLES) else "done"
        status_text = f"State: {self.state.name} | Angle: {angle_name} | Captured: {len(self.captured_images)}/12"
        cv2.putText(annotated, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if self.state == CaptureState.STABILIZING and self.stability_start_time:
            elapsed = time.time() - self.stability_start_time
            remaining = max(0, STABILITY_HOLD_SECONDS - elapsed)
            cv2.putText(annotated, f"Hold still: {remaining:.1f}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if self.state == CaptureState.VALIDATING:
            cv2.putText(annotated, "Checking pose...", (10, 50),
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

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                landmarks = results.pose_landmarks

                # State machine transitions
                if self.state == CaptureState.WAITING:
                    if landmarks and check_full_body_visible(landmarks):
                        self.state = CaptureState.VALIDATING
                        self.pose_buffer.clear()

                elif self.state == CaptureState.VALIDATING:
                    if not landmarks or not check_full_body_visible(landmarks):
                        self.state = CaptureState.WAITING
                        self.audio.speak("validation_fail")
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
