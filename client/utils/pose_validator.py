"""
Pose validation utilities for MediaPipe landmark data.
Checks full-body visibility and pose stability over time.
"""

import numpy as np

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

REQUIRED_INDICES = list(REQUIRED_LANDMARKS.values())


def check_full_body_visible(landmarks, threshold: float = 0.5) -> bool:
    """
    Check that all required landmarks are visible above a confidence threshold.

    Args:
        landmarks: MediaPipe pose landmarks (NormalizedLandmarkList)
        threshold: minimum visibility score (0-1)

    Returns:
        True if all required landmarks are sufficiently visible
    """
    if landmarks is None:
        return False
    for idx in REQUIRED_INDICES:
        if idx >= len(landmarks.landmark):
            return False
        if landmarks.landmark[idx].visibility < threshold:
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
            landmarks: MediaPipe pose landmarks
        """
        if landmarks is None:
            return
        coords = []
        for lm in landmarks.landmark:
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
