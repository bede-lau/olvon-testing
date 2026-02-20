"""
Text-to-speech audio feedback for the capture wizard.
Wraps pyttsx3 with graceful fallback if unavailable.
"""

import logging

logger = logging.getLogger(__name__)

MESSAGES = {
    "starting": "Starting body scan capture. Please stand in front of the camera.",
    "hold": "Hold still. Capturing in progress.",
    "captured": "Photo captured successfully.",
    "validation_fail": "Could not detect full body. Please step back and ensure your full body is visible.",
    "complete": "All captures complete. You may close the window.",
    "turn_front": "Face the camera directly.",
    "turn_right": "Turn to face your right side.",
    "turn_back": "Turn to face away from the camera.",
    "turn_left": "Turn to face your left side.",
    "turn_elevated": "Look slightly upward for the elevated capture.",
    "stabilizing": "Good pose detected. Hold still for one second.",
    "wrong_orientation": "Wrong orientation detected. Please adjust your position.",
}


class AudioFeedback:
    """TTS feedback wrapper. No-ops gracefully if pyttsx3 is unavailable."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.engine = None
        if enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 160)
            except Exception as e:
                logger.warning("pyttsx3 unavailable, audio feedback disabled: %s", e)
                self.engine = None

    def speak(self, key: str) -> None:
        """Speak a predefined message by key."""
        if not self.enabled or self.engine is None:
            return
        text = MESSAGES.get(key)
        if text:
            self._say(text)

    def speak_custom(self, text: str) -> None:
        """Speak arbitrary text."""
        if not self.enabled or self.engine is None:
            return
        self._say(text)

    def _say(self, text: str) -> None:
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.warning("TTS speak failed: %s", e)
