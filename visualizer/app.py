"""
Streamlit unified wizard for body scanning, garment input, pipeline execution,
and 3D result viewing. Runs entirely on vast.ai — users access via browser.

Body scan uses streamlit-webrtc for live video analysis with automatic capture
when the user's pose orientation is correct and stable.
"""

import base64
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from io import BytesIO

# Ensure project root is on sys.path so `server.*` / `client.*` imports work
# regardless of the working directory Streamlit uses.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="Olvon Physics Test", layout="wide")

OUTPUT_DIR = Path("server/outputs")
INPUT_DIR = Path("server/inputs")
GARMENT_PHOTO_DIR = Path("server/inputs/garment_photos")
DEFAULT_GLB_PATH = OUTPUT_DIR / "final_fitted_avatar.glb"
DEFAULT_SIZING_PATH = OUTPUT_DIR / "sizing_result.json"

for d in [OUTPUT_DIR, INPUT_DIR, GARMENT_PHOTO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 5 capture angles
SCAN_ANGLES = ["front", "right", "back", "left", "elevated"]
SCAN_INSTRUCTIONS = {
    "front": "Face the camera directly. Ensure your full body is visible.",
    "right": "Turn so your RIGHT side faces the camera.",
    "back": "Turn to face AWAY from the camera.",
    "left": "Turn so your LEFT side faces the camera.",
    "elevated": "Face the camera and look slightly UPWARD.",
}

STEP_LABELS = ["Body Input", "Body Scan", "Garment Input", "Generate", "Results"]

# Auto-capture timing
STABILITY_HOLD_SECONDS = 2.0  # hold correct pose for this long to auto-capture


# ---------------------------------------------------------------------------
# Body scan video processor (runs in a worker thread)
# ---------------------------------------------------------------------------
class BodyScanProcessor(VideoProcessorBase):
    """
    Processes each video frame: runs MediaPipe pose detection, checks orientation,
    draws overlay, and auto-captures when conditions are met.

    Communication with main thread:
    - capture_queue: sends (angle_name, bgr_image) when auto-capture triggers
    - target_angle / skip_validation: set by main thread to control behavior
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.capture_queue: queue.Queue = queue.Queue()

        # Controlled by main thread
        self._target_angle: str = "front"
        self._skip_validation: bool = False
        self._active: bool = True  # set False to stop processing

        # Internal state (worker thread only)
        self._stability_start: float | None = None
        self._detected_orientation: str = "unknown"
        self._body_visible: bool = False
        self._captured: bool = False  # prevents double-capture for same angle
        self._frame_count: int = 0

        # Lazy-init MediaPipe landmarker (heavy, do once)
        self._landmarker = None
        self._mp = None
        self._init_error: str | None = None

    def _init_mediapipe(self):
        """Initialize MediaPipe pose landmarker (called once on first frame)."""
        try:
            import mediapipe as mp
            from client.utils.pose_validator import _ensure_model, MODEL_PATH

            _ensure_model()

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )
            self._mp = mp
        except Exception as e:
            self._init_error = str(e)

    @property
    def target_angle(self) -> str:
        with self._lock:
            return self._target_angle

    @target_angle.setter
    def target_angle(self, value: str):
        with self._lock:
            if self._target_angle != value:
                self._target_angle = value
                self._stability_start = None
                self._captured = False

    @property
    def skip_validation(self) -> bool:
        with self._lock:
            return self._skip_validation

    @skip_validation.setter
    def skip_validation(self, value: bool):
        with self._lock:
            self._skip_validation = value

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @active.setter
    def active(self, value: bool):
        with self._lock:
            self._active = value

    def reset_for_angle(self, angle: str):
        """Reset state for a new angle (called from main thread)."""
        with self._lock:
            self._target_angle = angle
            self._stability_start = None
            self._captured = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        if not self.active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Lazy init
        if self._landmarker is None and self._init_error is None:
            self._init_mediapipe()

        if self._init_error:
            cv2.putText(
                img, f"MediaPipe error: {self._init_error[:60]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Run pose detection
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )
        timestamp_ms = self._frame_count * 33  # ~30fps
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

        # Analyze pose
        with self._lock:
            target = self._target_angle
            skip = self._skip_validation
            already_captured = self._captured

        body_visible = False
        orientation_ok = False
        detected = "unknown"

        if landmarks:
            from client.utils.pose_validator import (
                check_full_body_visible,
                detect_orientation,
            )

            body_visible = check_full_body_visible(landmarks)
            detected = detect_orientation(landmarks)
            orientation_ok = skip or (detected == target)

        self._body_visible = body_visible
        self._detected_orientation = detected

        # Stability tracking
        now = time.monotonic()
        if body_visible and orientation_ok and not already_captured:
            if self._stability_start is None:
                self._stability_start = now
            elapsed = now - self._stability_start
            remaining = max(0, STABILITY_HOLD_SECONDS - elapsed)

            if elapsed >= STABILITY_HOLD_SECONDS:
                # Auto-capture!
                with self._lock:
                    self._captured = True
                self._stability_start = None
                self.capture_queue.put((target, img.copy()))
        else:
            if not body_visible or not orientation_ok:
                self._stability_start = None

        # --- Draw overlay ---
        h, w = img.shape[:2]

        # Draw skeleton landmarks
        if landmarks:
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

        # Status bar background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Angle info
        status_color = (0, 255, 0) if orientation_ok and body_visible else (0, 0, 255)
        cv2.putText(
            img, f"Target: {target.upper()}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            img, f"Detected: {detected.upper()}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
        )

        # Body visibility
        vis_text = "Body: VISIBLE" if body_visible else "Body: NOT VISIBLE"
        vis_color = (0, 255, 0) if body_visible else (0, 0, 255)
        cv2.putText(
            img, vis_text, (w - 250, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2,
        )

        if already_captured:
            cv2.putText(
                img, "CAPTURED - waiting for next angle...",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        elif body_visible and orientation_ok and self._stability_start is not None:
            elapsed = now - self._stability_start
            remaining = max(0, STABILITY_HOLD_SECONDS - elapsed)
            # Countdown bar
            progress = min(1.0, elapsed / STABILITY_HOLD_SECONDS)
            bar_w = int(w * progress)
            cv2.rectangle(img, (0, h - 20), (bar_w, h), (0, 255, 0), -1)
            cv2.putText(
                img, f"Hold still: {remaining:.1f}s",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        elif not body_visible:
            cv2.putText(
                img, "Step back so full body is visible",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        elif not orientation_ok:
            cv2.putText(
                img, f"Please turn to show {target.upper()} view",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "step": 0,
        "height_cm": 170.0,
        "weight_kg": 0.0,
        "captured_images": {},       # {angle_name: image_bytes (PNG-encoded)}
        "scan_angle_idx": 0,
        "garment_photo_path": None,
        "garment_measurements": None,
        "fabric": "cotton",
        "sizing_result": None,
        "pipeline_ran": False,
        "skip_validation": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def render_step_indicator():
    cols = st.columns(len(STEP_LABELS))
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS)):
        current = st.session_state.step
        if i < current:
            col.markdown(f"**:white_check_mark: {label}**")
        elif i == current:
            col.markdown(f"**:arrow_right: {label}**")
        else:
            col.markdown(f":black_circle: {label}")
    st.divider()


def go_to_step(step: int):
    st.session_state.step = step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_glb_as_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:model/gltf-binary;base64,{b64}"


def render_model_viewer(glb_data_uri: str, height: int = 600):
    html = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer
        src="{glb_data_uri}"
        alt="3D Fitted Avatar"
        camera-controls
        auto-rotate
        auto-rotate-delay="0"
        rotation-per-second="15deg"
        shadow-intensity="1"
        style="width: 100%; height: {height}px; background-color: #1a1a2e;"
    >
        <div slot="poster" style="display:flex;align-items:center;justify-content:center;height:100%;color:white;">
            Loading 3D model...
        </div>
    </model-viewer>
    """
    components.html(html, height=height + 20)


def render_sizing_panel(sizing_data: dict):
    st.subheader("Size Recommendation")
    rec_size = sizing_data.get("recommended_size", "N/A")
    confidence = sizing_data.get("confidence_score", 0)
    fit_score = sizing_data.get("fit_score", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recommended Size", rec_size)
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")
    with col3:
        st.metric("Fit Score", f"{fit_score:.0%}")

    measurements = sizing_data.get("measurements", {})
    if measurements:
        st.subheader("Body Measurements")
        import pandas as pd
        df = pd.DataFrame([
            {"Measurement": k.replace("_", " ").title(), "Value": v}
            for k, v in measurements.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    all_scores = sizing_data.get("all_scores", {})
    if all_scores:
        st.subheader("All Size Scores")
        import pandas as pd
        df = pd.DataFrame([
            {"Size": k, "Score": f"{v:.3f}"}
            for k, v in all_scores.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_pipeline_log(sizing_data: dict):
    pipeline_log = sizing_data.get("pipeline_log", [])
    with st.expander("Pipeline Log", expanded=False):
        if pipeline_log:
            for entry in pipeline_log:
                gpu = f" | {entry['gpu_info']}" if entry.get("gpu_info") else ""
                st.text(f"[{entry['stage']}] {entry['error_type']}: {entry['message']}{gpu}")
        else:
            st.text("Pipeline completed successfully - no fallbacks triggered.")


def run_pipeline(height_cm, weight_kg, garment_photo_path, garment_measurements, fabric):
    from server.main_pipeline import PhysicsTestPipeline
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = PhysicsTestPipeline()
    return pipeline.run(
        user_height_cm=height_cm if height_cm > 0 else None,
        user_weight_kg=weight_kg if weight_kg and weight_kg > 0 else None,
        garment_photo_path=garment_photo_path,
        garment_measurements=garment_measurements if garment_measurements else None,
        fabric=fabric,
    )


# ---------------------------------------------------------------------------
# Step renderers
# ---------------------------------------------------------------------------
def render_step_body_input():
    st.header("Step 1: Body Measurements")
    st.markdown("Enter your height and (optionally) weight for accurate sizing.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.height_cm = st.number_input(
            "Height (cm) *", min_value=100.0, max_value=250.0,
            value=st.session_state.height_cm, step=1.0,
            help="Required. Your height in centimeters.",
        )
    with col2:
        st.session_state.weight_kg = st.number_input(
            "Weight (kg)", min_value=0.0, max_value=300.0,
            value=st.session_state.weight_kg, step=1.0,
            help="Optional. Set to 0 to skip.",
        )

    st.markdown("")
    if st.button("Next: Body Scan", type="primary"):
        go_to_step(1)
        st.rerun()


def render_step_body_scan():
    st.header("Step 2: Body Scan")
    st.markdown(
        "Stand in front of your camera with your **full body visible**. "
        "The system will automatically detect your orientation and capture "
        "each angle when you hold the correct pose for 2 seconds."
    )

    captured = st.session_state.captured_images
    angle_idx = st.session_state.scan_angle_idx

    # Show captured thumbnails
    if captured:
        st.markdown("**Captured so far:**")
        thumb_cols = st.columns(len(SCAN_ANGLES))
        for i, angle in enumerate(SCAN_ANGLES):
            with thumb_cols[i]:
                if angle in captured:
                    st.image(captured[angle], caption=angle, width=120)
                else:
                    st.markdown(f"*{angle}*")

    # All done?
    if angle_idx >= len(SCAN_ANGLES):
        st.success(f"All {len(SCAN_ANGLES)} angles captured!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous: Body Input"):
                go_to_step(0)
                st.rerun()
        with col2:
            if st.button("Next: Garment Input", type="primary"):
                go_to_step(2)
                st.rerun()
        return

    current_angle = SCAN_ANGLES[angle_idx]
    st.subheader(f"Angle {angle_idx + 1}/{len(SCAN_ANGLES)}: {current_angle.upper()}")
    st.info(SCAN_INSTRUCTIONS.get(current_angle, "Position yourself for this angle."))

    # Skip validation toggle
    st.session_state.skip_validation = st.checkbox(
        "Skip orientation validation",
        value=st.session_state.skip_validation,
        help="Enable if auto-detection isn't working (e.g., back view is unreliable).",
    )

    # WebRTC video stream with auto-capture
    ctx = webrtc_streamer(
        key="body_scan",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=BodyScanProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "facingMode": "user",
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

    # Update processor settings from main thread
    if ctx.video_processor:
        ctx.video_processor.target_angle = current_angle
        ctx.video_processor.skip_validation = st.session_state.skip_validation

    # Poll for auto-captured frames
    if ctx.state.playing and ctx.video_processor:
        try:
            angle_name, bgr_img = ctx.video_processor.capture_queue.get_nowait()
            # Encode as PNG bytes for storage
            _, png_buf = cv2.imencode(".png", bgr_img)
            png_bytes = png_buf.tobytes()
            st.session_state.captured_images[angle_name] = png_bytes
            st.session_state.scan_angle_idx += 1

            # Advance to next angle (processor will pick up new target on next rerun)
            st.success(f"Captured {angle_name} view!")
            time.sleep(0.5)  # brief pause so user sees success message
            st.rerun()
        except queue.Empty:
            pass

    # Manual capture fallback button
    if ctx.state.playing and ctx.video_processor:
        if st.button("Capture manually (skip auto-detect)"):
            # Grab latest frame from processor
            ctx.video_processor.skip_validation = True
            st.markdown("*Manual capture: hold still for 2 seconds...*")

    # Navigation
    st.markdown("---")
    if st.button("Previous: Body Input"):
        go_to_step(0)
        st.rerun()


def render_step_garment_input():
    st.header("Step 3: Garment Input")
    st.markdown("Upload a garment photo and/or enter garment measurements.")

    garment_photo = st.file_uploader(
        "Upload garment photo", type=["jpg", "jpeg", "png"],
        help="Upload a photo of the garment for 3D reconstruction via Garment3DGen.",
    )

    if garment_photo is not None:
        save_path = GARMENT_PHOTO_DIR / garment_photo.name
        save_path.write_bytes(garment_photo.read())
        st.session_state.garment_photo_path = str(save_path)
        st.image(str(save_path), caption="Garment reference", width=300)

    st.markdown("**Garment Measurements (cm)** — leave at 0 to skip")
    col_a, col_b = st.columns(2)
    with col_a:
        chest_width = st.number_input("Chest width", min_value=0.0, value=0.0, step=1.0)
        body_length = st.number_input("Body length", min_value=0.0, value=0.0, step=1.0)
    with col_b:
        sleeve_length = st.number_input("Sleeve length", min_value=0.0, value=0.0, step=1.0)
        waist_width = st.number_input("Waist width", min_value=0.0, value=0.0, step=1.0)

    if any(v > 0 for v in [chest_width, body_length, sleeve_length, waist_width]):
        st.session_state.garment_measurements = {
            "chest_width_cm": chest_width if chest_width > 0 else 50.0,
            "body_length_cm": body_length if body_length > 0 else 70.0,
            "sleeve_length_cm": sleeve_length if sleeve_length > 0 else 20.0,
            "waist_width_cm": waist_width if waist_width > 0 else 48.0,
        }
    else:
        st.session_state.garment_measurements = None

    st.session_state.fabric = st.selectbox(
        "Fabric type",
        ["cotton", "polyester", "spandex", "cotton-poly blend", "nylon"],
    )

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous: Body Scan"):
            go_to_step(1)
            st.rerun()
    with col2:
        if st.button("Next: Generate", type="primary"):
            go_to_step(3)
            st.rerun()


def render_step_generate():
    st.header("Step 4: Generate")

    # Summary
    st.subheader("Input Summary")
    st.markdown(f"- **Height:** {st.session_state.height_cm} cm")
    if st.session_state.weight_kg > 0:
        st.markdown(f"- **Weight:** {st.session_state.weight_kg} kg")
    st.markdown(f"- **Body scan photos:** {len(st.session_state.captured_images)}/{len(SCAN_ANGLES)}")
    if st.session_state.garment_photo_path:
        st.markdown(f"- **Garment photo:** {st.session_state.garment_photo_path}")
    if st.session_state.garment_measurements:
        st.markdown(f"- **Garment measurements:** provided")
    st.markdown(f"- **Fabric:** {st.session_state.fabric}")

    st.markdown("---")

    if st.button("Generate", type="primary", use_container_width=True):
        # Save captured images to server/inputs/
        for angle_name, img_bytes in st.session_state.captured_images.items():
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                idx = SCAN_ANGLES.index(angle_name)
                filepath = INPUT_DIR / f"capture_{idx:02d}_{angle_name}.png"
                cv2.imwrite(str(filepath), img)

        with st.spinner("Running pipeline... This may take a few minutes."):
            try:
                result = run_pipeline(
                    st.session_state.height_cm,
                    st.session_state.weight_kg,
                    st.session_state.garment_photo_path,
                    st.session_state.garment_measurements,
                    st.session_state.fabric,
                )
                st.session_state.sizing_result = result
                st.session_state.pipeline_ran = True
                go_to_step(4)
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    # Navigation
    st.markdown("---")
    if st.button("Previous: Garment Input"):
        go_to_step(2)
        st.rerun()


def render_step_results():
    st.header("Step 5: Results")

    # 3D viewer
    glb_data_uri = load_glb_as_base64(DEFAULT_GLB_PATH)
    if glb_data_uri:
        render_model_viewer(glb_data_uri)
    else:
        st.info("No 3D model generated. The pipeline may not have produced a GLB file.")
        uploaded_glb = st.file_uploader("Upload .glb file", type=["glb"])
        if uploaded_glb:
            b64 = base64.b64encode(uploaded_glb.read()).decode("utf-8")
            render_model_viewer(f"data:model/gltf-binary;base64,{b64}")

    # Sizing panel
    sizing_data = st.session_state.get("sizing_result")
    if sizing_data is None and DEFAULT_SIZING_PATH.exists():
        sizing_data = json.loads(DEFAULT_SIZING_PATH.read_text())

    if sizing_data:
        render_sizing_panel(sizing_data)
        render_pipeline_log(sizing_data)
    else:
        st.info("No sizing data available.")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous: Generate"):
            go_to_step(3)
            st.rerun()
    with col2:
        if st.button("Start Over", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STEP_RENDERERS = [
    render_step_body_input,
    render_step_body_scan,
    render_step_garment_input,
    render_step_generate,
    render_step_results,
]


def main():
    init_session()
    st.title("Olvon Physics Test")
    render_step_indicator()
    step = st.session_state.step
    if 0 <= step < len(STEP_RENDERERS):
        STEP_RENDERERS[step]()
    else:
        st.session_state.step = 0
        st.rerun()


if __name__ == "__main__":
    main()
