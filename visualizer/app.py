"""
Streamlit wizard for 2D virtual try-on using FASHN VTON v1.5.

4-step flow: Body Input → Body Scan (3 angles) → Garment Input → Results
Results show two tabs: Feed Video and Virtual Dressing Room.
"""

import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="Olvon VTON", layout="wide")

OUTPUT_DIR = Path("server/outputs")
INPUT_DIR = Path("server/inputs")
GARMENT_PHOTO_DIR = Path("server/inputs/garment_photos")
DEFAULT_SIZING_PATH = OUTPUT_DIR / "sizing_result.json"

for d in [OUTPUT_DIR, INPUT_DIR, GARMENT_PHOTO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Angles to capture — back can be disabled via BACK_SCAN_ENABLED in visualizer/config.py
from visualizer.config import BACK_SCAN_ENABLED as _BACK_SCAN_ENABLED
SCAN_ANGLES = ["front", "side", "back"] if _BACK_SCAN_ENABLED else ["front", "side"]
SCAN_INSTRUCTIONS = {
    "front": "Face the camera directly. Ensure your full body is visible.",
    "side": "Turn so your side faces the camera.",
    "back": "Turn to face AWAY from the camera.",
}

STEP_LABELS = ["Body Input", "Body Scan", "Garment Input", "Results"]

STABILITY_HOLD_SECONDS = 2.0
BACK_COUNTDOWN_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Body scan video processor (runs in a worker thread)
# ---------------------------------------------------------------------------
class BodyScanProcessor(VideoProcessorBase):
    """
    Processes each video frame: runs MediaPipe pose detection, checks orientation,
    draws overlay, and auto-captures when conditions are met.

    For back view: uses face-absence detection (no MediaPipe face = facing away)
    combined with a countdown timer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.capture_queue: queue.Queue = queue.Queue()

        self._target_angle: str = "front"
        self._skip_validation: bool = False
        self._active: bool = True

        self._stability_start: float | None = None
        self._detected_orientation: str = "unknown"
        self._body_visible: bool = False
        self._captured: bool = False
        self._frame_count: int = 0

        self._landmarker = None
        self._mp = None
        self._init_error: str | None = None

    def _init_mediapipe(self):
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
        with self._lock:
            self._target_angle = angle
            self._stability_start = None
            self._captured = False

    def _detect_back_view(self, landmarks) -> bool:
        """Check if person is facing away: nose visibility low + shoulders wide."""
        if landmarks is None or len(landmarks) < 29:
            return False
        nose_vis = landmarks[0].visibility
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        return nose_vis < 0.5 and shoulder_width > 0.10

    def _check_back_body_visible(self, landmarks) -> bool:
        """Body visibility for back view — excludes nose (not visible from behind)."""
        BACK_INDICES = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles
        if landmarks is None or len(landmarks) < 29:
            return False
        return all(landmarks[idx].visibility >= 0.5 for idx in BACK_INDICES)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        if not self.active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self._landmarker is None and self._init_error is None:
            self._init_mediapipe()

        if self._init_error:
            cv2.putText(
                img, f"MediaPipe error: {self._init_error[:60]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )
        timestamp_ms = self._frame_count * 33
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

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

            if target == "back":
                body_visible = self._check_back_body_visible(landmarks)
                is_back = self._detect_back_view(landmarks)
                detected = "back" if is_back else detect_orientation(landmarks)
                orientation_ok = skip or is_back
            else:
                body_visible = check_full_body_visible(landmarks)
                detected = detect_orientation(landmarks)
                # Map "right" or "left" to "side" for our 3-angle system
                if target == "side" and detected in ("right", "left"):
                    orientation_ok = True
                else:
                    orientation_ok = skip or (detected == target)

        self._body_visible = body_visible
        self._detected_orientation = detected

        # Stability tracking
        now = time.monotonic()
        hold_time = BACK_COUNTDOWN_SECONDS if target == "back" else STABILITY_HOLD_SECONDS

        if body_visible and orientation_ok and not already_captured:
            if self._stability_start is None:
                self._stability_start = now
            elapsed = now - self._stability_start

            if elapsed >= hold_time:
                with self._lock:
                    self._captured = True
                self._stability_start = None
                self.capture_queue.put((target, img.copy()))
        else:
            if not body_visible or not orientation_ok:
                self._stability_start = None

        # --- Draw overlay ---
        h, w = img.shape[:2]

        if landmarks:
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        status_color = (0, 255, 0) if orientation_ok and body_visible else (0, 0, 255)
        cv2.putText(
            img, f"Target: {target.upper()}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            img, f"Detected: {detected.upper()}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
        )

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
            remaining = max(0, hold_time - elapsed)
            progress = min(1.0, elapsed / hold_time)
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
        "captured_images": {},       # {angle_name: png_bytes}
        "scan_angle_idx": 0,
        "garment_files": [],         # list of UploadedFile-like dicts
        "garment_paths": [],         # list of saved paths
        "category": "tops",
        "garment_photo_type": "flat-lay",
        "fabric": "cotton",
        "sizing_result": None,
        "pipeline_ran": False,
        "skip_validation": False,
        "tryon_cache": {},           # {garment_idx: {"front": PIL.Image, "back": PIL.Image}}
        "active_garment": None,      # currently selected garment index or None
        "dressing_view": "front",    # "front" or "back"
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

    measurements = sizing_data.get("measurements", sizing_data.get("body_measurements", {}))
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


def run_pipeline(
    front_photo_path, side_photo_path, back_photo_path,
    garment_paths, category, garment_photo_type,
    height_cm, weight_kg, fabric,
):
    import datetime
    with open("/tmp/vton_debug.log", "a") as f:
        f.write(f"\n[{datetime.datetime.now()}] run_pipeline() called\n")
        f.write(f"  garment_paths={garment_paths}\n")
        f.write(f"  front={front_photo_path}\n")
    from server.main_pipeline import VTONPipeline
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = VTONPipeline()
    result = pipeline.run(
        front_photo=front_photo_path,
        garment_photos=garment_paths,
        category=category,
        garment_photo_type=garment_photo_type,
        side_photo=side_photo_path,
        back_photo=back_photo_path,
        height_cm=height_cm if height_cm and height_cm > 0 else None,
        weight_kg=weight_kg if weight_kg and weight_kg > 0 else None,
        fabric=fabric,
    )
    with open("/tmp/vton_debug.log", "a") as f:
        f.write(f"  pipeline returned, pipeline_log={result.get('pipeline_log')}\n")
        f.write(f"  tryon_results={result.get('tryon_results')}\n")
    return result


def run_single_tryon(person_path, garment_path, category, garment_photo_type, output_path):
    """Run a single try-on for the virtual dressing room (on-demand)."""
    # Prefer regenerated person image if available
    regen_path = Path("server/outputs/person_regenerated.png")
    enhanced_path = Path("server/outputs/person_enhanced.png")
    if regen_path.exists():
        person_path = str(regen_path)
    elif enhanced_path.exists():
        person_path = str(enhanced_path)

    from server.core.tryon_worker import TryOnWorker
    worker = TryOnWorker()
    return worker.generate(
        person_path=person_path,
        garment_path=garment_path,
        category=category,
        output_path=output_path,
        garment_photo_type=garment_photo_type,
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
        "The system will automatically capture each angle when you hold the correct pose. "
        "**3 captures needed:** front, side, and back."
    )

    captured = st.session_state.captured_images
    angle_idx = st.session_state.scan_angle_idx

    if captured:
        st.markdown("**Captured so far:**")
        thumb_cols = st.columns(len(SCAN_ANGLES))
        for i, angle in enumerate(SCAN_ANGLES):
            with thumb_cols[i]:
                if angle in captured:
                    st.image(captured[angle], caption=angle, width=160)
                    st.download_button(
                        label="⬇ Download",
                        data=captured[angle],
                        file_name=f"capture_{angle}.png",
                        mime="image/png",
                        key=f"dl_{angle}",
                    )
                else:
                    st.markdown(f"*{angle} — not yet captured*")

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

    if current_angle == "back":
        st.warning(
            "For back view: turn away from the camera. "
            "A 5-second countdown will start when no face is detected."
        )

    st.session_state.skip_validation = st.checkbox(
        "Skip orientation validation",
        value=st.session_state.skip_validation,
        help="Enable if auto-detection isn't working.",
    )

    tab_cam, tab_upload = st.tabs(["📷 Webcam Capture", "📁 Upload Photo"])

    with tab_cam:
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

        if ctx.video_processor:
            ctx.video_processor.target_angle = current_angle
            ctx.video_processor.skip_validation = st.session_state.skip_validation

        if ctx.state.playing and ctx.video_processor:
            @st.fragment(run_every=0.5)
            def _capture_poll():
                """Polls capture queue every 500 ms; advances angle on success."""
                if not (ctx.state.playing and ctx.video_processor):
                    return
                try:
                    angle_name, bgr_img = ctx.video_processor.capture_queue.get_nowait()
                    _, png_buf = cv2.imencode(".png", bgr_img)
                    st.session_state.captured_images[angle_name] = png_buf.tobytes()
                    st.session_state.scan_angle_idx += 1
                    try:
                        st.rerun(scope="app")   # Streamlit >= 1.37
                    except TypeError:
                        st.rerun()              # Streamlit < 1.37
                except queue.Empty:
                    pass

            _capture_poll()

        if ctx.state.playing and ctx.video_processor:
            if st.button("Capture manually (skip auto-detect)"):
                ctx.video_processor.skip_validation = True
                st.markdown("*Manual capture: hold still...*")

    with tab_upload:
        up = st.file_uploader(
            f"Upload {current_angle} photo (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            key=f"upload_{current_angle}",
            help="Full-body photo in correct orientation.",
        )
        if up:
            img_bytes = up.read()
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                _, png_buf = cv2.imencode(".png", decoded)
                st.session_state.captured_images[current_angle] = png_buf.tobytes()
                st.session_state.scan_angle_idx += 1
                st.success(f"{current_angle.title()} photo uploaded.")
                st.rerun()
            else:
                st.error("Could not decode image. Try a different file.")

    st.markdown("---")
    if st.button("Previous: Body Input"):
        go_to_step(0)
        st.rerun()


def render_step_garment_input():
    st.header("Step 3: Garment Input")
    st.markdown("Upload one or more garment photos, select category, and run the pipeline.")

    st.info(
        "**Tips for best try-on results:**\n"
        "- Use garment photos with a **contrasting background** (avoid white shirt on white background — "
        "low contrast causes segmentation to fail).\n"
        "- Wear a **fitted base garment** during scanning — being shirtless can reduce try-on accuracy.\n"
        "- Ensure your **full head and face are visible** in the scan — partial head crop reduces pose accuracy."
    )

    garment_files = st.file_uploader(
        "Upload garment photos", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload flat-lay or model-worn garment photos.",
    )

    if garment_files:
        saved_paths = []
        cols = st.columns(min(len(garment_files), 4))
        for i, gf in enumerate(garment_files):
            save_path = GARMENT_PHOTO_DIR / gf.name
            save_path.write_bytes(gf.read())
            saved_paths.append(str(save_path))
            with cols[i % len(cols)]:
                st.image(str(save_path), caption=gf.name, width=200)
        st.session_state.garment_paths = saved_paths

    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state.category = st.selectbox(
            "Garment category",
            ["tops", "bottoms", "one-pieces"],
        )
    with col_b:
        st.session_state.garment_photo_type = st.radio(
            "Photo type",
            ["flat-lay", "model"],
            horizontal=True,
        )

    st.session_state.fabric = st.selectbox(
        "Fabric type",
        ["cotton", "polyester", "spandex", "cotton-poly blend", "nylon"],
    )

    st.markdown("---")

    # Run pipeline button
    can_run = bool(st.session_state.garment_paths) and bool(st.session_state.captured_images.get("front"))
    if not can_run:
        st.warning("Need at least a front body photo and one garment photo to run.")

    if st.button("Run Pipeline", type="primary", use_container_width=True, disabled=not can_run):
        # Save captured body photos
        for angle_name, img_bytes in st.session_state.captured_images.items():
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                filepath = INPUT_DIR / f"capture_{angle_name}.png"
                cv2.imwrite(str(filepath), img)

        front_path = str(INPUT_DIR / "capture_front.png")
        side_path = str(INPUT_DIR / "capture_side.png") if "side" in st.session_state.captured_images else None
        back_path = str(INPUT_DIR / "capture_back.png") if "back" in st.session_state.captured_images else None

        with st.spinner("Running pipeline... This may take a few minutes."):
            try:
                result = run_pipeline(
                    front_path, side_path, back_path,
                    st.session_state.garment_paths,
                    st.session_state.category,
                    st.session_state.garment_photo_type,
                    st.session_state.height_cm,
                    st.session_state.weight_kg,
                    st.session_state.fabric,
                )
                st.session_state.sizing_result = result
                st.session_state.pipeline_ran = True
                # Reset dressing room state
                st.session_state.tryon_cache = {}
                st.session_state.active_garment = None
                go_to_step(3)
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous: Body Scan"):
            go_to_step(1)
            st.rerun()


def render_step_results():
    st.header("Step 4: Results")

    sizing_data = st.session_state.get("sizing_result")
    _from_session = sizing_data is not None
    if sizing_data is None and DEFAULT_SIZING_PATH.exists():
        sizing_data = json.loads(DEFAULT_SIZING_PATH.read_text())
    _plog = sizing_data.get("pipeline_log", []) if sizing_data else []
    st.warning(
        f"DEBUG: data_source={'session_state' if _from_session else 'file'}, "
        f"pipeline_ran={st.session_state.get('pipeline_ran')}, "
        f"pipeline_log_entries={len(_plog)}, "
        f"sizing_result.json_exists={DEFAULT_SIZING_PATH.exists()}"
    )

    if not sizing_data:
        st.info("No results yet. Run the pipeline first.")
        if st.button("Previous: Garment Input"):
            go_to_step(2)
            st.rerun()
        return

    tab_feed, tab_dressing = st.tabs(["Feed Video", "Virtual Dressing Room"])

    # --- Tab 1: Feed Video ---
    with tab_feed:
        feed_path = sizing_data.get("feed_video")
        if feed_path and Path(feed_path).exists():
            st.video(feed_path)
        else:
            st.info("No feed video generated. FASHN VTON or FFmpeg may not be available.")

            # Show any try-on images that were generated
            tryon_results = sizing_data.get("tryon_results", {})
            for idx, views in tryon_results.items():
                front_path = views.get("front")
                if front_path and Path(front_path).exists():
                    st.image(front_path, caption=f"Try-on result (garment {idx})", width=400)

        render_sizing_panel(sizing_data)
        render_pipeline_log(sizing_data)

    # --- Tab 2: Virtual Dressing Room ---
    with tab_dressing:
        st.subheader("Virtual Dressing Room")

        # View toggle
        col_flip, col_spacer = st.columns([1, 3])
        with col_flip:
            if st.button("Flip View"):
                st.session_state.dressing_view = (
                    "back" if st.session_state.dressing_view == "front" else "front"
                )
                st.rerun()

        current_view = st.session_state.dressing_view
        st.caption(f"Current view: **{current_view.upper()}**")

        # Get raw body photos
        front_capture = st.session_state.captured_images.get("front")
        back_capture = st.session_state.captured_images.get("back")

        active_garment = st.session_state.active_garment
        tryon_cache = st.session_state.tryon_cache

        # Display main image
        display_image = None

        if active_garment is not None and active_garment in tryon_cache:
            cached = tryon_cache[active_garment]
            if current_view in cached and cached[current_view] is not None:
                display_image = cached[current_view]

        if display_image is not None:
            st.image(display_image, caption=f"Try-on - {current_view} view", use_container_width=True)
        else:
            # Show raw body photo
            raw_photo = front_capture if current_view == "front" else back_capture
            if raw_photo:
                st.image(raw_photo, caption=f"Body - {current_view} view", use_container_width=True)
            else:
                st.info(f"No {current_view} photo captured.")

        # Garment cards
        garment_paths = st.session_state.garment_paths
        if garment_paths:
            st.markdown("**Select a garment:**")
            gcols = st.columns(min(len(garment_paths), 4))
            for i, gpath in enumerate(garment_paths):
                with gcols[i % len(gcols)]:
                    is_active = active_garment == i
                    border_style = "3px solid #00ff00" if is_active else "1px solid #ccc"
                    st.markdown(
                        f'<div style="border: {border_style}; padding: 4px; border-radius: 8px;">',
                        unsafe_allow_html=True,
                    )
                    if Path(gpath).exists():
                        st.image(gpath, width=150)
                    label = "Remove" if is_active else "Try on"
                    if st.button(label, key=f"garment_{i}"):
                        if is_active:
                            # Toggle off
                            st.session_state.active_garment = None
                            st.rerun()
                        else:
                            # Try on this garment
                            st.session_state.active_garment = i

                            if i not in tryon_cache:
                                # Run VTON for front + back
                                cache_entry = {}
                                front_path = INPUT_DIR / "capture_front.png"
                                back_path = INPUT_DIR / "capture_back.png"

                                with st.spinner(f"Running try-on for garment {i + 1}..."):
                                    # Front
                                    if front_path.exists():
                                        out_f = OUTPUT_DIR / f"dressing_front_{i}.png"
                                        result_f = run_single_tryon(
                                            str(front_path), gpath,
                                            st.session_state.category,
                                            st.session_state.garment_photo_type,
                                            str(out_f),
                                        )
                                        if result_f and Path(result_f).exists():
                                            cache_entry["front"] = str(result_f)

                                    # Back
                                    if back_path.exists():
                                        out_b = OUTPUT_DIR / f"dressing_back_{i}.png"
                                        result_b = run_single_tryon(
                                            str(back_path), gpath,
                                            st.session_state.category,
                                            st.session_state.garment_photo_type,
                                            str(out_b),
                                        )
                                        if result_b and Path(result_b).exists():
                                            cache_entry["back"] = str(result_b)

                                st.session_state.tryon_cache[i] = cache_entry

                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No garments uploaded.")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous: Garment Input"):
            go_to_step(2)
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
    render_step_results,
]


def main():
    init_session()
    st.title("Olvon VTON")
    st.caption("Build: 2026-03-12-debug")
    render_step_indicator()
    step = st.session_state.step
    if 0 <= step < len(STEP_RENDERERS):
        STEP_RENDERERS[step]()
    else:
        st.session_state.step = 0
        st.rerun()


if __name__ == "__main__":
    main()
