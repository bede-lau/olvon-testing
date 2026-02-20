"""
Streamlit unified wizard for body scanning, garment input, pipeline execution,
and 3D result viewing. Runs entirely on vast.ai — users access via browser.
"""

import base64
import json
import logging
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

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

STEP_LABELS = ["Body Input", "Body Scan", "Garment Input", "Generate", "Results"]


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "step": 0,
        "height_cm": 170.0,
        "weight_kg": 0.0,
        "captured_images": {},       # {angle_name: image_bytes}
        "scan_angle_idx": 0,
        "garment_photo_path": None,
        "garment_measurements": None,
        "fabric": "cotton",
        "sizing_result": None,
        "pipeline_ran": False,
        "capture_attempt": {},       # {angle_name: int} for unique widget keys
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


def resize_image_bytes(image_bytes: bytes, max_w: int = 1280, max_h: int = 720) -> np.ndarray:
    """Decode image bytes, resize to fit within max dimensions, return RGB numpy array."""
    import cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def validate_captured_image(image_rgb: np.ndarray, expected: str) -> tuple[bool, str, str]:
    """Server-side orientation validation for a captured image."""
    try:
        from client.utils.pose_validator import validate_image_orientation
        return validate_image_orientation(image_rgb, expected)
    except Exception as e:
        return True, "unknown", f"Validation unavailable: {e}"


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

    instructions = {
        "front": "Face the camera directly. Ensure your full body is visible.",
        "right": "Turn so your right side faces the camera.",
        "back": "Turn to face away from the camera.",
        "left": "Turn so your left side faces the camera.",
        "elevated": "Face the camera and look slightly upward.",
    }
    st.info(instructions.get(current_angle, "Position yourself for this angle."))

    # Skip validation toggle
    st.session_state.skip_validation = st.checkbox(
        "Skip orientation validation",
        value=st.session_state.skip_validation,
        help="Enable if auto-detection isn't working (e.g., back view is unreliable).",
    )

    # Unique key per angle + attempt
    if current_angle not in st.session_state.capture_attempt:
        st.session_state.capture_attempt[current_angle] = 0
    attempt = st.session_state.capture_attempt[current_angle]
    widget_key = f"cam_{current_angle}_{attempt}"

    photo = st.camera_input(f"Take {current_angle} photo", key=widget_key)

    if photo is not None:
        image_bytes = photo.getvalue()
        image_rgb = resize_image_bytes(image_bytes)

        if image_rgb is None:
            st.error("Failed to decode image. Please try again.")
            return

        # Validate orientation
        if st.session_state.skip_validation:
            accepted, detected, msg = True, "skipped", "Validation skipped"
        else:
            accepted, detected, msg = validate_captured_image(image_rgb, current_angle)

        if accepted:
            st.success(f"Accepted! {msg}")
            # Store the raw bytes for saving later
            st.session_state.captured_images[current_angle] = image_bytes
            st.session_state.scan_angle_idx = angle_idx + 1
            st.rerun()
        else:
            st.error(f"Rejected: {msg}")
            st.markdown(f"**Expected:** {current_angle.upper()} | **Detected:** {detected.upper()}")

            # Manual accept for difficult angles (especially back)
            col_retry, col_accept = st.columns(2)
            with col_retry:
                if st.button("Retake"):
                    st.session_state.capture_attempt[current_angle] = attempt + 1
                    st.rerun()
            with col_accept:
                if st.button("Accept anyway"):
                    st.session_state.captured_images[current_angle] = image_bytes
                    st.session_state.scan_angle_idx = angle_idx + 1
                    st.rerun()

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
        import cv2
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
