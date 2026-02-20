"""
Streamlit visualizer with Google model-viewer for 3D GLB display.
Full UI for body measurements, garment input, and in-process pipeline execution.
"""

import base64
import json
import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Olvon Physics Test - 3D Viewer", layout="wide")

OUTPUT_DIR = Path("server/outputs")
GARMENT_PHOTO_DIR = Path("server/inputs/garment_photos")
DEFAULT_GLB_PATH = OUTPUT_DIR / "final_fitted_avatar.glb"
DEFAULT_SIZING_PATH = OUTPUT_DIR / "sizing_result.json"

GARMENT_PHOTO_DIR.mkdir(parents=True, exist_ok=True)


def load_glb_as_base64(path: Path) -> str | None:
    """Load a GLB file and return as base64 data URI."""
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:model/gltf-binary;base64,{b64}"


def render_model_viewer(glb_data_uri: str, height: int = 600):
    """Render a Google model-viewer component with the GLB data."""
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
    """Display sizing recommendation results."""
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
    """Display pipeline fallback log in a collapsible expander."""
    pipeline_log = sizing_data.get("pipeline_log", [])
    with st.expander("Pipeline Log", expanded=False):
        if pipeline_log:
            for entry in pipeline_log:
                gpu = f" | {entry['gpu_info']}" if entry.get("gpu_info") else ""
                st.text(f"[{entry['stage']}] {entry['error_type']}: {entry['message']}{gpu}")
        else:
            st.text("Pipeline completed successfully — no fallbacks triggered.")


def run_pipeline(height_cm, weight_kg, garment_photo_path, garment_measurements, fabric):
    """Run the pipeline in-process and return the result."""
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


def main():
    st.title("Olvon Physics Test - 3D Avatar Viewer")

    # --- Sidebar ---
    st.sidebar.header("Body Measurements")
    height_cm = st.sidebar.number_input(
        "Height (cm) *", min_value=100.0, max_value=250.0, value=170.0, step=1.0,
        help="Required. Your height in centimeters.",
    )
    weight_kg = st.sidebar.number_input(
        "Weight (kg)", min_value=0.0, max_value=300.0, value=0.0, step=1.0,
        help="Optional. Your weight in kilograms. Set to 0 to skip.",
    )

    st.sidebar.header("Garment Input")
    garment_photo = st.sidebar.file_uploader(
        "Upload garment photo", type=["jpg", "jpeg", "png"],
        help="Upload a photo of the garment for 3D reconstruction via Garment3DGen.",
    )

    st.sidebar.markdown("**Garment Measurements (cm)**")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        chest_width = st.number_input("Chest width", min_value=0.0, value=0.0, step=1.0)
        body_length = st.number_input("Body length", min_value=0.0, value=0.0, step=1.0)
    with col_b:
        sleeve_length = st.number_input("Sleeve length", min_value=0.0, value=0.0, step=1.0)
        waist_width = st.number_input("Waist width", min_value=0.0, value=0.0, step=1.0)

    fabric = st.sidebar.selectbox(
        "Fabric type",
        ["cotton", "polyester", "spandex", "cotton-poly blend", "nylon"],
    )

    # Process garment photo
    garment_photo_path = None
    if garment_photo is not None:
        save_path = GARMENT_PHOTO_DIR / garment_photo.name
        save_path.write_bytes(garment_photo.read())
        garment_photo_path = str(save_path)

    # Build garment measurements dict if any provided
    garment_measurements = None
    if any(v > 0 for v in [chest_width, body_length, sleeve_length, waist_width]):
        garment_measurements = {
            "chest_width_cm": chest_width if chest_width > 0 else 50.0,
            "body_length_cm": body_length if body_length > 0 else 70.0,
            "sleeve_length_cm": sleeve_length if sleeve_length > 0 else 20.0,
            "waist_width_cm": waist_width if waist_width > 0 else 48.0,
        }

    # Generate button
    if st.sidebar.button("Generate", type="primary", use_container_width=True):
        with st.spinner("Running pipeline..."):
            try:
                result = run_pipeline(
                    height_cm, weight_kg,
                    garment_photo_path, garment_measurements, fabric,
                )
                st.session_state["sizing_result"] = result
                st.session_state["pipeline_ran"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    # --- Main content ---
    col_viewer, col_sizing = st.columns([3, 1])

    with col_viewer:
        glb_data_uri = load_glb_as_base64(DEFAULT_GLB_PATH)
        if glb_data_uri:
            render_model_viewer(glb_data_uri)
        else:
            st.info(
                "No 3D model loaded. Enter your measurements and click **Generate** to run the pipeline, "
                "or upload a GLB file."
            )
            uploaded_glb = st.file_uploader("Upload .glb file", type=["glb"])
            if uploaded_glb:
                b64 = base64.b64encode(uploaded_glb.read()).decode("utf-8")
                render_model_viewer(f"data:model/gltf-binary;base64,{b64}")

        # Show uploaded garment reference photo
        if garment_photo is not None:
            st.subheader("Garment Reference")
            st.image(garment_photo_path or garment_photo, width=300)

    with col_sizing:
        sizing_data = st.session_state.get("sizing_result")
        if sizing_data is None and DEFAULT_SIZING_PATH.exists():
            sizing_data = json.loads(DEFAULT_SIZING_PATH.read_text())

        if sizing_data:
            render_sizing_panel(sizing_data)
        else:
            st.info("No sizing data available. Click **Generate** to run the pipeline.")

    # Pipeline log at bottom
    if sizing_data:
        render_pipeline_log(sizing_data)


if __name__ == "__main__":
    main()
