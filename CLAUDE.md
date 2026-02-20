# Olvon Physics Test - Project Context

## Architecture

Unified browser-based system for 3D body scanning and virtual garment fitting, running on vast.ai:

- **Client** (`client/`): Standalone webcam capture wizard using MediaPipe + OpenCV + pyttsx3 (optional, for local use)
- **Server** (`server/`): ML inference pipeline + Blender physics simulation
  - `server/lib/Garment3DGen/` — git submodule for ML garment generation
  - `server/lib/InstantMesh/` — git submodule for photo → 3D reconstruction
- **Visualizer** (`visualizer/`): Streamlit step-by-step wizard — the unified interface for body scan capture (via browser webcam), garment input, pipeline execution, and 3D result viewing

## Tech Stack

- Python 3.10+, MediaPipe, OpenCV, pyttsx3, PyTorch, trimesh, Blender (bpy subprocess), Streamlit
- Garment3DGen + InstantMesh (optional GPU dependencies: PyTorch3D, nvdiffrast, CLIP)

## Key Pattern: Real-First-With-Fallback

Every ML stage tries to load real model weights / run real ML pipelines. If unavailable, it falls back to parametric mesh generation via trimesh. This lets the full pipeline run without any trained models or GPU.

## File Purposes

| File | Role |
|------|------|
| `client/capture_wizard.py` | State machine: WAITING→VALIDATING→STABILIZING→CAPTURING→ROTATING→COMPLETE (5 angles with orientation detection) |
| `client/utils/pose_validator.py` | MediaPipe landmark visibility + stability + orientation detection (`detect_orientation()`, `validate_image_orientation()`) |
| `client/utils/audio_feedback.py` | pyttsx3 TTS wrapper with graceful fallback |
| `server/core/anny_inference.py` | Body mesh: ANNY parametric model via `import anny` → trimesh cylinder fallback |
| `server/core/garment_generator.py` | T-shirt mesh: torch load attempt → trimesh box fallback + `generate_from_measurements()` |
| `server/core/garment_3dgen.py` | Garment3DGen + InstantMesh wrapper (photo → 3D garment) with staged fallback |
| `server/core/physics_sim.py` | Blender bpy script (subprocess only, not importable) |
| `server/core/sizing_logic.py` | Math-based size recommendation from body measurements + height/weight/BMI |
| `server/core/diagnostics.py` | Shared `PipelineLog`, `log_fallback()`, `get_gpu_snapshot()` for structured fallback logging |
| `server/main_pipeline.py` | CLI orchestrator calling all 4 stages, accepts garment photo/measurements/height/weight |
| `visualizer/app.py` | Streamlit 5-step wizard: body input → webcam scan (5 angles with orientation validation) → garment input → pipeline execution → 3D results |

## Sizing Logic

- `recommend_size()` accepts `height_cm` and `weight_kg` in `body_measurements` dict
- Height + Weight → BMI-derived estimates blended 60/40 with mesh measurements
- Height only → proportional estimates blended 70/30
- Neither → backward compatible, mesh-only behavior
- Returns `bmi`, `height_cm`, `weight_kg` in `measurements` when provided

## Garment Generation Flow

1. **Photo provided** → Try Garment3DGen + InstantMesh (GPU required)
2. **Measurements provided** → Parametric mesh from explicit dimensions via `generate_from_measurements()`
3. **Neither** → Body-relative parametric T-shirt (existing fallback)

## Vast.ai Deployment

```bash
git clone --recursive <repo-url>
cd olvon-testing
bash setup.sh
streamlit run visualizer/app.py  # accessible on port 8501 (0.0.0.0)
```

## Model Weights

| Model | Type | Location / Install |
|-------|------|--------------------|
| ANNY | pip package (~42 MB) | `pip install anny` — all model data ships with the package |
| InstantMesh | HuggingFace checkpoints (~3.24 GB) | `server/lib/InstantMesh/ckpts/` — download from `TencentARC/InstantMesh` |
| Garment3DGen | Optimization-based (no weights) | CLIP + Fashion-CLIP auto-download on first run (~690 MB) |

Auto-downloaded model caches: `~/.cache/huggingface/hub/`, `~/.cache/clip/`

## Known Gaps

- Garment model weights (`garment_checkpoint.pth`) are not available — garment fallback meshes always used
- Garment3DGen + InstantMesh require CUDA GPU and submodule init
- physics_sim.py requires Blender 3.6+ installed and on PATH
- Standalone capture wizard requires physical webcam + display; Streamlit wizard uses browser webcam via `st.camera_input`
- Browser webcam capture requires HTTPS (vast.ai typically provides HTTPS tunnels)
- Back-view orientation detection is unreliable (MediaPipe may not detect pose); "Accept anyway" / "Skip validation" provided as overrides

## Quick Commands

```bash
# Self-test sizing logic
python server/core/sizing_logic.py

# Run pipeline (stages 1-2 work without Blender)
python -m server.main_pipeline --input-dir server/inputs --output-dir server/outputs --height 175 --weight 75

# Run pipeline with garment photo
python -m server.main_pipeline --garment-photo path/to/photo.jpg --height 175

# Run tests (no Blender needed)
python -m pytest tests/test_pipeline_no_blender.py -v

# Launch visualizer (accessible on 0.0.0.0:8501)
streamlit run visualizer/app.py

# Capture wizard (needs webcam)
python -m client.capture_wizard
```

## Conventions

- Blender script uses `bpy.ops.wm.obj_import()` (3.6+ API), not deprecated `bpy.ops.import_scene.obj()`
- Garment positioned 0.3m above body for physics gravity drop
- Sizing from bounding box: chest = x_extent * pi, waist = chest * 0.85
- model-viewer loaded via CDN (unpkg.com), no local JS bundling
- Streamlit binds 0.0.0.0:8501 via `.streamlit/config.toml`
- All ML fallbacks logged via `server/core/diagnostics.py` with GPU state
- Pipeline log passed through to Streamlit UI (collapsible expander)
