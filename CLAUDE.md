# Olvon VTON - Project Context

## Architecture

Unified browser-based system for 2D virtual try-on and body measurement, running on vast.ai:

- **Client** (`client/`): Standalone webcam capture wizard using MediaPipe + OpenCV + pyttsx3 (optional, for local use)
- **Server** (`server/`): 2D VTON pipeline using FASHN VTON v1.5 + MediaPipe body measurements
- **Visualizer** (`visualizer/`): Streamlit 4-step wizard — body input, webcam scan (3 angles), garment upload, results (feed video + virtual dressing room)

## Tech Stack

- Python 3.10+, MediaPipe, OpenCV, PyTorch, FASHN VTON v1.5, FFmpeg, Streamlit, streamlit-webrtc

## Key Pattern: Real-First-With-Fallback

Every stage tries real inference first, then falls back gracefully:
- FASHN VTON → returns None (no try-on image)
- MediaPipe landmarks → height/weight empirical formulas → population averages
- FFmpeg → returns None (no feed video)

## File Purposes

| File | Role |
|------|------|
| `client/capture_wizard.py` | State machine: standalone webcam capture (5 angles with orientation detection) |
| `client/utils/pose_validator.py` | MediaPipe landmark visibility + stability + orientation detection |
| `client/utils/audio_feedback.py` | pyttsx3 TTS wrapper with graceful fallback |
| `server/core/body_measurements.py` | MediaPipe pose landmarks → body measurements (chest, waist, hip, shoulder_width) with fallback chain |
| `server/core/tryon_worker.py` | FASHN VTON v1.5 wrapper — person photo + garment photo → 2D try-on image |
| `server/core/feed_generator.py` | FFmpeg-based feed video generator (1080x1440, crossfade slideshow) |
| `server/core/sizing_logic.py` | Math-based size recommendation from body measurements + height/weight/BMI |
| `server/core/diagnostics.py` | Shared `PipelineLog`, `log_fallback()`, `get_gpu_snapshot()` for structured fallback logging |
| `server/main_pipeline.py` | CLI orchestrator: measurements → try-on → sizing → feed video |
| `visualizer/app.py` | Streamlit 4-step wizard: body input → 3-angle webcam scan → garment input → results (feed video + virtual dressing room) |

## Pipeline Flow

1. `body_measurements.extract(front_photo, side_photo, height, weight)` → measurements dict
2. `tryon_worker.generate()` per garment × per view (front + back) → try-on images
3. `sizing_logic.recommend_size(measurements)` → sizing result
4. `feed_generator.generate_feed_video(front_tryon_images)` → feed video

## Sizing Logic

- `recommend_size()` accepts `height_cm` and `weight_kg` in `body_measurements` dict
- Height + Weight → BMI-derived estimates blended 60/40 with measurement estimates
- Height only → proportional estimates blended 70/30
- Neither → backward compatible, measurement-only behavior
- Returns `bmi`, `height_cm`, `weight_kg` in `measurements` when provided

## Body Measurements

- Fallback chain: MediaPipe landmarks → height/weight empirical → population averages
- Landmarks: nose-to-ankle pixel distance + known height → pixel-to-cm ratio
- Circumferences: `width * π * correction_factor`
- Reuses `client/utils/pose_validator._ensure_model()` for model download

## Virtual Try-On

- FASHN VTON v1.5 (Apache 2.0 license)
- Input: person photo (full body) + garment photo → output: 2D photorealistic try-on image
- Handles face rendering natively — no compositing needed
- ~2GB weights, ~8GB VRAM
- Falls back to None when unavailable

## Vast.ai Deployment

```bash
git clone <repo-url>
cd olvon-testing
bash setup.sh
streamlit run visualizer/app.py --server.address=0.0.0.0
```

## Model Weights

| Model | Type | Location / Install |
|-------|------|--------------------|
| FASHN VTON v1.5 | Diffusion model (~2 GB) | `server/lib/fashn-vton/weights/` — downloaded via setup.sh |
| MediaPipe Pose | Task file (~26 MB) | `client/assets/pose_landmarker_heavy.task` — auto-downloaded on first use |

## Known Gaps

- FASHN VTON requires CUDA GPU (~8GB VRAM) for inference
- FFmpeg required for feed video generation
- Back-view orientation detection is unreliable (MediaPipe may not detect pose); uses face-absence + countdown
- Browser webcam capture requires HTTPS; WebRTC needs STUN server (Google's public STUN configured)
- `BodyScanProcessor` runs MediaPipe in a worker thread; communicates captures to main thread via `queue.Queue`

## Quick Commands

```bash
# Self-test sizing logic
python server/core/sizing_logic.py

# Run pipeline
python -m server.main_pipeline --front-photo test.jpg --height 175 --weight 75 --garment-photo garment.jpg --category tops

# Run tests
python -m pytest tests/test_pipeline.py -v

# Launch visualizer (accessible on 0.0.0.0:8501)
streamlit run visualizer/app.py

# Capture wizard (needs webcam)
python -m client.capture_wizard
```

## Conventions

- Streamlit binds 0.0.0.0:8501 via `.streamlit/config.toml`
- All ML fallbacks logged via `server/core/diagnostics.py` with GPU state
- Pipeline log passed through to Streamlit UI (collapsible expander)
- Feed video: 1080x1440, H.264, yuv420p, crossfade transitions
- VTON runs twice per garment in dressing room: front photo + back photo
- Dressing room results cached in session state per garment index
