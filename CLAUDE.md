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
| `server/core/person_regenerator.py` | SD1.5 + ControlNet + IP-Adapter person regeneration (webcam → neutral pose, white bg) |
| `server/core/person_enhancer.py` | Real-ESRGAN upscale + BiRefNet background removal (fallback for regenerator) |
| `server/core/tryon_worker.py` | FASHN VTON v1.5 wrapper — person photo + garment photo → 2D try-on image |
| `server/core/feed_generator.py` | FFmpeg-based feed video generator (1080x1440, crossfade slideshow, auto codec detection) |
| `server/core/sizing_logic.py` | Math-based size recommendation from body measurements + height/weight/BMI |
| `server/core/diagnostics.py` | Shared `PipelineLog`, `log_fallback()`, `get_gpu_snapshot()` for structured fallback logging |
| `server/main_pipeline.py` | CLI orchestrator: measurements → try-on → sizing → feed video |
| `visualizer/app.py` | Streamlit 4-step wizard: body input → 3-angle webcam scan → garment input → results (feed video + virtual dressing room) |

## Pipeline Flow

0. `person_regenerator.regenerate_person(webcam_img)` → neutral-pose person on white bg (fallback: `person_enhancer` → raw photo)
1. `body_measurements.extract(front_photo, side_photo, height, weight)` → measurements dict (uses ORIGINAL photo)
2. `tryon_worker.generate()` per garment × per view (front + back) → try-on images (uses REGENERATED person)
3. `sizing_logic.recommend_size(measurements)` → sizing result
4. `feed_generator.generate_feed_video(front_tryon_images)` → feed video

### Person Preparation Fallback Chain

1. **person_regenerator** (SD1.5 + ControlNet + IP-Adapter) — best quality, neutral pose, white bg
2. **person_enhancer** (Real-ESRGAN + BiRefNet) — medium quality, original pose, white bg
3. **Raw webcam photo** — last resort

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

| Model | Type | Location / Install | License |
|-------|------|--------------------|---------|
| FASHN VTON v1.5 | Diffusion model (~2 GB) | `server/lib/fashn-vton/weights/` — downloaded via setup.sh | Apache 2.0 |
| MediaPipe Pose | Task file (~26 MB) | `client/assets/pose_landmarker_heavy.task` — auto-downloaded on first use | Apache 2.0 |
| RealESRGAN_x4plus | Upscaler (~67 MB) | `~/.cache/realesrgan/` — auto-downloaded on first use | **BSD 3-Clause** |
| BiRefNet-portrait | Segmentation (~168 MB) | `~/.u2net/` — auto-downloaded by rembg on first use | Apache 2.0 |
| SD1.5 base | Diffusion model (~4 GB) | `~/.cache/huggingface/` — auto-downloaded on first use | CreativeML OpenRAIL-M |
| ControlNet OpenPose | ControlNet (~1.5 GB) | `~/.cache/huggingface/` — auto-downloaded on first use | Apache 2.0 |
| IP-Adapter FaceID Plus v2 | Adapter (~1.5 GB) | `~/.cache/huggingface/` — auto-downloaded on first use | Apache 2.0 |
| InsightFace buffalo_l | Face analysis (~300 MB) | `~/.insightface/` — auto-downloaded on first use | MIT |
| Neutral pose skeleton | Static asset (50 KB) | `client/assets/neutral_pose_skeleton.png` — committed | N/A |

**Real-ESRGAN BSD 3-Clause requirement:** When distributing an app, reproduce the copyright notice
(`Copyright (c) 2021, xinntao`) and the three BSD conditions in the documentation or About section.
Full license text is in `README.md` under "Third-Party Licenses".

## Vast.ai Deployment Notes

- **Disk:** Rent with 60GB+ disk. The container overlay (default ~16GB) is the only accessible
  filesystem — the host nvme is NOT mounted inside the container. `/workspace`, `/tmp`, and `/`
  all share the same overlay.
- **SSH:** Connect with `ssh -A` (agent forwarding) so `git clone` works without copying keys
  to the instance. Load key locally first: `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519`.
- **PyTorch version:** FASHN VTON v1.5 requires PyTorch >= 2.4.0. `setup.sh` auto-detects and
  upgrades if needed. If manual install is required, **always install torch and torchvision
  together** — mismatched versions cause `operator torchvision::nms does not exist` at import:
  `pip uninstall torch torchvision torchaudio -y && pip install "torch>=2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir`
- **CUDA wheels:** Use `cu126` index even on CUDA 13.x — PyTorch CUDA wheels are backward-compatible
  with newer drivers.
- **Weight download:** Must run AFTER PyTorch >= 2.4.0 is installed. Uses the script directly:
  `python server/lib/fashn-vton/scripts/download_weights.py --weights-dir server/lib/fashn-vton/weights`
  (not a Python API method — `TryOnPipeline.download_weights()` does not exist).
- **FASHN VTON repo:** `fashn-AI/fashn-vton-1.5` (not `fashn-ai/fashn-vton`).
- **Result API:** Pipeline returns `result.images[0]` (not a PIL Image directly).
- **Call API:** Pipeline accepts only `person_image`, `garment_image`, `category`, `garment_photo_type`. Parameters `num_inference_steps` and `seed` are **not valid** and will raise `TypeError`.

## Critical: Streamlit Must Run on Vast.ai

**Never run `streamlit run visualizer/app.py` on the local machine.** The user's local machine
(Windows) has a Python venv without `fashn_vton`, GPU, or model weights. If Streamlit runs
locally, the browser connects to the local process, and the pipeline silently fails with
"FASHN VTON pipeline not available" — because `fashn_vton` is only installed on the vast.ai
instance.

**Required setup:** SSH into vast.ai with `-L 8501:localhost:8501` to tunnel the remote
Streamlit port. Run `streamlit run visualizer/app.py` on the vast.ai instance only. Access
via `http://localhost:8501` in the browser.

**Quick diagnostic:** If VTON fails, check `sys.executable` in the Streamlit process. If it
shows a Windows path (e.g., `C:\Users\...\python.exe`), Streamlit is running locally.

## Known Gaps

- Person regeneration requires ~10 GB VRAM peak; recommended 24 GB GPU for full pipeline
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
