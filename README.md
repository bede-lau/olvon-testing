# Olvon VTON

2D virtual try-on system using FASHN VTON v1.5 for photorealistic garment fitting.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VAST.AI INSTANCE                               │
│                                                                         │
│  ┌─────────────────────────┐     ┌───────────────────────────────────┐  │
│  │     Streamlit Wizard    │     │             Server                │  │
│  │    (Browser-based UI)   │     │         (2D VTON Pipeline)        │  │
│  │                         │     │                                   │  │
│  │ Step 1: Body Input      │     │ body_measurements (MediaPipe)     │  │
│  │ Step 2: Webcam Scan     │────>│       ↓                           │  │
│  │   (front + side + back) │     │ tryon_worker (FASHN VTON v1.5)    │  │
│  │ Step 3: Garment Upload  │     │       ↓                           │  │
│  │ Step 4: Results         │<────│ sizing_logic (size recommendation)│  │
│  │   - Feed Video          │     │       ↓                           │  │
│  │   - Dressing Room       │     │ feed_generator (FFmpeg video)     │  │
│  └─────────────────────────┘     └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         ▲
         │ Browser (HTTPS)
         │
    ┌────┴────┐
    │  User   │
    │ (local) │
    └─────────┘
```

**Pattern:** Real-first-with-fallback. Each stage attempts real inference, then
falls back gracefully (VTON → None, landmarks → empirical formulas → population averages).

## Prerequisites

- **Python 3.10+**
- **CUDA GPU** (~8GB VRAM) for FASHN VTON inference
- **FFmpeg** for feed video generation
- **Vast.ai account** (for GPU instance)
- **Browser with webcam** (for body scan capture)

## Quick Start (Vast.ai)

### 1. Set Up the Instance

1. Rent a GPU instance on [vast.ai](https://vast.ai) (RTX 3090/4090/4070 Super Ti, 50GB+ disk)
2. SSH in with port forwarding:

```bash
ssh -A -p <port> root@<host> -L 8080:localhost:8080
```

3. Install FFmpeg:

```bash
apt update && apt install -y ffmpeg
```

4. Clone and set up:

```bash
cd ~
git clone git@github.com:bede-lau/olvon-testing.git && cd olvon-testing
bash setup.sh
```

### 2. Run the Wizard

```bash
streamlit run visualizer/app.py
```

Open `http://localhost:8501` in your browser. The wizard has 4 steps:

1. **Body Input** — Enter height (cm) and optional weight (kg)
2. **Body Scan** — 3 captures via webcam: front, side, back (auto-capture with pose detection)
3. **Garment Input** — Upload garment photos, select category, run pipeline
4. **Results** — Two tabs:
   - **Feed Video** — Short-form try-on slideshow video
   - **Virtual Dressing Room** — Flip between front/back views, click garments to try on

### 3. Clean Up

Destroy the vast.ai instance to stop billing.

## CLI Usage

```bash
# Single garment
python -m server.main_pipeline \
  --front-photo person_front.jpg \
  --back-photo person_back.jpg \
  --garment-photo garment.jpg \
  --category tops \
  --height 175 --weight 75

# Multiple garments
python -m server.main_pipeline \
  --front-photo person_front.jpg \
  --garment-photo shirt1.jpg \
  --garment-photo shirt2.jpg \
  --category tops
```

## Model Weights

| Model | Size | Location |
|-------|------|----------|
| FASHN VTON v1.5 | ~2 GB | `server/lib/fashn-vton/weights/` (downloaded by setup.sh) |
| MediaPipe Pose | ~26 MB | `client/assets/` (auto-downloaded on first use) |

### FASHN VTON License

FASHN VTON v1.5 is licensed under Apache 2.0. See [fashn-ai/fashn-vton](https://github.com/fashn-ai/fashn-vton) for details.

## Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

Tests cover body measurements, sizing logic, try-on worker fallback, and feed generator
command building. No GPU required for tests.

## Troubleshooting

**"FASHN VTON pipeline not available"** — Ensure `fashn-vton` is installed (`pip install -e server/lib/fashn-vton`) and weights are downloaded. Requires CUDA GPU.

**"FFmpeg not found on PATH"** — Install FFmpeg: `apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS). Feed video generation will be skipped without it.

**Webcam not opening in browser** — Browser webcam requires HTTPS. Use SSH port forwarding (`-L 8501:localhost:8501`) with vast.ai.

**Back view not detected** — The system uses face-absence detection with a 5-second countdown for back view. Use "Skip validation" if auto-detection fails.

**CUDA out of memory** — FASHN VTON needs ~8GB VRAM. Use a GPU with sufficient memory (RTX 3090+).

**mediapipe not installed** — Run `pip install mediapipe`. Included in `requirements.txt`.
