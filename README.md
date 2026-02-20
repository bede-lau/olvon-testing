# Olvon Physics Test

End-to-end 3D body scanning, garment generation, and physics-based cloth
simulation system.

## Architecture

```
┌──────────────┐     ┌───────────────────────────────────┐     ┌──────────────┐
│    Client    │     │             Server                │     │  Visualizer  │
│              │     │                                   │     │              │
│ capture_     │────>│ anny_inference (body mesh)        │     │ Streamlit +  │
│ wizard.py    │     │       ↓                           │────>│ model-viewer │
│ (webcam +    │     │ garment_generator (t-shirt mesh)  │     │ (3D GLB      │
│  MediaPipe)  │     │       ↓                           │     │  display)    │
│              │     │ physics_sim.py (Blender cloth)    │     │              │
│              │     │       ↓                           │     │              │
│              │     │ sizing_logic (size recommendation)│     │              │
└──────────────┘     └───────────────────────────────────┘     └──────────────┘
```

**Pattern:** Real-first-with-fallback. Each ML stage attempts to load real model
weights, then falls back to parametric mesh generation using trimesh.

## Prerequisites

- **Python 3.10+**
- **Blender 3.6+** (for physics simulation stage)
- **Webcam** (for client capture)
- **CUDA GPU** (optional, for ML inference acceleration)

## Vast.ai GPU Rental

For running with real ML models on a GPU:

1. **Create account** at [vast.ai](https://vast.ai)
2. **Add credits** — $5-10 is enough for testing
3. **Search instances** — click "Search" in the console
   - Filter: GPU Type = RTX 3090 or RTX 4090 or RTX 4090 Ti Super
   - Filter: Disk Space >= 50GB
   - Sort by: price (ascending)
4. **Select a template** — choose "PyTorch 2.0+" from the template dropdown
5. **Rent** — click "Rent" on your chosen instance
6. **Connect** — use the SSH command shown in your instances panel:
   ```bash
   ssh -p <port> root@<host> -L 8080:localhost:8080
   ```
7. **Clone and run** on the instance:
   ```bash
   git clone --recursive <your-repo-url> && cd olvon-testing
   bash setup.sh
   python -m server.main_pipeline --input-dir server/inputs --output-dir server/outputs --height 175 --weight 75
   ```
8. **Download results** — scp the GLB file back to your local machine:
   ```bash
   scp -P <port> root@<host>:~/olvon-testing/server/outputs/final_fitted_avatar.glb .
   ```

## System Dependencies

### Blender Installation

**Windows:**

1. Download from [blender.org/download](https://www.blender.org/download/)
2. Install and add to PATH: `C:\Program Files\Blender Foundation\Blender 3.6\`

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install blender
# Or for latest version:
sudo snap install blender --classic
```

**Verify:**

```bash
blender --version
```

### OpenGL Libraries (Linux headless servers)

```bash
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2
```

## Model Weights

Without model weights, the system uses parametric mesh fallbacks (trimesh
primitives). Install the models below for real ML inference.

### ANNY Body Model (~42 MB, installed via pip)

```bash
pip install anny
```

ANNY is a parametric body model (similar, but unrelated to SMPL-X). All model data ships with
the pip package — no separate weight download needed. The pipeline automatically
uses ANNY when installed, otherwise falls back to trimesh primitives.

**Optional SMPL-X interop weights:**

```bash
mkdir -p server/assets/weights/anny
wget -O server/assets/weights/anny/noncommercial.zip \
  http://download.europe.naverlabs.com/humans/Anny/noncommercial.zip
unzip server/assets/weights/anny/noncommercial.zip -d server/assets/weights/anny/
```

### InstantMesh (~3.24 GB download, ~9.3 GB with auto-deps)

```bash
mkdir -p server/lib/InstantMesh/ckpts
wget -O server/lib/InstantMesh/ckpts/diffusion_pytorch_model.bin \
  "https://huggingface.co/TencentARC/InstantMesh/resolve/main/diffusion_pytorch_model.bin"
wget -O server/lib/InstantMesh/ckpts/instant_mesh_large.ckpt \
  "https://huggingface.co/TencentARC/InstantMesh/resolve/main/instant_mesh_large.ckpt"
```

Requires CUDA GPU and the InstantMesh submodule (`git submodule update --init --recursive`).

### Garment3DGen (no download needed)

Optimization-based — no pretrained weights to download. CLIP ViT-B/32 (~340 MB)
and Fashion-CLIP (~350 MB) auto-download on first run. Pre-install Fashion-CLIP:

```bash
pip install fashion-clip
```

### Auto-Downloaded Models on First Run (~7 GB)

| Model | Size | Triggered by |
|-------|------|-------------|
| Zero123++ v1.2 pipeline | ~5.76 GB | InstantMesh |
| DINO ViT-B/16 | ~350 MB | InstantMesh |
| CLIP ViT-B/32 | ~340 MB | Garment3DGen |
| Fashion-CLIP | ~350 MB | Garment3DGen |

These cache to `~/.cache/huggingface/hub/` and `~/.cache/clip/`.

### Disk Space Requirements

| Setup | Space |
|-------|-------|
| Minimum (fallback mode) | ~1 GB |
| With InstantMesh weights | ~4 GB |
| Full setup (all auto-downloads) | ~15 GB |
| **Recommended** | **20 GB+** |

## Client Setup

```bash
cd client
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run capture wizard (requires webcam + display)
python -m client.capture_wizard
```

The wizard guides you through 12 capture angles with audio feedback. Press 'q'
to quit early.

## Server Setup

```bash
cd server
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run full pipeline
python -m server.main_pipeline --input-dir inputs --output-dir outputs
```

Stages 1-2 (body + garment mesh) run without Blender. Stage 3 (physics) requires
Blender on PATH. Stage 4 (sizing) always runs.

## Visualizer Setup

```bash
pip install -r visualizer/requirements.txt
streamlit run visualizer/app.py
```

Opens a browser with:

- 3D model viewer (drag to rotate 360 degrees, scroll to zoom)
- Size recommendation panel
- File upload if no pipeline output exists

## Troubleshooting

**"mediapipe not installed"** — Run `pip install mediapipe` in your client venv.

**"Blender not found on PATH"** — Install Blender 3.6+ and ensure
`blender --version` works in your terminal.

**"anny package not installed"** — Run `pip install anny` to use the real
parametric body model. Without it, the system falls back to trimesh primitives.

**Webcam not opening** — Check that no other application is using the webcam.
Try `cv2.VideoCapture(1)` if you have multiple cameras.

**Streamlit model-viewer blank** — Ensure your browser allows loading scripts
from unpkg.com (CDN for model-viewer).

**CUDA out of memory** — The fallback meshes don't require GPU. If using real
models, try reducing batch size or using a larger GPU.
