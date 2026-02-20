# Olvon Physics Test

End-to-end 3D body scanning, garment generation, and physics-based cloth
simulation system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VAST.AI INSTANCE                               │
│                                                                         │
│  ┌─────────────────────────┐     ┌───────────────────────────────────┐  │
│  │     Streamlit Wizard    │     │             Server                │  │
│  │    (Browser-based UI)   │     │         (ML Pipeline)             │  │
│  │                         │     │                                   │  │
│  │ Step 1: Body Input      │     │ anny_inference (body mesh)        │  │
│  │ Step 2: Webcam Scan     │────>│       ↓                           │  │
│  │   (5 angles + orient.)  │     │ garment_generator (t-shirt mesh)  │  │
│  │ Step 3: Garment Input   │     │       ↓                           │  │
│  │ Step 4: Generate        │     │ physics_sim.py (Blender cloth)    │  │
│  │ Step 5: 3D Results      │<────│       ↓                           │  │
│  │                         │     │ sizing_logic (size recommendation)│  │
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

**Pattern:** Real-first-with-fallback. Each ML stage attempts to load real model
weights, then falls back to parametric mesh generation using trimesh.

## Prerequisites

- **Python 3.10+**
- **Vast.ai account** (for GPU instance to run everything)
- **Browser with webcam** (for body scan capture via Streamlit)

## Full Workflow (Unified Browser-Based)

Everything runs on a **vast.ai GPU instance**. You access the Streamlit wizard
from your browser — no local setup required beyond SSH.

---

### Part 1: Set Up the Vast.ai Instance

**1.1. Rent an instance:**

1. Create account at [vast.ai](https://vast.ai)
2. Add credits — $5-10 is enough for testing
3. Search instances:
   - Filter: GPU Type = RTX 3090 or RTX 4090 or RTX 4090 Ti Super
   - Filter: Disk Space >= 50GB
   - Sort by: price (ascending)
4. Select template: "PyTorch 2.0+"
5. Click "Rent"

**1.2. Connect from your LOCAL terminal:**

```bash
ssh -A -p <port> root@<host> -L 8501:localhost:8501
```

> The `-L 8501:localhost:8501` flag forwards the Streamlit port to your local
> machine. The `-A` flag forwards your SSH key for git cloning.

**1.3. Install Blender 3.6+ and system libraries (on the instance):**

```bash
apt update && apt install -y libsm6 libice6 libgl1-mesa-glx libegl1-mesa \
  libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxext6 libxi6 \
  libxkbcommon0 libxxf86vm1
ldconfig

cd /opt
wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar xf blender-3.6.0-linux-x64.tar.xz
ln -sf /opt/blender-3.6.0-linux-x64/blender /usr/local/bin/blender
rm -f /usr/bin/blender
hash -r
blender --version  # Should show Blender 3.6.0
```

**1.4. Clone the repo and run setup (on the instance):**

```bash
cd ~
git clone --recursive git@github.com:bede-lau/olvon-testing.git && cd olvon-testing
bash setup.sh
```

---

### Part 2: Use the Streamlit Wizard (Browser)

**2.1. Start the Streamlit app (on the instance):**

```bash
streamlit run visualizer/app.py
```

**2.2. Open `http://localhost:8501` in your browser.** The wizard guides you through 5 steps:

1. **Body Input** — Enter height (cm) and optional weight (kg)
2. **Body Scan** — Take 5 photos using your browser's webcam (front, right, back, left, elevated). Each photo is validated for correct orientation. Use "Skip validation" or "Accept anyway" if auto-detection is unreliable (especially for back view).
3. **Garment Input** — Upload a garment photo and/or enter garment measurements
4. **Generate** — Review inputs and run the full pipeline (body mesh, garment, physics sim, sizing)
5. **Results** — View the 3D fitted avatar (model-viewer), sizing recommendation, and pipeline log

---

### Part 3: Clean Up

**Destroy the vast.ai instance** in your vast.ai console to stop billing.

---

### Alternative: Standalone Capture Wizard (LOCAL)

If you prefer to capture locally with the OpenCV-based wizard (with audio feedback
and live skeleton overlay):

```bash
cd <project-root>
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r client/requirements.txt
python -m client.capture_wizard
```

The wizard captures 5 angles (front, right, back, left, elevated) with orientation
detection. Captures are saved to `client/output_captures/`.
Upload them to the instance with `scp` and run the pipeline via CLI.

---

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

## Troubleshooting

**"mediapipe not installed"** — Run `pip install mediapipe`. Included in both
`client/requirements.txt` and root `requirements.txt`.

**"Blender not found on PATH"** — Install Blender 3.6+ and ensure
`blender --version` works in your terminal.

**"anny package not installed"** — Run `pip install anny` to use the real
parametric body model. Without it, the system falls back to trimesh primitives.

**Webcam not opening in browser** — Browser webcam requires HTTPS. Vast.ai
typically provides HTTPS tunnels, or use SSH port forwarding (`-L 8501:localhost:8501`).
For the standalone capture wizard, check that no other application is using the webcam.

**Back view not detected** — MediaPipe may not detect a pose when facing away.
Use "Skip validation" or "Accept anyway" in the Streamlit wizard.

**Streamlit model-viewer blank** — Ensure your browser allows loading scripts
from unpkg.com (CDN for model-viewer).

**CUDA out of memory** — The fallback meshes don't require GPU. If using real
models, try reducing batch size or using a larger GPU.

**torch-sparse/torch-scatter stuck compiling** — These compile CUDA extensions
from source if prebuilt wheels aren't found. The setup script auto-detects your
PyTorch/CUDA versions and uses prebuilt wheels from `data.pyg.org`. If it still
compiles from source, install manually:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>+cu<CUDA>.html
```

**nvdiffrast build fails** — Must be installed with `--no-build-isolation` so it
can find PyTorch. The setup script handles this automatically, but if you need to
install manually:
```bash
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/
```
