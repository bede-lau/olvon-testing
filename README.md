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
- **CUDA GPU** (24GB VRAM recommended; 12GB works with reduced quality — see VRAM table below)
- **PyTorch >= 2.4.0** (required by fashn-vton-1.5)
- **FFmpeg** for feed video generation
- **Vast.ai account** (for GPU instance)
- **Browser with webcam** (for body scan capture)

---

## Quick Start (Vast.ai)

> **Convention used in this guide:**
> - `[local]` — run this in a terminal on your own machine
> - `[vast.ai]` — run this in the SSH session connected to the instance

---

### 0. One-time: Set Up SSH Key

> **[local]** — do this once on your machine. Skip if `~/.ssh/id_ed25519` already exists.

GitHub no longer supports password authentication for git. You need an SSH key
forwarded via agent so the vast.ai instance can clone without storing any keys on it.

**Generate the key:**

```bash
# [local]
ssh-keygen -t ed25519 -C "your@email.com"
# Press Enter for all prompts
```

**Add the public key to GitHub:**

```bash
# [local]
cat ~/.ssh/id_ed25519.pub
# Copy the entire output line
```

Go to **GitHub → Settings → SSH and GPG keys → New SSH key**, paste, click **Add SSH key**.

**Load the key into your local SSH agent:**

```bash
# [local]
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**Verify GitHub auth works:**

```bash
# [local]
ssh -T git@github.com
# Expected: Hi bede-lau! You've successfully authenticated...
```

> The agent must be running and the key loaded **every time** you open a new local terminal
> before SSHing into vast.ai. On macOS the agent persists; on Windows/WSL you may need to
> repeat `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519` each session.

---

### 1. Rent a Vast.ai Instance

Two separate resources to configure — GPU VRAM and disk storage are independent:

| Resource | Requirement | Where to set it |
|----------|------------|-----------------|
| **GPU VRAM** | **24GB recommended** (RTX 3090, RTX 4090, A5000) | Choose a GPU model in the instance list |
| **Container Size** | **30GB recommended** | The "Container Size" field on the rent/create screen — increase it before confirming |

**Why 24GB VRAM?** The pipeline runs multiple ML models sequentially. Each stage unloads
before the next, but the peak during person regeneration reaches ~10 GB of static weights
plus ~2-4 GB of intermediate tensors and CUDA overhead:

| Stage | What runs | VRAM peak |
|-------|-----------|-----------|
| 0a. Face embedding | InsightFace buffalo_l | ~1 GB |
| 0b. Person regeneration | SD1.5 + ControlNet + IP-Adapter (fp16) | **~10 GB** |
| 1. Virtual try-on | FASHN VTON v1.5 | **~8 GB** |
| *(fallback)* Upscale | Real-ESRGAN x4plus | ~2 GB |
| *(fallback)* BG removal | BiRefNet-portrait | ~1.5 GB |

A 12 GB card (RTX 3060/4070) sits right at the edge during Stage 0b once you account for
PyTorch's CUDA context (~0.5-1 GB), memory fragmentation, and the CPU→GPU weight transfer
overlap during model loading. 24 GB eliminates OOM risk across all stages.

> **On a 12 GB GPU:** person regeneration is skipped automatically and the pipeline falls
> back to the enhancer path (Real-ESRGAN + BiRefNet, ~3.5 GB peak) then FASHN VTON (~8 GB
> peak). You get background removal but keep the original pose instead of a neutral one.

**Why 30GB disk?** The base PyTorch image uses ~13GB, PyTorch 2.4 upgrade adds ~2.5GB, FASHN
VTON weights ~2GB, other packages ~500MB — totalling ~18GB. 30GB gives ~12GB headroom for
output images and temp files during inference without paying for unnecessary space.

- Template: any PyTorch image

---

### 2. Connect from Your Local Machine

> **CRITICAL: Do NOT run Streamlit locally.** If you run `streamlit run visualizer/app.py`
> on your local Windows/Mac machine (even accidentally), the browser will connect to a local
> Streamlit process that has no GPU, no `fashn_vton`, and no model weights. The pipeline will
> silently fail with "FASHN VTON pipeline not available". Streamlit must **only** run on the
> **vast.ai instance** and be accessed via the SSH port tunnel described below.

**SSH session with port tunnel (single command):**

```bash
# [local]
ssh -p <port> root@<host> -L 8501:localhost:8501 -o ServerAliveInterval=30 -o ServerAliveCountMax=5
```

| Flag | Purpose |
|------|---------|
| `-p <port>` | The SSH port shown in your vast.ai instance dashboard |
| `root@<host>` | The IP address shown in your vast.ai instance dashboard |
| `-L 8501:localhost:8501` | Forwards the remote Streamlit port to your local browser |
| `-o ServerAliveInterval=30` | Sends keepalive pings every 30s to prevent "Broken pipe" disconnections |

Add `-A` if you need SSH agent forwarding for `git clone`.

**Alternative — separate tunnel terminal (if you prefer):**

```bash
# [local] — terminal 1 (SSH session)
ssh -p <port> root@<host> -o ServerAliveInterval=30

# [local] — terminal 2 (port tunnel, keep running)
ssh -N -L 8501:localhost:8501 -p <port> root@<host> -o ServerAliveInterval=30
```

Leave the tunnel running the entire time you use the app. `http://localhost:8501` in your
browser will only work while this tunnel is active.

> **Verify the connection:** After SSHing into the instance, run `ssh-add -l`.
> If it returns "The agent has no identities" or an error, exit and run `ssh-add ~/.ssh/id_ed25519`
> on your local machine before reconnecting with `-A`.

> **Common mistake:** Do not run either SSH command from inside the vast.ai terminal — that
> SSHes the instance into itself and hangs. Always run them from your local machine.

---

### 3. Install FFmpeg

> **[vast.ai]**

```bash
# [vast.ai]
apt update && apt install -y ffmpeg
```

---

### 4. Clone and Set Up

> **[vast.ai]**

```bash
# [vast.ai]
cd ~
git clone git@github.com:bede-lau/olvon-testing.git
cd olvon-testing
bash setup.sh
```

> **Do not** chain `git clone ... && cd olvon-testing` if you are already inside the
> repo directory — you will end up with nested duplicate directories
> (`~/olvon-testing/olvon-testing/...`). Run them as separate commands.

---

### 5. Verify PyTorch + torchvision

> **[vast.ai]** — `setup.sh` automatically checks PyTorch >= 2.4.0 and torchvision
> compatibility, upgrading both if needed. This step is just to verify.

```bash
# [vast.ai]
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# Expected: 2.x.x 0.x.x True
```

If either version looks wrong or CUDA shows `False`, reinstall manually:

```bash
# [vast.ai]
pip uninstall torch torchvision torchaudio -y
pip install "torch>=2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
```

> **`--no-cache-dir` is required** — without it pip fills the 16GB overlay with download
> cache and fails mid-install with `[Errno 28] No space left on device`.

> **CUDA version:** Use the `cu126` index even on CUDA 13.x — PyTorch CUDA wheels are
> backward-compatible with newer drivers.

> **PyTorch + torchvision must be installed together.** Installing only one causes an ABI
> mismatch (`operator torchvision::nms does not exist`). Always uninstall both before
> reinstalling.

---

### 6. Download FASHN VTON Weights

> **[vast.ai]** — `setup.sh` attempts this automatically but it requires PyTorch >= 2.4.0
> first. If it failed during setup, run manually:

```bash
# [vast.ai]
python server/lib/fashn-vton/scripts/download_weights.py --weights-dir server/lib/fashn-vton/weights
```

This downloads ~2GB from HuggingFace. Check weights are present after:

```bash
# [vast.ai]
ls server/lib/fashn-vton/weights/
```

---

### 7. Run the Wizard

> **[vast.ai]** — Run Streamlit on the vast.ai instance, **never** on your local machine.

```bash
# [vast.ai]
streamlit run visualizer/app.py
```

Then open **`http://localhost:8501`** in your browser. This works because the `-L 8501:localhost:8501`
flag in your SSH command (step 2) tunnels the remote port to your local machine.

> **How to verify you're connected to vast.ai (not local):** If the pipeline shows
> "FASHN VTON pipeline not available", you are likely running Streamlit locally. Kill any
> local Streamlit process (`pkill -f streamlit` on your local machine), ensure the SSH tunnel
> is active, and only run `streamlit run` on the vast.ai instance.

The wizard has 4 steps:

1. **Body Input** — Enter height (cm) and optional weight (kg)
2. **Body Scan** — 3 webcam captures: front, side, back (auto-capture with pose detection)
3. **Garment Input** — Upload garment photos, select category, run pipeline
4. **Results** — Two tabs:
   - **Feed Video** — Short-form try-on slideshow video
   - **Virtual Dressing Room** — Flip between front/back views, click garments to try on
---

### 8. Clean Up

Destroy the vast.ai instance to stop billing.

---

## CLI Usage

> **[vast.ai]**

```bash
# [vast.ai] — single garment
python -m server.main_pipeline \
  --front-photo person_front.jpg \
  --back-photo person_back.jpg \
  --garment-photo garment.jpg \
  --category tops \
  --height 175 --weight 75

# [vast.ai] — multiple garments
python -m server.main_pipeline \
  --front-photo person_front.jpg \
  --garment-photo shirt1.jpg \
  --garment-photo shirt2.jpg \
  --category tops
```

---

## Model Weights

| Model | Size | Location / Install | License |
|-------|------|--------------------|---------|
| FASHN VTON v1.5 | ~2 GB | `server/lib/fashn-vton/weights/` — downloaded via `setup.sh` | Apache 2.0 |
| MediaPipe Pose | ~26 MB | `client/assets/pose_landmarker_heavy.task` — auto-downloaded on first use | Apache 2.0 |
| RealESRGAN_x4plus | ~67 MB | `~/.cache/realesrgan/` — downloaded on first use by `person_enhancer.py` | BSD 3-Clause |
| BiRefNet-portrait | ~168 MB | `~/.u2net/` — downloaded on first use by `rembg` | Apache 2.0 |
| SD1.5 base | ~4 GB | `~/.cache/huggingface/` — downloaded on first use by `person_regenerator.py` | CreativeML OpenRAIL-M |
| ControlNet OpenPose | ~1.5 GB | `~/.cache/huggingface/` — downloaded on first use | Apache 2.0 |
| IP-Adapter FaceID Plus v2 | ~1.5 GB | `~/.cache/huggingface/` — downloaded on first use | Apache 2.0 |
| InsightFace buffalo_l | ~300 MB | `~/.insightface/` — downloaded on first use | MIT |

### Person Preparation Pipeline

Before FASHN VTON, the pipeline prepares a clean person image using a three-tier fallback:

1. **Person Regenerator** (`server/core/person_regenerator.py`) — SD1.5 + ControlNet OpenPose + IP-Adapter FaceID Plus v2. Generates a new image of the person in a neutral standing pose (arms at sides) on white background, preserving facial identity. Best VTON input quality.
2. **Person Enhancer** (`server/core/person_enhancer.py`) — Real-ESRGAN 4× upscale + BiRefNet background removal to white canvas. Original pose preserved. Used when regeneration deps are missing or fail.
3. **Raw webcam photo** — last resort if both above fail.

All models are unloaded between stages via `del` + `torch.cuda.empty_cache()`.

---

## Third-Party Licenses

### FASHN VTON v1.5

Licensed under Apache 2.0. See [fashn-AI/fashn-vton-1.5](https://github.com/fashn-AI/fashn-vton-1.5).

### Real-ESRGAN

Used in `server/core/person_enhancer.py` for 4× person photo upscaling.
Source: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

**BSD 3-Clause License**

```
Copyright (c) 2021, xinntao
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

> **For binary/app distribution:** this copyright notice and the three conditions
> above must be reproduced in the app's documentation or About section.

### BiRefNet-portrait

Used in `server/core/person_enhancer.py` via `rembg` for background removal.
Source: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — licensed under Apache 2.0.

---

## Running Tests

> **[vast.ai]** — no GPU required

```bash
# [vast.ai]
python -m pytest tests/test_pipeline.py -v
```

Tests cover body measurements, sizing logic, try-on worker fallback, and feed generator
command building.

---

## Troubleshooting

**`Permission denied (publickey)` when cloning**
This happens when the server cannot access your local SSH key. To fix:
1. **Local machine**: Ensure agent is running and key is loaded:
   `eval "$(ssh-agent -s)"` followed by `ssh-add ~/.ssh/id_ed25519`. 
   Verify with `ssh-add -l`.
2. **Reconnect**: Connect to the instance using the `-A` flag:
   `ssh -A -p <port> root@<host>`.
3. **Instance**: Verify the key is forwarded by running `ssh-add -l` on the server.
   It should list your key's fingerprint. Then try cloning again.

**`localhost refused to connect` in browser**
The port tunnel is not running. Open a second local terminal and run:
```bash
# [local] — terminal 2
ssh -N -L 8501:localhost:8501 -p <port> root@<host>
```
Leave it running. The `-L` flag in the main SSH connection alone is not reliable — always
use this dedicated tunnel terminal.

**SSH command hangs after running it inside the instance**
You ran the `ssh` command from the vast.ai terminal instead of your local machine. Press
`Ctrl+C` to cancel, then run it from a local terminal.

**`No space left on device` during pip install**
The vast.ai container disk (overlay filesystem) defaults to ~16GB and is the only storage
accessible inside the container — the host nvme is NOT mounted in. Free space first:
```bash
# [vast.ai]
pip uninstall torch torchvision torchaudio -y
conda clean --all -y
pip cache purge
```
Then reinstall with `--no-cache-dir`. Rent with 30GB+ container size to avoid this entirely.

**Weight download fails with `PyTorch >= 2.4 required`**
Upgrade PyTorch before running the download script. See step 5.

**`Invalid requirement: ''` or `command not found` when running pip install**
Do not use backslash line continuations (`\`) when pasting multi-line commands into the
terminal — paste as a single line instead.

**"FASHN VTON pipeline not available"**
**Most common cause:** Streamlit is running on your **local machine** instead of the vast.ai
instance. Kill any local Streamlit (`pkill -f streamlit` locally), verify the SSH tunnel
is active (`-L 8501:localhost:8501`), and run `streamlit run visualizer/app.py` only on
vast.ai. If the issue persists, check that `fashn-vton` is installed on the instance
(`pip install -e server/lib/fashn-vton`) and weights are present in
`server/lib/fashn-vton/weights/`. Requires CUDA GPU.

**"FFmpeg not found on PATH"**
Run `apt install ffmpeg` on the vast.ai instance. Feed video generation is skipped without it.

**Webcam not opening in browser**
Browser webcam requires HTTPS or localhost. Ensure the SSH tunnel (`-L 8501:localhost:8501`)
is active and you are accessing `http://localhost:8501`, not the external IP.

**Back view not detected**
The system uses face-absence detection with a 5-second countdown for back view. Use
"Skip validation" if auto-detection fails.

**CUDA out of memory**
Person regeneration peaks at ~10 GB and FASHN VTON needs ~8 GB. Use a 24 GB GPU (RTX 3090+)
for the full pipeline. On 12 GB GPUs, regeneration is skipped automatically and the enhancer
fallback (Real-ESRGAN + BiRefNet) is used instead.

**Nested duplicate directories (`olvon-testing/olvon-testing/...`)**
You ran `git clone ... && cd olvon-testing` while already inside the repo. Navigate back:
`cd ~` and work from `~/olvon-testing`.
