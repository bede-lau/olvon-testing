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
| **GPU VRAM** | 8GB+ (RTX 3090 = 24GB, RTX 4090 = 24GB) | Choose a GPU model in the instance list |
| **Container Size** | **30GB recommended** | The "Container Size" field on the rent/create screen — increase it before confirming |

**Why 30GB?** The base PyTorch image uses ~13GB, PyTorch 2.4 upgrade adds ~2.5GB, FASHN
VTON weights ~2GB, other packages ~500MB — totalling ~18GB. 30GB gives ~12GB headroom for
output images and temp files during inference without paying for unnecessary space.

- Template: any PyTorch image

---

### 2. Connect from Your Local Machine

Two terminals are required — one for the SSH session and one for the port tunnel.

**Terminal 1 — main SSH session:**

```bash
# [local] — terminal 1
ssh -A -p <port> root@<host>
```

| Flag | Purpose |
|------|---------|
| `-A` | Forwards your local SSH agent so `git clone` works on the instance |
| `-p <port>` | The SSH port shown in your vast.ai instance dashboard |
| `root@<host>` | The IP address shown in your vast.ai instance dashboard |

**Terminal 2 — port tunnel (open this after starting Streamlit in terminal 1):**

```bash
# [local] — terminal 2
ssh -N -L 8501:localhost:8501 -p <port> root@<host>
```

Leave terminal 2 running the entire time you use the app. `http://localhost:8501` in your
browser will only work while this tunnel is active.

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

### 5. Upgrade PyTorch to >= 2.4.0

> **[vast.ai]** — FASHN VTON v1.5 requires PyTorch >= 2.4.0. Default vast.ai images
> often ship 2.2.x which will cause the weight download to fail.

Check the installed version first:

```bash
# [vast.ai]
python -c "import torch; print(torch.__version__)"
```

If it is below 2.4.0:

```bash
# [vast.ai]
pip uninstall torch torchvision torchaudio -y
pip install "torch>=2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
```

> **`--no-cache-dir` is required** — without it pip fills the 16GB overlay with download
> cache and fails mid-install with `[Errno 28] No space left on device`.

> **CUDA version:** Use the `cu126` index even on CUDA 13.x — PyTorch CUDA wheels are
> backward-compatible with newer drivers.

Verify GPU is detected:

```bash
# [vast.ai]
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x True
```

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

> **[vast.ai]**

```bash
# [vast.ai]
streamlit run visualizer/app.py
```

Then open a second local terminal and start the port tunnel (if not already running):

```bash
# [local] — terminal 2
ssh -N -L 8501:localhost:8501 -p <port> root@<host>
```

Then open **`http://localhost:8501`** in your browser. The tunnel must stay open the entire
time you use the app — do not close terminal 2.

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

### Person Enhancement Pipeline

`server/core/person_enhancer.py` runs two models sequentially before FASHN VTON inference:

1. **Real-ESRGAN x4plus** — 4× upscale of the webcam photo (identity-preserving, no face hallucination)
2. **BiRefNet-portrait** — background removal to a clean white canvas

VRAM usage is sequential (models unloaded between steps):

| Step | Model | VRAM peak | Duration |
|------|-------|-----------|---------|
| Real-ESRGAN 4× | realesrgan-x4plus | ~2 GB | ~2 s |
| BiRefNet cutout | birefnet-portrait | ~1.5 GB | ~1 s |
| FASHN VTON | diffusion model | ~8 GB | ~30–60 s |
| **Peak (FASHN VTON)** | | **~8 GB** | |

Both stages fall back gracefully if the libraries are unavailable.

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
Your SSH agent is not forwarded. On your local machine: `ssh-add ~/.ssh/id_ed25519`, then
reconnect with `ssh -A ...`.

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
Ensure `fashn-vton` is installed (`pip install -e server/lib/fashn-vton`) and weights
are present in `server/lib/fashn-vton/weights/`. Requires CUDA GPU.

**"FFmpeg not found on PATH"**
Run `apt install ffmpeg` on the vast.ai instance. Feed video generation is skipped without it.

**Webcam not opening in browser**
Browser webcam requires HTTPS or localhost. Ensure the SSH tunnel (`-L 8501:localhost:8501`)
is active and you are accessing `http://localhost:8501`, not the external IP.

**Back view not detected**
The system uses face-absence detection with a 5-second countdown for back view. Use
"Skip validation" if auto-detection fails.

**CUDA out of memory**
FASHN VTON needs ~8GB VRAM. Use a GPU with sufficient memory (RTX 3090+).

**Nested duplicate directories (`olvon-testing/olvon-testing/...`)**
You ran `git clone ... && cd olvon-testing` while already inside the repo. Navigate back:
`cd ~` and work from `~/olvon-testing`.
