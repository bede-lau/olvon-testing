#!/bin/bash
set -e

echo "=== Olvon VTON - Setup ==="

# ---------------------------------------------------------------------------
# Step 1: Ensure PyTorch >= 2.4.0 and matching torchvision
# ---------------------------------------------------------------------------
echo "[1/5] Checking PyTorch + torchvision compatibility..."

NEED_PYTORCH_INSTALL=false

# Check PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "$TORCH_VERSION" = "none" ]; then
    echo "  PyTorch not installed. Will install."
    NEED_PYTORCH_INSTALL=true
else
    # Extract major.minor as comparable integer (e.g. 2.4 → 204, 2.10 → 210)
    TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1)
    TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2)
    TORCH_NUM=$((TORCH_MAJOR * 100 + TORCH_MINOR))
    if [ "$TORCH_NUM" -lt 204 ]; then
        echo "  PyTorch $TORCH_VERSION is below 2.4.0. Will upgrade."
        NEED_PYTORCH_INSTALL=true
    else
        echo "  PyTorch $TORCH_VERSION OK."
    fi
fi

# Check torchvision compatibility (import test — catches ABI mismatch)
if [ "$NEED_PYTORCH_INSTALL" = false ]; then
    if ! python -c "import torchvision" 2>/dev/null; then
        echo "  torchvision import failed (version mismatch). Will reinstall both."
        NEED_PYTORCH_INSTALL=true
    else
        TV_VERSION=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
        echo "  torchvision $TV_VERSION OK."
    fi
fi

if [ "$NEED_PYTORCH_INSTALL" = true ]; then
    echo "  Uninstalling old PyTorch packages..."
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    echo "  Installing PyTorch >= 2.4.0 + torchvision (cu126)..."
    pip install "torch>=2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
    echo "  Verifying install..."
    python -c "import torch, torchvision; print(f'  PyTorch {torch.__version__}, torchvision {torchvision.__version__}, CUDA: {torch.cuda.is_available()}')"
fi

# ---------------------------------------------------------------------------
# Step 2: Install base requirements
# ---------------------------------------------------------------------------
echo "[2/5] Installing base requirements..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Step 3: Install FASHN VTON
# ---------------------------------------------------------------------------
echo "[3/5] Installing FASHN VTON..."
if [ -d "server/lib/fashn-vton" ]; then
    echo "  fashn-vton directory already exists."
else
    git clone https://github.com/fashn-AI/fashn-vton-1.5.git server/lib/fashn-vton
fi
pip install -e server/lib/fashn-vton || echo "  fashn-vton install failed. VTON fallback will be used."

# ---------------------------------------------------------------------------
# Step 4: Download FASHN VTON weights
# ---------------------------------------------------------------------------
echo "[4/5] Downloading FASHN VTON weights..."
WEIGHTS_DIR="server/lib/fashn-vton/weights"
if [ -d "$WEIGHTS_DIR" ] && [ "$(ls -A $WEIGHTS_DIR 2>/dev/null)" ]; then
    echo "  Weights already present. Skipping."
else
    python server/lib/fashn-vton/scripts/download_weights.py --weights-dir "$WEIGHTS_DIR" \
        || echo "  Weight download failed. VTON fallback will be used."
fi

# ---------------------------------------------------------------------------
# Step 5: Create directories and check prerequisites
# ---------------------------------------------------------------------------
echo "[5/5] Creating directories and checking prerequisites..."
mkdir -p server/inputs/garment_photos
mkdir -p server/outputs

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -1)
    echo "  Found: $FFMPEG_VERSION"
else
    echo "  FFmpeg not found on PATH. Feed video generation will be skipped."
    echo "  Install FFmpeg: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  streamlit run visualizer/app.py          # Launch UI (accessible on port 8501)"
echo "  python -m pytest tests/ -v               # Run tests"
echo "  python -m server.main_pipeline --front-photo test.jpg --height 175 --weight 75 --garment-photo garment.jpg --category tops"
