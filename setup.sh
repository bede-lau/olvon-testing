#!/bin/bash
set -e

echo "=== Olvon VTON - Setup ==="

# Install base requirements
echo "[1/4] Installing base requirements..."
pip install -r requirements.txt

# Install FASHN VTON
echo "[2/4] Installing FASHN VTON..."
if [ -d "server/lib/fashn-vton" ]; then
    echo "  fashn-vton directory already exists."
else
    git clone https://github.com/fashn-AI/fashn-vton-1.5.git server/lib/fashn-vton
fi
pip install -e server/lib/fashn-vton || echo "  fashn-vton install failed. VTON fallback will be used."

# Download FASHN VTON weights
echo "[3/4] Downloading FASHN VTON weights..."
WEIGHTS_DIR="server/lib/fashn-vton/weights"
if [ -d "$WEIGHTS_DIR" ] && [ "$(ls -A $WEIGHTS_DIR 2>/dev/null)" ]; then
    echo "  Weights already present. Skipping."
else
    python -c "
from fashn_vton import TryOnPipeline
TryOnPipeline.download_weights('$WEIGHTS_DIR')
print('  Weights downloaded.')
" 2>/dev/null || echo "  Weight download failed. VTON fallback will be used."
fi

# Create directories and check prerequisites
echo "[4/4] Creating directories and checking prerequisites..."
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
