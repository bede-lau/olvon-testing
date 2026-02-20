#!/bin/bash
set -e

echo "=== Olvon Physics Test - Setup ==="

# Install base requirements
echo "[1/7] Installing base requirements..."
pip install -r requirements.txt

# Install Garment3DGen dependencies
echo "[2/7] Installing Garment3DGen dependencies..."
if [ -d "server/lib/Garment3DGen" ] && [ -f "server/lib/Garment3DGen/requirements.txt" ]; then
    pip install -r server/lib/Garment3DGen/requirements.txt
else
    echo "  Garment3DGen not found (submodule not initialized?). Skipping."
    echo "  Run: git submodule update --init --recursive"
fi

# Install InstantMesh dependencies
echo "[3/7] Installing InstantMesh dependencies..."
if [ -d "server/lib/InstantMesh" ] && [ -f "server/lib/InstantMesh/requirements.txt" ]; then
    pip install -r server/lib/InstantMesh/requirements.txt
else
    echo "  InstantMesh not found (submodule not initialized?). Skipping."
fi

# Install GPU-dependent packages
echo "[4/7] Installing PyTorch3D, nvdiffrast, CLIP..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null && {
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" || echo "  PyTorch3D install failed (may need CUDA toolkit). Skipping."
    pip install "git+https://github.com/NVlabs/nvdiffrast.git" || echo "  nvdiffrast install failed. Skipping."
    pip install "git+https://github.com/openai/CLIP.git" || echo "  CLIP install failed. Skipping."
} || echo "  No CUDA available. Skipping GPU-dependent packages (fallback mode will be used)."

# Check Blender
echo "[5/7] Checking Blender..."
if command -v blender &> /dev/null; then
    BLENDER_VERSION=$(blender --version 2>/dev/null | head -1)
    echo "  Found: $BLENDER_VERSION"
else
    echo "  Blender not found on PATH. Physics simulation will be skipped."
    echo "  Install Blender 3.6+ for full pipeline support."
fi

# Create directories
echo "[6/7] Creating directories..."
mkdir -p server/inputs/garment_photos
mkdir -p server/outputs
mkdir -p server/assets/weights

# Download model weights
echo "[7/7] Downloading model weights..."
pip install anny || echo "  ANNY install failed. Fallback mode will be used."

if [ -d "server/lib/InstantMesh" ]; then
    mkdir -p server/lib/InstantMesh/ckpts
    if [ ! -f "server/lib/InstantMesh/ckpts/instant_mesh_large.ckpt" ]; then
        echo "  Downloading InstantMesh checkpoints (~3.24 GB)..."
        wget -q --show-progress -O server/lib/InstantMesh/ckpts/diffusion_pytorch_model.bin \
          "https://huggingface.co/TencentARC/InstantMesh/resolve/main/diffusion_pytorch_model.bin"
        wget -q --show-progress -O server/lib/InstantMesh/ckpts/instant_mesh_large.ckpt \
          "https://huggingface.co/TencentARC/InstantMesh/resolve/main/instant_mesh_large.ckpt"
    else
        echo "  InstantMesh checkpoints already present. Skipping."
    fi
else
    echo "  InstantMesh submodule not found. Skipping checkpoint download."
fi

pip install fashion-clip || echo "  Fashion-CLIP install failed. Skipping."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  streamlit run visualizer/app.py          # Launch UI (accessible on port 8501)"
echo "  python -m pytest tests/ -v               # Run tests"
echo "  python -m server.main_pipeline --height 175 --weight 75  # Run pipeline"
