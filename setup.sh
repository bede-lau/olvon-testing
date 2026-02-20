#!/bin/bash
set -e

echo "=== Olvon Physics Test - Setup ==="

# Detect PyTorch and CUDA versions for prebuilt wheels
echo "[0/7] Detecting PyTorch and CUDA versions..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.',''))" 2>/dev/null || echo "")

if [ -n "$TORCH_VERSION" ] && [ -n "$CUDA_VERSION" ]; then
    PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"
    echo "  PyTorch ${TORCH_VERSION} + CUDA ${CUDA_VERSION}"
    echo "  PyG wheel URL: ${PYG_WHEEL_URL}"
else
    PYG_WHEEL_URL=""
    echo "  Could not detect PyTorch/CUDA. torch-sparse/torch-scatter will compile from source (slow)."
fi

# Install base requirements
echo "[1/7] Installing base requirements..."
pip install -r requirements.txt

# Install torch-sparse and torch-scatter from prebuilt wheels (avoids 1hr+ compile)
echo "[1.5/7] Installing torch-sparse and torch-scatter..."
if [ -n "$PYG_WHEEL_URL" ]; then
    pip install torch-scatter torch-sparse -f "$PYG_WHEEL_URL" \
        || echo "  Prebuilt wheels not found for this PyTorch/CUDA combo. Falling back to source build (this may take a while)..."
else
    pip install torch-scatter torch-sparse \
        || echo "  torch-scatter/torch-sparse install failed. Skipping."
fi

# Install Garment3DGen dependencies
echo "[2/7] Installing Garment3DGen dependencies..."
if [ -d "server/lib/Garment3DGen" ] && [ -f "server/lib/Garment3DGen/requirements.txt" ]; then
    pip install -r server/lib/Garment3DGen/requirements.txt
else
    echo "  Garment3DGen not found (submodule not initialized?). Skipping."
    echo "  Run: git submodule update --init --recursive"
fi

# Install GPU-dependent packages (nvdiffrast BEFORE InstantMesh so it's already available)
echo "[3/7] Installing PyTorch3D, nvdiffrast, CLIP..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null && {
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" || echo "  PyTorch3D install failed (may need CUDA toolkit). Skipping."
    pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git" || echo "  nvdiffrast install failed. Skipping."
    pip install "git+https://github.com/openai/CLIP.git" || echo "  CLIP install failed. Skipping."
} || echo "  No CUDA available. Skipping GPU-dependent packages (fallback mode will be used)."

# Install InstantMesh dependencies (nvdiffrast already installed above, skip it here)
echo "[4/7] Installing InstantMesh dependencies..."
if [ -d "server/lib/InstantMesh" ] && [ -f "server/lib/InstantMesh/requirements.txt" ]; then
    # Filter out nvdiffrast from requirements since it's already installed with --no-build-isolation
    grep -v "nvdiffrast" server/lib/InstantMesh/requirements.txt > /tmp/instantmesh_reqs_filtered.txt || true
    pip install -r /tmp/instantmesh_reqs_filtered.txt || echo "  Some InstantMesh deps failed. Continuing."
    rm -f /tmp/instantmesh_reqs_filtered.txt
else
    echo "  InstantMesh not found (submodule not initialized?). Skipping."
fi

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
