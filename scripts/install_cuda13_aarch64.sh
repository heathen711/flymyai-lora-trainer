#!/bin/bash
# install_cuda13_aarch64.sh
# CUDA 13.0 installation for ARM64 SBSA platforms (GH200, DGX SPARK, etc.)

set -e

echo "============================================"
echo "CUDA 13.0 Installation for aarch64-sbsa"
echo "============================================"
echo ""

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: This script is for aarch64 (ARM64) systems only."
    echo "Current architecture: $ARCH"
    echo "For x86_64, use: ./scripts/install_cuda13.sh"
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. Ensure CUDA 13.0 toolkit is installed."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Prerequisites
echo "Step 0: Installing build prerequisites..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build git curl
elif command -v yum &> /dev/null; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y cmake ninja-build git curl
fi

# Install Rust (required for tokenizers)
if ! command -v rustc &> /dev/null; then
    echo ""
    echo "Step 0.5: Installing Rust compiler (required for tokenizers)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(rustc --version)"
fi

# Step 1: PyTorch from CUDA 13.0 sbsa index
echo ""
echo "Step 1: Installing PyTorch with CUDA 13.0 sbsa support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"

# Step 2: Pure Python packages (no compilation needed)
echo ""
echo "Step 2: Installing pure Python packages..."
pip install \
    accelerate \
    peft \
    omegaconf \
    einops \
    huggingface-hub \
    datasets \
    loguru \
    matplotlib \
    sentencepiece

# Step 3: Transformers (tokenizers will compile with Rust)
echo ""
echo "Step 3: Installing transformers (tokenizers will compile)..."
pip install transformers

# Step 4: Diffusers from specific commit
echo ""
echo "Step 4: Installing diffusers..."
pip install git+https://github.com/huggingface/diffusers@7a2b78bf0f788d311cc96b61e660a8e13e3b1e63

# Step 5: bitsandbytes (has sbsa wheels)
echo ""
echo "Step 5: Installing bitsandbytes..."
pip install bitsandbytes || {
    echo "Pre-built wheel not found, attempting to build from source..."
    git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bitsandbytes
    cd /tmp/bitsandbytes
    cmake -DCOMPUTE_BACKEND=cuda -S .
    make -j$(nproc)
    pip install -e .
    cd -
}

# Step 6: DeepSpeed (must build from source for aarch64)
echo ""
echo "Step 6: Installing DeepSpeed from source..."
echo "This may take several minutes..."
DS_BUILD_OPS=1 pip install deepspeed --no-cache-dir || {
    echo "WARNING: DeepSpeed build failed. Trying minimal install..."
    pip install deepspeed --no-build-isolation || {
        echo "ERROR: DeepSpeed installation failed."
        echo "You may need to build manually or skip DeepSpeed features."
    }
}

# Step 7: optimum-quanto (check compatibility)
echo ""
echo "Step 7: Installing optimum-quanto..."
pip install optimum-quanto || {
    echo "WARNING: optimum-quanto may not fully support aarch64."
    echo "Some quantization features may not work."
}

# Step 8: Other utilities
echo ""
echo "Step 8: Installing remaining utilities..."
pip install \
    opencv-python-headless \
    timm \
    qwen_vl_utils \
    fastsafetensors \
    onnxruntime

# Step 9: Flash Attention (OPTIONAL - NOT RECOMMENDED for CUDA 13.0)
echo ""
echo "Step 9: Flash Attention installation..."
echo "WARNING: Flash Attention has known issues with CUDA 13.0 on aarch64."
echo "There are GCC13 bugs that can cause compilation failures."
echo ""
echo "Do you want to attempt flash-attn installation? (y/n)"
echo "(This may take 2+ hours and could fail)"
read -r install_flash
if [ "$install_flash" == "y" ]; then
    echo "Installing ninja for faster compilation..."
    pip install ninja
    echo "Building flash-attn (this will take a LONG time)..."
    MAX_JOBS=4 pip install flash-attn --no-build-isolation || {
        echo "ERROR: Flash Attention build failed."
        echo "This is expected for CUDA 13.0 on aarch64."
        echo "Consider using CUDA 12.8 for flash-attn support."
    }
else
    echo "Skipping flash-attn installation."
    echo "cuDNN SDP will be used instead of Flash Attention."
fi

# Final verification
echo ""
echo "============================================"
echo "Installation Complete! Verifying..."
echo "============================================"
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability()
    print(f'Compute Capability: {cap}')
    if cap[0] >= 9:
        print('Architecture: Hopper+ (Full CUDA 13.0 features)')
    elif cap[0] >= 8:
        print('Architecture: Ampere+ (Partial CUDA 13.0 features)')
    print()
    print('Checking installed packages...')

import sys
packages = ['transformers', 'accelerate', 'peft', 'bitsandbytes', 'diffusers', 'deepspeed']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg}: {ver}')
    except ImportError:
        print(f'  {pkg}: NOT INSTALLED')

try:
    import flash_attn
    print(f'  flash-attn: {flash_attn.__version__}')
except ImportError:
    print(f'  flash-attn: NOT INSTALLED (using cuDNN SDP)')
"

echo ""
echo "============================================"
echo "IMPORTANT NOTES FOR AARCH64-SBSA:"
echo "============================================"
echo "1. Flash Attention 3 is NOT STABLE on CUDA 13.0 + aarch64"
echo "   Set enable_flash_attention_3: false in your config"
echo ""
echo "2. TF32 and cuDNN SDP are fully supported"
echo ""
echo "3. If you need stable Flash Attention, consider CUDA 12.8:"
echo "   pip install torch --index-url https://download.pytorch.org/whl/cu128"
echo ""
echo "4. For GH200/DGX SPARK, use train_configs/train_aarch64_optimized.yaml"
echo "============================================"
