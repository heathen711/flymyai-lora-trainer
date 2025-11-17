# AARCH64-SBSA (ARM64) Compatibility Guide

## Overview

This document details package compatibility for ARM64 Server Base System Architecture (SBSA) platforms like NVIDIA Grace Hopper (GH200), DGX SPARK, and Jetson Thor with CUDA 13.0.

## Package Compatibility Matrix

| Package | Pre-built Wheels | Build from Source | CUDA 13.0 Support | Notes |
|---------|-----------------|-------------------|-------------------|-------|
| **PyTorch** | ✅ Yes | ✅ Yes | ✅ Yes | Official sbsa nightly builds since Aug 2025 |
| **torchvision** | ✅ Yes | ✅ Yes | ✅ Yes | Follows PyTorch releases |
| **torchaudio** | ✅ Yes | ✅ Yes | ✅ Yes | Follows PyTorch releases |
| **bitsandbytes** | ✅ Yes | ✅ Yes | ✅ Yes | Official sbsa wheels, SM75+ support |
| **transformers** | ⚠️ Partial | ✅ Yes | ✅ Yes | Pure Python, but tokenizers needs Rust |
| **tokenizers** | ❌ No | ✅ Yes | ✅ Yes | Requires Rust compiler |
| **accelerate** | ✅ Yes | ✅ Yes | ✅ Yes | Pure Python |
| **peft** | ✅ Yes | ✅ Yes | ✅ Yes | Pure Python |
| **deepspeed** | ❌ No | ✅ Yes | ⚠️ Unknown | Must build from source |
| **flash-attn** | ⚠️ Partial | ⚠️ Issues | ❌ Not stable | CUDA 12.8 recommended, GCC13 bugs |
| **xformers** | ❌ No | ✅ Yes | ⚠️ Unknown | Must build from source |
| **diffusers** | ✅ Yes | ✅ Yes | ✅ Yes | Pure Python |
| **omegaconf** | ✅ Yes | ✅ Yes | N/A | Pure Python |
| **einops** | ✅ Yes | ✅ Yes | N/A | Pure Python |
| **opencv-python** | ⚠️ Partial | ✅ Yes | N/A | May need headless variant |
| **onnxruntime** | ⚠️ Partial | ✅ Yes | ⚠️ Unknown | Check official releases |

## Installation Script for AARCH64-SBSA

```bash
#!/bin/bash
# install_cuda13_aarch64.sh
# CUDA 13.0 installation for ARM64 SBSA platforms

set -e

echo "Installing CUDA 13.0 packages for aarch64-sbsa..."

# Prerequisites
echo "Step 0: Installing build prerequisites..."
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build

# Install Rust (required for tokenizers)
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust compiler..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Step 1: PyTorch from CUDA 13.0 sbsa index
echo ""
echo "Step 1: Installing PyTorch with CUDA 13.0 sbsa support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Step 2: Pure Python packages (no compilation needed)
echo ""
echo "Step 2: Installing pure Python packages..."
pip install accelerate peft diffusers omegaconf einops huggingface-hub datasets loguru matplotlib

# Step 3: Packages that may need compilation
echo ""
echo "Step 3: Installing transformers (tokenizers will compile with Rust)..."
pip install transformers

# Step 4: bitsandbytes (has sbsa wheels)
echo ""
echo "Step 4: Installing bitsandbytes..."
# Try pre-built wheel first
pip install bitsandbytes || {
    echo "Pre-built wheel not found, building from source..."
    pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git
}

# Step 5: DeepSpeed (must build from source)
echo ""
echo "Step 5: Installing DeepSpeed from source..."
DS_BUILD_OPS=1 pip install deepspeed --no-cache-dir

# Step 6: Optional - Build flash-attention (EXPERIMENTAL)
# WARNING: CUDA 13.0 has known issues with flash-attention on aarch64
# echo ""
# echo "Step 6: Building flash-attention (this may take 2+ hours)..."
# pip install ninja
# MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Step 7: Other utilities
echo ""
echo "Step 7: Installing remaining utilities..."
pip install sentencepiece opencv-python-headless timm qwen_vl_utils fastsafetensors

# Optional: optimum-quanto (check for aarch64 support)
pip install optimum-quanto || echo "optimum-quanto may not support aarch64"

echo ""
echo "Installation complete! Verifying..."
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
        print('Architecture: Hopper+ (Full CUDA 13.0 features supported)')
"
```

## Key Issues and Workarounds

### 1. Flash Attention 3 on CUDA 13.0 + aarch64

**Status**: NOT RECOMMENDED for CUDA 13.0 on aarch64

Known issues:
- GCC13 compiler bug with SVE/SVE256 and `-march=native`
- Compilation may fail or crash
- CUDA 12.8 is more stable

**Workaround**: Use CUDA 12.8 instead, or disable Flash Attention 3:
```yaml
cuda_13_features:
  enable_flash_attention_3: false  # Disable until stable
```

### 2. DeepSpeed on aarch64

**Status**: Must build from source

```bash
# Build with CPU ops
DS_BUILD_OPS=1 pip install deepspeed --no-cache-dir

# Or minimal build without CUDA ops
DS_BUILD_CPU_ADAM=1 pip install deepspeed
```

### 3. Tokenizers Compilation

**Requires**: Rust compiler installed

```bash
# Install Rust first
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Then install transformers
pip install transformers
```

### 4. OpenCV on Headless Servers

Use the headless variant:
```bash
pip install opencv-python-headless
```

## Recommended Configuration for GH200/DGX SPARK

```yaml
# train_configs/train_aarch64_optimized.yaml
pretrained_model_name_or_path: "Qwen/Qwen2.5-VL-3B-Instruct"
output_dir: "output/aarch64_optimized"
mixed_precision: "bf16"

cuda_13_features:
  enable_flash_attention_3: false  # Unstable on aarch64+CUDA13
  enable_fp8_training: false       # Test stability first
  enable_tf32_compute: true        # Supported
  enable_cudnn_sdp: true           # Use cuDNN SDP instead of FA3
```

## Alternative: Use CUDA 12.8 for Better Stability

Given the current state of CUDA 13.0 support on aarch64-sbsa, consider using CUDA 12.8:

```bash
# More stable for aarch64-sbsa
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

CUDA 12.8 offers:
- Better flash-attention compatibility
- More tested on ARM64 platforms
- Official SBSA wheel support

## Hardware Compatibility

### Supported SBSA Platforms
- NVIDIA Grace Hopper (GH200) - SM90
- DGX SPARK - SM121
- Jetson Thor - SM110
- GB300/B300 servers

### Architecture Notes
- CUDA 13.0 unifies ARM platforms (L4T and SBSA)
- Single install for all ARM architectures
- Compute capabilities: SM75+ for bitsandbytes

## References

- PyTorch CUDA 13.0 aarch64 nightly: GitHub PR #161257
- bitsandbytes SBSA wheels: GitHub releases
- Flash Attention CUDA 13 issues: GitHub Issue #1815
- NVIDIA CUDA 13.0 unified ARM support documentation
