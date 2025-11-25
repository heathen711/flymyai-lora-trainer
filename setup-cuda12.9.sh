#!/bin/bash
set -euo pipefail

# Setup script for flymyai-lora-trainer with CUDA 12.9 wheels

echo "=========================================="
echo "Setting up flymyai-lora-trainer (CUDA 12.9)"
echo "=========================================="
echo

# Activate venv
source venv-cuda12.9/bin/activate

# Set CUDA 12.9 paths
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH

echo "[1/6] Installing remaining dependencies (will pull in PyPI PyTorch temporarily)..."
if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt || true
fi

echo "[2/6] Removing PyPI PyTorch (will be replaced with CUDA 12.9 custom wheels)..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true

echo "[3/6] Installing PyTorch CUDA 12.9 wheels..."
pip install --no-cache-dir \
    /home/jay/Documents/DGX-Spark-PyTorch/wheels/triton-*+cu129*.whl \
    /home/jay/Documents/DGX-Spark-PyTorch/wheels/torch-*+cu129*.whl \
    /home/jay/Documents/DGX-Spark-PyTorch/wheels/torchvision-*+cu129*.whl \
    /home/jay/Documents/DGX-Spark-PyTorch/wheels/torchaudio-*+cu129*.whl

echo "[4/6] Installing Flash Attention CUDA 12.9 wheel..."
pip install --no-cache-dir /home/jay/Documents/DGX-Spark-FlashAttention/wheels/flash_attn-*+cu129*.whl

echo "[5/6] Installing SageAttention CUDA 12.9 wheel..."
pip install --no-cache-dir /home/jay/Documents/DGX-Spark-SageAttention/wheels/sageattention-*+cu129*.whl

echo "[6/6] Installing ONNX Runtime CUDA 12.9 wheel..."
pip install --no-cache-dir /home/jay/Documents/DGX-Spark-ONNX/wheels/onnxruntime*+cu129*.whl

echo
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "CUDA Version: 12.9"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
echo
echo "Installed wheels:"
echo "  - PyTorch 2.10.0a0+cu129.ubuntu2404.sm121"
echo "  - Triton 3.5.0+cu129.ubuntu2404.sm121"
echo "  - TorchVision 0.25.0a0+cu129.ubuntu2404.sm121"
echo "  - TorchAudio 2.10.0a0+cu129.ubuntu2404.sm121"
echo "  - Flash Attention 2.8.0+cu129.ubuntu2404.sm121"
echo "  - SageAttention 2.2.0+cu129.ubuntu2404.sm121"
echo "  - ONNX Runtime 1.23.2+cu129.ubuntu2404.sm121"
echo
echo "To activate this environment later:"
echo "  source activate-cuda12.9.sh"
echo "=========================================="
