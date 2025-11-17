#!/bin/bash
# setup_cuda13.sh
# Setup script for CUDA 13.0 environment

set -e

echo "Setting up CUDA 13.0 environment..."

# Check if CUDA 13.0 is installed
CUDA_PATH="/usr/local/cuda-13.0"
if [ ! -d "$CUDA_PATH" ]; then
    echo "ERROR: CUDA 13.0 not found at $CUDA_PATH"
    echo "Please install CUDA 13.0 toolkit first"
    exit 1
fi

# Set environment variables
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA version
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "NVCC Version: $NVCC_VERSION"

if [[ ! "$NVCC_VERSION" == "13."* ]]; then
    echo "WARNING: NVCC version is not 13.x, got $NVCC_VERSION"
fi

# Check NVIDIA driver version
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "NVIDIA Driver: $DRIVER_VERSION"

# Driver 550.x+ required for CUDA 13.0
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d'.' -f1)
if [ "$DRIVER_MAJOR" -lt 550 ]; then
    echo "WARNING: Driver version $DRIVER_VERSION may not support CUDA 13.0"
    echo "Recommended: Driver 550.x or newer"
fi

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability()}')
"

echo ""
echo "CUDA 13.0 environment setup complete!"
echo ""
echo "To persist these settings, add to your ~/.bashrc:"
echo "  export CUDA_HOME=$CUDA_PATH"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
