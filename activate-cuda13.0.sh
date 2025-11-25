#!/bin/bash
# Activation script for CUDA 13.0 environment
# Usage: source activate-cuda13.0.sh

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "=========================================="
    echo "ERROR: Script must be sourced, not executed!"
    echo "=========================================="
    echo "Usage: source activate-cuda13.0.sh"
    echo "   or: . activate-cuda13.0.sh"
    echo "=========================================="
    exit 1
fi

# Check if already in virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Warning: Already in virtual environment: $VIRTUAL_ENV"
    echo "Deactivating current environment first..."
    deactivate
fi

# Activate CUDA 13.0 virtual environment
source venv/bin/activate

# Set CUDA 13.0 environment variables
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH

echo "=========================================="
echo "CUDA 13.0 Environment Activated"
echo "=========================================="
echo "Virtual Environment: venv"
echo "CUDA_HOME: $CUDA_HOME"
echo "Python: $(which python)"
echo

# Check PyTorch installation
PYTORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')
CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')
CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda if hasattr(torch.version, "cuda") else "N/A")' 2>/dev/null || echo 'N/A')
DEVICE_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'N/A')
HAS_SM121=$(python -c 'import torch; print("sm_121" in torch.cuda.get_arch_list() if torch.cuda.is_available() else False)' 2>/dev/null || echo 'False')

echo "PyTorch: $PYTORCH_VERSION"
echo "CUDA Version: $CUDA_VERSION"
echo "CUDA Available: $CUDA_AVAILABLE"
echo "Device: $DEVICE_NAME"
echo "SM 121 Support: $HAS_SM121"

# Verify correct PyTorch version
if [[ "$CUDA_VERSION" == "13.0" && "$HAS_SM121" == "True" ]]; then
    echo "✓ Correct CUDA 13.0 PyTorch with sm_121 support"
elif [[ "$CUDA_VERSION" == "N/A" ]] || [[ "$PYTORCH_VERSION" == *"cpu"* ]]; then
    echo "✗ WARNING: CPU-only PyTorch detected!"
    echo "  Reinstall custom PyTorch wheels for CUDA 13.0"
elif [[ "$CUDA_VERSION" != "13.0" ]]; then
    echo "⚠ WARNING: Wrong CUDA version: $CUDA_VERSION (expected 13.0)"
    echo "  Reinstall custom PyTorch wheels for CUDA 13.0"
fi

echo "=========================================="
