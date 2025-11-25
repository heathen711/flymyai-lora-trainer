#!/bin/bash
# Activation script for CUDA 12.9 environment
# Usage: source activate-cuda12.9.sh

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "=========================================="
    echo "ERROR: Script must be sourced, not executed!"
    echo "=========================================="
    echo "Usage: source activate-cuda12.9.sh"
    echo "   or: . activate-cuda12.9.sh"
    echo "=========================================="
    exit 1
fi

# Check if already in virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Warning: Already in virtual environment: $VIRTUAL_ENV"
    echo "Deactivating current environment first..."
    deactivate
fi

# Activate CUDA 12.9 virtual environment
source venv-cuda12.9/bin/activate

# Set CUDA 12.9 environment variables
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH

echo "=========================================="
echo "CUDA 12.9 Environment Activated"
echo "=========================================="
echo "Virtual Environment: venv-cuda12.9"
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
if [[ "$CUDA_VERSION" == "12.9" && "$HAS_SM121" == "True" ]]; then
    echo "✓ Correct CUDA 12.9 PyTorch with sm_121 support"
elif [[ "$CUDA_VERSION" == "N/A" ]] || [[ "$PYTORCH_VERSION" == *"cpu"* ]]; then
    echo "✗ WARNING: CPU-only PyTorch detected!"
    echo "  Run: pip uninstall torch torchvision torchaudio triton -y"
    echo "  Then reinstall with setup-cuda12.9.sh"
elif [[ "$CUDA_VERSION" != "12.9" ]]; then
    echo "⚠ WARNING: Wrong CUDA version: $CUDA_VERSION (expected 12.9)"
    echo "  Run: pip uninstall torch torchvision torchaudio triton -y"
    echo "  Then reinstall with setup-cuda12.9.sh"
fi

echo "=========================================="
