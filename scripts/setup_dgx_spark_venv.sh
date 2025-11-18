#!/bin/bash
# setup_dgx_spark_venv.sh
# Setup virtual environment for DGX Spark with custom sm_121 PyTorch wheels

set -e

echo "=== DGX Spark Training Environment Setup ==="
echo

# Step 1: Create venv
echo "Step 1: Creating Python 3.12 virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
echo "✓ Virtual environment created"
echo

# Step 2: Verify cuDNN
echo "Step 2: Verifying cuDNN installation..."
if ! ldconfig -p | grep -q "libcudnn.so.9"; then
    echo "❌ cuDNN 9.x not found!"
    echo "Please install cuDNN 9.x for CUDA 13.0:"
    echo "  See: ~/.claude/knowledge/dgx-spark-custom-wheels.md"
    exit 1
fi
echo "✓ cuDNN 9.x found"
echo

# Step 3: Install DGX-Spark PyTorch wheels
echo "Step 3: Installing DGX-Spark custom PyTorch wheels (sm_121)..."
WHEELS_DIR="$HOME/Documents/DGX-Spark-PyTorch/wheels"
if [ ! -d "$WHEELS_DIR" ]; then
    echo "❌ Custom wheels not found at $WHEELS_DIR"
    echo "Expected wheel files:"
    echo "  - torch-2.10.0a0+cu130.ubuntu2404.sm121-cp312-cp312-linux_aarch64.whl"
    echo "  - triton-3.5.0+git4ccf8a95-cp312-cp312-linux_aarch64.whl"
    echo "  - torchvision-0.25.0a0+617079d-cp312-cp312-linux_aarch64.whl"
    echo "  - torchaudio-2.10.0a0+ee1a135-cp312-cp312-linux_aarch64.whl"
    exit 1
fi

pip install \
  "$WHEELS_DIR"/torch-2.10.0a0+cu130.ubuntu2404.sm121-cp312-cp312-linux_aarch64.whl \
  "$WHEELS_DIR"/triton-3.5.0+git4ccf8a95-cp312-cp312-linux_aarch64.whl \
  "$WHEELS_DIR"/torchvision-0.25.0a0+617079d-cp312-cp312-linux_aarch64.whl \
  "$WHEELS_DIR"/torchaudio-2.10.0a0+ee1a135-cp312-cp312-linux_aarch64.whl

echo "✓ PyTorch wheels installed"
echo

# Step 4: Install training dependencies
echo "Step 4: Installing training dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo

# Step 5: Reinstall custom wheels (in case requirements.txt overwrote them)
echo "Step 5: Ensuring custom PyTorch wheels are active..."
pip uninstall torch torchvision torchaudio triton -y
pip install \
  "$WHEELS_DIR"/torch-2.10.0a0+cu130.ubuntu2404.sm121-cp312-cp312-linux_aarch64.whl \
  "$WHEELS_DIR"/triton-3.5.0+git4ccf8a95-cp312-cp312-linux_aarch64.whl \
  "$WHEELS_DIR"/torchvision-0.25.0a0+617079d-cp312-cp312-linux_aarch64.whl \
  "$WHEELS_DIR"/torchaudio-2.10.0a0+ee1a135-cp312-cp312-linux_aarch64.whl

echo "✓ Custom wheels verified"
echo

# Step 6: Verification
echo "Step 6: Verifying installation..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability()
    print(f'Compute Capability: {cap}')
    print(f'Architectures: {torch.cuda.get_arch_list()}')
    sm121_support = 'sm_121' in torch.cuda.get_arch_list()
    print(f'sm_121 support: {sm121_support}')
    if not sm121_support:
        print('❌ WARNING: sm_121 support not found!')
        exit(1)
    print()
    print('Testing CUDA operations...')
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x + y
    print('✓ CUDA tensor operations working')
"

echo
echo "=== Setup Complete! ==="
echo
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo
echo "To start training:"
echo "  python train_4090.py --config train_configs/train_dgx_spark.yaml"
