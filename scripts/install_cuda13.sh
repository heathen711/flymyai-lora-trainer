#!/bin/bash
# install_cuda13.sh
# Install CUDA 13.0 compatible packages in correct order

set -e

echo "Installing CUDA 13.0 compatible packages..."

# Step 1: Install PyTorch from CUDA 13.0 index
echo ""
echo "Step 1: Installing PyTorch with CUDA 13.0 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Step 2: Install core training libraries
echo ""
echo "Step 2: Installing core training libraries..."
pip install accelerate deepspeed transformers peft

# Step 3: Install quantization libraries
echo ""
echo "Step 3: Installing quantization libraries..."
pip install bitsandbytes optimum-quanto

# Step 4: Install diffusers from specific commit
echo ""
echo "Step 4: Installing diffusers..."
pip install git+https://github.com/huggingface/diffusers@7a2b78bf0f788d311cc96b61e660a8e13e3b1e63

# Step 5: Install data and utility libraries
echo ""
echo "Step 5: Installing data and utility libraries..."
pip install einops huggingface-hub datasets omegaconf sentencepiece opencv-python matplotlib onnxruntime timm qwen_vl_utils loguru

# Step 6: Install FastSafeTensors
echo ""
echo "Step 6: Installing FastSafeTensors..."
pip install fastsafetensors

# Optional: Flash Attention (uncomment when available for CUDA 13.0)
# echo ""
# echo "Step 7: Installing Flash Attention..."
# pip install flash-attn --no-build-isolation
# pip install xformers

echo ""
echo "CUDA 13.0 package installation complete!"
echo ""
echo "Verify installation:"
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
"
