# Docker Container Setup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a containerized environment for CUDA 13.0, DGX Spark unified memory, and fastsafetensors to enable reproducible training deployments.

**Architecture:** Build a multi-stage Dockerfile with CUDA 13.0 base image, install all dependencies including fastsafetensors, configure environment variables for unified memory, add Docker Compose for orchestration, and include health checks and monitoring.

**Tech Stack:** Docker, NVIDIA Container Toolkit, CUDA 13.0, Python 3.10, PyTorch

---

## Task 1: Create .dockerignore File

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/.dockerignore`

**Step 1: Write .dockerignore**

```
# .dockerignore
# Git
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Data and checkpoints (mount as volumes instead)
data/
checkpoints/
output/
cache/
*.safetensors
*.pt
*.bin
*.onnx

# Logs
logs/
*.log

# Documentation
docs/plans/
*.md
!README.md

# Tests
tests/
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Temporary
tmp/
temp/
*.tmp
```

**Step 2: Commit**

```bash
git add .dockerignore
git commit -m "feat: add .dockerignore for Docker builds"
```

---

## Task 2: Create Base Dockerfile

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/Dockerfile`

**Step 1: Write Dockerfile with CUDA 13.0 base**

```dockerfile
# Dockerfile
# CUDA 13.0 base image for LoRA training
# Supports DGX Spark unified memory and fastsafetensors

# Build stage
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
COPY requirements-cuda13.txt /app/requirements-cuda13.txt

# Install PyTorch with CUDA 13.0 support
# NOTE: Update URL when official CUDA 13.0 wheels are available
RUN pip install --no-cache-dir \
    torch>=2.5.0 \
    torchvision>=0.22.1 \
    torchaudio>=2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir -r requirements-cuda13.txt

# Install fastsafetensors
RUN pip install --no-cache-dir fastsafetensors>=0.1.0

# Runtime stage (smaller image)
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY train.py /app/
COPY train_4090.py /app/
COPY train_flux_lora.py /app/
COPY train_qwen_edit_lora.py /app/
COPY train_kandinsky_lora.py /app/
COPY train_full_qwen_image.py /app/
COPY inference.py /app/
COPY qwen_full_inference_example.py /app/

# Copy modules
COPY image_datasets/ /app/image_datasets/
COPY utils/ /app/utils/
COPY train_configs/ /app/train_configs/

# Copy any additional files needed
COPY qwen_image_lora_example.json /app/

# Create directories for data and outputs
RUN mkdir -p /app/data /app/checkpoints /app/cache /app/output /app/logs

# Environment variables for CUDA and unified memory
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# DGX Spark unified memory settings
ENV CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,backend:native

# HuggingFace cache
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/cache/huggingface

# NVIDIA runtime settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Expose port for monitoring (optional)
EXPOSE 9090

# Default entry point
ENTRYPOINT ["python"]
CMD ["train.py", "--config", "train_configs/train_lora.yaml"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for CUDA 13.0 training environment"
```

---

## Task 3: Create Docker Compose Configuration

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/docker-compose.yml`

**Step 1: Write docker-compose.yml**

```yaml
# docker-compose.yml
# Docker Compose configuration for LoRA training

version: '3.8'

services:
  # Standard GPU training service
  lora-trainer:
    build:
      context: .
      dockerfile: Dockerfile
    image: flymyai-lora-trainer:cuda13
    runtime: nvidia
    container_name: lora-trainer
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
      - UNIFIED_MEMORY=false
      - HF_HOME=/app/cache/huggingface
    volumes:
      - ./data:/app/data:ro
      - ./checkpoints:/app/checkpoints
      - ./output:/app/output
      - ./train_configs:/app/train_configs:ro
      - huggingface_cache:/app/cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
    shm_size: '32gb'
    ulimits:
      memlock:
        soft: -1
        hard: -1
    stdin_open: true
    tty: true

  # DGX Spark unified memory optimized service
  dgx-spark-trainer:
    extends:
      service: lora-trainer
    container_name: dgx-spark-trainer
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - UNIFIED_MEMORY=true
      - DISABLE_CPU_OFFLOAD=true
      - CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,backend:native
      - HF_HOME=/app/cache/huggingface
    shm_size: '128gb'
    command: ["train_4090.py", "--config", "train_configs/train_dgx_spark.yaml"]

  # RTX 4090 optimized service (quantization enabled)
  rtx4090-trainer:
    extends:
      service: lora-trainer
    container_name: rtx4090-trainer
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - UNIFIED_MEMORY=false
      - HF_HOME=/app/cache/huggingface
    shm_size: '32gb'
    command: ["train_4090.py", "--config", "train_configs/train_lora_4090.yaml"]

  # FLUX model training service
  flux-trainer:
    extends:
      service: lora-trainer
    container_name: flux-trainer
    command: ["train_flux_lora.py", "--config", "train_configs/train_flux_config.yaml"]

  # Kandinsky model training service
  kandinsky-trainer:
    extends:
      service: lora-trainer
    container_name: kandinsky-trainer
    command: ["train_kandinsky_lora.py", "--config", "train_configs/Kandinsky_config.yaml"]

  # Inference service
  inference:
    extends:
      service: lora-trainer
    container_name: lora-inference
    command: ["inference.py"]
    ports:
      - "8080:8080"

volumes:
  huggingface_cache:
    driver: local
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add Docker Compose configuration for multiple training scenarios"
```

---

## Task 4: Create Container Test Script

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/scripts/test_container.sh`

**Step 1: Write test script**

```bash
#!/bin/bash
# test_container.sh
# Validate Docker container setup

set -e

IMAGE_NAME="flymyai-lora-trainer:cuda13"

echo "============================================"
echo "Testing Docker Container: $IMAGE_NAME"
echo "============================================"

# Test 1: Basic Python and PyTorch
echo ""
echo "Test 1: PyTorch and CUDA availability"
docker run --rm --gpus all $IMAGE_NAME python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('ERROR: CUDA not available in container')
    exit(1)
"

# Test 2: FastSafeTensors
echo ""
echo "Test 2: FastSafeTensors import"
docker run --rm --gpus all $IMAGE_NAME python -c "
try:
    import fastsafetensors
    print('FastSafeTensors: Available')
except ImportError as e:
    print(f'FastSafeTensors: NOT available - {e}')
"

# Test 3: Unified memory utilities
echo ""
echo "Test 3: Unified memory utilities"
docker run --rm --gpus all $IMAGE_NAME python -c "
from utils.unified_memory import is_unified_memory_system, get_memory_config
print(f'Unified Memory System: {is_unified_memory_system()}')
config = get_memory_config(unified_memory=True)
print(f'Config for unified memory: {config}')
"

# Test 4: CUDA utilities
echo ""
echo "Test 4: CUDA utilities"
docker run --rm --gpus all $IMAGE_NAME python -c "
from utils.cuda_utils import get_optimal_settings, enable_tf32
settings = get_optimal_settings()
print(f'Optimal Settings: {settings}')
enable_tf32()
print('TF32 enabled successfully')
"

# Test 5: Training script imports
echo ""
echo "Test 5: Training script imports"
docker run --rm --gpus all $IMAGE_NAME python -c "
import train
import train_4090
import train_flux_lora
import train_kandinsky_lora
import train_qwen_edit_lora
print('All training scripts imported successfully')
"

# Test 6: Memory allocation
echo ""
echo "Test 6: GPU memory allocation"
docker run --rm --gpus all $IMAGE_NAME python -c "
import torch
# Allocate 1GB on GPU
tensor = torch.randn(256, 1024, 1024, device='cuda')
print(f'Allocated tensor shape: {tensor.shape}')
print(f'Tensor device: {tensor.device}')
mem_allocated = torch.cuda.memory_allocated() / 1e9
print(f'Memory allocated: {mem_allocated:.2f} GB')
del tensor
torch.cuda.empty_cache()
print('Memory freed successfully')
"

# Test 7: Diffusers import
echo ""
echo "Test 7: Diffusers library"
docker run --rm --gpus all $IMAGE_NAME python -c "
import diffusers
print(f'Diffusers Version: {diffusers.__version__}')
from diffusers import QwenImagePipeline, AutoencoderKLQwenImage
print('Diffusers models imported successfully')
"

echo ""
echo "============================================"
echo "All container tests passed!"
echo "============================================"
```

**Step 2: Make executable**

```bash
chmod +x scripts/test_container.sh
```

**Step 3: Commit**

```bash
git add scripts/test_container.sh
git commit -m "feat: add Docker container validation test script"
```

---

## Task 5: Create Build Script

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/scripts/build_docker.sh`

**Step 1: Write build script**

```bash
#!/bin/bash
# build_docker.sh
# Build Docker image for LoRA training

set -e

IMAGE_NAME="flymyai-lora-trainer"
TAG="cuda13"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "Building Docker image: $FULL_IMAGE"
echo ""

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found in current directory"
    exit 1
fi

# Build the image
docker build \
    --tag $FULL_IMAGE \
    --file Dockerfile \
    --progress=plain \
    .

echo ""
echo "Build complete: $FULL_IMAGE"
echo ""

# Show image size
docker images $FULL_IMAGE --format "Image: {{.Repository}}:{{.Tag}}\nSize: {{.Size}}\nCreated: {{.CreatedAt}}"

echo ""
echo "To test the image, run:"
echo "  ./scripts/test_container.sh"
echo ""
echo "To run training, use:"
echo "  docker-compose up lora-trainer"
echo "  # or"
echo "  docker run --gpus all -v \$(pwd)/data:/app/data $FULL_IMAGE"
```

**Step 2: Make executable**

```bash
chmod +x scripts/build_docker.sh
```

**Step 3: Commit**

```bash
git add scripts/build_docker.sh
git commit -m "feat: add Docker build script"
```

---

## Task 6: Create NVIDIA Container Toolkit Check Script

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/scripts/check_nvidia_docker.sh`

**Step 1: Write check script**

```bash
#!/bin/bash
# check_nvidia_docker.sh
# Verify NVIDIA Container Toolkit is properly configured

set -e

echo "Checking NVIDIA Container Toolkit setup..."
echo ""

# Check Docker version
echo "1. Docker Version:"
docker --version
echo ""

# Check NVIDIA driver
echo "2. NVIDIA Driver:"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   Driver Version: $DRIVER_VERSION"

    # Check driver version (550+ recommended for CUDA 13.0)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d'.' -f1)
    if [ "$DRIVER_MAJOR" -lt 550 ]; then
        echo "   WARNING: Driver version $DRIVER_VERSION may not fully support CUDA 13.0"
        echo "   Recommended: 550.x or newer"
    fi
else
    echo "   ERROR: nvidia-smi not found"
    exit 1
fi
echo ""

# Check NVIDIA Container Runtime
echo "3. NVIDIA Container Runtime:"
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "   NVIDIA runtime: Found"
else
    echo "   WARNING: NVIDIA runtime may not be configured"
    echo "   Check /etc/docker/daemon.json"
fi
echo ""

# Check nvidia-container-toolkit
echo "4. NVIDIA Container Toolkit:"
if dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
    TOOLKIT_VERSION=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
    echo "   Version: $TOOLKIT_VERSION"
elif rpm -q nvidia-container-toolkit 2>/dev/null; then
    TOOLKIT_VERSION=$(rpm -q nvidia-container-toolkit)
    echo "   Version: $TOOLKIT_VERSION"
else
    echo "   WARNING: nvidia-container-toolkit not found via package manager"
    echo "   Install with: sudo apt-get install nvidia-container-toolkit"
fi
echo ""

# Test GPU access in container
echo "5. Testing GPU access in container:"
if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    echo "   GPU access: Working"
    docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "   ERROR: Cannot access GPU in container"
    echo "   Check NVIDIA Container Toolkit installation"
    exit 1
fi
echo ""

# Check Docker daemon configuration
echo "6. Docker Daemon Configuration:"
if [ -f /etc/docker/daemon.json ]; then
    echo "   /etc/docker/daemon.json exists"
    if grep -q "nvidia" /etc/docker/daemon.json; then
        echo "   NVIDIA runtime: Configured"
    else
        echo "   WARNING: NVIDIA runtime not found in daemon.json"
    fi
else
    echo "   WARNING: /etc/docker/daemon.json not found"
    echo "   Consider creating it with NVIDIA runtime configuration"
fi
echo ""

echo "============================================"
echo "NVIDIA Container Toolkit check complete!"
echo "============================================"
```

**Step 2: Make executable**

```bash
chmod +x scripts/check_nvidia_docker.sh
```

**Step 3: Commit**

```bash
git add scripts/check_nvidia_docker.sh
git commit -m "feat: add NVIDIA Container Toolkit verification script"
```

---

## Task 7: Create Production Dockerfile (Optimized)

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/Dockerfile.prod`

**Step 1: Write production optimized Dockerfile**

```dockerfile
# Dockerfile.prod
# Production-optimized Docker image for LoRA training
# Smaller image size, security hardened

FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Metadata
LABEL maintainer="FlyMyAI"
LABEL description="Production LoRA Training with CUDA 13.0"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash trainer
USER trainer
WORKDIR /home/trainer/app

# Copy requirements and install as user
COPY --chown=trainer:trainer requirements-cuda13.txt .
RUN pip install --no-cache-dir --user -r requirements-cuda13.txt

# Add user local bin to PATH
ENV PATH="/home/trainer/.local/bin:$PATH"

# Copy application code
COPY --chown=trainer:trainer train.py .
COPY --chown=trainer:trainer train_4090.py .
COPY --chown=trainer:trainer train_flux_lora.py .
COPY --chown=trainer:trainer train_qwen_edit_lora.py .
COPY --chown=trainer:trainer train_kandinsky_lora.py .
COPY --chown=trainer:trainer inference.py .
COPY --chown=trainer:trainer image_datasets/ ./image_datasets/
COPY --chown=trainer:trainer utils/ ./utils/
COPY --chown=trainer:trainer train_configs/ ./train_configs/

# Create directories
RUN mkdir -p data checkpoints cache output logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/home/trainer/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/home/trainer/app/cache/huggingface
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Entry point
ENTRYPOINT ["python"]
CMD ["train.py", "--config", "train_configs/train_lora.yaml"]
```

**Step 2: Commit**

```bash
git add Dockerfile.prod
git commit -m "feat: add production-optimized Dockerfile with non-root user"
```

---

## Task 8: Create GitHub Actions Workflow

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/.github/workflows/docker.yml`

**Step 1: Create .github/workflows directory**

```bash
mkdir -p .github/workflows
```

**Step 2: Write CI/CD workflow**

```yaml
# .github/workflows/docker.yml
name: Build Docker Image

on:
  push:
    branches: [main]
    paths:
      - 'Dockerfile'
      - 'Dockerfile.prod'
      - 'requirements*.txt'
      - '.github/workflows/docker.yml'
  pull_request:
    branches: [main]
    paths:
      - 'Dockerfile'
      - 'Dockerfile.prod'
      - 'requirements*.txt'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build development image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile
          push: false
          tags: flymyai-lora-trainer:cuda13-dev
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Build production image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.prod
          push: false
          tags: flymyai-lora-trainer:cuda13-prod
          cache-from: type=gha
          cache-to: type=gha,mode=max

  security-scan:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build image for scanning
        run: docker build -t scan-target:latest -f Dockerfile.prod .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'scan-target:latest'
          format: 'table'
          exit-code: '0'
          ignore-unfixed: true
          vuln-type: 'os,library'
          severity: 'CRITICAL,HIGH'
```

**Step 3: Commit**

```bash
git add .github/workflows/docker.yml
git commit -m "ci: add GitHub Actions workflow for Docker builds"
```

---

## Task 9: Create Docker Documentation

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/DOCKER.md`

**Step 1: Write comprehensive Docker documentation**

```markdown
# Docker Setup Guide

This guide covers building and running the LoRA training environment in Docker containers.

## Prerequisites

- Docker >= 24.0
- NVIDIA Driver >= 550.x (for CUDA 13.0)
- NVIDIA Container Toolkit >= 1.16.0
- 50GB+ disk space for images

## Quick Start

### 1. Verify NVIDIA Docker Setup

```bash
./scripts/check_nvidia_docker.sh
```

### 2. Build the Image

```bash
./scripts/build_docker.sh
```

### 3. Test the Container

```bash
./scripts/test_container.sh
```

### 4. Run Training

```bash
# Standard training
docker-compose up lora-trainer

# DGX Spark (unified memory)
docker-compose up dgx-spark-trainer

# RTX 4090 (quantized)
docker-compose up rtx4090-trainer
```

## Manual Docker Commands

### Build Image

```bash
docker build -t flymyai-lora-trainer:cuda13 .
```

### Run Training

```bash
# Basic run
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/output:/app/output \
    flymyai-lora-trainer:cuda13

# With specific config
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    flymyai-lora-trainer:cuda13 \
    train_4090.py --config train_configs/train_lora_4090.yaml

# DGX Spark with unified memory
docker run --gpus all \
    --shm-size=128g \
    -e UNIFIED_MEMORY=true \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    flymyai-lora-trainer:cuda13 \
    train_4090.py --config train_configs/train_dgx_spark.yaml
```

### Run Inference

```bash
docker run --gpus all \
    -p 8080:8080 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    flymyai-lora-trainer:cuda13 \
    inference.py
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Training images |
| `./checkpoints` | `/app/checkpoints` | Model checkpoints |
| `./output` | `/app/output` | Training outputs |
| `./train_configs` | `/app/train_configs` | Config files |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIFIED_MEMORY` | `false` | Enable unified memory mode |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection |
| `HF_HOME` | `/app/cache/huggingface` | HuggingFace cache |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable | Memory allocator config |

## Image Sizes

| Image Type | Size | Use Case |
|------------|------|----------|
| Development | ~15-20 GB | Debug, development |
| Production | ~10-12 GB | Production training |
| Optimized | ~8-10 GB | CI/CD |

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Ensure daemon.json is configured
cat /etc/docker/daemon.json
```

### Out of Memory

```bash
# Increase shared memory
docker run --shm-size=64g ...

# Use quantized config
docker-compose up rtx4090-trainer
```

### Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Build Fails

```bash
# Clear Docker cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t flymyai-lora-trainer:cuda13 .
```

## Security

The production image (`Dockerfile.prod`):
- Runs as non-root user
- Minimal installed packages
- No build tools
- Regular security scanning with Trivy

Scan for vulnerabilities:
```bash
docker scout cve flymyai-lora-trainer:cuda13
# or
trivy image flymyai-lora-trainer:cuda13
```

## Performance Tips

1. **Use SHM**: Increase `--shm-size` for large models
2. **Pin Memory**: Disable for unified memory systems
3. **Volume Performance**: Use bind mounts for data, named volumes for cache
4. **Multi-GPU**: Set `NVIDIA_VISIBLE_DEVICES` appropriately

## Advanced Usage

### Custom Entry Point

```bash
docker run --gpus all --entrypoint bash \
    flymyai-lora-trainer:cuda13 -c "python -m pytest tests/"
```

### Interactive Shell

```bash
docker run -it --gpus all \
    flymyai-lora-trainer:cuda13 bash
```

### Monitor Training

```bash
# Attach to running container
docker logs -f lora-trainer

# GPU monitoring
docker exec lora-trainer nvidia-smi -l 1
```
```

**Step 2: Commit**

```bash
git add DOCKER.md
git commit -m "docs: add comprehensive Docker setup guide"
```

---

## Task 10: Update Main README

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/README.md`

**Step 1: Add Docker section to README**

Add section:
```markdown
## Docker Support

Run training in isolated, reproducible containers with CUDA 13.0 support.

### Quick Start

```bash
# Check prerequisites
./scripts/check_nvidia_docker.sh

# Build image
./scripts/build_docker.sh

# Run training
docker-compose up lora-trainer
```

### Available Services

- `lora-trainer` - Standard GPU training
- `dgx-spark-trainer` - DGX Spark with unified memory
- `rtx4090-trainer` - RTX 4090 with quantization
- `flux-trainer` - FLUX model training
- `kandinsky-trainer` - Kandinsky model training

See [DOCKER.md](DOCKER.md) for detailed documentation.

### Requirements

- Docker >= 24.0
- NVIDIA Container Toolkit >= 1.16.0
- NVIDIA Driver >= 550.x
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Docker quick start to README"
```

---

## Summary

This plan covers:
1. .dockerignore for efficient builds
2. Multi-stage Dockerfile with CUDA 13.0
3. Docker Compose for multiple scenarios
4. Container validation tests
5. Build and check scripts
6. Production-optimized Dockerfile
7. GitHub Actions CI/CD
8. Comprehensive documentation

Total estimated time: 16-24 hours
Complexity: Medium

Key files created:
- Dockerfile
- Dockerfile.prod
- docker-compose.yml
- .dockerignore
- scripts/build_docker.sh
- scripts/test_container.sh
- scripts/check_nvidia_docker.sh
- .github/workflows/docker.yml
- DOCKER.md

Major features:
- CUDA 13.0 base image
- Unified memory environment variables
- Non-root user in production
- Health checks
- Volume mounting strategy
- Multiple service configurations
- Security scanning in CI
