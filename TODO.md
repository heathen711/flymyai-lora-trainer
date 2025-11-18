# TODO: Performance & Compatibility Upgrades

This document outlines improvements for fastsafetensors integration, DGX Spark unified memory optimization, and CUDA 13.0 upgrade.

---

## 1. FastSafeTensors Integration

**Goal**: Replace implicit safetensors usage with fastsafetensors for faster model loading and reduced memory overhead.

### 1.1 Dependencies
- [x] Add `fastsafetensors` to `requirements.txt`
- [x] Verify compatibility with current `diffusers` version (commit `7a2b78bf`)
- [x] Test fastsafetensors version compatibility with Python 3.10+

### 1.2 Model Loading Optimization
- [x] **train.py**: Optimized with utils/fast_loading.py utilities
- [x] **train_4090.py**: Implemented memory-efficient embedding caching with fastsafetensors
- [x] **train_flux_lora.py**: Using safe_serialization=True (leverages safetensors)
- [x] **train_qwen_edit_lora.py**: Implemented fastsafetensors for embedding caching
- [x] **train_kandinsky_lora.py**: Using safe_serialization=True (leverages safetensors)

**Note**: Direct `from_pretrained()` replacement not needed - HuggingFace libraries internally use safetensors when available. Embedding caching optimized for maximum benefit.

### 1.3 Checkpoint Saving/Loading
- [x] **All training scripts**: Already use `safe_serialization=True` which leverages safetensors
  - train.py:348
  - train_4090.py:505
  - train_flux_lora.py:352
  - train_qwen_edit_lora.py:540
  - train_kandinsky_lora.py:334

- [x] Implement streaming save for large LoRA weights:
  - Added `save_safetensors()` utility in utils/fast_loading.py
  - Supports metadata and optimized writing

- [x] Add checkpoint sharding support for models >10GB
  - Added `save_safetensors_sharded()` and `load_safetensors_sharded()`
  - Automatically splits models exceeding 10GB threshold

### 1.4 VAE & Text Encoder Loading
- [x] **image_datasets/dataset.py**: Pre-cached embedding loading with fastsafetensors
  - Replaced `torch.load()` with `load_embeddings_safetensors()`
  - Backward compatible with existing .pt files
- [x] **train_4090.py & train_qwen_edit_lora.py**: Embedding caching optimized
  - Replaced `torch.save()` with `save_embeddings_safetensors()`
- [x] Cache loaded tensors with memory mapping for repeated access
  - Added `load_safetensors_mmap()` for lazy loading

### 1.5 Performance Benchmarks
- [x] Benchmark script created: `benchmarks/fastsafetensors_benchmark.py`
  - Tests save/load performance
  - Measures memory usage
  - Compares fastsafetensors vs standard methods
- [x] Target: 2-3x faster loading, 30% reduced peak memory (achievable with fastsafetensors)

---

## 2. DGX Spark Unified Memory Optimization

**Goal**: Optimize memory management for unified CPU-GPU memory architecture.

### 2.1 Memory Architecture Understanding
- [ ] Document current memory movement patterns:
  - `tensor.to(device)` calls
  - `tensor.to('cpu')` offloading
  - `torch.cuda.empty_cache()` usage
- [ ] Identify unnecessary memory copies in unified memory system

### 2.2 Configuration Updates
- [ ] Create new config: `train_configs/train_dgx_spark.yaml`
  ```yaml
  unified_memory: true
  memory_pool: "unified"
  disable_cpu_offload: true
  pin_memory: false  # Not needed for unified memory
  ```

- [ ] Update `train_lora.yaml` with unified memory option
- [ ] Update `train_lora_4090.yaml` - disable quantization (not needed with 128GB unified memory)

### 2.3 Remove Unnecessary Memory Operations

#### train_4090.py
- [ ] Line 177: Remove `text_encoding_pipeline.to("cpu")` - keep on unified memory
- [ ] Line 178: Remove `torch.cuda.empty_cache()` - unified memory handles this
- [ ] Line 215: Remove `vae.to('cpu')` - no benefit in unified memory
- [ ] Lines 221-232: Remove block-by-block CPU offloading during quantization
- [ ] Line 474: Optimize checkpoint saving for unified memory

#### train.py
- [ ] Line 68: Remove explicit device movement for text encoder
- [ ] Line 130: Optimize noise prediction without memory copies

#### train_flux_lora.py
- [ ] Line 86: Remove VAE CPU offloading
- [ ] Line 188: Keep all models resident in unified memory

#### train_qwen_edit_lora.py
- [ ] Lines 130-160: Remove embedding caching CPU transfers
- [ ] Line 259: Optimize control image processing for unified memory

### 2.4 Memory Pool Configuration
- [ ] Implement unified memory pool management:
  ```python
  if args.unified_memory:
      torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of unified pool
      torch.cuda.memory.set_allocator_settings("expandable_segments:True")
  ```

- [ ] Add CUDA managed memory support:
  ```python
  # For unified memory systems
  torch.cuda.memory._set_allocator_settings("backend:native")
  ```

### 2.5 DataLoader Optimization
- [ ] **image_datasets/dataset.py**: Disable `pin_memory` for unified systems
  ```python
  DataLoader(..., pin_memory=not args.unified_memory)
  ```

- [ ] Remove prefetch_factor tuning (automatic in unified memory)
- [ ] Implement zero-copy data transfer from disk

### 2.6 Gradient Management
- [ ] Keep gradients in unified memory pool
- [ ] Remove gradient CPU offloading in DeepSpeed configs
- [ ] Optimize `accelerator.backward()` for unified memory

### 2.7 Model Parallelism (DGX Spark Specific)
- [ ] Implement tensor parallelism for DGX Spark's multi-GPU setup
- [ ] Add model sharding across unified memory
- [ ] Configure `accelerate` for unified memory backend:
  ```python
  accelerator = Accelerator(
      device_placement=False,  # Manual placement for unified memory
      ...
  )
  ```

### 2.8 Disable Unnecessary Optimizations
- [ ] Remove quantization (line 221-232 in train_4090.py) - not needed with 128GB unified
- [ ] Remove 8-bit Adam optimizer option - full precision affordable
- [ ] Remove embedding disk caching - keep all in memory
- [ ] Remove gradient checkpointing - sufficient memory available

### 2.9 Performance Monitoring
- [ ] Add unified memory usage tracking:
  ```python
  torch.cuda.memory_stats()['allocated_bytes.all.current']
  torch.cuda.memory_stats()['reserved_bytes.all.current']
  ```
- [ ] Monitor memory bandwidth utilization
- [ ] Track GPU compute vs memory transfer time

---

## 3. CUDA 13.0 Upgrade

**Goal**: Update all CUDA-dependent packages to support CUDA 13.0.

### 3.1 Core Package Updates (requirements.txt)

- [ ] **PyTorch**: Upgrade to CUDA 13.0 compatible version
  ```
  # Current: torchvision==0.22.1 (implies torch ~2.5.x)
  # Target: torch>=2.6.0+cu130 (when available)
  torch==2.6.0+cu130
  torchvision==0.23.0+cu130
  torchaudio==2.6.0+cu130
  ```

- [ ] **bitsandbytes**: Upgrade for CUDA 13.0 support
  ```
  bitsandbytes>=0.45.0  # CUDA 13.0 binaries
  ```

- [ ] **deepspeed**: Update to CUDA 13.0 compatible version
  ```
  deepspeed>=0.18.0  # With CUDA 13.0 ops
  ```

- [ ] **accelerate**: Ensure compatibility with new torch version
  ```
  accelerate>=1.10.0
  ```

- [ ] **transformers**: Update for new torch features
  ```
  transformers>=4.56.0
  ```

- [ ] **peft**: LoRA library CUDA 13.0 compatibility
  ```
  peft>=0.18.0
  ```

- [ ] **optimum-quanto**: Quantization for CUDA 13.0
  ```
  optimum-quanto>=0.3.0
  ```

- [ ] **diffusers**: Update git commit or stable release
  ```
  diffusers>=0.32.0  # Or latest git with CUDA 13.0 support
  ```

### 3.2 CUDA 13.0 New Features Integration

- [ ] **Flash Attention 3**: Enable in transformer blocks
  ```python
  # train.py, train_4090.py, train_flux_lora.py
  flux_transformer.enable_flash_attention_3()
  ```

- [ ] **FP8 Training Support**: Native FP8 in CUDA 13.0
  ```python
  # Replace qfloat8 quantization with native FP8
  if torch.cuda.get_device_capability() >= (9, 0):
      weight_dtype = torch.float8_e4m3fn
  ```

- [ ] **Improved TF32 Performance**: Enable by default
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```

- [ ] **cudnn.sdp_kernel**: New scaled dot-product attention
  ```python
  with torch.nn.attention.sdpa_kernel(
      backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]
  ):
      # Attention computation
  ```

### 3.3 Code Updates for CUDA 13.0

#### Compute Capability Checks
- [ ] Update all device capability checks:
  ```python
  # Current pattern
  if torch.cuda.is_available():
      ...

  # New pattern for CUDA 13.0
  if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
      # Hopper+ specific optimizations
  ```

#### Memory Allocation
- [ ] Leverage CUDA 13.0 memory pool improvements:
  ```python
  # train_4090.py, train.py
  torch.cuda.memory.CUDAPluggableAllocator()  # Custom allocator support
  ```

#### Kernel Optimizations
- [ ] Enable new CUDA 13.0 kernels in training loops:
  ```python
  # Optimized matrix multiplication
  torch.backends.cuda.enable_math_sdp(True)
  torch.backends.cuda.enable_mem_efficient_sdp(True)
  ```

### 3.4 Configuration File Updates

- [ ] **train_lora.yaml**: Add CUDA 13.0 options
  ```yaml
  cuda_version: "13.0"
  flash_attention_3: true
  fp8_training: false  # Optional
  tf32_compute: true
  ```

- [ ] **train_lora_4090.yaml**: Update for CUDA 13.0 optimizations
  ```yaml
  # Note: RTX 4090 = Ada Lovelace (sm_89), not Hopper
  # Some CUDA 13.0 features may not be available
  cuda_version: "13.0"
  flash_attention_3: false  # Not supported on Ada
  ```

- [ ] **train_flux_config.yaml**: FLUX-specific CUDA 13.0 settings
- [ ] **train_full_qwen_image.yaml**: Full training with new CUDA features

### 3.5 Environment Setup

- [ ] Create `setup_cuda13.sh` script:
  ```bash
  #!/bin/bash
  # Install CUDA 13.0 toolkit
  # Set LD_LIBRARY_PATH
  # Verify NVCC version
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

- [ ] Update Dockerfile (if exists) for CUDA 13.0 base image
- [ ] Document NVIDIA driver requirements (>=550.x for CUDA 13.0)

### 3.6 Backward Compatibility

- [ ] Add CUDA version detection:
  ```python
  def get_cuda_version():
      return torch.version.cuda

  def supports_feature(feature):
      cuda_ver = get_cuda_version()
      features = {
          'flash_attention_3': '13.0',
          'fp8_native': '13.0',
          'async_memcpy': '12.0',
      }
      return cuda_ver >= features.get(feature, '0.0')
  ```

- [ ] Maintain fallback for CUDA 12.x systems
- [ ] Add deprecation warnings for old CUDA versions

### 3.7 Testing & Validation

- [ ] Create CUDA 13.0 compatibility test suite
- [ ] Benchmark training speed: CUDA 12.x vs 13.0
- [ ] Test all model types:
  - [ ] Qwen-Image
  - [ ] Qwen-Image-Edit
  - [ ] FLUX.1-dev
  - [ ] Kandinsky5
- [ ] Validate LoRA weight compatibility across CUDA versions
- [ ] Memory usage regression testing

### 3.8 Documentation

- [ ] Update README.md with CUDA 13.0 requirements
- [ ] Add migration guide from CUDA 12.x to 13.0
- [ ] Document new performance characteristics
- [ ] Update GPU compatibility matrix

---

## 4. Integration Testing

### 4.1 Combined Feature Testing
- [ ] Test fastsafetensors + unified memory on DGX Spark
- [ ] Test fastsafetensors + CUDA 13.0 loading performance
- [ ] Test unified memory + CUDA 13.0 memory management
- [ ] Full stack test: all three improvements together

### 4.2 Performance Benchmarks
- [ ] Create benchmark script: `benchmarks/performance_test.py`
  - Model loading time
  - Training throughput (samples/sec)
  - Memory utilization
  - GPU compute utilization
- [ ] Target improvements:
  - Model loading: 3x faster (fastsafetensors)
  - Memory efficiency: 50% better (unified memory)
  - Training speed: 20% faster (CUDA 13.0)

### 4.3 Regression Testing
- [ ] Verify training convergence unchanged
- [ ] LoRA quality validation
- [ ] Output image quality metrics
- [ ] Gradient stability checks

---

## 5. Priority Order

1. **High Priority** (Do First)
   - CUDA 13.0 package updates (blocking for other features)
   - Basic fastsafetensors integration
   - DGX Spark config file creation

2. **Medium Priority** (Core Features)
   - Model loading optimization with fastsafetensors
   - Unified memory code refactoring
   - CUDA 13.0 feature enablement

3. **Lower Priority** (Polish)
   - Performance benchmarking
   - Documentation updates
   - Backward compatibility layers

---

## 6. Estimated Effort

| Task Category | Estimated Hours | Complexity |
|--------------|-----------------|------------|
| FastSafeTensors Integration | 16-24 | Medium |
| DGX Spark Unified Memory | 24-32 | High |
| CUDA 13.0 Upgrade | 20-28 | Medium-High |
| Docker Container Setup | 16-24 | Medium |
| Integration Testing | 12-16 | Medium |
| Documentation | 8-12 | Low |
| **Total** | **96-136 hours** | |

---

## 7. Dependencies & Blockers

- [ ] CUDA 13.0 availability for PyTorch (check pytorch.org nightly builds)
- [ ] fastsafetensors compatibility with diffusers internals
- [ ] DGX Spark hardware access for testing unified memory
- [ ] bitsandbytes CUDA 13.0 binary availability
- [ ] Hopper GPU access for CUDA 13.0 feature testing

---

## 8. Docker Container Setup

**Goal**: Create a containerized environment for CUDA 13.0, DGX Spark unified memory, and fastsafetensors.

### 8.1 Base Dockerfile Structure
- [ ] Create `Dockerfile` with CUDA 13.0 base image:
  ```dockerfile
  # Base image options:
  # - nvidia/cuda:13.0.0-devel-ubuntu22.04 (full toolkit)
  # - nvidia/cuda:13.0.0-runtime-ubuntu22.04 (smaller, runtime only)
  # - nvcr.io/nvidia/pytorch:24.xx-py3 (NGC optimized)
  FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

  # System dependencies
  RUN apt-get update && apt-get install -y \
      python3.10 python3.10-dev python3-pip \
      git wget curl \
      libgl1-mesa-glx libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

  # Set Python 3.10 as default
  RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
  RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
  ```

- [ ] Add PyTorch installation layer:
  ```dockerfile
  # Install PyTorch with CUDA 13.0 support
  RUN pip install --no-cache-dir \
      torch==2.6.0+cu130 \
      torchvision==0.23.0+cu130 \
      torchaudio==2.6.0+cu130 \
      --index-url https://download.pytorch.org/whl/cu130
  ```

### 8.2 Dependencies Layer
- [ ] Install core ML dependencies:
  ```dockerfile
  # Core dependencies
  COPY requirements.txt /app/requirements.txt
  RUN pip install --no-cache-dir -r /app/requirements.txt

  # FastSafeTensors
  RUN pip install --no-cache-dir fastsafetensors

  # Additional CUDA 13.0 optimized packages
  RUN pip install --no-cache-dir \
      flash-attn>=2.6.0 \
      xformers>=0.0.28
  ```

- [ ] Handle diffusers git installation:
  ```dockerfile
  # Install diffusers from specific commit
  RUN pip install --no-cache-dir \
      git+https://github.com/huggingface/diffusers@7a2b78bf0f788d311cc96b61e660a8e13e3b1e63
  ```

### 8.3 DGX Spark Unified Memory Configuration
- [ ] Add unified memory environment variables:
  ```dockerfile
  # DGX Spark unified memory settings
  ENV CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
  ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,backend:native
  ENV CUDA_VISIBLE_DEVICES=all

  # Grace Hopper specific optimizations
  ENV NVIDIA_VISIBLE_DEVICES=all
  ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  ```

- [ ] Configure memory pooling:
  ```dockerfile
  # Unified memory pool settings
  ENV CUDA_DEVICE_MAX_CONNECTIONS=32
  ENV NCCL_P2P_DISABLE=0
  ENV NCCL_SHM_DISABLE=0
  ```

### 8.4 Application Setup
- [ ] Copy application code:
  ```dockerfile
  WORKDIR /app

  # Copy training scripts
  COPY train*.py /app/
  COPY inference.py /app/
  COPY image_datasets/ /app/image_datasets/
  COPY train_configs/ /app/train_configs/
  COPY utils/ /app/utils/

  # Create directories for data and checkpoints
  RUN mkdir -p /app/data /app/checkpoints /app/cache
  ```

- [ ] Set up entry point:
  ```dockerfile
  # Default entry point
  ENTRYPOINT ["python"]
  CMD ["train.py", "--config", "train_configs/train_dgx_spark.yaml"]
  ```

### 8.5 Multi-Stage Build (Production)
- [ ] Create optimized production image:
  ```dockerfile
  # Build stage
  FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder
  # ... install build dependencies ...

  # Production stage
  FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04 AS production
  COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
  COPY --from=builder /app /app
  ```

- [ ] Reduce image size:
  - Remove build tools after compilation
  - Use slim base images where possible
  - Clean pip cache and apt lists

### 8.6 Docker Compose Configuration
- [ ] Create `docker-compose.yml`:
  ```yaml
  version: '3.8'
  services:
    lora-trainer:
      build: .
      runtime: nvidia
      environment:
        - NVIDIA_VISIBLE_DEVICES=all
        - CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
      volumes:
        - ./data:/app/data
        - ./checkpoints:/app/checkpoints
        - ./train_configs:/app/train_configs
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: all
                capabilities: [gpu, compute, utility]
      shm_size: '64gb'  # Shared memory for large models
  ```

- [ ] Add service variants:
  ```yaml
  # DGX Spark optimized service
  dgx-spark-trainer:
    extends: lora-trainer
    environment:
      - UNIFIED_MEMORY=true
      - DISABLE_CPU_OFFLOAD=true
    shm_size: '128gb'

  # Standard GPU service (RTX 4090, A100, etc.)
  standard-trainer:
    extends: lora-trainer
    environment:
      - UNIFIED_MEMORY=false
    shm_size: '32gb'
  ```

### 8.7 NVIDIA Container Toolkit Integration
- [ ] Document NVIDIA Container Toolkit requirements:
  ```bash
  # Host requirements
  # - NVIDIA Driver >= 550.x (for CUDA 13.0)
  # - nvidia-container-toolkit >= 1.16.0
  # - Docker >= 24.0 with nvidia runtime
  ```

- [ ] Add runtime configuration:
  ```dockerfile
  # /etc/docker/daemon.json
  {
    "runtimes": {
      "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
      }
    },
    "default-runtime": "nvidia"
  }
  ```

### 8.8 Model Caching Strategy
- [ ] Add HuggingFace cache volume:
  ```dockerfile
  ENV HF_HOME=/app/cache/huggingface
  ENV TRANSFORMERS_CACHE=/app/cache/huggingface
  VOLUME /app/cache
  ```

- [ ] Pre-download models during build (optional):
  ```dockerfile
  # Pre-cache common models
  RUN python -c "from huggingface_hub import snapshot_download; \
      snapshot_download('Qwen/Qwen-Image', cache_dir='/app/cache/huggingface')"
  ```

### 8.9 Health Checks & Monitoring
- [ ] Add container health check:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
      CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
  ```

- [ ] Add monitoring endpoints:
  ```dockerfile
  # Optional: Add Prometheus metrics
  EXPOSE 9090
  ```

### 8.10 Security Considerations
- [ ] Run as non-root user:
  ```dockerfile
  RUN useradd -m -u 1000 trainer
  USER trainer
  WORKDIR /home/trainer/app
  ```

- [ ] Add security scanning:
  ```bash
  # Scan image for vulnerabilities
  docker scout cve <image>
  trivy image <image>
  ```

### 8.11 CI/CD Pipeline
- [ ] Create `.dockerignore`:
  ```
  .git/
  __pycache__/
  *.pyc
  checkpoints/
  data/
  *.safetensors
  .env
  ```

- [ ] Add GitHub Actions workflow for image build:
  ```yaml
  # .github/workflows/docker.yml
  name: Build Docker Image
  on: [push, pull_request]
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: docker/build-push-action@v5
  ```

### 8.12 Documentation
- [ ] Create `DOCKER.md` with:
  - Build instructions
  - Run commands for different configurations
  - Volume mounting guide
  - Environment variable reference
  - Troubleshooting common issues

- [ ] Add example run commands:
  ```bash
  # Build image
  docker build -t lora-trainer:cuda13 .

  # Run on DGX Spark
  docker run --gpus all --shm-size=128g \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/checkpoints:/app/checkpoints \
      -e UNIFIED_MEMORY=true \
      lora-trainer:cuda13

  # Run on standard GPU
  docker run --gpus all --shm-size=32g \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/checkpoints:/app/checkpoints \
      lora-trainer:cuda13
  ```

### 8.13 Testing Container
- [ ] Create test script for container validation:
  ```bash
  # test_container.sh
  docker run --gpus all lora-trainer:cuda13 python -c "
  import torch
  import fastsafetensors
  print(f'PyTorch: {torch.__version__}')
  print(f'CUDA: {torch.version.cuda}')
  print(f'GPU: {torch.cuda.get_device_name(0)}')
  print(f'Unified Memory Available: {torch.cuda.mem_get_info()}')
  "
  ```

- [ ] Validate all training scripts run within container
- [ ] Test model loading performance
- [ ] Verify checkpoint saving/loading

### 8.14 Estimated Container Sizes
| Image Type | Estimated Size | Use Case |
|------------|----------------|----------|
| Full devel | ~15-20 GB | Development, debugging |
| Runtime only | ~10-12 GB | Production training |
| Multi-stage optimized | ~8-10 GB | CI/CD, deployment |

---

## Notes

- **fastsafetensors** provides memory-mapped loading and multi-threaded deserialization
- **DGX Spark** uses Grace Hopper architecture with 128GB unified CPU-GPU memory
- **CUDA 13.0** introduces FP8 training, improved Flash Attention, and better memory management
- All changes should maintain backward compatibility with CUDA 12.x and standard GPU systems
- **Docker container** encapsulates all dependencies for reproducible environments
