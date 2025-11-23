# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoRA (Low-Rank Adaptation) training implementation for Qwen-Image, Qwen-Image-Edit, and FLUX.1-dev models. The codebase is optimized for multiple GPU architectures including NVIDIA DGX Spark (ARM64 Blackwell GB10) with unified memory support.

## DGX Spark System Context

**Hardware:** NVIDIA DGX Spark with ARM64 Blackwell GB10 (SM 121, compute capability 12.1)
- 20-core heterogeneous CPU (Cortex-X925 + Cortex-A725)
- 119 GiB RAM
- NVIDIA GB10 GPU with CUDA 13.0
- Custom PyTorch wheels required (see ~/.claude/knowledge/dgx-spark-custom-wheels.md)

**Critical Setup Requirements:**
1. **Custom PyTorch Installation:** Must use custom-built wheels with sm_121 support from ~/Documents/DGX-Spark-PyTorch/wheels/
2. **cuDNN 9.x Required:** CUDA 13.0 requires cuDNN 9.x installed on host system
3. **Install Order:** cuDNN → ONNX → PyTorch → SageAttention → requirements.txt
4. **Never mix PyPI PyTorch with custom wheels** - This causes symbol conflicts

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install custom PyTorch wheels (DGX Spark only)
pip install \
  ~/Documents/DGX-Spark-PyTorch/wheels/torch-2.10.0a0+cu130.ubuntu2404.sm121-cp312-cp312-linux_aarch64.whl \
  ~/Documents/DGX-Spark-PyTorch/wheels/triton-3.5.0+git4ccf8a95-cp312-cp312-linux_aarch64.whl \
  ~/Documents/DGX-Spark-PyTorch/wheels/torchvision-0.25.0a0+617079d-cp312-cp312-linux_aarch64.whl \
  ~/Documents/DGX-Spark-PyTorch/wheels/torchaudio-2.10.0a0+ee1a135-cp312-cp312-linux_aarch64.whl

# Install project dependencies
pip install -r requirements.txt

# Verify sm_121 support (DGX Spark)
python -c "import torch; print(f'sm_121: {\"sm_121\" in torch.cuda.get_arch_list()}')"
```

### Training Commands

**Qwen-Image LoRA:**
```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

**Qwen-Image Full Training:**
```bash
accelerate launch train_full_qwen_image.py --config ./train_configs/train_full_qwen_image.yaml
```

**Qwen-Image-Edit LoRA:**
```bash
accelerate launch train_qwen_edit_lora.py --config ./train_configs/train_lora_qwen_edit.yaml
```

**FLUX.1-dev LoRA:**
```bash
accelerate launch train_flux_lora.py --config ./train_configs/train_flux_config.yaml
```

**DGX Spark Optimized (24GB VRAM constraint):**
```bash
accelerate launch train_4090.py --config ./train_configs/train_dgx_spark.yaml
```

**ARM64 Optimized (DGX Spark):**
```bash
python train.py --config train_configs/train_aarch64_optimized.yaml
```

### Dataset Validation
```bash
python utils/validate_dataset.py --path path/to/your/dataset
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_unified_memory.py -v

# Run with coverage
pytest --cov=utils tests/
```

## Training Configuration Architecture

### Configuration File Hierarchy

All training configs are YAML files in `train_configs/` with a common structure but different optimization profiles:

1. **train_lora.yaml** - Standard LoRA training (high-VRAM GPUs)
2. **train_lora_4090.yaml** - Memory-optimized for 24GB GPUs (quantization + 8-bit Adam)
3. **train_dgx_spark.yaml** - DGX Spark unified memory optimized
4. **train_aarch64_optimized.yaml** - ARM64 specific optimizations
5. **train_cuda13_optimized.yaml** - CUDA 13.0 feature enablement
6. **train_flux_config.yaml** - FLUX model specific settings

### Critical Configuration Options

#### Memory Management Options

**unified_memory** (bool, default: false)
- **Purpose:** Enable unified CPU-GPU memory mode (DGX Spark/Grace Hopper)
- **When to use:** Only on Grace Hopper (sm_90+) or Blackwell (sm_121+) systems with unified memory
- **Implications:**
  - Disables CPU offloading (counterproductive with unified memory)
  - Disables pin_memory (not needed)
  - Keeps models resident in unified memory pool
  - Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - Configures 90% memory fraction for training
- **Config location:** train_dgx_spark.yaml (true), train_aarch64_optimized.yaml (true)
- **Implementation:** utils/unified_memory.py:is_unified_memory_system()

**disable_cpu_offload** (bool, default: false)
- **Purpose:** Prevent moving models to CPU during training
- **When to use:** Systems with sufficient GPU/unified memory (>80GB)
- **Implications:**
  - Text encoder stays on GPU
  - VAE stays on GPU
  - Faster training (no transfer overhead)
  - Higher memory usage
- **Config location:** All configs support this; enabled in train_dgx_spark.yaml

**pin_memory** (bool, default: true)
- **Purpose:** Use pinned memory for DataLoader
- **When to use:** Standard discrete GPU systems
- **When to disable:** Unified memory systems (DGX Spark, Grace Hopper)
- **Implications:**
  - True: Faster CPU-GPU transfers (standard systems)
  - False: Required for unified memory (no transfers needed)
- **Config location:** Controlled by unified_memory setting

**disable_quantization** (bool, default: false)
- **Purpose:** Skip model quantization
- **When to use:** Systems with 128GB+ unified memory or 80GB+ VRAM
- **Implications:**
  - Full precision models (higher quality)
  - 3-4x more memory required
  - Faster inference (no dequantization overhead)
- **Config location:** train_dgx_spark.yaml (true)

**quantize** (bool, default: false)
- **Purpose:** Enable 8-bit quantization via optimum.quanto
- **When to use:** 24GB VRAM systems (RTX 4090, RTX 3090)
- **Implications:**
  - Reduces memory by ~3-4x
  - Slight quality degradation (negligible for LoRA)
  - Required for FLUX on 24GB GPUs
- **Config location:** train_lora_4090.yaml (true)

**adam8bit** (bool, default: false)
- **Purpose:** Use 8-bit Adam optimizer (bitsandbytes)
- **When to use:** Memory-constrained systems (<32GB VRAM)
- **Implications:**
  - Reduces optimizer state memory by 75%
  - Minimal impact on convergence
  - Enables training on 24GB GPUs
- **Config location:** train_lora_4090.yaml (true)

**disable_gradient_checkpointing** (bool, default: false)
- **Purpose:** Skip gradient checkpointing
- **When to use:** Systems with abundant memory (128GB+ unified, 80GB+ VRAM)
- **Implications:**
  - Faster training (no recomputation)
  - Higher memory usage
  - DGX Spark can disable due to 128GB unified memory
- **Config location:** train_dgx_spark.yaml (true)

#### CUDA 13.0 Feature Flags

**cuda_features.enable_flash_attention_3** (bool)
- **Purpose:** Enable Flash Attention 3 (Hopper/Blackwell only)
- **Requirements:** CUDA 13.0 + sm_90 or higher
- **WARNING:** Unstable on ARM64 with GCC13 (see docs/AARCH64_SBSA_COMPATIBILITY.md)
- **Implications:**
  - 2-4x faster attention
  - Lower memory usage
  - Known issues with head_dim in (192, 224] on sm_121
- **Config location:** train_dgx_spark.yaml (true), train_aarch64_optimized.yaml (false - unstable)

**cuda_features.enable_fp8_training** (bool)
- **Purpose:** Enable native FP8 training (Hopper/Blackwell)
- **Requirements:** CUDA 13.0 + sm_90+
- **Implications:**
  - 2x faster compute
  - Experimental, test stability first
  - Not needed for most workloads (bf16 sufficient)
- **Config location:** train_dgx_spark.yaml (false), train_cuda13_optimized.yaml (false)

**cuda_features.enable_tf32_compute** (bool)
- **Purpose:** Enable TensorFloat-32 compute (Ampere+)
- **Requirements:** sm_80+ (A100, RTX 30xx/40xx, Hopper, Blackwell)
- **Implications:**
  - ~3x faster matmul operations
  - Minimal precision impact
  - Safe to enable on all Ampere+ GPUs
- **Config location:** All configs (true for sm_80+)

**cuda_features.enable_cudnn_sdp** (bool)
- **Purpose:** Use cuDNN scaled dot-product attention
- **When to use:** Alternative to Flash Attention (more stable on ARM64)
- **Implications:**
  - Stable fallback when FA3 is unstable
  - Good performance on ARM64 platforms
- **Config location:** train_dgx_spark.yaml (true), train_aarch64_optimized.yaml (true)

#### Data Processing Options

**precompute_text_embeddings** (bool, default: true)
- **Purpose:** Pre-encode text prompts before training
- **Implications:**
  - Faster training (no repeated encoding)
  - Higher initial memory usage
  - Recommended for all configs

**precompute_image_embeddings** (bool, default: true)
- **Purpose:** Pre-encode images via VAE
- **Implications:**
  - Faster training loop
  - Requires more memory
  - Can be disabled on memory-constrained systems

**save_cache_on_disk** (bool, default: true)
- **Purpose:** Save precomputed embeddings to disk
- **When to disable:** Unified memory systems (keep in memory)
- **Implications:**
  - True: Saves memory, slower first epoch
  - False: Faster but higher memory (good for unified memory)
- **Config location:** train_dgx_spark.yaml (false), others (true)

#### FastSafeTensors Options

**use_fastsafetensors** (bool, default: true)
- **Purpose:** Use optimized safetensors loading
- **Implications:**
  - 2-3x faster checkpoint loading
  - Automatic fallback if not available
- **Config location:** All configs (true)

**fastsafetensors_num_threads** (int, default: 8)
- **Purpose:** Parallel threads for safetensors I/O
- **Tuning:** Match CPU core count (DGX Spark: 8-12 threads optimal)

### Configuration Decision Tree for DGX Spark

```
Is this DGX Spark (sm_121)?
├─ Yes → Use train_dgx_spark.yaml or train_aarch64_optimized.yaml
│   ├─ unified_memory: true
│   ├─ disable_cpu_offload: true
│   ├─ pin_memory: false
│   ├─ disable_quantization: true (128GB unified memory)
│   ├─ disable_gradient_checkpointing: true
│   ├─ cuda_features.enable_flash_attention_3: false (unstable on ARM64)
│   └─ cuda_features.enable_cudnn_sdp: true (stable alternative)
│
└─ No → Is VRAM >= 80GB?
    ├─ Yes → Use train_lora.yaml (standard config)
    │   └─ All optimizations disabled
    │
    └─ No → Is VRAM == 24GB?
        ├─ Yes → Use train_lora_4090.yaml
        │   ├─ quantize: true
        │   ├─ adam8bit: true
        │   └─ save_cache_on_disk: true
        │
        └─ No → Insufficient VRAM, need 24GB minimum
```

## Architecture Overview

### Training Scripts (Entry Points)

- **train.py** - Standard Qwen-Image LoRA training (high-VRAM)
- **train_4090.py** - Memory-optimized for 24GB GPUs with quantization
- **train_flux_lora.py** - FLUX model LoRA training
- **train_qwen_edit_lora.py** - Qwen-Image-Edit control-based training
- **train_full_qwen_image.py** - Full model fine-tuning (not LoRA)

All scripts support:
1. Unified memory detection and configuration (utils/unified_memory.py)
2. CUDA feature detection (utils/cuda_utils.py)
3. Memory monitoring (utils/memory_monitor.py)
4. FastSafeTensors integration (utils/fast_loading.py)

### Key Utilities

**utils/unified_memory.py**
- Auto-detects Grace Hopper/Blackwell unified memory systems
- Configures PyTorch memory allocator for unified memory
- Provides memory configuration presets (pin_memory, num_workers, etc.)
- Called at training script initialization when unified_memory: true

**utils/cuda_utils.py**
- Detects CUDA version and compute capability
- Feature support checking (FA3, FP8, TF32, FA2)
- Provides optimal settings based on GPU architecture
- Version comparison utilities

**utils/memory_monitor.py**
- Tracks GPU memory allocation/reserved/peak
- Logs memory usage at training checkpoints
- Automatic monitoring when unified_memory: true

**utils/fast_loading.py**
- Optimized safetensors loading with fastsafetensors
- Automatic fallback to standard safetensors
- Configurable thread count for parallel I/O

### Dataset Structure

**Standard Training (Qwen/FLUX):**
```
dataset/
├── img1.png
├── img1.txt
├── img2.jpg
├── img2.txt
└── ...
```

**Qwen-Image-Edit (Control-based):**
```
dataset/
├── images/           # Target images + captions
│   ├── image_001.jpg
│   ├── image_001.txt
│   └── ...
└── control/          # Control images (same names)
    ├── image_001.jpg
    └── ...
```

## Important Patterns

### Conditional CPU Offloading Pattern

All training scripts implement conditional offloading:

```python
# After text encoding
if not getattr(args, 'unified_memory', False):
    text_encoding_pipeline.to("cpu")
    torch.cuda.empty_cache()
else:
    logger.info("Unified memory: keeping text encoding pipeline on device")
```

This pattern appears in:
- Text encoder offloading (after embedding precomputation)
- VAE offloading (after image encoding)
- Transformer block offloading (during quantization)

### Unified Memory Initialization Pattern

Training scripts follow this initialization sequence:

```python
# 1. Load config
args = OmegaConf.load(config_path)

# 2. Setup unified memory environment (if enabled)
if getattr(args, 'unified_memory', False):
    setup_unified_memory_env()
    logger.info("Unified memory mode enabled")

# 3. Initialize accelerator
accelerator = Accelerator(...)

# 4. Conditional memory monitoring
if getattr(args, 'unified_memory', False):
    reset_peak_memory_stats()
    log_memory_usage("before_training")
```

### Quantization Pattern

For memory-constrained systems (train_4090.py):

```python
if args.quantize and not getattr(args, 'disable_quantization', False):
    # Quantize transformer blocks
    for block in tqdm(all_blocks):
        block.to(device, dtype=torch_dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        if not getattr(args, 'unified_memory', False):
            block.to('cpu')  # Only offload if not unified memory
```

## DGX Spark Specific Considerations

### Known Limitations

1. **Flash Attention 3 on ARM64:** Unstable with GCC13 on CUDA 13.0 + aarch64. Use cuDNN SDP instead.
2. **Head Dimension Constraints on sm_121:** FA with gradients fails for head_dim in (192, 224] or >224 with dropout.
3. **Custom Wheels Required:** PyPI PyTorch lacks sm_121 kernels. Always use custom wheels from ~/Documents/DGX-Spark-PyTorch/.

### Verification Commands

```bash
# Check CUDA/GPU configuration
python -c "from utils.cuda_utils import get_optimal_settings; import json; print(json.dumps(get_optimal_settings(), indent=2))"

# Check unified memory detection
python -c "from utils.unified_memory import is_unified_memory_system, get_memory_config; print(f'Unified: {is_unified_memory_system()}'); import json; print(json.dumps(get_memory_config(), indent=2))"

# Verify PyTorch sm_121 support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'sm_121: {\"sm_121\" in torch.cuda.get_arch_list()}')"

# Check cuDNN installation
ldconfig -p | grep cudnn
```

### Memory Optimization Guidelines for DGX Spark

With 128GB unified memory, the optimal configuration is:
- **No quantization:** Full precision models
- **No CPU offloading:** Keep all models on device
- **No gradient checkpointing:** Abundant memory allows full computation graphs
- **Disable pin_memory:** Unified memory makes pinning unnecessary
- **Lower num_workers:** Less data loading parallelism needed (4 vs 8)
- **save_cache_on_disk: false:** Keep precomputed embeddings in unified memory

This configuration maximizes training speed by eliminating all memory-saving techniques that add overhead.

## Common Issues and Solutions

### PyPI PyTorch Overwrites Custom Wheels

**Problem:** Installing requirements.txt replaces custom PyTorch with PyPI version.

**Solution:**
```bash
pip uninstall torch torchvision torchaudio triton -y
pip install ~/Documents/DGX-Spark-PyTorch/wheels/torch-*.whl \
            ~/Documents/DGX-Spark-PyTorch/wheels/triton-*.whl \
            ~/Documents/DGX-Spark-PyTorch/wheels/torchvision-*.whl \
            ~/Documents/DGX-Spark-PyTorch/wheels/torchaudio-*.whl
```

### cuDNN Library Not Found

**Problem:** "libcudnn.so.9: cannot open shared object file"

**Solution:** Install cuDNN 9.x for CUDA 13.0 (see ~/.claude/knowledge/dgx-spark-custom-wheels.md)

### Flash Attention 3 Crashes on ARM64

**Problem:** FA3 causes CUDA errors on DGX Spark.

**Solution:** Use train_aarch64_optimized.yaml with enable_flash_attention_3: false and enable_cudnn_sdp: true.

## File Naming Conventions

- **Training scripts:** `train_*.py` (main entry points)
- **Config files:** `train_configs/train_*.yaml`
- **Utilities:** `utils/*.py` (helper modules)
- **Datasets:** `image_datasets/*.py` (dataset loaders)
- **Tests:** `tests/test_*.py` (pytest test files)

## Documentation References

- **DGX Spark Setup:** ~/.claude/knowledge/dgx-spark-custom-wheels.md
- **System Info:** ~/.claude/knowledge/system.md
- **GPU Compatibility:** docs/GPU_COMPATIBILITY.md
- **ARM64 Compatibility:** docs/AARCH64_SBSA_COMPATIBILITY.md
- **Unified Memory Plan:** docs/plans/2025-11-16-dgx-spark-unified-memory.md (completed)
- **FastSafeTensors:** docs/FASTSAFETENSORS_INTEGRATION.md
- Not allowed to run the training, only the user can run training