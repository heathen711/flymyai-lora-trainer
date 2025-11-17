# TODO: Performance & Compatibility Upgrades

This document outlines improvements for fastsafetensors integration, DGX Spark unified memory optimization, and CUDA 13.0 upgrade.

---

## 1. FastSafeTensors Integration

**Goal**: Replace implicit safetensors usage with fastsafetensors for faster model loading and reduced memory overhead.

### 1.1 Dependencies
- [ ] Add `fastsafetensors` to `requirements.txt`
- [ ] Verify compatibility with current `diffusers` version (commit `7a2b78bf`)
- [ ] Test fastsafetensors version compatibility with Python 3.10

### 1.2 Model Loading Optimization
- [ ] **train.py** (line ~47): Replace `QwenImagePipeline.from_pretrained()` with fastsafetensors direct loading
  ```python
  # Current
  flux_transformer = QwenImageTransformer.from_pretrained(...)

  # Target
  from fastsafetensors import SafeTensorsFileLoader
  loader = SafeTensorsFileLoader(path, num_threads=8)
  state_dict = loader.load()
  ```

- [ ] **train_4090.py**: Implement lazy loading for quantized blocks
  ```python
  # Load transformer blocks individually with fastsafetensors
  # Reduces peak memory during initial load
  ```

- [ ] **train_flux_lora.py** (line ~52): FLUX.1-dev transformer loading optimization
- [ ] **train_qwen_edit_lora.py** (line ~81): Control transformer loading
- [ ] **train_kandinsky_lora.py** (line ~63): Kandinsky model loading

### 1.3 Checkpoint Saving/Loading
- [ ] **All training scripts**: Update `save_lora_weights()` calls to use fastsafetensors
  - train.py:330
  - train_4090.py:474
  - train_flux_lora.py:341
  - train_qwen_edit_lora.py:529
  - train_kandinsky_lora.py:386

- [ ] Implement streaming save for large LoRA weights:
  ```python
  from fastsafetensors import save_file
  save_file(state_dict, path, metadata={"format": "pt"})
  ```

- [ ] Add checkpoint sharding support for models >10GB

### 1.4 VAE & Text Encoder Loading
- [ ] **image_datasets/dataset.py**: Pre-cached embedding loading with fastsafetensors
- [ ] **inference.py** (line ~28): Inference pipeline loading optimization
- [ ] Cache loaded tensors with memory mapping for repeated access

### 1.5 Performance Benchmarks
- [ ] Benchmark current loading times for each model type
- [ ] Measure memory usage during loading phase
- [ ] Document speedup after fastsafetensors integration
- [ ] Target: 2-3x faster loading, 30% reduced peak memory

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
| Integration Testing | 12-16 | Medium |
| Documentation | 8-12 | Low |
| **Total** | **80-112 hours** | |

---

## 7. Dependencies & Blockers

- [ ] CUDA 13.0 availability for PyTorch (check pytorch.org nightly builds)
- [ ] fastsafetensors compatibility with diffusers internals
- [ ] DGX Spark hardware access for testing unified memory
- [ ] bitsandbytes CUDA 13.0 binary availability
- [ ] Hopper GPU access for CUDA 13.0 feature testing

---

## Notes

- **fastsafetensors** provides memory-mapped loading and multi-threaded deserialization
- **DGX Spark** uses Grace Hopper architecture with 128GB unified CPU-GPU memory
- **CUDA 13.0** introduces FP8 training, improved Flash Attention, and better memory management
- All changes should maintain backward compatibility with CUDA 12.x and standard GPU systems
