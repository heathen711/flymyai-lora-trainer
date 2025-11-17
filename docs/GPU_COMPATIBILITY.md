# GPU Compatibility Matrix

## Supported GPUs

### Hopper Architecture (Compute Capability 9.0)
- H100 PCIe / SXM
- H200
- DGX H100
- **Grace Hopper (GH200) - aarch64-sbsa**

**Full CUDA 13.0 features:**
- Flash Attention 3 (⚠️ unstable on aarch64)
- Native FP8 training
- TF32 compute
- Unified memory (DGX Spark, GH200)

### Grace Architecture - aarch64-sbsa (ARM64 SBSA)
- **Grace Hopper GH200** (SM90, 480GB unified memory)
- **DGX SPARK** (SM121)
- **Jetson Thor** (SM110)

**CUDA 13.0 features (aarch64-sbsa):**
- TF32 compute ✅
- BF16 training ✅
- cuDNN SDP attention ✅
- Flash Attention 3 ⚠️ (unstable, GCC13 bugs)
- Unified memory ✅

**Installation:** Use `./scripts/install_cuda13_aarch64.sh`
**Config:** Use `train_configs/train_aarch64_optimized.yaml`

See [AARCH64_SBSA_COMPATIBILITY.md](./AARCH64_SBSA_COMPATIBILITY.md) for details.

### Ampere Architecture (Compute Capability 8.0-8.6)
- A100 (8.0)
- A6000 (8.6)
- RTX 3090/3080 (8.6)

**Partial CUDA 13.0 features:**
- TF32 compute
- Flash Attention 2
- BF16 training

### Ada Lovelace Architecture (Compute Capability 8.9)
- RTX 4090/4080 (8.9)
- L40

**Partial CUDA 13.0 features:**
- TF32 compute
- Flash Attention 2
- BF16 training
- Note: NOT Hopper, so no FA3 or FP8

### Turing Architecture (Compute Capability 7.5)
- RTX 2080 Ti
- T4

**Limited features:**
- FP16 training only
- No TF32
- Basic attention only

## Memory Requirements

| GPU | VRAM | Recommended Config |
|-----|------|-------------------|
| H100 80GB | 80GB | train_cuda13_optimized.yaml |
| A100 80GB | 80GB | train_lora.yaml |
| A100 40GB | 40GB | train_lora.yaml |
| RTX 4090 | 24GB | train_lora_4090.yaml (quantized) |
| RTX 3090 | 24GB | train_lora_4090.yaml (quantized) |

## Detecting Your GPU

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

Or in Python:
```python
from utils.cuda_utils import get_optimal_settings
print(get_optimal_settings())
```
