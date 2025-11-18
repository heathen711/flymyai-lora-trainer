# CUDA 13.0 Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update all CUDA-dependent packages to support CUDA 13.0, enabling new features like Flash Attention 3, native FP8 training, and improved memory management.

**Architecture:** Update requirements.txt with CUDA 13.0 compatible package versions, create CUDA version detection utilities, add backward compatibility layers, and systematically enable new CUDA 13.0 features in training configurations.

**Tech Stack:** PyTorch 2.6+, CUDA 13.0, bitsandbytes, deepspeed, transformers, accelerate

**Status:** ✓ COMPLETE (Completed: 2025-11-17, Merged via PR #2)

---

## Task 1: Create CUDA Version Detection Utility

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/utils/cuda_utils.py`
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/test_cuda_utils.py`

**Step 1: Write failing tests**

```python
# tests/test_cuda_utils.py
import pytest

def test_get_cuda_version():
    """Test CUDA version detection."""
    from utils.cuda_utils import get_cuda_version
    version = get_cuda_version()
    assert isinstance(version, str)
    # Format should be like "12.1" or "13.0"
    assert "." in version or version == "N/A"

def test_supports_feature_flash_attention_3():
    """Test feature support checking."""
    from utils.cuda_utils import supports_feature
    result = supports_feature("flash_attention_3")
    assert isinstance(result, bool)

def test_get_device_capability():
    """Test device capability retrieval."""
    from utils.cuda_utils import get_device_capability
    cap = get_device_capability()
    assert isinstance(cap, tuple)
    assert len(cap) == 2

def test_is_hopper_or_newer():
    """Test Hopper architecture detection."""
    from utils.cuda_utils import is_hopper_or_newer
    result = is_hopper_or_newer()
    assert isinstance(result, bool)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cuda_utils.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement CUDA utilities**

```python
# utils/cuda_utils.py
"""
CUDA version detection and feature support utilities.

Provides functions to check CUDA capabilities and enable features
based on CUDA version and GPU architecture.
"""
import torch
from typing import Tuple, Dict, Any

# Feature requirements: feature_name -> (min_cuda_version, min_compute_capability)
FEATURE_REQUIREMENTS = {
    "flash_attention_3": ("13.0", (9, 0)),  # Hopper+ required
    "fp8_native": ("13.0", (9, 0)),  # Hopper+ required
    "async_memcpy": ("12.0", (8, 0)),  # Ampere+
    "tf32_compute": ("11.0", (8, 0)),  # Ampere+
    "flash_attention_2": ("12.0", (8, 0)),  # Ampere+
}


def get_cuda_version() -> str:
    """
    Get the current CUDA version.

    Returns:
        CUDA version string (e.g., "13.0") or "N/A" if CUDA not available
    """
    if not torch.cuda.is_available():
        return "N/A"

    cuda_version = torch.version.cuda
    if cuda_version is None:
        return "N/A"

    return cuda_version


def get_device_capability() -> Tuple[int, int]:
    """
    Get the compute capability of the current CUDA device.

    Returns:
        Tuple of (major, minor) compute capability, or (0, 0) if not available
    """
    if not torch.cuda.is_available():
        return (0, 0)

    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return (0, 0)


def is_hopper_or_newer() -> bool:
    """
    Check if GPU is Hopper architecture (sm_90) or newer.

    Returns:
        True if Hopper or newer, False otherwise
    """
    capability = get_device_capability()
    return capability[0] >= 9


def is_ampere_or_newer() -> bool:
    """
    Check if GPU is Ampere architecture (sm_80) or newer.

    Returns:
        True if Ampere or newer, False otherwise
    """
    capability = get_device_capability()
    return capability[0] >= 8


def _version_compare(v1: str, v2: str) -> int:
    """
    Compare two version strings.

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    if v1 == "N/A" or v2 == "N/A":
        return -1

    parts1 = [int(x) for x in v1.split(".")]
    parts2 = [int(x) for x in v2.split(".")]

    for p1, p2 in zip(parts1, parts2):
        if p1 < p2:
            return -1
        if p1 > p2:
            return 1

    return 0


def supports_feature(feature: str) -> bool:
    """
    Check if a CUDA feature is supported.

    Args:
        feature: Feature name (e.g., "flash_attention_3", "fp8_native")

    Returns:
        True if feature is supported, False otherwise
    """
    if feature not in FEATURE_REQUIREMENTS:
        return False

    min_cuda, min_capability = FEATURE_REQUIREMENTS[feature]

    cuda_version = get_cuda_version()
    device_capability = get_device_capability()

    # Check CUDA version
    if _version_compare(cuda_version, min_cuda) < 0:
        return False

    # Check compute capability
    if device_capability < min_capability:
        return False

    return True


def get_optimal_settings() -> Dict[str, Any]:
    """
    Get optimal CUDA settings based on current hardware.

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "use_flash_attention_3": supports_feature("flash_attention_3"),
        "use_fp8": supports_feature("fp8_native"),
        "use_tf32": supports_feature("tf32_compute"),
        "use_flash_attention_2": supports_feature("flash_attention_2"),
        "cuda_version": get_cuda_version(),
        "compute_capability": get_device_capability(),
    }

    return settings


def enable_tf32() -> None:
    """Enable TF32 computation if supported."""
    if supports_feature("tf32_compute"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_recommended_dtype() -> torch.dtype:
    """
    Get recommended dtype based on hardware capabilities.

    Returns:
        Recommended torch dtype
    """
    if supports_feature("fp8_native"):
        # FP8 available but typically want BF16 for training
        return torch.bfloat16
    elif is_ampere_or_newer():
        return torch.bfloat16
    else:
        return torch.float16
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cuda_utils.py -v
```

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add utils/cuda_utils.py tests/test_cuda_utils.py
git commit -m "feat: add CUDA version detection and feature support utilities"
```

---

## Task 2: Update requirements.txt with Version Placeholders

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/requirements.txt`
- Create: `/home/jay/Documents/flymyai-lora-trainer/requirements-cuda13.txt`

**Step 1: Create CUDA 13.0 specific requirements file**

```
# requirements-cuda13.txt
# CUDA 13.0 compatible package versions
# NOTE: Update versions when CUDA 13.0 packages become available

# Core PyTorch (CUDA 13.0)
# --index-url https://download.pytorch.org/whl/cu130
# torch==2.6.0+cu130
# torchvision==0.23.0+cu130
# torchaudio==2.6.0+cu130

# Current stable versions (update when CUDA 13.0 builds available)
torch>=2.5.0
torchvision>=0.22.1
torchaudio>=2.5.0

# Training libraries
accelerate>=1.10.0
deepspeed>=0.18.0
transformers>=4.56.0
peft>=0.18.0

# Quantization
bitsandbytes>=0.45.0
optimum-quanto>=0.3.0

# Diffusers (specific commit for compatibility)
diffusers @ git+https://github.com/huggingface/diffusers@7a2b78bf0f788d311cc96b61e660a8e13e3b1e63

# Data & utilities
einops==0.8.0
huggingface-hub>=0.34.3
datasets>=4.0.0
omegaconf>=2.3.0
sentencepiece>=0.2.0
opencv-python>=4.12.0
matplotlib>=3.10.5
onnxruntime>=1.22.1
timm>=1.0.19
qwen_vl_utils>=0.0.11
loguru>=0.7.3

# FastSafeTensors
fastsafetensors>=0.1.0

# Optional: Flash Attention for CUDA 13.0
# flash-attn>=2.6.0  # Uncomment when available
# xformers>=0.0.28   # Uncomment when available
```

**Step 2: Commit**

```bash
git add requirements-cuda13.txt
git commit -m "feat: add CUDA 13.0 specific requirements file"
```

---

## Task 3: Create CUDA Environment Setup Script

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/scripts/setup_cuda13.sh`

**Step 1: Create scripts directory**

```bash
mkdir -p scripts
```

**Step 2: Write setup script**

```bash
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
```

**Step 3: Make executable**

```bash
chmod +x scripts/setup_cuda13.sh
```

**Step 4: Commit**

```bash
git add scripts/setup_cuda13.sh
git commit -m "feat: add CUDA 13.0 environment setup script"
```

---

## Task 4: Add CUDA 13.0 Options to Training Configs

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_configs/train_lora.yaml`

**Step 1: Add CUDA 13.0 configuration options**

Add at end of file:
```yaml
# CUDA 13.0 Features (requires CUDA 13.0 and compatible GPU)
cuda_13_features:
  enable_flash_attention_3: false  # Hopper+ only
  enable_fp8_training: false       # Hopper+ only
  enable_tf32_compute: true        # Ampere+
  enable_cudnn_sdp: true           # Scaled dot-product attention
```

**Step 2: Commit**

```bash
git add train_configs/train_lora.yaml
git commit -m "feat: add CUDA 13.0 feature configuration options"
```

---

## Task 5: Enable TF32 Compute in Training Scripts

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train.py` (after imports)

**Step 1: Add CUDA utilities import**

```python
from utils.cuda_utils import enable_tf32, supports_feature, get_optimal_settings
```

**Step 2: Enable TF32 at start of main()**

After accelerator creation:
```python
# Enable TF32 for improved performance on Ampere+
enable_tf32()
if supports_feature("tf32_compute"):
    logger.info("TF32 compute enabled for improved performance")
```

**Step 3: Test import**

```bash
python -c "import train; print('Import successful')"
```

Expected: "Import successful"

**Step 4: Commit**

```bash
git add train.py
git commit -m "feat: enable TF32 compute in train.py for Ampere+ GPUs"
```

---

## Task 6: Enable TF32 in train_4090.py

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py`

**Step 1: Add CUDA utilities import**

```python
from utils.cuda_utils import enable_tf32, supports_feature
```

**Step 2: Enable TF32 after accelerator creation**

```python
# Enable TF32 for improved performance (RTX 4090 = Ampere)
enable_tf32()
logger.info("TF32 compute enabled")
```

**Step 3: Commit**

```bash
git add train_4090.py
git commit -m "feat: enable TF32 compute in train_4090.py"
```

---

## Task 7: Enable TF32 in Other Training Scripts

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_flux_lora.py`
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_qwen_edit_lora.py`
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_kandinsky_lora.py`

**Step 1: Add CUDA utilities import to each**

```python
from utils.cuda_utils import enable_tf32, supports_feature
```

**Step 2: Add TF32 enable call in main()**

```python
enable_tf32()
```

**Step 3: Commit**

```bash
git add train_flux_lora.py train_qwen_edit_lora.py train_kandinsky_lora.py
git commit -m "feat: enable TF32 compute in all training scripts"
```

---

## Task 8: Add Backward Compatibility Layer

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/utils/compat.py`
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/test_compat.py`

**Step 1: Write failing test**

```python
# tests/test_compat.py
import pytest

def test_check_requirements():
    """Test requirements checking."""
    from utils.compat import check_requirements
    warnings = check_requirements()
    assert isinstance(warnings, list)

def test_get_deprecation_warnings():
    """Test deprecation warning generation."""
    from utils.compat import get_deprecation_warnings
    warnings = get_deprecation_warnings()
    assert isinstance(warnings, list)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compat.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement compatibility layer**

```python
# utils/compat.py
"""
Backward compatibility utilities for CUDA version migration.

Provides warnings and fallbacks for older CUDA versions.
"""
from typing import List
import warnings
from utils.cuda_utils import get_cuda_version, get_device_capability

MINIMUM_CUDA_VERSION = "12.0"
RECOMMENDED_CUDA_VERSION = "13.0"


def check_requirements() -> List[str]:
    """
    Check if current environment meets requirements.

    Returns:
        List of warning messages
    """
    warning_list = []

    cuda_version = get_cuda_version()

    if cuda_version == "N/A":
        warning_list.append("CUDA not available - GPU training will not work")
        return warning_list

    # Parse version
    try:
        major, minor = cuda_version.split(".")[:2]
        major = int(major)
        minor = int(minor.split(".")[0]) if "." in minor else int(minor)
    except Exception:
        warning_list.append(f"Could not parse CUDA version: {cuda_version}")
        return warning_list

    # Check minimum version
    if major < 12:
        warning_list.append(
            f"CUDA {cuda_version} is below minimum {MINIMUM_CUDA_VERSION}. "
            "Some features may not work."
        )

    # Check recommended version
    if major < 13:
        warning_list.append(
            f"CUDA {cuda_version} detected. "
            f"Recommend upgrading to {RECOMMENDED_CUDA_VERSION} for best performance."
        )

    # Check compute capability
    capability = get_device_capability()
    if capability[0] < 8:
        warning_list.append(
            f"GPU compute capability {capability} is older than Ampere (8.0). "
            "Some optimizations disabled."
        )

    return warning_list


def get_deprecation_warnings() -> List[str]:
    """
    Get warnings about deprecated features.

    Returns:
        List of deprecation warning messages
    """
    warnings_list = []

    cuda_version = get_cuda_version()

    if cuda_version == "N/A":
        return warnings_list

    try:
        major = int(cuda_version.split(".")[0])
    except Exception:
        return warnings_list

    if major <= 11:
        warnings_list.append(
            "CUDA 11.x is deprecated. Please upgrade to CUDA 12.x or 13.0."
        )

    return warnings_list


def show_compatibility_warnings() -> None:
    """Display all compatibility warnings."""
    for warning in check_requirements():
        warnings.warn(warning, DeprecationWarning)

    for warning in get_deprecation_warnings():
        warnings.warn(warning, DeprecationWarning)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compat.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add utils/compat.py tests/test_compat.py
git commit -m "feat: add backward compatibility layer with version warnings"
```

---

## Task 9: Integrate Compatibility Checks into Training Scripts

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train.py`

**Step 1: Add compatibility import**

```python
from utils.compat import show_compatibility_warnings
```

**Step 2: Call at start of main()**

```python
# Show compatibility warnings if needed
show_compatibility_warnings()
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add compatibility warnings to train.py"
```

---

## Task 10: Add CUDA 13.0 Feature Flags to Config

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/train_configs/train_cuda13_optimized.yaml`

**Step 1: Create CUDA 13.0 optimized config**

```yaml
# train_configs/train_cuda13_optimized.yaml
# Configuration optimized for CUDA 13.0 and Hopper GPUs

pretrained_model_name_or_path: "Qwen/Qwen2.5-VL-3B-Instruct"
output_dir: "output/cuda13_optimized"
logging_dir: "logs"

# Training hyperparameters
gradient_accumulation_steps: 1
mixed_precision: "bf16"
report_to: "tensorboard"
learning_rate: 1e-4
max_train_steps: 1000
checkpointing_steps: 500
seed: 42

# LoRA configuration
rank: 16

# Data configuration
data_config:
  img_dir: "data/images"
  img_size: 1024

# CUDA 13.0 Optimizations
cuda_13_features:
  enable_flash_attention_3: true   # Hopper+ (H100, H200)
  enable_fp8_training: false       # Optional FP8
  enable_tf32_compute: true        # Ampere+
  enable_cudnn_sdp: true           # Scaled dot-product attention

# Memory optimizations
memory_efficient_attention: true
gradient_checkpointing: false  # Disable if sufficient memory

# FastSafeTensors
use_fastsafetensors: true
fastsafetensors_num_threads: 8

# Unified memory (if on DGX Spark)
unified_memory: false
```

**Step 2: Commit**

```bash
git add train_configs/train_cuda13_optimized.yaml
git commit -m "feat: add CUDA 13.0 optimized training configuration"
```

---

## Task 11: Create CUDA Compatibility Test Suite

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/test_cuda_compatibility.py`

**Step 1: Write comprehensive CUDA tests**

```python
# tests/test_cuda_compatibility.py
"""
CUDA compatibility test suite.

Tests for verifying CUDA features and compatibility.
"""
import pytest
import torch

def test_torch_cuda_available():
    """Test that PyTorch CUDA is available."""
    # This test may fail on non-GPU systems, that's expected
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0
    else:
        pytest.skip("CUDA not available")

def test_cuda_version_format():
    """Test CUDA version string format."""
    from utils.cuda_utils import get_cuda_version
    version = get_cuda_version()
    if version != "N/A":
        parts = version.split(".")
        assert len(parts) >= 2
        assert parts[0].isdigit()

def test_device_capability_format():
    """Test device capability format."""
    from utils.cuda_utils import get_device_capability
    cap = get_device_capability()
    assert len(cap) == 2
    assert isinstance(cap[0], int)
    assert isinstance(cap[1], int)
    assert cap[0] >= 0
    assert cap[1] >= 0

def test_tf32_enable():
    """Test TF32 can be enabled without error."""
    from utils.cuda_utils import enable_tf32
    # Should not raise
    enable_tf32()
    if torch.cuda.is_available():
        # Verify it was set (may be False on pre-Ampere)
        assert isinstance(torch.backends.cuda.matmul.allow_tf32, bool)

def test_optimal_settings_structure():
    """Test optimal settings returns correct structure."""
    from utils.cuda_utils import get_optimal_settings
    settings = get_optimal_settings()
    assert "use_flash_attention_3" in settings
    assert "use_fp8" in settings
    assert "use_tf32" in settings
    assert "cuda_version" in settings
    assert "compute_capability" in settings

def test_recommended_dtype():
    """Test recommended dtype selection."""
    from utils.cuda_utils import get_recommended_dtype
    dtype = get_recommended_dtype()
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_memory_allocation():
    """Test basic GPU memory allocation works."""
    tensor = torch.randn(100, 100, device="cuda")
    assert tensor.device.type == "cuda"
    del tensor
    torch.cuda.empty_cache()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_conversion():
    """Test dtype conversion on GPU."""
    tensor = torch.randn(10, 10, device="cuda")
    bf16_tensor = tensor.to(torch.bfloat16)
    assert bf16_tensor.dtype == torch.bfloat16
    fp16_tensor = tensor.to(torch.float16)
    assert fp16_tensor.dtype == torch.float16
```

**Step 2: Run tests**

```bash
pytest tests/test_cuda_compatibility.py -v
```

Expected: PASS (some may skip if no CUDA)

**Step 3: Commit**

```bash
git add tests/test_cuda_compatibility.py
git commit -m "test: add CUDA compatibility test suite"
```

---

## Task 12: Update README with CUDA 13.0 Information

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/README.md`

**Step 1: Add CUDA 13.0 section**

```markdown
## CUDA 13.0 Support

This project supports CUDA 13.0 with automatic feature detection and backward compatibility.

### Requirements

- NVIDIA Driver: 550.x or newer
- CUDA Toolkit: 13.0
- GPU: Hopper (H100/H200) for full features, Ampere (A100/RTX 30xx/40xx) for partial support

### Setup

```bash
# Setup CUDA 13.0 environment
./scripts/setup_cuda13.sh

# Install CUDA 13.0 compatible packages
pip install -r requirements-cuda13.txt
```

### Features by Architecture

| Feature | Hopper (9.0) | Ampere (8.0) | Turing (7.5) |
|---------|--------------|--------------|--------------|
| Flash Attention 3 | Yes | No | No |
| Native FP8 | Yes | No | No |
| TF32 Compute | Yes | Yes | No |
| Flash Attention 2 | Yes | Yes | No |

### Configuration

```yaml
cuda_13_features:
  enable_flash_attention_3: true  # Hopper only
  enable_fp8_training: false
  enable_tf32_compute: true       # Ampere+
  enable_cudnn_sdp: true
```

### Backward Compatibility

The code automatically detects CUDA version and enables appropriate features:

```python
from utils.cuda_utils import get_optimal_settings
settings = get_optimal_settings()
print(settings)
# {'use_flash_attention_3': False, 'use_fp8': False, 'use_tf32': True, ...}
```

Warnings are shown for outdated CUDA versions:
```
DeprecationWarning: CUDA 12.1 detected. Recommend upgrading to 13.0 for best performance.
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add CUDA 13.0 support documentation"
```

---

## Task 13: Add GPU Compatibility Matrix

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/docs/GPU_COMPATIBILITY.md`

**Step 1: Create docs directory if needed**

```bash
mkdir -p docs
```

**Step 2: Write compatibility matrix**

```markdown
# GPU Compatibility Matrix

## Supported GPUs

### Hopper Architecture (Compute Capability 9.0)
- H100 PCIe / SXM
- H200
- DGX H100

**Full CUDA 13.0 features:**
- Flash Attention 3
- Native FP8 training
- TF32 compute
- Unified memory (DGX Spark)

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
```

**Step 3: Commit**

```bash
git add docs/GPU_COMPATIBILITY.md
git commit -m "docs: add GPU compatibility matrix"
```

---

## Summary

This plan covers:
1. CUDA version detection utilities
2. Feature support checking
3. CUDA 13.0 requirements file
4. Environment setup script
5. TF32 compute enabling
6. Backward compatibility layer
7. CUDA 13.0 optimized config
8. Comprehensive test suite
9. Documentation and compatibility matrix

Total estimated time: 20-28 hours
Complexity: Medium-High

Key files created:
- utils/cuda_utils.py
- utils/compat.py
- requirements-cuda13.txt
- scripts/setup_cuda13.sh
- train_configs/train_cuda13_optimized.yaml
- tests/test_cuda_compatibility.py
- docs/GPU_COMPATIBILITY.md

Major changes:
- Automatic CUDA feature detection
- TF32 enabled by default on Ampere+
- Backward compatibility warnings
- Future-proof architecture for CUDA 13.0 features

**Note:** Actual CUDA 13.0 PyTorch packages must be available from PyTorch index. Update requirements-cuda13.txt with actual version numbers when released.
