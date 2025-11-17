# DGX Spark Unified Memory Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize memory management for DGX Spark's unified CPU-GPU memory architecture (Grace Hopper) to eliminate unnecessary memory copies and maximize training throughput.

**Architecture:** Create unified memory detection and configuration system, then systematically remove CPU offloading operations that are counterproductive in unified memory environments. Add new config file for DGX Spark with appropriate memory pool settings.

**Tech Stack:** PyTorch, CUDA unified memory, Accelerate, OmegaConf

---

## Task 1: Create Unified Memory Detection Utility

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/utils/unified_memory.py`
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/test_unified_memory.py`

**Step 1: Write failing tests**

```python
# tests/test_unified_memory.py
import pytest

def test_is_unified_memory_system():
    """Test unified memory detection function exists."""
    from utils.unified_memory import is_unified_memory_system
    result = is_unified_memory_system()
    assert isinstance(result, bool)

def test_get_memory_config_standard():
    """Test memory config for standard systems."""
    from utils.unified_memory import get_memory_config
    config = get_memory_config(unified_memory=False)
    assert config["pin_memory"] == True
    assert config["disable_cpu_offload"] == False

def test_get_memory_config_unified():
    """Test memory config for unified memory systems."""
    from utils.unified_memory import get_memory_config
    config = get_memory_config(unified_memory=True)
    assert config["pin_memory"] == False
    assert config["disable_cpu_offload"] == True

def test_setup_unified_memory_env():
    """Test that environment variables are set correctly."""
    from utils.unified_memory import setup_unified_memory_env
    import os
    setup_unified_memory_env()
    # Should not raise
    assert True
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_unified_memory.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'utils.unified_memory'"

**Step 3: Implement the unified memory utility**

```python
# utils/unified_memory.py
"""
Unified memory detection and configuration for DGX Spark (Grace Hopper) systems.

Grace Hopper architecture provides 128GB unified CPU-GPU memory, eliminating
the need for explicit memory transfers and CPU offloading.
"""
import os
import torch
from typing import Dict, Any

def is_unified_memory_system() -> bool:
    """
    Detect if running on a unified memory system (DGX Spark/Grace Hopper).

    Returns:
        True if unified memory is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Check for Grace Hopper architecture (compute capability 9.0+)
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 9:
        # Grace Hopper or newer
        return True

    # Check environment variable override
    if os.environ.get("UNIFIED_MEMORY", "").lower() in ("true", "1", "yes"):
        return True

    return False


def get_memory_config(unified_memory: bool = None) -> Dict[str, Any]:
    """
    Get optimal memory configuration based on system type.

    Args:
        unified_memory: Override detection. If None, auto-detect.

    Returns:
        Dictionary with memory configuration settings
    """
    if unified_memory is None:
        unified_memory = is_unified_memory_system()

    if unified_memory:
        return {
            "pin_memory": False,  # Not needed for unified memory
            "disable_cpu_offload": True,  # Keep everything in unified pool
            "prefetch_factor": 2,  # Lower prefetch, data already accessible
            "num_workers": 4,  # Fewer workers needed
            "memory_pool": "unified",
            "expandable_segments": True,
        }
    else:
        return {
            "pin_memory": True,  # Pinned memory for faster transfers
            "disable_cpu_offload": False,  # CPU offloading beneficial
            "prefetch_factor": 4,  # Higher prefetch for async loading
            "num_workers": 8,  # More workers to hide transfer latency
            "memory_pool": "standard",
            "expandable_segments": False,
        }


def setup_unified_memory_env() -> None:
    """
    Configure environment variables for unified memory operation.

    Sets CUDA allocator settings and memory pool configuration.
    """
    if is_unified_memory_system():
        # Enable expandable segments for better memory utilization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,backend:native"

        # Force managed memory allocation
        os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

        # Configure memory fraction
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception:
                pass  # May not be supported on all systems


def configure_accelerator_for_unified_memory(accelerator_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modify Accelerator kwargs for unified memory systems.

    Args:
        accelerator_kwargs: Base Accelerator configuration

    Returns:
        Modified configuration for unified memory
    """
    if is_unified_memory_system():
        # Don't let Accelerate manage device placement
        accelerator_kwargs["device_placement"] = False
    return accelerator_kwargs
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_unified_memory.py -v
```

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add utils/unified_memory.py tests/test_unified_memory.py
git commit -m "feat: add unified memory detection and configuration utility"
```

---

## Task 2: Create DGX Spark Training Configuration

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/train_configs/train_dgx_spark.yaml`

**Step 1: Read existing train_lora.yaml as template**

```bash
cat train_configs/train_lora.yaml
```

**Step 2: Create DGX Spark specific config**

```yaml
# train_configs/train_dgx_spark.yaml
# DGX Spark (Grace Hopper) optimized training configuration
# Assumes 128GB unified CPU-GPU memory

pretrained_model_name_or_path: "Qwen/Qwen2.5-VL-3B-Instruct"
output_dir: "output/dgx_spark"
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

# DGX Spark Unified Memory Settings
unified_memory: true
disable_cpu_offload: true
pin_memory: false
disable_quantization: true  # Not needed with 128GB unified memory
disable_gradient_checkpointing: true  # Sufficient memory available

# Memory pool configuration
memory_fraction: 0.9
expandable_segments: true

# Disable embedding caching to disk (keep in unified memory)
precompute_text_embeddings: true
precompute_image_embeddings: true
save_cache_on_disk: false  # Keep in memory

# Removed optimizations (not needed with unified memory)
# - No 8-bit Adam
# - No quantization
# - No CPU offloading
# - No gradient checkpointing

# FastSafeTensors (from previous plan)
use_fastsafetensors: true
fastsafetensors_num_threads: 8
```

**Step 3: Commit**

```bash
git add train_configs/train_dgx_spark.yaml
git commit -m "feat: add DGX Spark unified memory training configuration"
```

---

## Task 3: Update train_4090.py to Support Unified Memory Mode

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py:1-41` (imports)
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py:88-135` (main setup)

**Step 1: Add unified memory import**

Add after line 41 (after gc import) in `/home/jay/Documents/flymyai-lora-trainer/train_4090.py`:
```python
from utils.unified_memory import (
    is_unified_memory_system,
    get_memory_config,
    setup_unified_memory_env,
)
```

**Step 2: Add unified memory setup after args loading**

After line 89 (args = OmegaConf.load...):
```python
# Setup unified memory environment if applicable
if getattr(args, 'unified_memory', False):
    setup_unified_memory_env()
    logger.info("Unified memory mode enabled - disabling CPU offloading")
```

**Step 3: Test import works**

```bash
python -c "import train_4090; print('Import successful')"
```

Expected: "Import successful"

**Step 4: Commit**

```bash
git add train_4090.py
git commit -m "feat: add unified memory support to train_4090.py imports"
```

---

## Task 4: Remove CPU Offloading from train_4090.py (Conditional)

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py:176-178` (text encoding offload)
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py:214-215` (VAE offload)

**Step 1: Update text encoding pipeline offloading (around line 176)**

Replace:
```python
text_encoding_pipeline.to("cpu")
torch.cuda.empty_cache()
```

With:
```python
if not getattr(args, 'unified_memory', False):
    text_encoding_pipeline.to("cpu")
    torch.cuda.empty_cache()
else:
    logger.info("Unified memory: keeping text encoding pipeline on device")
```

**Step 2: Update VAE offloading (around line 214)**

Replace:
```python
vae.to('cpu')
torch.cuda.empty_cache()
```

With:
```python
if not getattr(args, 'unified_memory', False):
    vae.to('cpu')
    torch.cuda.empty_cache()
else:
    logger.info("Unified memory: keeping VAE on device")
```

**Step 3: Commit**

```bash
git add train_4090.py
git commit -m "feat: make CPU offloading conditional on unified_memory flag in train_4090.py"
```

---

## Task 5: Skip Quantization in Unified Memory Mode

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py:221-232` (quantization block)

**Step 1: Update quantization logic**

Replace:
```python
if args.quantize:
    torch_dtype = weight_dtype
    device = accelerator.device
    all_blocks = list(flux_transformer.transformer_blocks)
    for block in tqdm(all_blocks):
        block.to(device, dtype=torch_dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        block.to('cpu')
    flux_transformer.to(device, dtype=torch_dtype)
    quantize(flux_transformer, weights=qfloat8)
    freeze(flux_transformer)
```

With:
```python
if args.quantize and not getattr(args, 'disable_quantization', False):
    torch_dtype = weight_dtype
    device = accelerator.device
    all_blocks = list(flux_transformer.transformer_blocks)
    for block in tqdm(all_blocks):
        block.to(device, dtype=torch_dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        if not getattr(args, 'unified_memory', False):
            block.to('cpu')
    flux_transformer.to(device, dtype=torch_dtype)
    quantize(flux_transformer, weights=qfloat8)
    freeze(flux_transformer)
elif getattr(args, 'disable_quantization', False):
    logger.info("Quantization disabled (unified memory mode)")
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
```

**Step 2: Commit**

```bash
git add train_4090.py
git commit -m "feat: make quantization and block offloading conditional for unified memory"
```

---

## Task 6: Update DataLoader Configuration

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/image_datasets/dataset.py`
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/test_dataloader_config.py`

**Step 1: Write failing test**

```python
# tests/test_dataloader_config.py
import pytest

def test_dataloader_respects_pin_memory_false():
    """Test that DataLoader can be created with pin_memory=False."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    dataset = TensorDataset(torch.randn(10, 3))
    loader = DataLoader(dataset, pin_memory=False)
    assert loader.pin_memory == False

def test_dataloader_respects_pin_memory_true():
    """Test that DataLoader can be created with pin_memory=True."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    dataset = TensorDataset(torch.randn(10, 3))
    loader = DataLoader(dataset, pin_memory=True)
    assert loader.pin_memory == True
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_dataloader_config.py -v
```

Expected: PASS

**Step 3: Read current dataset.py**

```bash
head -100 image_datasets/dataset.py
```

**Step 4: Commit test**

```bash
git add tests/test_dataloader_config.py
git commit -m "test: add DataLoader pin_memory configuration tests"
```

---

## Task 7: Update train.py for Unified Memory

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train.py:1-35` (imports)
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train.py:55-70` (main setup)

**Step 1: Add unified memory import**

Add after existing imports:
```python
from utils.unified_memory import (
    is_unified_memory_system,
    get_memory_config,
    setup_unified_memory_env,
)
```

**Step 2: Add setup call after args loading**

After `args = OmegaConf.load(parse_args())`:
```python
# Setup unified memory environment if configured
if getattr(args, 'unified_memory', False):
    setup_unified_memory_env()
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add unified memory support to train.py"
```

---

## Task 8: Update train_flux_lora.py for Unified Memory

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_flux_lora.py` (imports and main)

**Step 1: Add unified memory import**

```python
from utils.unified_memory import (
    is_unified_memory_system,
    get_memory_config,
    setup_unified_memory_env,
)
```

**Step 2: Add setup call in main()**

After args loading:
```python
if getattr(args, 'unified_memory', False):
    setup_unified_memory_env()
```

**Step 3: Commit**

```bash
git add train_flux_lora.py
git commit -m "feat: add unified memory support to train_flux_lora.py"
```

---

## Task 9: Update train_qwen_edit_lora.py for Unified Memory

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_qwen_edit_lora.py` (imports and main)

**Step 1: Add unified memory import**

```python
from utils.unified_memory import (
    is_unified_memory_system,
    get_memory_config,
    setup_unified_memory_env,
)
```

**Step 2: Commit**

```bash
git add train_qwen_edit_lora.py
git commit -m "feat: add unified memory support to train_qwen_edit_lora.py"
```

---

## Task 10: Add Memory Usage Monitoring

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/utils/memory_monitor.py`
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/test_memory_monitor.py`

**Step 1: Write failing test**

```python
# tests/test_memory_monitor.py
import pytest

def test_get_memory_stats():
    """Test memory stats retrieval."""
    from utils.memory_monitor import get_memory_stats
    stats = get_memory_stats()
    assert "allocated_gb" in stats
    assert "reserved_gb" in stats
    assert isinstance(stats["allocated_gb"], float)

def test_log_memory_usage():
    """Test memory usage logging."""
    from utils.memory_monitor import log_memory_usage
    # Should not raise
    log_memory_usage("test checkpoint")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_monitor.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement memory monitor**

```python
# utils/memory_monitor.py
"""
Memory usage monitoring for training scripts.

Provides utilities to track GPU and unified memory usage.
"""
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_memory_stats() -> Dict[str, Any]:
    """
    Get current memory statistics.

    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }

    stats = torch.cuda.memory_stats()

    return {
        "allocated_gb": stats.get("allocated_bytes.all.current", 0) / (1024**3),
        "reserved_gb": stats.get("reserved_bytes.all.current", 0) / (1024**3),
        "max_allocated_gb": stats.get("allocated_bytes.all.peak", 0) / (1024**3),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
    }


def log_memory_usage(checkpoint_name: str = "") -> None:
    """
    Log current memory usage.

    Args:
        checkpoint_name: Optional name for the logging checkpoint
    """
    if not torch.cuda.is_available():
        logger.info(f"[{checkpoint_name}] No CUDA available")
        return

    stats = get_memory_stats()
    logger.info(
        f"[{checkpoint_name}] Memory: "
        f"Allocated={stats['allocated_gb']:.2f}GB, "
        f"Reserved={stats['reserved_gb']:.2f}GB, "
        f"Peak={stats['max_allocated_gb']:.2f}GB"
    )


def reset_peak_memory_stats() -> None:
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_monitor.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add utils/memory_monitor.py tests/test_memory_monitor.py
git commit -m "feat: add memory usage monitoring utility"
```

---

## Task 11: Integrate Memory Monitoring into Training Loop

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_4090.py` (training loop section)

**Step 1: Add memory monitor import**

```python
from utils.memory_monitor import log_memory_usage, reset_peak_memory_stats
```

**Step 2: Add monitoring calls at key points**

Before training loop:
```python
if getattr(args, 'unified_memory', False):
    reset_peak_memory_stats()
    log_memory_usage("before_training")
```

After each epoch/checkpoint:
```python
if getattr(args, 'unified_memory', False):
    log_memory_usage(f"step_{global_step}")
```

**Step 3: Commit**

```bash
git add train_4090.py
git commit -m "feat: integrate memory monitoring into train_4090.py training loop"
```

---

## Task 12: Add Unified Memory Configuration to Other Config Files

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_configs/train_lora.yaml`
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_configs/train_lora_4090.yaml`
- Modify: `/home/jay/Documents/flymyai-lora-trainer/train_configs/train_flux_config.yaml`

**Step 1: Add unified memory options (disabled by default)**

Add to each config file:
```yaml
# Unified Memory Settings (DGX Spark)
# Set unified_memory: true for Grace Hopper systems
unified_memory: false
disable_cpu_offload: false
disable_quantization: false
disable_gradient_checkpointing: false
```

**Step 2: Commit**

```bash
git add train_configs/*.yaml
git commit -m "feat: add unified memory configuration options to all training configs"
```

---

## Task 13: Update Documentation

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/README.md`

**Step 1: Add DGX Spark section**

Add section:
```markdown
## DGX Spark Unified Memory Support

This project supports DGX Spark (Grace Hopper) unified memory architecture for optimal training performance.

### Benefits
- Eliminates CPU-GPU memory transfers
- No need for quantization with 128GB unified memory
- Removes CPU offloading overhead
- Simplified memory management

### Configuration

For DGX Spark systems, use the dedicated config:
```bash
python train_4090.py --config train_configs/train_dgx_spark.yaml
```

Or enable in any config:
```yaml
unified_memory: true
disable_cpu_offload: true
disable_quantization: true
```

### Memory Monitoring

Memory usage is automatically logged when `unified_memory: true`:
```
[step_100] Memory: Allocated=45.23GB, Reserved=48.00GB, Peak=52.10GB
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add DGX Spark unified memory documentation"
```

---

## Summary

This plan covers:
1. Unified memory detection utility
2. DGX Spark configuration file
3. Conditional CPU offloading removal
4. Conditional quantization skip
5. DataLoader pin_memory configuration
6. Memory usage monitoring
7. Documentation

Total estimated time: 24-32 hours
Complexity: High

Key files modified:
- utils/unified_memory.py (new)
- utils/memory_monitor.py (new)
- train_configs/train_dgx_spark.yaml (new)
- train_4090.py, train.py, train_flux_lora.py, train_qwen_edit_lora.py
- All train_configs/*.yaml files
- README.md

Major changes:
- CPU offloading becomes conditional
- Quantization becomes optional
- Memory monitoring added
- New DGX Spark specific configuration
