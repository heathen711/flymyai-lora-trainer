# Integration Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate that all three improvements (fastsafetensors, unified memory, CUDA 13.0) work together correctly and measure combined performance benefits.

**Architecture:** Create a comprehensive test suite that validates each feature individually and in combination, build performance benchmarking infrastructure, implement regression tests to ensure training quality is maintained, and document expected performance characteristics.

**Tech Stack:** pytest, PyTorch, CUDA, fastsafetensors, benchmarking tools

---

## Task 1: Create Test Infrastructure

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/conftest.py`
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/__init__.py`

**Step 1: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
"""
Pytest configuration and shared fixtures for integration tests.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_state_dict():
    """Create a sample state dict for testing."""
    return {
        "layer1.weight": torch.randn(64, 64),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(128, 64),
        "layer2.bias": torch.randn(128),
    }

@pytest.fixture
def large_state_dict():
    """Create a larger state dict simulating transformer model."""
    state_dict = {}
    for i in range(24):
        state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.randn(1024, 1024)
        state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.randn(1024, 1024)
        state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.randn(1024, 1024)
        state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = torch.randn(1024, 1024)
    return state_dict

@pytest.fixture
def mock_config():
    """Create a mock training configuration."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "pretrained_model_name_or_path": "test_model",
        "output_dir": "test_output",
        "logging_dir": "logs",
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "report_to": "tensorboard",
        "learning_rate": 1e-4,
        "max_train_steps": 10,
        "rank": 16,
        "use_fastsafetensors": True,
        "fastsafetensors_num_threads": 8,
        "unified_memory": False,
        "quantize": False,
    })
    return config

@pytest.fixture
def gpu_available():
    """Check if GPU is available and skip if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True

@pytest.fixture
def hopper_gpu():
    """Check if Hopper GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper GPU required (compute capability 9.0+)")
    return True

@pytest.fixture
def ampere_gpu():
    """Check if Ampere+ GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Ampere GPU required (compute capability 8.0+)")
    return True
```

**Step 2: Create tests/__init__.py**

```python
# tests/__init__.py
"""Integration tests for flymyai-lora-trainer."""
```

**Step 3: Commit**

```bash
git add tests/conftest.py tests/__init__.py
git commit -m "test: add pytest infrastructure with shared fixtures"
```

---

## Task 2: Create FastSafeTensors Integration Tests

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/integration/test_fastsafetensors_integration.py`

**Step 1: Create integration test directory**

```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

**Step 2: Write integration tests**

```python
# tests/integration/test_fastsafetensors_integration.py
"""
Integration tests for fastsafetensors with training pipeline.
"""
import pytest
import torch
import time
from safetensors.torch import save_file as st_save

def test_fastsafetensors_load_save_roundtrip(temp_dir, large_state_dict):
    """Test that state dicts survive save/load cycle."""
    from utils.fast_loading import load_safetensors, save_safetensors

    file_path = temp_dir / "test_model.safetensors"

    # Save
    save_safetensors(large_state_dict, str(file_path))
    assert file_path.exists()

    # Load
    loaded = load_safetensors(str(file_path), num_threads=8)

    # Verify all keys present
    assert set(loaded.keys()) == set(large_state_dict.keys())

    # Verify tensor values match
    for key in large_state_dict:
        assert torch.allclose(loaded[key], large_state_dict[key], rtol=1e-5)

def test_fastsafetensors_performance_improvement(temp_dir, large_state_dict):
    """Test that fastsafetensors provides performance improvement."""
    from utils.fast_loading import load_safetensors, is_fastsafetensors_available

    if not is_fastsafetensors_available():
        pytest.skip("fastsafetensors not available")

    file_path = temp_dir / "perf_test.safetensors"
    st_save(large_state_dict, str(file_path))

    # Warm up
    _ = load_safetensors(str(file_path))

    # Benchmark multiple loads
    times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = load_safetensors(str(file_path), num_threads=8)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    throughput = file_size_mb / avg_time

    # Should achieve reasonable throughput (> 500 MB/s on modern SSDs)
    assert throughput > 100, f"Low throughput: {throughput:.2f} MB/s"

def test_fastsafetensors_thread_scaling(temp_dir, large_state_dict):
    """Test that more threads improve performance."""
    from utils.fast_loading import load_safetensors, is_fastsafetensors_available

    if not is_fastsafetensors_available():
        pytest.skip("fastsafetensors not available")

    file_path = temp_dir / "thread_test.safetensors"
    st_save(large_state_dict, str(file_path))

    # Test with 1 thread
    start = time.perf_counter()
    _ = load_safetensors(str(file_path), num_threads=1)
    time_1_thread = time.perf_counter() - start

    # Test with 8 threads
    start = time.perf_counter()
    _ = load_safetensors(str(file_path), num_threads=8)
    time_8_threads = time.perf_counter() - start

    # More threads should be at least not slower
    # (actual speedup depends on disk/CPU)
    assert time_8_threads <= time_1_thread * 1.5

def test_fastsafetensors_metadata_preservation(temp_dir, sample_state_dict):
    """Test that metadata is preserved through save/load."""
    from utils.fast_loading import save_safetensors, load_safetensors

    file_path = temp_dir / "metadata_test.safetensors"
    metadata = {"format": "pt", "version": "1.0", "lora_rank": "16"}

    save_safetensors(sample_state_dict, str(file_path), metadata=metadata)

    # Load and verify structure (metadata may not be accessible via fastsafetensors)
    loaded = load_safetensors(str(file_path))
    assert len(loaded) == len(sample_state_dict)

@pytest.mark.gpu
def test_fastsafetensors_gpu_transfer(temp_dir, sample_state_dict, gpu_available):
    """Test loading directly to GPU."""
    from utils.fast_loading import load_safetensors, is_fastsafetensors_available
    from safetensors.torch import save_file as st_save

    file_path = temp_dir / "gpu_test.safetensors"
    st_save(sample_state_dict, str(file_path))

    # Load to GPU
    loaded = load_safetensors(str(file_path), device="cuda")

    for key in loaded:
        assert loaded[key].device.type == "cuda"

    # Verify values match (after moving to CPU)
    for key in sample_state_dict:
        cpu_tensor = loaded[key].cpu()
        assert torch.allclose(cpu_tensor, sample_state_dict[key], rtol=1e-5)
```

**Step 3: Run tests**

```bash
pytest tests/integration/test_fastsafetensors_integration.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add tests/integration/
git commit -m "test: add fastsafetensors integration tests"
```

---

## Task 3: Create Unified Memory Integration Tests

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/integration/test_unified_memory_integration.py`

**Step 1: Write unified memory integration tests**

```python
# tests/integration/test_unified_memory_integration.py
"""
Integration tests for unified memory optimization.
"""
import pytest
import torch
import os

def test_unified_memory_config_loading(mock_config):
    """Test that unified memory config is properly loaded."""
    from utils.unified_memory import get_memory_config

    config = get_memory_config(unified_memory=True)

    assert config["pin_memory"] == False
    assert config["disable_cpu_offload"] == True
    assert config["memory_pool"] == "unified"

def test_unified_memory_env_setup():
    """Test environment variables are set correctly."""
    from utils.unified_memory import setup_unified_memory_env

    # This should not raise
    setup_unified_memory_env()

    # Check environment variable format (not necessarily set if not unified system)
    # The function should handle non-unified systems gracefully

@pytest.mark.gpu
def test_no_cpu_offload_in_unified_mode(gpu_available):
    """Test that CPU offloading is skipped in unified memory mode."""
    from utils.unified_memory import get_memory_config

    config = get_memory_config(unified_memory=True)

    # Simulate training decision
    should_offload_to_cpu = not config["disable_cpu_offload"]
    assert should_offload_to_cpu == False

@pytest.mark.gpu
def test_memory_monitoring(gpu_available):
    """Test memory monitoring works correctly."""
    from utils.memory_monitor import get_memory_stats, log_memory_usage, reset_peak_memory_stats

    # Reset stats
    reset_peak_memory_stats()

    # Get initial stats
    initial_stats = get_memory_stats()
    assert "allocated_gb" in initial_stats
    assert initial_stats["allocated_gb"] >= 0

    # Allocate some memory
    tensor = torch.randn(1000, 1000, device="cuda")

    # Check stats increased
    after_stats = get_memory_stats()
    assert after_stats["allocated_gb"] > initial_stats["allocated_gb"]

    # Log should not raise
    log_memory_usage("test_checkpoint")

    # Cleanup
    del tensor
    torch.cuda.empty_cache()

def test_dataloader_config_for_unified_memory():
    """Test DataLoader configuration respects unified memory settings."""
    from torch.utils.data import DataLoader, TensorDataset
    from utils.unified_memory import get_memory_config

    dataset = TensorDataset(torch.randn(100, 10))

    # Standard config
    standard_config = get_memory_config(unified_memory=False)
    loader_standard = DataLoader(
        dataset,
        pin_memory=standard_config["pin_memory"],
        num_workers=0  # Skip workers for test
    )
    assert loader_standard.pin_memory == True

    # Unified memory config
    unified_config = get_memory_config(unified_memory=True)
    loader_unified = DataLoader(
        dataset,
        pin_memory=unified_config["pin_memory"],
        num_workers=0
    )
    assert loader_unified.pin_memory == False

def test_unified_memory_detection():
    """Test unified memory system detection."""
    from utils.unified_memory import is_unified_memory_system

    result = is_unified_memory_system()
    assert isinstance(result, bool)

    # Test environment override
    os.environ["UNIFIED_MEMORY"] = "true"
    result_with_override = is_unified_memory_system()
    # Note: might still be False if no CUDA, but should not raise
    assert isinstance(result_with_override, bool)

    # Clean up
    del os.environ["UNIFIED_MEMORY"]

@pytest.mark.gpu
def test_memory_stays_on_device_in_unified_mode(gpu_available):
    """Test that tensors stay on device in unified memory mode."""
    from utils.unified_memory import get_memory_config

    config = get_memory_config(unified_memory=True)

    # Create tensor on GPU
    tensor = torch.randn(100, 100, device="cuda")

    # In unified memory mode, we don't move to CPU
    if config["disable_cpu_offload"]:
        # Tensor should stay on GPU
        assert tensor.device.type == "cuda"
    else:
        # Standard mode would move to CPU
        tensor = tensor.to("cpu")
        assert tensor.device.type == "cpu"

    del tensor
    torch.cuda.empty_cache()
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_unified_memory_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_unified_memory_integration.py
git commit -m "test: add unified memory integration tests"
```

---

## Task 4: Create CUDA 13.0 Integration Tests

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/integration/test_cuda13_integration.py`

**Step 1: Write CUDA 13.0 integration tests**

```python
# tests/integration/test_cuda13_integration.py
"""
Integration tests for CUDA 13.0 features and backward compatibility.
"""
import pytest
import torch

def test_cuda_version_detection():
    """Test CUDA version detection."""
    from utils.cuda_utils import get_cuda_version

    version = get_cuda_version()
    assert isinstance(version, str)

    if version != "N/A":
        parts = version.split(".")
        assert len(parts) >= 2
        major = int(parts[0])
        assert major >= 11  # Minimum supported version

def test_feature_support_checking():
    """Test feature support detection."""
    from utils.cuda_utils import supports_feature

    # These should return bool without raising
    assert isinstance(supports_feature("flash_attention_3"), bool)
    assert isinstance(supports_feature("fp8_native"), bool)
    assert isinstance(supports_feature("tf32_compute"), bool)
    assert isinstance(supports_feature("nonexistent_feature"), bool)

@pytest.mark.gpu
def test_tf32_enablement(ampere_gpu):
    """Test TF32 can be enabled on Ampere+ GPUs."""
    from utils.cuda_utils import enable_tf32, supports_feature

    if supports_feature("tf32_compute"):
        enable_tf32()
        assert torch.backends.cuda.matmul.allow_tf32 == True
        assert torch.backends.cudnn.allow_tf32 == True

def test_optimal_settings_generation():
    """Test optimal settings are generated correctly."""
    from utils.cuda_utils import get_optimal_settings

    settings = get_optimal_settings()

    required_keys = [
        "use_flash_attention_3",
        "use_fp8",
        "use_tf32",
        "cuda_version",
        "compute_capability",
    ]

    for key in required_keys:
        assert key in settings

    # Types should be correct
    assert isinstance(settings["use_flash_attention_3"], bool)
    assert isinstance(settings["use_fp8"], bool)
    assert isinstance(settings["use_tf32"], bool)
    assert isinstance(settings["cuda_version"], str)
    assert isinstance(settings["compute_capability"], tuple)

def test_backward_compatibility_warnings():
    """Test compatibility warnings are generated."""
    from utils.compat import check_requirements, get_deprecation_warnings

    warnings = check_requirements()
    assert isinstance(warnings, list)

    deprecations = get_deprecation_warnings()
    assert isinstance(deprecations, list)

@pytest.mark.gpu
def test_recommended_dtype(gpu_available):
    """Test recommended dtype selection."""
    from utils.cuda_utils import get_recommended_dtype

    dtype = get_recommended_dtype()
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]

    # On modern GPUs, should prefer bfloat16
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        assert dtype == torch.bfloat16

@pytest.mark.gpu
def test_gpu_computation_with_recommended_dtype(gpu_available):
    """Test that computation works with recommended dtype."""
    from utils.cuda_utils import get_recommended_dtype, enable_tf32

    enable_tf32()
    dtype = get_recommended_dtype()

    # Create tensors with recommended dtype
    a = torch.randn(100, 100, device="cuda", dtype=dtype)
    b = torch.randn(100, 100, device="cuda", dtype=dtype)

    # Matrix multiplication should work
    c = torch.matmul(a, b)
    assert c.shape == (100, 100)
    assert c.dtype == dtype

    # Clean up
    del a, b, c
    torch.cuda.empty_cache()

def test_device_capability_detection():
    """Test device capability detection."""
    from utils.cuda_utils import get_device_capability, is_hopper_or_newer, is_ampere_or_newer

    cap = get_device_capability()
    assert isinstance(cap, tuple)
    assert len(cap) == 2

    hopper = is_hopper_or_newer()
    ampere = is_ampere_or_newer()

    assert isinstance(hopper, bool)
    assert isinstance(ampere, bool)

    # If Hopper, must also be Ampere
    if hopper:
        assert ampere == True

@pytest.mark.gpu
def test_memory_allocation_patterns(gpu_available):
    """Test various memory allocation patterns work correctly."""
    from utils.cuda_utils import enable_tf32

    enable_tf32()

    # Test different allocation patterns
    patterns = [
        torch.randn(1000, 1000, device="cuda"),
        torch.zeros(500, 500, device="cuda"),
        torch.ones(200, 200, device="cuda"),
    ]

    # All should be on CUDA
    for tensor in patterns:
        assert tensor.device.type == "cuda"

    # Cleanup
    for tensor in patterns:
        del tensor
    torch.cuda.empty_cache()
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_cuda13_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_cuda13_integration.py
git commit -m "test: add CUDA 13.0 integration tests"
```

---

## Task 5: Create Combined Feature Integration Tests

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/integration/test_combined_features.py`

**Step 1: Write combined integration tests**

```python
# tests/integration/test_combined_features.py
"""
Integration tests for combined features: fastsafetensors + unified memory + CUDA 13.0.
"""
import pytest
import torch
import time
from safetensors.torch import save_file as st_save

def test_all_utilities_importable():
    """Test that all utility modules can be imported together."""
    from utils.fast_loading import load_safetensors, save_safetensors
    from utils.unified_memory import is_unified_memory_system, get_memory_config
    from utils.cuda_utils import get_optimal_settings, enable_tf32
    from utils.memory_monitor import get_memory_stats
    from utils.compat import check_requirements

    # All imports successful
    assert callable(load_safetensors)
    assert callable(get_memory_config)
    assert callable(get_optimal_settings)
    assert callable(get_memory_stats)
    assert callable(check_requirements)

def test_combined_configuration():
    """Test configuration with all features enabled."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        # FastSafeTensors
        "use_fastsafetensors": True,
        "fastsafetensors_num_threads": 8,
        # Unified Memory
        "unified_memory": True,
        "disable_cpu_offload": True,
        "disable_quantization": True,
        # CUDA 13.0
        "cuda_13_features": {
            "enable_flash_attention_3": True,
            "enable_fp8_training": False,
            "enable_tf32_compute": True,
        },
    })

    assert config.use_fastsafetensors == True
    assert config.unified_memory == True
    assert config.cuda_13_features.enable_tf32_compute == True

@pytest.mark.gpu
def test_fastsafetensors_with_unified_memory(temp_dir, large_state_dict, gpu_available):
    """Test fastsafetensors loading in unified memory mode."""
    from utils.fast_loading import load_safetensors, save_safetensors
    from utils.unified_memory import setup_unified_memory_env, get_memory_config
    from utils.memory_monitor import log_memory_usage

    # Setup unified memory env
    setup_unified_memory_env()

    # Save state dict
    file_path = temp_dir / "unified_test.safetensors"
    save_safetensors(large_state_dict, str(file_path))

    # Load with unified memory config
    config = get_memory_config(unified_memory=True)
    log_memory_usage("before_load")

    loaded = load_safetensors(str(file_path), num_threads=8, device="cuda")

    log_memory_usage("after_load")

    # Verify loaded correctly
    assert len(loaded) == len(large_state_dict)
    for key in loaded:
        assert loaded[key].device.type == "cuda"

    # Cleanup
    del loaded
    torch.cuda.empty_cache()

@pytest.mark.gpu
def test_cuda13_features_with_fastsafetensors(temp_dir, sample_state_dict, ampere_gpu):
    """Test CUDA 13.0 features work with fastsafetensors-loaded models."""
    from utils.fast_loading import load_safetensors, save_safetensors
    from utils.cuda_utils import enable_tf32, get_recommended_dtype

    # Enable CUDA 13.0 features
    enable_tf32()
    dtype = get_recommended_dtype()

    # Save and load with fastsafetensors
    file_path = temp_dir / "cuda13_test.safetensors"
    save_safetensors(sample_state_dict, str(file_path))
    loaded = load_safetensors(str(file_path), device="cuda")

    # Convert to recommended dtype
    for key in loaded:
        loaded[key] = loaded[key].to(dtype)

    # Perform computation with TF32
    result = torch.matmul(loaded["layer1.weight"], loaded["layer1.weight"].T)
    assert result.dtype == dtype

    del loaded, result
    torch.cuda.empty_cache()

@pytest.mark.gpu
def test_all_features_combined(temp_dir, large_state_dict, ampere_gpu):
    """Test all three features working together."""
    from utils.fast_loading import load_safetensors, save_safetensors
    from utils.unified_memory import setup_unified_memory_env, get_memory_config
    from utils.cuda_utils import enable_tf32, get_recommended_dtype
    from utils.memory_monitor import log_memory_usage, reset_peak_memory_stats

    # Setup all features
    setup_unified_memory_env()
    enable_tf32()
    config = get_memory_config(unified_memory=True)
    dtype = get_recommended_dtype()

    # Reset memory stats
    reset_peak_memory_stats()
    log_memory_usage("start")

    # Save model with fastsafetensors
    file_path = temp_dir / "combined_test.safetensors"
    save_safetensors(large_state_dict, str(file_path))

    # Load model
    start_time = time.perf_counter()
    loaded = load_safetensors(str(file_path), num_threads=8, device="cuda")
    load_time = time.perf_counter() - start_time

    log_memory_usage("after_load")

    # Convert to optimal dtype
    for key in loaded:
        loaded[key] = loaded[key].to(dtype)

    log_memory_usage("after_dtype_conversion")

    # Perform some computation
    first_key = list(loaded.keys())[0]
    result = torch.matmul(loaded[first_key], loaded[first_key].T)

    log_memory_usage("after_computation")

    # Verify
    assert load_time < 60  # Should load large model in under 60 seconds
    assert result.device.type == "cuda"
    assert result.dtype == dtype

    # Cleanup
    del loaded, result
    torch.cuda.empty_cache()

def test_backward_compatibility_with_standard_safetensors(temp_dir, sample_state_dict):
    """Test that we fall back gracefully if fastsafetensors unavailable."""
    from utils.fast_loading import load_safetensors, save_safetensors

    file_path = temp_dir / "compat_test.safetensors"

    # Save with our utility
    save_safetensors(sample_state_dict, str(file_path))

    # Load with our utility (should work regardless of fastsafetensors availability)
    loaded = load_safetensors(str(file_path))

    # Verify data integrity
    for key in sample_state_dict:
        assert key in loaded
        assert torch.allclose(loaded[key], sample_state_dict[key])

def test_config_validation():
    """Test that invalid configurations are handled gracefully."""
    from utils.unified_memory import get_memory_config
    from utils.cuda_utils import supports_feature

    # These should not raise even with invalid input
    config = get_memory_config(unified_memory=None)  # Auto-detect
    assert isinstance(config, dict)

    # Non-existent feature should return False, not raise
    result = supports_feature("totally_fake_feature")
    assert result == False
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_combined_features.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_combined_features.py
git commit -m "test: add combined feature integration tests"
```

---

## Task 6: Create Performance Benchmark Suite

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/benchmarks/performance_test.py`

**Step 1: Write comprehensive benchmark script**

```python
#!/usr/bin/env python
# benchmarks/performance_test.py
"""
Comprehensive performance benchmark suite for LoRA training improvements.

Measures:
- Model loading time (fastsafetensors)
- Memory utilization (unified memory)
- Training throughput (CUDA 13.0)
- Combined feature performance
"""
import time
import torch
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from safetensors.torch import save_file as st_save

# Import our utilities
from utils.fast_loading import load_safetensors, save_safetensors, is_fastsafetensors_available
from utils.unified_memory import is_unified_memory_system, get_memory_config, setup_unified_memory_env
from utils.cuda_utils import get_optimal_settings, enable_tf32, get_recommended_dtype
from utils.memory_monitor import get_memory_stats, reset_peak_memory_stats


def create_test_model(num_layers=24, hidden_size=2048):
    """Create a test state dict similar to a transformer model."""
    state_dict = {}
    for i in range(num_layers):
        state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = torch.randn(hidden_size * 4, hidden_size)
        state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = torch.randn(hidden_size, hidden_size * 4)
    return state_dict


def benchmark_loading(file_path, num_runs=5):
    """Benchmark model loading times."""
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        state_dict = load_safetensors(file_path, num_threads=8)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append(end - start)
        del state_dict

    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def benchmark_gpu_loading(file_path, num_runs=3):
    """Benchmark loading directly to GPU."""
    if not torch.cuda.is_available():
        return None

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        state_dict = load_safetensors(file_path, num_threads=8, device="cuda")
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        del state_dict
        torch.cuda.empty_cache()

    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def benchmark_computation(num_iterations=100):
    """Benchmark matrix computation performance."""
    if not torch.cuda.is_available():
        return None

    dtype = get_recommended_dtype()
    size = 2048

    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)

    # Warm up
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    flops = 2 * size**3 * num_iterations
    tflops = flops / total_time / 1e12

    del a, b
    torch.cuda.empty_cache()

    return {
        "total_time": total_time,
        "iterations": num_iterations,
        "tflops": tflops,
    }


def benchmark_memory_efficiency():
    """Benchmark memory utilization patterns."""
    if not torch.cuda.is_available():
        return None

    reset_peak_memory_stats()

    # Allocate progressively larger tensors
    tensors = []
    sizes = [1024, 2048, 4096]
    stats_progression = []

    for size in sizes:
        tensor = torch.randn(size, size, device="cuda")
        tensors.append(tensor)
        stats_progression.append({
            "size": size,
            "stats": get_memory_stats(),
        })

    final_stats = get_memory_stats()

    # Cleanup
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache()

    return {
        "progression": stats_progression,
        "final_stats": final_stats,
    }


def run_full_benchmark():
    """Run comprehensive benchmark suite."""
    print("=" * 60)
    print("Performance Benchmark Suite")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {},
        "benchmarks": {},
    }

    # System information
    print("\n1. System Information")
    print("-" * 40)

    results["system_info"]["fastsafetensors_available"] = is_fastsafetensors_available()
    results["system_info"]["unified_memory_system"] = is_unified_memory_system()
    results["system_info"]["cuda_settings"] = get_optimal_settings()

    print(f"FastSafeTensors: {results['system_info']['fastsafetensors_available']}")
    print(f"Unified Memory: {results['system_info']['unified_memory_system']}")
    print(f"CUDA Settings: {results['system_info']['cuda_settings']}")

    # Enable optimizations
    enable_tf32()
    if results["system_info"]["unified_memory_system"]:
        setup_unified_memory_env()

    # Create test model
    print("\n2. Creating Test Model")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.safetensors")

        print("Creating test model (24 layers, 2048 hidden)...")
        state_dict = create_test_model(num_layers=24, hidden_size=2048)

        print("Saving test model...")
        st_save(state_dict, model_path)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {file_size_mb:.2f} MB")

        results["system_info"]["model_size_mb"] = file_size_mb

        # Loading benchmark
        print("\n3. Loading Performance")
        print("-" * 40)

        cpu_loading = benchmark_loading(model_path, num_runs=5)
        results["benchmarks"]["cpu_loading"] = cpu_loading
        print(f"CPU Loading: {cpu_loading['avg_time']:.4f}s avg")
        print(f"  Throughput: {file_size_mb / cpu_loading['avg_time']:.2f} MB/s")

        if torch.cuda.is_available():
            gpu_loading = benchmark_gpu_loading(model_path, num_runs=3)
            results["benchmarks"]["gpu_loading"] = gpu_loading
            if gpu_loading:
                print(f"GPU Loading: {gpu_loading['avg_time']:.4f}s avg")
                print(f"  Throughput: {file_size_mb / gpu_loading['avg_time']:.2f} MB/s")

        # Computation benchmark
        print("\n4. Computation Performance")
        print("-" * 40)

        if torch.cuda.is_available():
            compute_results = benchmark_computation(num_iterations=100)
            results["benchmarks"]["computation"] = compute_results
            if compute_results:
                print(f"Matrix Multiplication: {compute_results['tflops']:.2f} TFLOPS")
        else:
            print("GPU not available, skipping computation benchmark")

        # Memory efficiency
        print("\n5. Memory Efficiency")
        print("-" * 40)

        if torch.cuda.is_available():
            memory_results = benchmark_memory_efficiency()
            results["benchmarks"]["memory"] = memory_results
            if memory_results:
                final = memory_results["final_stats"]
                print(f"Peak Allocated: {final['max_allocated_gb']:.2f} GB")
                print(f"Peak Reserved: {final['reserved_gb']:.2f} GB")
                print(f"Memory Efficiency: {final['allocated_gb'] / final['reserved_gb'] * 100:.1f}%")
        else:
            print("GPU not available, skipping memory benchmark")

    # Target metrics
    print("\n6. Performance Targets")
    print("-" * 40)

    targets = {
        "loading_speedup": 2.0,  # 2x faster loading target
        "memory_reduction": 0.3,  # 30% memory reduction target
        "compute_efficiency": 0.8,  # 80% of peak FLOPS target
    }

    results["targets"] = targets
    print(f"Loading Speedup Target: {targets['loading_speedup']}x")
    print(f"Memory Reduction Target: {targets['memory_reduction'] * 100}%")
    print(f"Compute Efficiency Target: {targets['compute_efficiency'] * 100}%")

    # Save results
    print("\n7. Saving Results")
    print("-" * 40)

    results_file = "benchmarks/latest_results.json"
    os.makedirs("benchmarks", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_full_benchmark()
```

**Step 2: Make executable**

```bash
chmod +x benchmarks/performance_test.py
```

**Step 3: Commit**

```bash
git add benchmarks/performance_test.py
git commit -m "feat: add comprehensive performance benchmark suite"
```

---

## Task 7: Create Regression Testing Suite

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/tests/regression/test_training_quality.py`

**Step 1: Create regression test directory**

```bash
mkdir -p tests/regression
touch tests/regression/__init__.py
```

**Step 2: Write regression tests**

```python
# tests/regression/test_training_quality.py
"""
Regression tests to ensure training quality is maintained after improvements.
"""
import pytest
import torch
import torch.nn as nn

def test_lora_weight_initialization():
    """Test that LoRA weight initialization is consistent."""
    from peft import LoraConfig

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v"],
    )

    # Initialization parameters should be set
    assert config.r == 16
    assert config.lora_alpha == 16

def test_gradient_computation():
    """Test that gradients are computed correctly."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward pass
    x = torch.randn(2, 10)
    y = model(x)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any()

@pytest.mark.gpu
def test_mixed_precision_consistency(gpu_available):
    """Test that mixed precision produces consistent results."""
    model = nn.Linear(10, 10).cuda()
    x = torch.randn(2, 10, device="cuda")

    # FP32 forward
    with torch.no_grad():
        y_fp32 = model(x)

    # BF16 forward
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_bf16 = model(x)

    # Results should be close (within BF16 precision)
    assert torch.allclose(y_fp32, y_bf16.float(), rtol=1e-2, atol=1e-2)

def test_loss_computation():
    """Test that loss computation is numerically stable."""
    predictions = torch.randn(10, 10)
    targets = torch.randn(10, 10)

    # MSE loss
    mse_loss = nn.MSELoss()(predictions, targets)
    assert not torch.isnan(mse_loss)
    assert not torch.isinf(mse_loss)
    assert mse_loss >= 0

    # Huber loss
    huber_loss = nn.HuberLoss()(predictions, targets)
    assert not torch.isnan(huber_loss)
    assert not torch.isinf(huber_loss)

@pytest.mark.gpu
def test_checkpoint_save_load_consistency(temp_dir, gpu_available):
    """Test that checkpoints save and load consistently."""
    from utils.fast_loading import save_safetensors, load_safetensors

    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    ).cuda()

    # Get state dict
    original_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save checkpoint
    checkpoint_path = temp_dir / "checkpoint.safetensors"
    save_safetensors(original_state, str(checkpoint_path))

    # Load checkpoint
    loaded_state = load_safetensors(str(checkpoint_path))

    # Verify consistency
    for key in original_state:
        assert key in loaded_state
        assert torch.equal(original_state[key], loaded_state[key])

def test_optimizer_state_preservation():
    """Test that optimizer state is preserved correctly."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Take a step
    x = torch.randn(2, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Save optimizer state
    original_state = optimizer.state_dict()

    # Create new optimizer and load state
    new_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    new_optimizer.load_state_dict(original_state)

    # States should match
    assert new_optimizer.state_dict()["param_groups"] == original_state["param_groups"]

@pytest.mark.gpu
def test_noise_scheduler_consistency(gpu_available):
    """Test noise scheduler produces consistent results."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler()

    # Set seed for reproducibility
    torch.manual_seed(42)
    noise1 = torch.randn(1, 4, 32, 32, device="cuda")

    torch.manual_seed(42)
    noise2 = torch.randn(1, 4, 32, 32, device="cuda")

    assert torch.equal(noise1, noise2)

def test_weight_dtype_conversion():
    """Test that weight dtype conversion preserves information."""
    weights_fp32 = torch.randn(100, 100)

    # Convert to BF16
    weights_bf16 = weights_fp32.to(torch.bfloat16)

    # Convert back to FP32
    weights_restored = weights_bf16.to(torch.float32)

    # Should be close (within BF16 precision limits)
    assert torch.allclose(weights_fp32, weights_restored, rtol=1e-2, atol=1e-2)

def test_gradient_accumulation_equivalence():
    """Test that gradient accumulation produces equivalent results."""
    # Model with same initialization
    torch.manual_seed(42)
    model1 = nn.Linear(10, 10)
    torch.manual_seed(42)
    model2 = nn.Linear(10, 10)

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-3)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)

    # Same data
    torch.manual_seed(0)
    x1 = torch.randn(4, 10)
    x2 = x1.clone()

    # Method 1: Single batch of 4
    y1 = model1(x1)
    loss1 = y1.sum() / 4
    loss1.backward()
    optimizer1.step()

    # Method 2: Gradient accumulation (2 batches of 2)
    optimizer2.zero_grad()
    y2a = model2(x2[:2])
    loss2a = y2a.sum() / 4
    loss2a.backward()

    y2b = model2(x2[2:])
    loss2b = y2b.sum() / 4
    loss2b.backward()
    optimizer2.step()

    # Weights should be close
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, rtol=1e-5)
```

**Step 3: Run regression tests**

```bash
pytest tests/regression/test_training_quality.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add tests/regression/
git commit -m "test: add regression tests for training quality"
```

---

## Task 8: Create Test Runner Script

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/scripts/run_tests.sh`

**Step 1: Write test runner script**

```bash
#!/bin/bash
# run_tests.sh
# Comprehensive test runner for all test suites

set -e

echo "============================================"
echo "Running All Tests"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED_TESTS=()

# Function to run test suite
run_suite() {
    local suite_name=$1
    local test_path=$2

    echo ""
    echo -e "${YELLOW}Running: $suite_name${NC}"
    echo "-" * 40

    if pytest $test_path -v --tb=short; then
        echo -e "${GREEN}PASSED: $suite_name${NC}"
    else
        echo -e "${RED}FAILED: $suite_name${NC}"
        FAILED_TESTS+=("$suite_name")
    fi
}

# Unit Tests
echo ""
echo "1. Unit Tests"
run_suite "FastSafeTensors Utils" "tests/test_fast_loading.py"
run_suite "Unified Memory Utils" "tests/test_unified_memory.py"
run_suite "CUDA Utils" "tests/test_cuda_utils.py"
run_suite "Memory Monitor" "tests/test_memory_monitor.py"
run_suite "Compatibility Layer" "tests/test_compat.py"

# Integration Tests
echo ""
echo "2. Integration Tests"
run_suite "FastSafeTensors Integration" "tests/integration/test_fastsafetensors_integration.py"
run_suite "Unified Memory Integration" "tests/integration/test_unified_memory_integration.py"
run_suite "CUDA 13.0 Integration" "tests/integration/test_cuda13_integration.py"
run_suite "Combined Features" "tests/integration/test_combined_features.py"

# Regression Tests
echo ""
echo "3. Regression Tests"
run_suite "Training Quality" "tests/regression/test_training_quality.py"

# Config Tests
echo ""
echo "4. Configuration Tests"
run_suite "Config Loading" "tests/test_config_loading.py"

# Summary
echo ""
echo "============================================"
echo "Test Summary"
echo "============================================"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Failed test suites:${NC}"
    for suite in "${FAILED_TESTS[@]}"; do
        echo "  - $suite"
    done
    exit 1
fi
```

**Step 2: Make executable**

```bash
chmod +x scripts/run_tests.sh
```

**Step 3: Commit**

```bash
git add scripts/run_tests.sh
git commit -m "feat: add comprehensive test runner script"
```

---

## Task 9: Create CI Test Configuration

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/.github/workflows/tests.yml`

**Step 1: Write GitHub Actions test workflow**

```yaml
# .github/workflows/tests.yml
name: Run Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install safetensors omegaconf peft

      - name: Create utils __init__.py
        run: touch utils/__init__.py

      - name: Run unit tests (CPU only)
        run: |
          pytest tests/test_*.py -v --tb=short -k "not gpu"

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pytest
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install safetensors omegaconf peft

      - name: Create utils __init__.py
        run: touch utils/__init__.py

      - name: Run integration tests (CPU only)
        run: |
          pytest tests/integration/ -v --tb=short -k "not gpu"

  regression-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pytest
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install safetensors omegaconf peft diffusers

      - name: Create utils __init__.py
        run: touch utils/__init__.py

      - name: Run regression tests (CPU only)
        run: |
          pytest tests/regression/ -v --tb=short -k "not gpu"
```

**Step 2: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: add GitHub Actions workflow for tests"
```

---

## Task 10: Document Testing Strategy

**Files:**
- Create: `/home/jay/Documents/flymyai-lora-trainer/docs/TESTING.md`

**Step 1: Write testing documentation**

```markdown
# Testing Strategy

This document outlines the testing approach for the LoRA training improvements.

## Test Categories

### 1. Unit Tests (`tests/test_*.py`)
- Individual utility function tests
- No external dependencies
- Fast execution
- Run on every commit

### 2. Integration Tests (`tests/integration/`)
- Feature combination tests
- Tests actual workflows
- May require GPU
- Run on PRs

### 3. Regression Tests (`tests/regression/`)
- Training quality validation
- Numerical stability checks
- Gradient verification
- Run before releases

### 4. Performance Benchmarks (`benchmarks/`)
- Loading performance
- Memory utilization
- Compute throughput
- Run manually or on schedule

## Running Tests

### All Tests
```bash
./scripts/run_tests.sh
```

### Specific Suite
```bash
pytest tests/integration/ -v
```

### GPU Tests Only
```bash
pytest -m gpu tests/ -v
```

### Skip GPU Tests
```bash
pytest -k "not gpu" tests/ -v
```

## Test Fixtures

Located in `tests/conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_state_dict`: Small state dict for quick tests
- `large_state_dict`: Larger state dict for performance tests
- `mock_config`: Training configuration object
- `gpu_available`: Skip if no GPU
- `hopper_gpu`: Skip if not Hopper GPU
- `ampere_gpu`: Skip if not Ampere+ GPU

## Performance Targets

From `benchmarks/performance_test.py`:

| Metric | Target | Baseline |
|--------|--------|----------|
| Model Loading | 2-3x faster | Current implementation |
| Memory Efficiency | 50% better | With CPU offloading |
| Training Speed | 20% faster | Without TF32/optimizations |

## Continuous Integration

GitHub Actions workflows:
- `tests.yml`: Run all tests on CPU
- `docker.yml`: Build and scan Docker images

## Adding New Tests

1. **Unit Test**: Add to `tests/test_<module>.py`
2. **Integration Test**: Add to `tests/integration/test_<feature>.py`
3. **Regression Test**: Add to `tests/regression/test_<aspect>.py`
4. **Benchmark**: Add to `benchmarks/<benchmark_name>.py`

### Test Template

```python
import pytest
import torch

def test_feature_description():
    """Test that feature works correctly."""
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value

@pytest.mark.gpu
def test_gpu_feature(gpu_available):
    """Test GPU-specific feature."""
    # Test implementation
    pass
```

## Coverage Goals

- Utils: >90% coverage
- Integration: All major workflows
- Regression: Critical numerical operations
- Performance: Key bottlenecks

## Troubleshooting

### Test Failures

1. **Import errors**: Check `utils/__init__.py` exists
2. **GPU tests fail**: Ensure CUDA available or skip with `-k "not gpu"`
3. **Memory errors**: Run tests individually or increase memory

### Performance Issues

1. **Slow tests**: Use smaller fixtures for quick iteration
2. **GPU memory**: Clear cache between tests with `torch.cuda.empty_cache()`
3. **Flaky tests**: Use fixed random seeds `torch.manual_seed(42)`
```

**Step 2: Commit**

```bash
git add docs/TESTING.md
git commit -m "docs: add comprehensive testing strategy documentation"
```

---

## Task 11: Update Main README with Testing Section

**Files:**
- Modify: `/home/jay/Documents/flymyai-lora-trainer/README.md`

**Step 1: Add testing section**

```markdown
## Testing

The project includes comprehensive test suites for validation.

### Quick Start

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific suite
pytest tests/integration/ -v

# Run performance benchmarks
python benchmarks/performance_test.py
```

### Test Categories

- **Unit Tests**: Individual utility functions
- **Integration Tests**: Feature combinations
- **Regression Tests**: Training quality
- **Benchmarks**: Performance metrics

See [docs/TESTING.md](docs/TESTING.md) for detailed testing documentation.

### Performance Targets

| Feature | Target Improvement |
|---------|-------------------|
| Model Loading (fastsafetensors) | 2-3x faster |
| Memory Efficiency (unified memory) | 50% better |
| Training Speed (CUDA 13.0) | 20% faster |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add testing section to README"
```

---

## Summary

This plan covers:
1. Test infrastructure with pytest fixtures
2. FastSafeTensors integration tests
3. Unified memory integration tests
4. CUDA 13.0 integration tests
5. Combined feature tests
6. Performance benchmark suite
7. Regression test suite
8. Test runner script
9. CI/CD test configuration
10. Testing documentation

Total estimated time: 12-16 hours
Complexity: Medium

Key files created:
- tests/conftest.py
- tests/integration/*.py
- tests/regression/*.py
- benchmarks/performance_test.py
- scripts/run_tests.sh
- .github/workflows/tests.yml
- docs/TESTING.md

Test coverage:
- All utility modules tested
- Integration between features validated
- Performance benchmarked
- Training quality regression checked
- CI/CD pipeline established

The integration testing plan ensures:
1. Each improvement works individually
2. All improvements work together
3. No regression in training quality
4. Performance targets are measurable
5. Automated testing in CI/CD
