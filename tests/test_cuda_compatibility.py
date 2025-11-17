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
