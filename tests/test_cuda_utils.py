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
