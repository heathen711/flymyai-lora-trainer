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
