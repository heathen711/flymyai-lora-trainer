import pytest
import torch
import tempfile
import os
from safetensors.torch import save_file as st_save


def test_load_safetensors_file():
    """Test loading a safetensors file with fastsafetensors."""
    from utils.fast_loading import load_safetensors

    # Create a temporary safetensors file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.safetensors")
        test_tensor = {"weight": torch.randn(10, 10)}
        st_save(test_tensor, test_path)

        # Load with our utility
        loaded = load_safetensors(test_path)
        assert "weight" in loaded
        assert loaded["weight"].shape == (10, 10)


def test_save_safetensors_file():
    """Test saving a state dict with fastsafetensors."""
    from utils.fast_loading import save_safetensors

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.safetensors")
        test_tensor = {"weight": torch.randn(10, 10)}

        save_safetensors(test_tensor, test_path)
        assert os.path.exists(test_path)


def test_load_with_num_threads():
    """Test that num_threads parameter is respected."""
    from utils.fast_loading import load_safetensors

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.safetensors")
        test_tensor = {"weight": torch.randn(10, 10)}
        st_save(test_tensor, test_path)

        # Should not raise with num_threads parameter
        loaded = load_safetensors(test_path, num_threads=4)
        assert "weight" in loaded
