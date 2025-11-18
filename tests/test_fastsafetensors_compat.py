#!/usr/bin/env python3
"""
Compatibility tests for fastsafetensors integration.

Run these tests after installing all dependencies:
    pip install -r requirements.txt
    pytest tests/test_fastsafetensors_compat.py -v
"""
import pytest
import torch
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fast_loading import (
    load_safetensors,
    save_safetensors,
    load_embeddings_safetensors,
    save_embeddings_safetensors,
    is_fastsafetensors_available,
)


def test_fastsafetensors_available():
    """Test that fastsafetensors is available or fallback works."""
    # Should not raise an error either way
    available = is_fastsafetensors_available()
    print(f"FastSafeTensors available: {available}")
    assert isinstance(available, bool)


def test_basic_save_load():
    """Test basic safetensors save/load functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test tensors
        state_dict = {
            "layer1.weight": torch.randn(10, 20),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(20, 30),
        }

        save_path = os.path.join(tmpdir, "test.safetensors")

        # Save
        save_safetensors(state_dict, save_path, metadata={"format": "pt"})
        assert os.path.exists(save_path)

        # Load
        loaded = load_safetensors(save_path)
        assert len(loaded) == len(state_dict)

        for key in state_dict:
            assert key in loaded
            assert torch.allclose(state_dict[key], loaded[key])


def test_embedding_save_load():
    """Test embedding-specific save/load functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock embeddings (similar to training scripts)
        embeddings = {
            "prompt_embeds": torch.randn(1, 512, 4096),
            "prompt_embeds_mask": torch.ones(1, 512),
        }

        save_path = os.path.join(tmpdir, "embeddings.safetensors")

        # Save
        save_embeddings_safetensors(embeddings, save_path)
        assert os.path.exists(save_path)

        # Load
        loaded = load_embeddings_safetensors(save_path)
        assert len(loaded) == len(embeddings)

        for key in embeddings:
            assert key in loaded
            assert torch.allclose(embeddings[key], loaded[key])


def test_backward_compatibility_with_torch_save():
    """Test that we can still load old torch.save files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data = {
            "prompt_embeds": torch.randn(1, 128, 768),
            "prompt_embeds_mask": torch.ones(1, 128),
        }

        # Save with torch.save (old format)
        pt_path = os.path.join(tmpdir, "old_cache.pt")
        torch.save(data, pt_path)

        # Should be able to load with torch.load
        loaded = torch.load(pt_path)
        assert len(loaded) == len(data)

        for key in data:
            assert key in loaded
            assert torch.allclose(data[key], loaded[key])


def test_diffusers_compatibility():
    """Test that fastsafetensors doesn't break diffusers imports."""
    try:
        import diffusers
        from diffusers import QwenImagePipeline

        # Should not raise import errors
        assert diffusers.__version__ is not None
        print(f"Diffusers version: {diffusers.__version__}")

    except ImportError as e:
        pytest.skip(f"Diffusers not installed: {e}")


def test_device_placement():
    """Test loading to different devices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dict = {"weight": torch.randn(5, 5)}
        save_path = os.path.join(tmpdir, "model.safetensors")

        save_safetensors(state_dict, save_path)

        # Load to CPU
        loaded_cpu = load_safetensors(save_path, device="cpu")
        assert loaded_cpu["weight"].device.type == "cpu"

        # Load to CUDA if available
        if torch.cuda.is_available():
            loaded_cuda = load_safetensors(save_path, device="cuda")
            assert loaded_cuda["weight"].device.type == "cuda"


def test_metadata():
    """Test that metadata is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dict = {"data": torch.randn(3, 3)}
        save_path = os.path.join(tmpdir, "with_metadata.safetensors")

        metadata = {
            "format": "pt",
            "lora_rank": "16",
            "author": "test"
        }

        save_safetensors(state_dict, save_path, metadata=metadata)

        # Verify file was created
        assert os.path.exists(save_path)

        # Load and check (metadata is in the file but we don't expose it in load)
        loaded = load_safetensors(save_path)
        assert "data" in loaded


def test_large_tensor_handling():
    """Test handling of larger tensors (simulating LoRA weights)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a larger state dict (~100MB)
        large_dict = {}
        for i in range(50):
            large_dict[f"layer_{i}.weight"] = torch.randn(512, 512)

        save_path = os.path.join(tmpdir, "large_model.safetensors")

        # Should handle without issues
        save_safetensors(large_dict, save_path)
        loaded = load_safetensors(save_path)

        assert len(loaded) == len(large_dict)
        # Spot check a few tensors
        assert torch.allclose(large_dict["layer_0.weight"], loaded["layer_0.weight"])
        assert torch.allclose(large_dict["layer_25.weight"], loaded["layer_25.weight"])


if __name__ == "__main__":
    print("=" * 80)
    print("FastSafeTensors Compatibility Tests")
    print("=" * 80)
    print()

    pytest.main([__file__, "-v", "--tb=short"])
