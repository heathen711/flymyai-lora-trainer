import pytest
import torch
import tempfile
import os


def test_save_lora_weights_with_fastsafetensors():
    """Test that LoRA weights can be saved with fastsafetensors."""
    from utils.fast_loading import save_safetensors, load_safetensors

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock LoRA weights
        lora_weights = {
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(16, 64),
            "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(64, 16),
            "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": torch.randn(16, 64),
            "base_model.model.transformer_blocks.0.attn.to_k.lora_B.weight": torch.randn(64, 16),
        }

        save_path = os.path.join(tmpdir, "lora_weights.safetensors")
        save_safetensors(lora_weights, save_path, metadata={"format": "pt", "lora_rank": "16"})

        assert os.path.exists(save_path)

        # Verify we can load it back
        loaded = load_safetensors(save_path)
        assert len(loaded) == 4
        for key in lora_weights:
            assert key in loaded
            assert loaded[key].shape == lora_weights[key].shape
