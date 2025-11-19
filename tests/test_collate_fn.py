#!/usr/bin/env python3
"""
Test script to verify the custom collate function handles variable-sized latents correctly.
"""
import torch
import pytest
from image_datasets.dataset import custom_collate_fn


def test_variable_sized_latents():
    """Test that collate function correctly pads variable-sized image latents."""
    # Simulate batch with variable-sized latents (as seen in the error)
    # Entry 0: [16, 1, 84, 128]
    # Entry 3: [16, 1, 128, 84]
    latent1 = torch.randn(16, 1, 84, 128)
    latent2 = torch.randn(16, 1, 100, 100)
    latent3 = torch.randn(16, 1, 64, 128)
    latent4 = torch.randn(16, 1, 128, 84)

    text_embed1 = torch.randn(256, 3584)
    text_embed2 = torch.randn(256, 3584)
    text_embed3 = torch.randn(256, 3584)
    text_embed4 = torch.randn(256, 3584)

    text_mask1 = torch.ones(256, dtype=torch.bool)
    text_mask2 = torch.ones(256, dtype=torch.bool)
    text_mask3 = torch.ones(256, dtype=torch.bool)
    text_mask4 = torch.ones(256, dtype=torch.bool)

    # Create batch
    batch = [
        (latent1, text_embed1, text_mask1),
        (latent2, text_embed2, text_mask2),
        (latent3, text_embed3, text_mask3),
        (latent4, text_embed4, text_mask4),
    ]

    # Test collate function
    imgs, text_embeds, text_masks = custom_collate_fn(batch)

    # Verify all images are padded to the same size
    assert imgs.shape[0] == 4, f"Expected batch size 4, got {imgs.shape[0]}"
    assert imgs.shape[1] == 16, f"Expected 16 channels, got {imgs.shape[1]}"
    assert imgs.shape[2] == 1, f"Expected 1 time dim, got {imgs.shape[2]}"

    # Check that spatial dimensions are the maximum from the batch
    max_h_expected = max(84, 100, 64, 128)
    max_w_expected = max(128, 100, 128, 84)
    assert imgs.shape[3] == max_h_expected, f"Expected height {max_h_expected}, got {imgs.shape[3]}"
    assert imgs.shape[4] == max_w_expected, f"Expected width {max_w_expected}, got {imgs.shape[4]}"

    # Verify text embeddings and masks are batched correctly
    assert text_embeds.shape == (4, 256, 3584)
    assert text_masks.shape == (4, 256)


def test_uniform_sized_latents():
    """Test that collate function works with uniform-sized latents (no padding needed)."""
    # All latents same size
    latent1 = torch.randn(16, 1, 128, 128)
    latent2 = torch.randn(16, 1, 128, 128)
    latent3 = torch.randn(16, 1, 128, 128)

    text_embed1 = torch.randn(256, 3584)
    text_embed2 = torch.randn(256, 3584)
    text_embed3 = torch.randn(256, 3584)

    text_mask1 = torch.ones(256, dtype=torch.bool)
    text_mask2 = torch.ones(256, dtype=torch.bool)
    text_mask3 = torch.ones(256, dtype=torch.bool)

    batch = [
        (latent1, text_embed1, text_mask1),
        (latent2, text_embed2, text_mask2),
        (latent3, text_embed3, text_mask3),
    ]

    imgs, text_embeds, text_masks = custom_collate_fn(batch)

    # Verify shapes
    assert imgs.shape == (3, 16, 1, 128, 128)
    assert text_embeds.shape == (3, 256, 3584)
    assert text_masks.shape == (3, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
