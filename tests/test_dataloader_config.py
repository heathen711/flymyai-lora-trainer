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
