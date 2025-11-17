"""
FastSafeTensors utility functions for optimized model loading and saving.

Provides multi-threaded loading and memory-mapped access for safetensors files.
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from fastsafetensors import fastsafe_open
    FASTSAFETENSORS_AVAILABLE = True
except ImportError:
    FASTSAFETENSORS_AVAILABLE = False
    # Fallback to standard safetensors
    from safetensors.torch import load_file as st_load_file

# Always use standard safetensors for saving (fastsafetensors is read-optimized)
from safetensors.torch import save_file as st_save_file


def load_safetensors(
    path: str,
    num_threads: int = 8,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load a safetensors file using fastsafetensors for optimized performance.

    Args:
        path: Path to the safetensors file
        num_threads: Number of threads for parallel loading (default: 8)
                     Note: fastsafetensors uses its own optimization, this param is for API compatibility
        device: Device to load tensors to (default: "cpu")

    Returns:
        Dictionary of tensor name to tensor
    """
    if FASTSAFETENSORS_AVAILABLE:
        state_dict = {}
        with fastsafe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    else:
        # Fallback to standard safetensors
        return st_load_file(path, device=device)


def save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Save a state dict to a safetensors file.

    Args:
        state_dict: Dictionary of tensor name to tensor
        path: Output path for the safetensors file
        metadata: Optional metadata dictionary
    """
    if metadata is None:
        metadata = {"format": "pt"}

    st_save_file(state_dict, path, metadata=metadata)


def is_fastsafetensors_available() -> bool:
    """Check if fastsafetensors is available."""
    return FASTSAFETENSORS_AVAILABLE
