"""
FastSafeTensors utility functions for optimized model loading and saving.

Provides multi-threaded loading and memory-mapped access for safetensors files.
"""
import os
import torch
from typing import Dict, Optional

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
        num_threads: Number of threads (reserved for future API compatibility)
        device: Device to load tensors to (default: "cpu")

    Returns:
        Dictionary of tensor name to tensor

    Raises:
        FileNotFoundError: If the file does not exist
        RuntimeError: If the file is corrupted or cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    try:
        if FASTSAFETENSORS_AVAILABLE:
            state_dict = {}
            with fastsafe_open(path, framework="pt", device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            # Fallback to standard safetensors
            return st_load_file(path, device=device)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors file {path}: {e}") from e


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

    Raises:
        RuntimeError: If the file cannot be saved
    """
    if metadata is None:
        metadata = {"format": "pt"}

    try:
        st_save_file(state_dict, path, metadata=metadata)
    except Exception as e:
        raise RuntimeError(f"Failed to save safetensors file to {path}: {e}") from e


def is_fastsafetensors_available() -> bool:
    """Check if fastsafetensors is available."""
    return FASTSAFETENSORS_AVAILABLE
