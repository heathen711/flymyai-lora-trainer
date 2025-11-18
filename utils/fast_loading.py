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


def load_safetensors_mmap(
    path: str,
    keys: Optional[list] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load specific tensors from a safetensors file using memory mapping.

    Provides lazy loading - tensors are loaded on-demand from disk.
    Useful for large models where you only need specific layers.

    Args:
        path: Path to the safetensors file
        keys: List of specific tensor keys to load (None = load all)
        device: Device to load tensors to

    Returns:
        Dictionary of tensor name to tensor
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    try:
        if FASTSAFETENSORS_AVAILABLE:
            state_dict = {}
            with fastsafe_open(path, framework="pt", device=device) as f:
                load_keys = keys if keys is not None else f.keys()
                for key in load_keys:
                    if key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            # Fallback to standard safetensors
            full_dict = st_load_file(path, device=device)
            if keys is not None:
                return {k: v for k, v in full_dict.items() if k in keys}
            return full_dict
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors file {path}: {e}") from e


def save_safetensors_sharded(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    max_shard_size: int = 10 * 1024 * 1024 * 1024,  # 10GB default
    metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Save a large state dict to multiple sharded safetensors files.

    Splits the state dict into multiple files if it exceeds max_shard_size.
    Useful for models >10GB.

    Args:
        state_dict: Dictionary of tensor name to tensor
        output_dir: Output directory for sharded files
        max_shard_size: Maximum size per shard in bytes
        metadata: Optional metadata dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    if metadata is None:
        metadata = {"format": "pt"}

    # Calculate tensor sizes
    tensor_sizes = {k: v.numel() * v.element_size() for k, v in state_dict.items()}
    total_size = sum(tensor_sizes.values())

    # If total size is under limit, save as single file
    if total_size <= max_shard_size:
        save_path = os.path.join(output_dir, "model.safetensors")
        save_safetensors(state_dict, save_path, metadata)
        return

    # Split into shards
    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor_sizes[key]

        # Start new shard if adding this tensor would exceed limit
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append((shard_idx, current_shard))
            current_shard = {}
            current_size = 0
            shard_idx += 1

        current_shard[key] = tensor
        current_size += tensor_size

    # Add final shard
    if current_shard:
        shards.append((shard_idx, current_shard))

    # Save each shard
    shard_metadata = metadata.copy()
    shard_metadata["total_shards"] = str(len(shards))

    for idx, shard_dict in shards:
        shard_metadata["shard_index"] = str(idx)
        shard_path = os.path.join(output_dir, f"model-{idx:05d}-of-{len(shards):05d}.safetensors")
        save_safetensors(shard_dict, shard_path, shard_metadata)


def load_safetensors_sharded(
    model_dir: str,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load a sharded model from multiple safetensors files.

    Args:
        model_dir: Directory containing sharded safetensors files
        device: Device to load tensors to

    Returns:
        Combined dictionary of all tensors
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find all shard files
    shard_files = sorted([
        f for f in os.listdir(model_dir)
        if f.endswith(".safetensors")
    ])

    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    # Load and merge all shards
    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        shard_dict = load_safetensors(shard_path, device=device)
        state_dict.update(shard_dict)

    return state_dict


def save_embeddings_safetensors(
    embeddings: Dict[str, torch.Tensor],
    path: str
) -> None:
    """
    Save embeddings (text/image) to a safetensors file.

    Optimized replacement for torch.save() for caching embeddings.

    Args:
        embeddings: Dictionary of embedding tensors
        path: Output path
    """
    # Ensure all tensors are on CPU for saving
    cpu_embeddings = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in embeddings.items()}
    save_safetensors(cpu_embeddings, path, metadata={"type": "embeddings"})


def load_embeddings_safetensors(
    path: str,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load embeddings from a safetensors file.

    Optimized replacement for torch.load() for loading cached embeddings.

    Args:
        path: Path to embeddings file
        device: Device to load to

    Returns:
        Dictionary of embedding tensors
    """
    return load_safetensors(path, device=device)
