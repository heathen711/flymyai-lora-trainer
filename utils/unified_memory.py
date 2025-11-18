# utils/unified_memory.py
"""
Unified memory detection and configuration for DGX Spark (Grace Hopper) systems.

Grace Hopper architecture provides 128GB unified CPU-GPU memory, eliminating
the need for explicit memory transfers and CPU offloading.
"""
import os
import torch
from typing import Dict, Any

def is_unified_memory_system() -> bool:
    """
    Detect if running on a unified memory system (DGX Spark/Grace Hopper).

    Returns:
        True if unified memory is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Check for Grace Hopper architecture (compute capability 9.0+)
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 9:
        # Grace Hopper or newer
        return True

    # Check environment variable override
    if os.environ.get("UNIFIED_MEMORY", "").lower() in ("true", "1", "yes"):
        return True

    return False


def get_memory_config(unified_memory: bool = None) -> Dict[str, Any]:
    """
    Get optimal memory configuration based on system type.

    Args:
        unified_memory: Override detection. If None, auto-detect.

    Returns:
        Dictionary with memory configuration settings
    """
    if unified_memory is None:
        unified_memory = is_unified_memory_system()

    if unified_memory:
        return {
            "pin_memory": False,  # Not needed for unified memory
            "disable_cpu_offload": True,  # Keep everything in unified pool
            "prefetch_factor": 2,  # Lower prefetch, data already accessible
            "num_workers": 4,  # Fewer workers needed
            "memory_pool": "unified",
            "expandable_segments": True,
        }
    else:
        return {
            "pin_memory": True,  # Pinned memory for faster transfers
            "disable_cpu_offload": False,  # CPU offloading beneficial
            "prefetch_factor": 4,  # Higher prefetch for async loading
            "num_workers": 8,  # More workers to hide transfer latency
            "memory_pool": "standard",
            "expandable_segments": False,
        }


def setup_unified_memory_env() -> None:
    """
    Configure environment variables for unified memory operation.

    Sets CUDA allocator settings and memory pool configuration.
    """
    if is_unified_memory_system():
        # Enable expandable segments for better memory utilization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,backend:native"

        # Force managed memory allocation
        os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

        # Configure memory fraction
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception:
                pass  # May not be supported on all systems


def configure_accelerator_for_unified_memory(accelerator_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modify Accelerator kwargs for unified memory systems.

    Args:
        accelerator_kwargs: Base Accelerator configuration

    Returns:
        Modified configuration for unified memory
    """
    if is_unified_memory_system():
        # Don't let Accelerate manage device placement
        accelerator_kwargs["device_placement"] = False
    return accelerator_kwargs
