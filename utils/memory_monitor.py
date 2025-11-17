# utils/memory_monitor.py
"""
Memory usage monitoring for training scripts.

Provides utilities to track GPU and unified memory usage.
"""
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_memory_stats() -> Dict[str, Any]:
    """
    Get current memory statistics.

    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }

    stats = torch.cuda.memory_stats()

    return {
        "allocated_gb": stats.get("allocated_bytes.all.current", 0) / (1024**3),
        "reserved_gb": stats.get("reserved_bytes.all.current", 0) / (1024**3),
        "max_allocated_gb": stats.get("allocated_bytes.all.peak", 0) / (1024**3),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
    }


def log_memory_usage(checkpoint_name: str = "") -> None:
    """
    Log current memory usage.

    Args:
        checkpoint_name: Optional name for the logging checkpoint
    """
    if not torch.cuda.is_available():
        logger.info(f"[{checkpoint_name}] No CUDA available")
        return

    stats = get_memory_stats()
    logger.info(
        f"[{checkpoint_name}] Memory: "
        f"Allocated={stats['allocated_gb']:.2f}GB, "
        f"Reserved={stats['reserved_gb']:.2f}GB, "
        f"Peak={stats['max_allocated_gb']:.2f}GB"
    )


def reset_peak_memory_stats() -> None:
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
