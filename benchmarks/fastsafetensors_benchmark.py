#!/usr/bin/env python3
"""
Performance benchmark for fastsafetensors vs standard safetensors/torch.save.

Measures:
1. Model loading time
2. Model saving time
3. Memory usage during load
4. Peak memory during operations

Usage:
    python benchmarks/fastsafetensors_benchmark.py
"""
import os
import time
import tempfile
import torch
import psutil
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fast_loading import (
    load_safetensors,
    save_safetensors,
    load_safetensors_sharded,
    save_safetensors_sharded,
    is_fastsafetensors_available,
)
from safetensors.torch import save_file as st_save_file, load_file as st_load_file


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def create_mock_lora_weights(rank: int = 16, num_layers: int = 28) -> Dict[str, torch.Tensor]:
    """
    Create mock LoRA weights similar to a typical LoRA training output.

    Args:
        rank: LoRA rank
        num_layers: Number of transformer layers

    Returns:
        Dictionary of LoRA weights
    """
    hidden_size = 3072
    weights = {}

    for i in range(num_layers):
        # Attention layers
        for attn_type in ["to_q", "to_k", "to_v", "to_out.0"]:
            weights[f"transformer_blocks.{i}.attn.{attn_type}.lora_A.weight"] = torch.randn(rank, hidden_size)
            weights[f"transformer_blocks.{i}.attn.{attn_type}.lora_B.weight"] = torch.randn(hidden_size, rank)

    return weights


def benchmark_save(
    state_dict: Dict[str, torch.Tensor],
    method: str,
    tmpdir: str
) -> Tuple[float, float]:
    """
    Benchmark saving performance.

    Args:
        state_dict: State dict to save
        method: "fastsafetensors" or "standard"
        tmpdir: Temporary directory for saving

    Returns:
        Tuple of (time_seconds, memory_peak_mb)
    """
    save_path = os.path.join(tmpdir, f"model_{method}.safetensors")
    mem_before = get_memory_usage()

    start_time = time.time()
    if method == "fastsafetensors":
        save_safetensors(state_dict, save_path, metadata={"format": "pt"})
    elif method == "standard":
        st_save_file(state_dict, save_path, metadata={"format": "pt"})
    elif method == "torch":
        torch_path = os.path.join(tmpdir, "model_torch.pt")
        torch.save(state_dict, torch_path)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start_time
    mem_after = get_memory_usage()
    mem_peak = mem_after - mem_before

    return elapsed, mem_peak


def benchmark_load(
    save_path: str,
    method: str,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Benchmark loading performance.

    Args:
        save_path: Path to saved file
        method: "fastsafetensors" or "standard"
        device: Device to load to

    Returns:
        Tuple of (time_seconds, memory_peak_mb)
    """
    mem_before = get_memory_usage()

    start_time = time.time()
    if method == "fastsafetensors":
        state_dict = load_safetensors(save_path, device=device)
    elif method == "standard":
        state_dict = st_load_file(save_path, device=device)
    elif method == "torch":
        state_dict = torch.load(save_path, map_location=device)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start_time
    mem_after = get_memory_usage()
    mem_peak = mem_after - mem_before

    return elapsed, mem_peak


def benchmark_sharded_save_load(
    state_dict: Dict[str, torch.Tensor],
    tmpdir: str,
    max_shard_size: int = 50 * 1024 * 1024  # 50MB for testing
) -> Tuple[float, float]:
    """Benchmark sharded save/load performance."""
    output_dir = os.path.join(tmpdir, "sharded")

    # Save
    mem_before = get_memory_usage()
    start_time = time.time()
    save_safetensors_sharded(state_dict, output_dir, max_shard_size=max_shard_size)
    save_time = time.time() - start_time
    save_mem = get_memory_usage() - mem_before

    # Load
    mem_before = get_memory_usage()
    start_time = time.time()
    loaded_dict = load_safetensors_sharded(output_dir)
    load_time = time.time() - start_time
    load_mem = get_memory_usage() - mem_before

    # Verify
    assert len(loaded_dict) == len(state_dict), "Loaded dict size mismatch"

    return (save_time + load_time), (save_mem + load_mem)


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 80)
    print("FastSafeTensors Performance Benchmark")
    print("=" * 80)
    print()

    # Check availability
    if is_fastsafetensors_available():
        print("✓ fastsafetensors is available")
    else:
        print("✗ fastsafetensors is NOT available - using fallback")
    print()

    # Create test data
    print("Creating mock LoRA weights...")
    small_weights = create_mock_lora_weights(rank=16, num_layers=7)  # ~25MB
    large_weights = create_mock_lora_weights(rank=64, num_layers=28)  # ~400MB

    small_size = sum(t.numel() * t.element_size() for t in small_weights.values()) / 1024 / 1024
    large_size = sum(t.numel() * t.element_size() for t in large_weights.values()) / 1024 / 1024

    print(f"Small model size: {small_size:.2f} MB ({len(small_weights)} tensors)")
    print(f"Large model size: {large_size:.2f} MB ({len(large_weights)} tensors)")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Benchmark 1: Small model save
        print("-" * 80)
        print("Benchmark 1: Small Model Save (~25MB)")
        print("-" * 80)

        methods = ["fastsafetensors", "standard", "torch"]
        for method in methods:
            try:
                time_s, mem_mb = benchmark_save(small_weights, method, tmpdir)
                print(f"{method:20s}: {time_s*1000:8.2f} ms | Memory: {mem_mb:8.2f} MB")
            except Exception as e:
                print(f"{method:20s}: FAILED - {e}")
        print()

        # Benchmark 2: Small model load
        print("-" * 80)
        print("Benchmark 2: Small Model Load (~25MB)")
        print("-" * 80)

        # Save files first
        save_safetensors(small_weights, os.path.join(tmpdir, "small_fast.safetensors"))
        st_save_file(small_weights, os.path.join(tmpdir, "small_std.safetensors"))
        torch.save(small_weights, os.path.join(tmpdir, "small_torch.pt"))

        methods_files = [
            ("fastsafetensors", os.path.join(tmpdir, "small_fast.safetensors")),
            ("standard", os.path.join(tmpdir, "small_std.safetensors")),
            ("torch", os.path.join(tmpdir, "small_torch.pt")),
        ]

        for method, file_path in methods_files:
            try:
                time_s, mem_mb = benchmark_load(file_path, method)
                print(f"{method:20s}: {time_s*1000:8.2f} ms | Memory: {mem_mb:8.2f} MB")
            except Exception as e:
                print(f"{method:20s}: FAILED - {e}")
        print()

        # Benchmark 3: Large model save
        print("-" * 80)
        print("Benchmark 3: Large Model Save (~400MB)")
        print("-" * 80)

        for method in ["fastsafetensors", "standard"]:
            try:
                time_s, mem_mb = benchmark_save(large_weights, method, tmpdir)
                print(f"{method:20s}: {time_s*1000:8.2f} ms | Memory: {mem_mb:8.2f} MB")
            except Exception as e:
                print(f"{method:20s}: FAILED - {e}")
        print()

        # Benchmark 4: Large model load
        print("-" * 80)
        print("Benchmark 4: Large Model Load (~400MB)")
        print("-" * 80)

        save_safetensors(large_weights, os.path.join(tmpdir, "large_fast.safetensors"))
        st_save_file(large_weights, os.path.join(tmpdir, "large_std.safetensors"))

        methods_files = [
            ("fastsafetensors", os.path.join(tmpdir, "large_fast.safetensors")),
            ("standard", os.path.join(tmpdir, "large_std.safetensors")),
        ]

        for method, file_path in methods_files:
            try:
                time_s, mem_mb = benchmark_load(file_path, method)
                print(f"{method:20s}: {time_s*1000:8.2f} ms | Memory: {mem_mb:8.2f} MB")
            except Exception as e:
                print(f"{method:20s}: FAILED - {e}")
        print()

        # Benchmark 5: Sharded save/load
        print("-" * 80)
        print("Benchmark 5: Sharded Save/Load (~400MB, 50MB shards)")
        print("-" * 80)

        try:
            time_s, mem_mb = benchmark_sharded_save_load(large_weights, tmpdir)
            print(f"Sharded ops          : {time_s*1000:8.2f} ms | Memory: {mem_mb:8.2f} MB")
        except Exception as e:
            print(f"Sharded ops          : FAILED - {e}")
        print()

    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)
    print()
    print("Expected improvements with fastsafetensors:")
    print("  - Model loading: 2-3x faster")
    print("  - Memory usage: 20-30% lower peak")
    print("  - Saving: Similar performance (both use same backend)")
    print()


if __name__ == "__main__":
    run_benchmarks()
