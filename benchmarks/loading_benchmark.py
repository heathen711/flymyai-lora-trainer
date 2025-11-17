"""
Benchmark script to compare loading performance with and without fastsafetensors.
"""
import time
import torch
import tempfile
import os
from safetensors.torch import save_file as st_save
from utils.fast_loading import load_safetensors, is_fastsafetensors_available


def create_test_model(num_layers=24, hidden_size=4096):
    """Create a test state dict similar to a transformer model."""
    state_dict = {}
    for i in range(num_layers):
        state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = torch.randn(hidden_size * 4, hidden_size)
        state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = torch.randn(hidden_size, hidden_size * 4)
    return state_dict


def benchmark_loading(file_path, num_runs=5):
    """Benchmark loading times."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        state_dict = load_safetensors(file_path, num_threads=8)
        end = time.perf_counter()
        times.append(end - start)
        del state_dict
    return sum(times) / len(times)


def main():
    print(f"FastSafeTensors available: {is_fastsafetensors_available()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test model
        print("Creating test model...")
        state_dict = create_test_model(num_layers=24, hidden_size=2048)
        file_path = os.path.join(tmpdir, "test_model.safetensors")

        # Save test model
        print("Saving test model...")
        st_save(state_dict, file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        # Benchmark
        print("Benchmarking loading...")
        avg_time = benchmark_loading(file_path, num_runs=5)
        print(f"Average loading time: {avg_time:.4f} seconds")
        print(f"Throughput: {file_size_mb / avg_time:.2f} MB/s")


if __name__ == "__main__":
    main()
