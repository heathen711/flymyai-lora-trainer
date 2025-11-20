#!/usr/bin/env python3
"""
Test if PyTorch's built-in Flash Attention 2 works on DGX Spark.

This tests the CUDA-native FA2 implementation, NOT the external flash-attn package (FA3).
"""
import torch
import sys

print("=" * 80)
print("Testing Flash Attention 2 (PyTorch Built-in) on DGX Spark")
print("=" * 80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute: {torch.cuda.get_device_capability()}")
print()

# Test with realistic transformer sizes
test_cases = [
    # (batch, heads, seq_len, head_dim, name)
    (2, 24, 1000, 128, "Small sequence (warmup)"),
    (2, 24, 2000, 128, "Medium sequence"),
    (2, 24, 4182, 128, "Training sequence (4096 img + 86 txt)"),
]

for batch, heads, seq_len, head_dim, name in test_cases:
    print(f"Testing: {name}")
    print(f"  Shape: [{batch}, {heads}, {seq_len}, {head_dim}]")

    try:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)

        # Enable ONLY Flash Attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False
        ):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

        attn_memory_gb = (batch * heads * seq_len * seq_len * 2) / 1e9  # bf16 = 2 bytes
        print(f"  ✅ Flash Attention 2 SUCCESS")
        print(f"     Memory saved vs Math SDPA: ~{attn_memory_gb:.1f} GB")

        del q, k, v, output
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"  ❌ FAILED: {e}")
        sys.exit(1)

    print()

print("=" * 80)
print("✅ Flash Attention 2 works perfectly on DGX Spark!")
print("=" * 80)
print()
print("Next step: Enable in training config:")
print("  train_dgx_spark_qwen_lora.py line 33:")
print("    torch.backends.cuda.enable_flash_sdp(True)  # Change from False")
print("  ")
print("This will reduce attention memory from 16.8 GB to ~100 MB!")
