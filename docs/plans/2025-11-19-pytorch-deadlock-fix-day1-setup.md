# PyTorch Deadlock Fix - Day 1 Setup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up repository structure, build scripts, test harness, and Dockerfile for PyTorch SDPA deadlock debugging on DGX Spark

**Architecture:** Create standalone repository with Docker-based PyTorch build system, test scripts for SDPA validation, and automated testing harness. Uses existing DGX-Spark-ONNX image as base.

**Tech Stack:** Bash, Python 3.12, Docker, PyTorch (to be built), CUDA 13.0

---

## Task 1: Create Repository Structure

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/` (directory)
- Create: `/home/jay/Documents/pytorch-deadlock-fix/patches/` (directory)
- Create: `/home/jay/Documents/pytorch-deadlock-fix/logs/` (directory)
- Create: `/home/jay/Documents/pytorch-deadlock-fix/wheels/` (directory)
- Create: `/home/jay/Documents/pytorch-deadlock-fix/analysis/` (directory)

**Step 1: Create directory structure**

```bash
mkdir -p ~/Documents/pytorch-deadlock-fix/{patches,logs,wheels,analysis}
```

**Step 2: Verify structure**

Run: `ls -la ~/Documents/pytorch-deadlock-fix/`
Expected: Shows patches/, logs/, wheels/, analysis/ directories

**Step 3: Initialize git repository**

```bash
cd ~/Documents/pytorch-deadlock-fix
git init
```

**Step 4: Create .gitignore**

Create `/home/jay/Documents/pytorch-deadlock-fix/.gitignore`:
```
# Build artifacts
pytorch/
*.pyc
__pycache__/
*.whl
*.log
test_venv/

# Logs (keep directory, ignore contents)
logs/*.log
logs/*.txt

# Wheels (too large for git)
wheels/*.whl

# Docker
.dockerignore

# IDE
.vscode/
.idea/
*.swp
*.swo
```

**Step 5: Initial commit**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add .gitignore
git commit -m "chore: initialize pytorch-deadlock-fix repository"
```

---

## Task 2: Create README Documentation

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/README.md`

**Step 1: Write README**

Create `/home/jay/Documents/pytorch-deadlock-fix/README.md`:
```markdown
# PyTorch SDPA Deadlock Fix for DGX Spark

Fix for Math SDPA backend deadlock on ARM64 Blackwell GB10 with CUDA 13.0 and unified memory.

## Problem

Training Qwen-Image LoRA on DGX Spark deadlocks during first transformer forward pass in Math SDPA backend.

## Approach

1. **Instrument** PyTorch SDPA with extensive logging to identify exact deadlock location
2. **Analyze** logs to determine which operation hangs (cuBLAS, softmax, transpose, or allocator)
3. **Fix** root cause with ARM64+unified memory specific patches
4. **Validate** with progressive testing (standalone → 1 step → 10 steps → full training)

## Quick Start

### Phase 1: Build Instrumented Wheel

```bash
# Get current PyTorch commit
python3 -c "import torch; print(torch.version.git_version)"

# Build instrumented wheel
./build_instrumented.sh instrumentation

# Test installation
./test_fix.sh wheels/torch-*-instrumented.whl instrumentation
```

### Phase 2: Analyze Deadlock

Review `training_instrumented.log` to identify which scenario:
- Scenario 1: cuBLAS stream ordering
- Scenario 2: Softmax reduction kernel
- Scenario 3: Transpose memory layout
- Scenario 4: CUDA driver page fault handler

### Phase 3: Apply Fix and Validate

```bash
# Build fixed wheel (replace X with scenario number)
./build_instrumented.sh fix 02_fix_scenario_X.patch

# Validate fix
./test_fix.sh wheels/torch-*-fixed.whl fix
```

## Repository Structure

```
pytorch-deadlock-fix/
├── README.md                     # This file
├── Dockerfile                    # Build environment
├── build_instrumented.sh         # Build script
├── test_fix.sh                   # Testing harness
├── test_sdpa.py                  # Standalone SDPA test
├── benchmark_fix.py              # Performance benchmark
├── patches/                      # Patch files for PyTorch
│   ├── 01_sdpa_instrumentation.patch
│   ├── 02_fix_scenario_1.patch
│   ├── 02_fix_scenario_2.patch
│   ├── 02_fix_scenario_3.patch
│   └── 02_fix_scenario_4.patch
├── logs/                         # Build and test logs
├── wheels/                       # Built wheels
└── analysis/                     # Root cause analysis docs
    ├── deadlock_location.md
    ├── root_cause.md
    └── fix_explanation.md
```

## Requirements

- DGX Spark (ARM64 Blackwell GB10)
- CUDA 13.0+
- Docker with ARM64 support
- DGX-Spark-ONNX base image

## Timeline

- Day 1 (4-6h): Setup & instrumentation build
- Day 2 (2-4h): Analysis & root cause identification
- Day 3 (6-8h): Fix implementation
- Day 4 (4-6h): Validation & documentation

## References

- Design: `/home/jay/Documents/flymyai-lora-trainer/docs/plans/2025-11-19-pytorch-deadlock-fix-design.md`
- Related: `/home/jay/Documents/DGX-Spark-PyTorch/`
```

**Step 2: Commit README**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add README.md
git commit -m "docs: add README with project overview and quick start"
```

---

## Task 3: Create Standalone SDPA Test Script

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/test_sdpa.py`

**Step 1: Write test script**

Create `/home/jay/Documents/pytorch-deadlock-fix/test_sdpa.py`:
```python
#!/usr/bin/env python3
"""
Standalone SDPA test to verify Math backend works without deadlock.

This script tests the Math SDPA backend in isolation to identify if
the deadlock occurs in basic attention computation or only during
full training.
"""
import torch
import sys

def main():
    print("=" * 60)
    print("Standalone SDPA Test - Math Backend Only")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Architectures: {torch.cuda.get_arch_list()}")
    print("")

    print("Testing SDPA with instrumentation...")
    sys.stderr.flush()
    sys.stdout.flush()

    # Create test tensors
    print("Creating test tensors...")
    q = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.bfloat16)

    print(f"  Q shape: {q.shape}, dtype: {q.dtype}, device: {q.device}")
    print(f"  K shape: {k.shape}, dtype: {k.dtype}, device: {k.device}")
    print(f"  V shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
    print("")

    print("Calling SDPA with Math backend only...")
    sys.stderr.flush()
    sys.stdout.flush()

    # Force Math backend, disable all others
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False
    ):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    print(f"✅ SDPA completed! Output shape: {out.shape}")
    print(f"   Output dtype: {out.dtype}, device: {out.device}")
    print(f"   Output min: {out.min().item():.4f}, max: {out.max().item():.4f}")
    print("=" * 60)
    print("Test PASSED - No deadlock detected")
    print("=" * 60)
    sys.stderr.flush()
    sys.stdout.flush()

if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x ~/Documents/pytorch-deadlock-fix/test_sdpa.py
```

**Step 3: Test script runs (will use current PyTorch)**

Run: `cd ~/Documents/pytorch-deadlock-fix && python test_sdpa.py 2>&1 | head -20`
Expected: Should show PyTorch version and device info (may deadlock with current wheel)

**Step 4: Commit test script**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add test_sdpa.py
git commit -m "test: add standalone SDPA test for Math backend"
```

---

## Task 4: Create Performance Benchmark Script

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/benchmark_fix.py`

**Step 1: Write benchmark script**

Create `/home/jay/Documents/pytorch-deadlock-fix/benchmark_fix.py`:
```python
#!/usr/bin/env python3
"""
Benchmark SDPA performance to measure overhead of fixes.

Compares throughput and latency before/after applying fixes.
Target: <20% overhead acceptable, <10% ideal.
"""
import torch
import time
import sys

def benchmark_sdpa(num_iterations=100):
    """Benchmark SDPA throughput and latency"""

    print("Initializing tensors...")
    q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.bfloat16)

    print(f"Running {num_iterations} iterations...")
    print(f"  Tensor shape: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"  Dtype: {q.dtype}, Device: {q.device}")
    print("")

    # Warmup
    print("Warming up (10 iterations)...")
    for _ in range(10):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    print("Warmup complete")
    print("")

    # Benchmark
    print("Benchmarking...")
    start = time.time()

    for _ in range(num_iterations):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = num_iterations / elapsed
    latency_ms = (elapsed / num_iterations) * 1000

    print("")
    print("=" * 60)
    print("SDPA Benchmark Results")
    print("=" * 60)
    print(f"Iterations:   {num_iterations}")
    print(f"Total time:   {elapsed:.3f} seconds")
    print(f"Throughput:   {throughput:.2f} iter/s")
    print(f"Avg latency:  {latency_ms:.2f} ms")
    print("=" * 60)
    print("")

    return throughput, latency_ms

def main():
    print("=" * 60)
    print("PyTorch SDPA Performance Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print("")

    benchmark_sdpa(100)

if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x ~/Documents/pytorch-deadlock-fix/benchmark_fix.py
```

**Step 3: Commit benchmark script**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add benchmark_fix.py
git commit -m "perf: add SDPA benchmark script"
```

---

## Task 5: Create Build Script

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/build_instrumented.sh`

**Step 1: Write build script**

Create `/home/jay/Documents/pytorch-deadlock-fix/build_instrumented.sh`:
```bash
#!/bin/bash
set -euo pipefail

PHASE=$1  # "instrumentation" or "fix"
DATE_TAG="debug-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "Building Instrumented PyTorch - Phase: $PHASE"
echo "=========================================="

# Get current wheel commit hash
CURRENT_COMMIT=$(python3 -c "import torch; print(torch.version.git_version)" 2>/dev/null || echo "main")
echo "Current PyTorch commit: $CURRENT_COMMIT"

# Select patch based on phase
if [ "$PHASE" = "instrumentation" ]; then
    PATCH_FILE="01_sdpa_instrumentation.patch"
elif [ "$PHASE" = "fix" ]; then
    # User specifies which fix after analysis
    PATCH_FILE="${2:-02_fix_scenario_1.patch}"
    echo "Using fix patch: $PATCH_FILE"
else
    echo "Usage: $0 <instrumentation|fix> [fix_patch_name]"
    exit 1
fi

# Verify patch exists
if [ ! -f "patches/$PATCH_FILE" ]; then
    echo "ERROR: Patch file patches/$PATCH_FILE not found"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build \
    --platform=linux/arm64 \
    --build-arg PYTORCH_COMMIT=$CURRENT_COMMIT \
    --build-arg PATCH_FILE=$PATCH_FILE \
    -t pytorch-debug:$DATE_TAG \
    -t pytorch-debug:latest \
    . 2>&1 | tee logs/build_${DATE_TAG}.log

echo "✓ Docker build complete"

# Extract wheel
echo "Extracting wheel..."
mkdir -p wheels
docker create --name pytorch-debug-temp pytorch-debug:$DATE_TAG
docker cp pytorch-debug-temp:/wheels/. wheels/
docker rm pytorch-debug-temp

# Identify wheel file
WHEEL_FILE=$(ls -t wheels/torch-*+cu130*.whl | head -1 || echo "")
if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: No wheel file found in wheels/"
    exit 1
fi
echo "Built wheel: $WHEEL_FILE"

# Test installation
echo "Testing wheel installation..."
python3 -m venv test_venv
source test_venv/bin/activate
pip install "$WHEEL_FILE"

# Verify sm_121 support
python3 -c "
import torch
assert 'sm_121' in torch.cuda.get_arch_list(), 'sm_121 support missing!'
print('✅ sm_121 support confirmed')
print(f'PyTorch version: {torch.__version__}')
print(f'Git version: {torch.version.git_version}')
"

deactivate
rm -rf test_venv

echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Wheel: $WHEEL_FILE"
echo "Log: logs/build_${DATE_TAG}.log"
echo ""
echo "Next steps:"
echo "  1. Install wheel: ./test_fix.sh $WHEEL_FILE $PHASE"
echo "  2. Run training to capture logs"
echo "  3. Analyze deadlock location"
```

**Step 2: Make executable**

```bash
chmod +x ~/Documents/pytorch-deadlock-fix/build_instrumented.sh
```

**Step 3: Commit build script**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add build_instrumented.sh
git commit -m "build: add PyTorch instrumented build script"
```

---

## Task 6: Create Test Harness Script

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/test_fix.sh`

**Step 1: Write test harness**

Create `/home/jay/Documents/pytorch-deadlock-fix/test_fix.sh`:
```bash
#!/bin/bash
set -euo pipefail

WHEEL_PATH=$1
TEST_PHASE=$2  # "instrumentation" or "fix"

echo "=========================================="
echo "Testing PyTorch Wheel"
echo "=========================================="
echo "Wheel: $(basename $WHEEL_PATH)"
echo "Phase: $TEST_PHASE"
echo ""

# Navigate to training directory
cd /home/jay/Documents/flymyai-lora-trainer
source venv/bin/activate

# Uninstall old PyTorch
echo "Uninstalling old PyTorch..."
pip uninstall torch torchvision torchaudio triton -y

# Install new wheel + dependencies
echo "Installing new wheel..."
pip install $WHEEL_PATH
pip install ~/Documents/DGX-Spark-PyTorch/wheels/triton-*.whl
pip install ~/Documents/DGX-Spark-PyTorch/wheels/torchvision-*.whl
pip install ~/Documents/DGX-Spark-PyTorch/wheels/torchaudio-*.whl

# Verify installation
python -c "
import torch
assert 'sm_121' in torch.cuda.get_arch_list(), 'sm_121 support missing!'
print(f'✅ PyTorch {torch.__version__} installed with sm_121 support')
"

echo ""

# Phase-specific testing
if [ "$TEST_PHASE" = "instrumentation" ]; then
    echo "=========================================="
    echo "Phase 1: Standalone SDPA Test"
    echo "=========================================="

    python ~/Documents/pytorch-deadlock-fix/test_sdpa.py 2> ~/Documents/pytorch-deadlock-fix/logs/sdpa_instrumentation.log || true

    echo "Debug output written to logs/sdpa_instrumentation.log"
    echo "Sample output:"
    head -20 ~/Documents/pytorch-deadlock-fix/logs/sdpa_instrumentation.log
    echo ""

    echo "=========================================="
    echo "Phase 2: Training Single Step"
    echo "=========================================="
    echo "Running training with 5-minute timeout..."

    timeout 300 python train_dgx_spark_qwen_lora.py \
        --config train_configs/train_dgx_spark_qwen_lora_rebecka.yaml \
        2>&1 | tee ~/Documents/pytorch-deadlock-fix/logs/training_instrumented.log || true

    echo ""
    echo "=========================================="
    echo "Analysis"
    echo "=========================================="
    echo "Last 10 SDPA debug messages:"
    grep -E "SDPA_DEBUG|CUDA_EVENT|ALLOC_DEBUG|CUDA_DEBUG" ~/Documents/pytorch-deadlock-fix/logs/training_instrumented.log | tail -10 || echo "No debug messages found"

    echo ""
    echo "Deadlock occurred after last message above"
    echo "Review logs/training_instrumented.log and logs/sdpa_instrumentation.log"
    echo "Identify which scenario (1-4) applies"

elif [ "$TEST_PHASE" = "fix" ]; then
    echo "=========================================="
    echo "Phase 1: Standalone SDPA Test"
    echo "=========================================="

    timeout 60 python ~/Documents/pytorch-deadlock-fix/test_sdpa.py 2> ~/Documents/pytorch-deadlock-fix/logs/sdpa_fixed.log

    if grep -q "SDPA completed" ~/Documents/pytorch-deadlock-fix/logs/sdpa_fixed.log; then
        echo "✅ SDPA test passed"
    else
        echo "❌ SDPA test failed - check logs/sdpa_fixed.log"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Phase 2: Training Single Step"
    echo "=========================================="

    timeout 60 python train_dgx_spark_qwen_lora.py \
        --config train_configs/train_dgx_spark_qwen_lora_rebecka.yaml \
        --max_train_steps 1 \
        2>&1 | tee ~/Documents/pytorch-deadlock-fix/logs/training_fix_step1.log

    if grep -q "step=0" ~/Documents/pytorch-deadlock-fix/logs/training_fix_step1.log; then
        echo "✅ Training step 1 passed"
    else
        echo "❌ Training step 1 failed"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Phase 3: Training 10 Steps"
    echo "=========================================="

    timeout 600 python train_dgx_spark_qwen_lora.py \
        --config train_configs/train_dgx_spark_qwen_lora_rebecka.yaml \
        --max_train_steps 10 \
        2>&1 | tee ~/Documents/pytorch-deadlock-fix/logs/training_fix_step10.log

    if grep -q "step=9" ~/Documents/pytorch-deadlock-fix/logs/training_fix_step10.log; then
        echo "✅ Training 10 steps passed"
    else
        echo "❌ Training 10 steps failed"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Phase 4: Memory Leak Check"
    echo "=========================================="

    echo "Memory usage across steps:"
    grep "Memory:" ~/Documents/pytorch-deadlock-fix/logs/training_fix_step10.log | tail -5 || echo "No memory stats found"

    echo ""
    echo "=========================================="
    echo "✅ All tests passed! Fix is validated."
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run full training: python train_dgx_spark_qwen_lora.py --config ..."
    echo "  2. Benchmark performance: python ~/Documents/pytorch-deadlock-fix/benchmark_fix.py"
    echo "  3. Document fix in analysis/"
fi
```

**Step 2: Make executable**

```bash
chmod +x ~/Documents/pytorch-deadlock-fix/test_fix.sh
```

**Step 3: Commit test harness**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add test_fix.sh
git commit -m "test: add comprehensive test harness for wheel validation"
```

---

## Task 7: Create Dockerfile

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/Dockerfile`

**Step 1: Write Dockerfile**

Create `/home/jay/Documents/pytorch-deadlock-fix/Dockerfile`:
```dockerfile
FROM ghcr.io/heathen711/dgx-spark-onnx:latest

# Build arguments
ARG PYTORCH_COMMIT=main
ARG PATCH_FILE=01_sdpa_instrumentation.patch

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git ccache ninja-build \
    vim gdb \
    && rm -rf /var/lib/apt/lists/*

# Clone PyTorch at specific commit
WORKDIR /workspace
RUN git clone --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout ${PYTORCH_COMMIT} && \
    git submodule sync && \
    git submodule update --init --recursive

# Copy patches
COPY patches/ /workspace/patches/

# Apply patch
WORKDIR /workspace/pytorch
RUN if [ -f "/workspace/patches/${PATCH_FILE}" ]; then \
      echo "Applying patch: ${PATCH_FILE}"; \
      patch -p1 < /workspace/patches/${PATCH_FILE}; \
    else \
      echo "Warning: Patch file ${PATCH_FILE} not found"; \
    fi

# Build configuration
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV BUILD_TEST=0
ENV DEBUG=1
ENV CMAKE_BUILD_TYPE=RelWithDebInfo
ENV MAX_JOBS=8
ENV CXXFLAGS="-O3"

# Disable optimizations for attention kernels only (better debugging)
RUN echo 'aten/src/ATen/native/transformers/*.cpp: -O0 -g' >> .compile_flags && \
    echo 'aten/src/ATen/native/transformers/*.cu: -O0 -g' >> .compile_flags

# Build PyTorch
RUN --mount=type=cache,target=/root/.ccache \
    python3 setup.py bdist_wheel

# Extract wheel
RUN mkdir -p /wheels && \
    cp dist/torch-*.whl /wheels/

# Verification
RUN pip install /wheels/torch-*.whl && \
    python3 -c "import torch; assert 'sm_121' in torch.cuda.get_arch_list(), 'sm_121 support missing!'; print('✅ sm_121 support confirmed')"

CMD ["/bin/bash"]
```

**Step 2: Commit Dockerfile**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add Dockerfile
git commit -m "build: add Dockerfile for PyTorch instrumented build"
```

---

## Task 8: Create Patch File Templates

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/patches/README.md`
- Create: `/home/jay/Documents/pytorch-deadlock-fix/patches/01_sdpa_instrumentation.patch` (placeholder)

**Step 1: Create patch directory README**

Create `/home/jay/Documents/pytorch-deadlock-fix/patches/README.md`:
```markdown
# PyTorch Patch Files

## Instrumentation Patch

**01_sdpa_instrumentation.patch** - Adds comprehensive logging to PyTorch SDPA and unified memory allocator

Creates debug header `pytorch/c10/util/SDPADebug.h` with macros:
- `LOG_SDPA_DEBUG(msg)` - SDPA operation logging
- `LOG_ALLOC_DEBUG(msg)` - Memory allocator logging
- `LOG_CUDA_DEBUG(msg)` - CUDA operation logging
- `CUDA_EVENT_RECORD(name)` - Timestamp recording

Instruments:
- `aten/src/ATen/native/transformers/attention.cpp` - Entry point
- `aten/src/ATen/native/transformers/cuda/attention.cu` - Math backend
- `aten/src/ATen/native/cuda/Blas.cpp` - cuBLAS operations
- `c10/cuda/CUDACachingAllocator.cpp` - Memory allocation

## Fix Patches

**02_fix_scenario_1.patch** - cuBLAS stream ordering fix
- Adds explicit stream synchronization before/after cuBLAS calls
- Only applies on ARM64 + CUDA 13.0+ with unified memory

**02_fix_scenario_2.patch** - Softmax reduction kernel fix
- Adds cudaMemPrefetchAsync before softmax kernel launch
- Prevents page faults during reduction operations

**02_fix_scenario_3.patch** - Transpose contiguous copy fix
- Forces .contiguous() on key.transpose() for unified memory
- Eliminates complex stride patterns in cuBLAS

**02_fix_scenario_4.patch** - Unified memory allocator fix
- Replaces cudaMallocManaged with cudaMallocAsync
- Prevents global page fault handler contention

## Creating Patches

Patches must be created from actual PyTorch source modifications.

1. Clone PyTorch
2. Make modifications
3. Generate patch: `git diff > patches/patch_name.patch`
4. Test patch: `git apply --check patches/patch_name.patch`
```

**Step 2: Create placeholder instrumentation patch**

Create `/home/jay/Documents/pytorch-deadlock-fix/patches/01_sdpa_instrumentation.patch`:
```patch
# Placeholder for SDPA instrumentation patch
#
# This file will be replaced with actual patch after:
# 1. Cloning PyTorch source
# 2. Adding instrumentation code (see design doc Section 2)
# 3. Running: git diff > patches/01_sdpa_instrumentation.patch
#
# IMPORTANT: This must be created from actual PyTorch source
# See: /home/jay/Documents/flymyai-lora-trainer/docs/plans/2025-11-19-pytorch-deadlock-fix-design.md
# Section 2 for exact code to add
```

**Step 3: Commit patch templates**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add patches/
git commit -m "docs: add patch file templates and documentation"
```

---

## Task 9: Create Analysis Document Templates

**Files:**
- Create: `/home/jay/Documents/pytorch-deadlock-fix/analysis/deadlock_location.md`
- Create: `/home/jay/Documents/pytorch-deadlock-fix/analysis/root_cause.md`
- Create: `/home/jay/Documents/pytorch-deadlock-fix/analysis/fix_explanation.md`

**Step 1: Create deadlock location template**

Create `/home/jay/Documents/pytorch-deadlock-fix/analysis/deadlock_location.md`:
```markdown
# Deadlock Location Analysis

**Date:** [To be filled after instrumentation run]
**Status:** [PENDING|IDENTIFIED]

## Last Debug Messages

```
[Paste last 20 debug messages from training_instrumented.log]
```

## Scenario Classification

- [ ] Scenario 1: cuBLAS stream ordering
- [ ] Scenario 2: Softmax reduction kernel
- [ ] Scenario 3: Transpose operation
- [ ] Scenario 4: CUDA driver page fault handler

## Evidence

**Last successful operation:**
```
[OPERATION_NAME] completed at timestamp [XXXX]
```

**First failed operation:**
```
[OPERATION_NAME] started but never completed
```

## Conclusion

Deadlock occurs in: [EXACT LOCATION]

Applied fix: `patches/02_fix_scenario_X.patch`
```

**Step 2: Create root cause template**

Create `/home/jay/Documents/pytorch-deadlock-fix/analysis/root_cause.md`:
```markdown
# Root Cause Analysis

**Scenario:** [1|2|3|4]
**Operation:** [Exact operation that deadlocks]

## Why It Deadlocks

[Technical explanation]

## ARM64 + Blackwell + Unified Memory Specific Behavior

[How this platform combination triggers the issue]

## CUDA/PyTorch Code Path

1. [Function call 1]
2. [Function call 2]
3. → Deadlock here

## References

- PyTorch issue: [link if found]
- CUDA documentation: [link]
- Similar reports: [links]
```

**Step 3: Create fix explanation template**

Create `/home/jay/Documents/pytorch-deadlock-fix/analysis/fix_explanation.md`:
```markdown
# Fix Explanation

**Patch:** `patches/02_fix_scenario_X.patch`
**Lines changed:** [N]
**Performance impact:** [X]%

## Code Changes

```diff
[Paste git diff output]
```

## How It Fixes The Deadlock

[Detailed explanation]

## Performance Impact

**Benchmark results:**
- Baseline: [X] iter/s, [Y] ms latency
- Fixed: [X] iter/s, [Y] ms latency
- Overhead: [Z]%

## Alternative Approaches Considered

1. [Approach 1] - Rejected because [reason]
2. [Approach 2] - Rejected because [reason]

## Upstream Submission Plan

- [ ] Create PyTorch GitHub issue
- [ ] Draft pull request
- [ ] Add tests
- [ ] Submit for review
```

**Step 4: Commit analysis templates**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add analysis/
git commit -m "docs: add analysis document templates"
```

---

## Task 10: Final Verification and Next Steps

**Step 1: Verify repository structure**

Run: `cd ~/Documents/pytorch-deadlock-fix && tree -L 2`
Expected: Shows complete directory structure with all files

**Step 2: Verify all scripts are executable**

```bash
cd ~/Documents/pytorch-deadlock-fix
ls -la *.sh *.py
```
Expected: All .sh and .py files have execute permissions

**Step 3: Create final commit**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add -A
git commit -m "chore: complete Day 1 setup - ready for PyTorch build" --allow-empty
```

**Step 4: Document next steps in README**

Add to `/home/jay/Documents/pytorch-deadlock-fix/README.md` at the end:
```markdown
## Current Status

✅ Day 1 Setup Complete
- Repository structure created
- Test scripts ready (test_sdpa.py, benchmark_fix.py)
- Build scripts ready (build_instrumented.sh, test_fix.sh)
- Dockerfile ready
- Analysis templates ready

⏳ Next: Create Instrumentation Patch
1. Clone PyTorch source (matches current wheel commit)
2. Add instrumentation code from design doc Section 2
3. Generate patch: `git diff > patches/01_sdpa_instrumentation.patch`
4. Build instrumented wheel: `./build_instrumented.sh instrumentation`
5. Run tests: `./test_fix.sh wheels/torch-*-instrumented.whl instrumentation`
6. Analyze logs in `logs/training_instrumented.log`
7. Identify scenario (1-4)
8. Create appropriate fix patch
9. Build fixed wheel: `./build_instrumented.sh fix 02_fix_scenario_X.patch`
10. Validate: `./test_fix.sh wheels/torch-*-fixed.whl fix`
```

**Step 5: Final commit**

```bash
cd ~/Documents/pytorch-deadlock-fix
git add README.md
git commit -m "docs: document Day 1 completion and next steps"
```

**Step 6: Show git log**

Run: `cd ~/Documents/pytorch-deadlock-fix && git log --oneline`
Expected: Shows all commits from Day 1 setup

---

## Completion Checklist

After all tasks complete:

- [x] Repository structure created
- [x] README.md with project overview
- [x] test_sdpa.py for standalone SDPA testing
- [x] benchmark_fix.py for performance measurement
- [x] build_instrumented.sh for Docker builds
- [x] test_fix.sh for automated validation
- [x] Dockerfile for PyTorch build environment
- [x] Patch file templates and documentation
- [x] Analysis document templates
- [x] All scripts executable
- [x] Git repository initialized and committed

**Next:** Create actual instrumentation patch by cloning PyTorch and adding code from design doc Section 2.
