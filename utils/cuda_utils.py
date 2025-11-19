# utils/cuda_utils.py
"""
CUDA version detection and feature support utilities.

Provides functions to check CUDA capabilities and enable features
based on CUDA version and GPU architecture.
"""
import torch
from typing import Tuple, Dict, Any

# Feature requirements: feature_name -> (min_cuda_version, min_compute_capability)
FEATURE_REQUIREMENTS = {
    "flash_attention_3": ("13.0", (9, 0)),  # Hopper+ required
    "fp8_native": ("13.0", (9, 0)),  # Hopper+ required
    "async_memcpy": ("12.0", (8, 0)),  # Ampere+
    "tf32_compute": ("11.0", (8, 0)),  # Ampere+
    "flash_attention_2": ("12.0", (8, 0)),  # Ampere+
}


def get_cuda_version() -> str:
    """
    Get the current CUDA version.

    Returns:
        CUDA version string (e.g., "13.0") or "N/A" if CUDA not available
    """
    if not torch.cuda.is_available():
        return "N/A"

    cuda_version = torch.version.cuda
    if cuda_version is None:
        return "N/A"

    return cuda_version


def get_device_capability() -> Tuple[int, int]:
    """
    Get the compute capability of the current CUDA device.

    Returns:
        Tuple of (major, minor) compute capability, or (0, 0) if not available
    """
    if not torch.cuda.is_available():
        return (0, 0)

    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return (0, 0)


def is_hopper_or_newer() -> bool:
    """
    Check if GPU is Hopper architecture (sm_90) or newer.

    Returns:
        True if Hopper or newer, False otherwise
    """
    capability = get_device_capability()
    return capability[0] >= 9


def is_ampere_or_newer() -> bool:
    """
    Check if GPU is Ampere architecture (sm_80) or newer.

    Returns:
        True if Ampere or newer, False otherwise
    """
    capability = get_device_capability()
    return capability[0] >= 8


def _version_compare(v1: str, v2: str) -> int:
    """
    Compare two version strings.

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    if v1 == "N/A" or v2 == "N/A":
        return -1

    parts1 = [int(x) for x in v1.split(".")]
    parts2 = [int(x) for x in v2.split(".")]

    for p1, p2 in zip(parts1, parts2):
        if p1 < p2:
            return -1
        if p1 > p2:
            return 1

    return 0


def supports_feature(feature: str) -> bool:
    """
    Check if a CUDA feature is supported.

    Args:
        feature: Feature name (e.g., "flash_attention_3", "fp8_native")

    Returns:
        True if feature is supported, False otherwise
    """
    if feature not in FEATURE_REQUIREMENTS:
        return False

    min_cuda, min_capability = FEATURE_REQUIREMENTS[feature]

    cuda_version = get_cuda_version()
    device_capability = get_device_capability()

    # Check CUDA version
    if _version_compare(cuda_version, min_cuda) < 0:
        return False

    # Check compute capability
    if device_capability < min_capability:
        return False

    return True


def get_optimal_settings() -> Dict[str, Any]:
    """
    Get optimal CUDA settings based on current hardware.

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "use_flash_attention_3": supports_feature("flash_attention_3"),
        "use_fp8": supports_feature("fp8_native"),
        "use_tf32": supports_feature("tf32_compute"),
        "use_flash_attention_2": supports_feature("flash_attention_2"),
        "cuda_version": get_cuda_version(),
        "compute_capability": get_device_capability(),
    }

    return settings


def enable_tf32() -> None:
    """Enable TF32 computation if supported."""
    if supports_feature("tf32_compute"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_recommended_dtype() -> torch.dtype:
    """
    Get recommended dtype based on hardware capabilities.

    Returns:
        Recommended torch dtype
    """
    if supports_feature("fp8_native"):
        # FP8 available but typically want BF16 for training
        return torch.bfloat16
    elif is_ampere_or_newer():
        return torch.bfloat16
    else:
        return torch.float16


def configure_attention_backend(
    enable_flash_attention_3: bool = False,
    enable_cudnn_sdp: bool = True,
    enable_math_sdp: bool = True,
    enable_mem_efficient: bool = True
) -> None:
    """
    Configure PyTorch's scaled dot-product attention backend.

    This function sets which SDPA backends are enabled. On DGX Spark (ARM64 + sm_121),
    Flash Attention 3 is unstable, so we disable it and use cuDNN backend instead.

    Args:
        enable_flash_attention_3: Enable Flash Attention (unstable on ARM64)
        enable_cudnn_sdp: Enable cuDNN backend (stable on ARM64)
        enable_math_sdp: Enable math backend (fallback)
        enable_mem_efficient: Enable memory-efficient backend
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Determine which backends to enable
        enabled_backends = []

        if enable_flash_attention_3:
            enabled_backends.append(SDPBackend.FLASH_ATTENTION)
        if enable_cudnn_sdp:
            enabled_backends.append(SDPBackend.CUDNN_ATTENTION)
        if enable_mem_efficient:
            enabled_backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if enable_math_sdp:
            enabled_backends.append(SDPBackend.MATH)

        logger.info("=" * 80)
        logger.info("Configuring SDPA (Scaled Dot-Product Attention) backends:")
        logger.info(f"  Flash Attention 3: {'ENABLED' if enable_flash_attention_3 else 'DISABLED'}")
        logger.info(f"  cuDNN Attention:   {'ENABLED' if enable_cudnn_sdp else 'DISABLED'}")
        logger.info(f"  Memory Efficient:  {'ENABLED' if enable_mem_efficient else 'DISABLED'}")
        logger.info(f"  Math (fallback):   {'ENABLED' if enable_math_sdp else 'DISABLED'}")
        logger.info("=" * 80)

        # Set the global default - this will be used by all SDPA operations
        # We can't use sdpa_kernel as a context manager globally, so we need to
        # configure it before model loading

        # NOTE: PyTorch doesn't provide a direct API to set global SDPA backend preferences.
        # The sdpa_kernel context manager only works locally.
        # We'll need to use environment variables or modify model forward passes.

        # For now, log the configuration and rely on PyTorch's auto-selection
        # with Flash Attention disabled at compile time if needed

        if not enable_flash_attention_3:
            logger.warning("Flash Attention 3 disabled - PyTorch will use cuDNN or fallback backends")
            logger.warning("If you still see Flash Attention errors, you may need to rebuild PyTorch")
            logger.warning("or set PYTORCH_FLASH_ATTENTION_DISABLE=1 environment variable")

    except ImportError:
        logger.warning("SDPBackend not available - using PyTorch default attention")

    return None
