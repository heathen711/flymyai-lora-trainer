# utils/compat.py
"""
Backward compatibility utilities for CUDA version migration.

Provides warnings and fallbacks for older CUDA versions.
"""
from typing import List
import warnings
from utils.cuda_utils import get_cuda_version, get_device_capability

MINIMUM_CUDA_VERSION = "12.0"
RECOMMENDED_CUDA_VERSION = "13.0"


def check_requirements() -> List[str]:
    """
    Check if current environment meets requirements.

    Returns:
        List of warning messages
    """
    warning_list = []

    cuda_version = get_cuda_version()

    if cuda_version == "N/A":
        warning_list.append("CUDA not available - GPU training will not work")
        return warning_list

    # Parse version
    try:
        major, minor = cuda_version.split(".")[:2]
        major = int(major)
        minor = int(minor.split(".")[0]) if "." in minor else int(minor)
    except Exception:
        warning_list.append(f"Could not parse CUDA version: {cuda_version}")
        return warning_list

    # Check minimum version
    if major < 12:
        warning_list.append(
            f"CUDA {cuda_version} is below minimum {MINIMUM_CUDA_VERSION}. "
            "Some features may not work."
        )

    # Check recommended version
    if major < 13:
        warning_list.append(
            f"CUDA {cuda_version} detected. "
            f"Recommend upgrading to {RECOMMENDED_CUDA_VERSION} for best performance."
        )

    # Check compute capability
    capability = get_device_capability()
    if capability[0] < 8:
        warning_list.append(
            f"GPU compute capability {capability} is older than Ampere (8.0). "
            "Some optimizations disabled."
        )

    return warning_list


def get_deprecation_warnings() -> List[str]:
    """
    Get warnings about deprecated features.

    Returns:
        List of deprecation warning messages
    """
    warnings_list = []

    cuda_version = get_cuda_version()

    if cuda_version == "N/A":
        return warnings_list

    try:
        major = int(cuda_version.split(".")[0])
    except Exception:
        return warnings_list

    if major <= 11:
        warnings_list.append(
            "CUDA 11.x is deprecated. Please upgrade to CUDA 12.x or 13.0."
        )

    return warnings_list


def show_compatibility_warnings() -> None:
    """Display all compatibility warnings."""
    for warning in check_requirements():
        warnings.warn(warning, DeprecationWarning)

    for warning in get_deprecation_warnings():
        warnings.warn(warning, DeprecationWarning)
