#!/usr/bin/env python3
"""
train_dgx_spark_qwen_lora.py

DGX Spark (sm_121) optimized Qwen-Image LoRA training script.

This script is EXCLUSIVELY for NVIDIA DGX Spark systems with:
- ARM64 Blackwell GB10 GPU (compute capability 12.1)
- 128GB unified CPU-GPU memory
- CUDA 13.0
- Custom PyTorch with sm_121 support

NOT for general use. Enforces maximum performance settings.

Author: Generated for DGX Spark optimization
Date: 2025-11-18
"""

import argparse
import copy
import logging
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Core imports
import datasets
import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# Local utilities
from image_datasets.dataset import loader
from utils.cuda_utils import enable_tf32, supports_feature, get_optimal_settings
from utils.unified_memory import setup_unified_memory_env
from utils.memory_monitor import log_memory_usage, reset_peak_memory_stats, get_memory_stats

logger = get_logger(__name__, log_level="INFO")

# ============================================================================
# DGX SPARK HARD-CODED SETTINGS
# ============================================================================

# Hard-coded batch size (tune through testing)
QWEN_LORA_BATCH_SIZE = 4

# Memory limits (128GB total, reserve headroom)
SAFE_MEMORY_LIMIT_GB = 115
WARNING_THRESHOLD_GB = 103.5  # 90% of safe limit

# DGX Spark optimal settings (override config)
DGX_SPARK_OVERRIDES = {
    # Unified Memory
    "unified_memory": True,
    "disable_cpu_offload": True,
    "pin_memory": False,
    "disable_quantization": True,
    "disable_gradient_checkpointing": True,

    # Precision
    "quantize": False,
    "adam8bit": False,
    "mixed_precision": "bf16",

    # Data Loading
    "save_cache_on_disk": True,

    # CUDA 13.0 Features
    "cuda_13_features": {
        "enable_flash_attention_3": False,  # Unstable on ARM64
        "enable_cudnn_sdp": True,
        "enable_tf32_compute": True,
        "enable_fp8_training": False,
    },
}

# ============================================================================
# DGX SPARK VALIDATION
# ============================================================================

def validate_dgx_spark():
    """
    Validate that we're running on DGX Spark hardware.
    Fail fast if not.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. This script requires DGX Spark with NVIDIA GPU."
        )

    # Check compute capability
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(
            f"This script requires DGX Spark (sm_121, compute capability 12.1). "
            f"Detected: sm_{capability[0]}{capability[1]} (compute capability {capability[0]}.{capability[1]})\n"
            f"Use train.py or train_4090.py for other GPUs."
        )

    # Check unified memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory < 120_000_000_000:  # Less than 120GB
        raise RuntimeError(
            f"Expected 128GB unified memory for DGX Spark. "
            f"Found {total_memory / 1e9:.1f}GB. "
            f"This may not be a DGX Spark system."
        )

    logger.info("✓ DGX Spark validation passed")
    logger.info(f"  GPU: sm_121 (Blackwell GB10)")
    logger.info(f"  Memory: {total_memory / 1e9:.1f}GB unified")
    logger.info(f"  CUDA: {torch.version.cuda}")

# ============================================================================
# CONFIG OVERRIDE
# ============================================================================

def override_config_for_dgx_spark(user_config):
    """
    Override user config with DGX Spark optimal settings.
    Log all changes verbosely.
    """
    logger.warning("=" * 80)
    logger.warning("DGX SPARK MODE: Overriding config for maximum performance")
    logger.warning("=" * 80)

    # Deep copy to avoid modifying original
    final_config = OmegaConf.create(user_config)

    # Apply overrides and log changes
    for key, value in DGX_SPARK_OVERRIDES.items():
        if key == "cuda_13_features":
            # Handle nested dict
            if not hasattr(final_config, "cuda_13_features"):
                final_config.cuda_13_features = {}
            for sub_key, sub_value in value.items():
                old_value = getattr(final_config.cuda_13_features, sub_key, None)
                if old_value != sub_value:
                    logger.info(f"  cuda_13_features.{sub_key}: {old_value} → {sub_value}")
                setattr(final_config.cuda_13_features, sub_key, sub_value)
        else:
            old_value = getattr(final_config, key, None)
            if old_value != value:
                logger.info(f"  {key}: {old_value} → {value}")
            setattr(final_config, key, value)

    # Override batch size in data_config
    if hasattr(final_config, "data_config"):
        old_bs = getattr(final_config.data_config, "train_batch_size", None)
        if old_bs != QWEN_LORA_BATCH_SIZE:
            logger.info(f"  data_config.train_batch_size: {old_bs} → {QWEN_LORA_BATCH_SIZE}")
        final_config.data_config.train_batch_size = QWEN_LORA_BATCH_SIZE

        # Override num_workers
        old_workers = getattr(final_config.data_config, "num_workers", None)
        if old_workers != 4:
            logger.info(f"  data_config.num_workers: {old_workers} → 4")
        final_config.data_config.num_workers = 4

    logger.warning("=" * 80)

    return final_config

# ============================================================================
# MEMORY MONITORING
# ============================================================================

class DGXSparkMemoryMonitor:
    """Memory monitoring for DGX Spark unified memory."""

    def __init__(self, safe_limit_gb=115, warning_threshold_gb=103.5):
        self.safe_limit_bytes = safe_limit_gb * 1e9
        self.warning_threshold_bytes = warning_threshold_gb * 1e9

    def check_memory_pressure(self, step, allocated_bytes=None):
        """
        Check memory pressure and warn/fail if thresholds exceeded.

        Args:
            step: Current training step
            allocated_bytes: Optional override for allocated bytes (for testing)
        """
        if allocated_bytes is None:
            stats = get_memory_stats()
            allocated_bytes = stats["allocated_gb"] * 1e9

        allocated_gb = allocated_bytes / 1e9

        # Critical: exceeding safe limit
        if allocated_bytes > self.safe_limit_bytes:
            raise RuntimeError(
                f"[Step {step}] Memory limit exceeded: {allocated_gb:.1f}GB / {self.safe_limit_bytes/1e9:.0f}GB safe limit.\n"
                f"Reduce QWEN_LORA_BATCH_SIZE from {QWEN_LORA_BATCH_SIZE} in train_dgx_spark_qwen_lora.py and restart."
            )

        # Warning: approaching limit
        if allocated_bytes > self.warning_threshold_bytes:
            usage_pct = (allocated_bytes / self.safe_limit_bytes) * 100
            logger.warning(
                f"[Step {step}] Memory pressure high: {allocated_gb:.1f}GB / {self.safe_limit_bytes/1e9:.0f}GB "
                f"({usage_pct:.1f}%)"
            )

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models_for_dgx_spark(args, weight_dtype, device):
    """
    Load all models into unified memory at startup.
    Models NEVER move during training.

    Args:
        args: Training configuration
        weight_dtype: torch.bfloat16 for DGX Spark
        device: CUDA device (unified memory)

    Returns:
        tuple: (text_encoding_pipeline, vae, transformer, noise_scheduler, lora_config)
    """
    logger.info("Loading models into unified memory (no offloading)...")

    # Load text encoding pipeline (stays on device)
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(device)
    logger.info("  ✓ Text encoding pipeline loaded to unified memory")

    # Load VAE (stays on device)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    vae.to(device)
    logger.info("  ✓ VAE loaded to unified memory")

    # Load transformer (stays on device)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype
    )
    transformer.to(device)
    logger.info("  ✓ Transformer loaded to unified memory")

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    logger.info(f"  ✓ LoRA adapter added (rank={args.rank})")

    # Load noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Log memory after loading
    log_memory_usage("after_model_loading")

    logger.info("All models resident in unified memory (no CPU offloading)")

    return text_encoding_pipeline, vae, transformer, noise_scheduler, lora_config

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DGX Spark optimized Qwen-Image LoRA training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (settings will be overridden for DGX Spark)",
    )
    args = parser.parse_args()
    return args.config

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function."""

    # Step 1: Validate DGX Spark hardware
    validate_dgx_spark()

    # Step 2: Load and override config
    config_path = parse_args()
    user_config = OmegaConf.load(config_path)
    args = override_config_for_dgx_spark(user_config)

    # Step 3: Setup unified memory environment
    setup_unified_memory_env()
    logger.info("Unified memory environment configured")

    # Step 4: Initialize memory monitor
    memory_monitor = DGXSparkMemoryMonitor(
        safe_limit_gb=SAFE_MEMORY_LIMIT_GB,
        warning_threshold_gb=WARNING_THRESHOLD_GB
    )

    # Step 5: Setup logging
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    # Step 6: Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Enable TF32
    enable_tf32()
    logger.info("TF32 compute enabled")

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Step 7: Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f"Weight dtype: {weight_dtype}")

    # Step 8: Log initial memory state
    reset_peak_memory_stats()
    log_memory_usage("before_model_loading")

    # Step 9: Load models into unified memory (zero movement)
    text_encoding_pipeline, vae, transformer, noise_scheduler, lora_config = \
        load_models_for_dgx_spark(args, weight_dtype, accelerator.device)

    # Verify no models on CPU
    assert str(transformer.device) != "cpu", "Transformer moved to CPU!"
    assert str(vae.device) != "cpu", "VAE moved to CPU!"

    logger.info("Model loading complete - all models in unified memory")

    # TODO: Implement data loading, training loop
    logger.info("Data loading and training loop implementation pending...")

if __name__ == "__main__":
    main()
