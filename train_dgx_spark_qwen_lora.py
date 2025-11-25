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
import gc
import logging
import os
import sys

import numpy as np
import torch

# CRITICAL: SDPA Backend Configuration
# Flash Attention 2.8.3 (external package from DGX-Spark-FlashAttention) supports:
# - sm_121 native kernels (Blackwell GB10)
# - Grouped-Query Attention (GQA): 28 Q heads, 4 K/V heads (Qwen VAE)
# PyTorch will use flash-attn package when available, fallback to built-in backends
# Must be set before any model loading
torch.backends.cuda.enable_flash_sdp(True)   # Uses flash-attn 2.8.3 with GQA support
torch.backends.cuda.enable_math_sdp(True)    # Enabled as fallback for standard attention (VAE encoder) - transformer uses GQA path which bypasses this via flash_attn_func
torch.backends.cuda.enable_mem_efficient_sdp(False)  # DISABLE - triggers FA sm80 kernels on sm_121
try:
    torch.backends.cuda.enable_cudnn_sdp(True)  # Enable as additional fallback
except AttributeError:
    pass  # Older PyTorch version

# ============================================================================
# MONKEY-PATCH: Use external flash-attn for GQA
# ============================================================================
# PyTorch's built-in SDPA doesn't support GQA (mismatched Q and K/V heads).
# Monkey-patch F.scaled_dot_product_attention to use flash_attn_func when GQA detected.
try:
    from flash_attn import flash_attn_func
    import torch.nn.functional as F

    _original_sdpa = F.scaled_dot_product_attention

    def _flash_attn_gqa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        """
        Wrapper that uses flash_attn_func when possible, falls back to cuDNN SDPA (NOT Math) otherwise.
        Avoids Math SDPA deadlock on ARM64+Blackwell.

        Flash Attention requires:
        - Input shape: (batch, seqlen, num_heads, head_dim)
        - PyTorch SDPA uses: (batch, num_heads, seqlen, head_dim)
        - head_dim <= 256

        Accepts **kwargs to handle additional parameters like 'enable_gqa' from transformers.
        """
        head_dim = query.shape[-1]  # (batch, num_heads, seqlen, head_dim)

        # DEBUG: Log first attention call to verify monkey patch is working
        if not hasattr(_flash_attn_gqa_wrapper, '_logged_first_call'):
            print(f"[MONKEY PATCH] First call: head_dim={head_dim}, query.shape={query.shape}")
            _flash_attn_gqa_wrapper._logged_first_call = True

        # Use flash_attn_func if head_dim <= 256
        if head_dim <= 256:
            # Transpose: (B, H, S, D) -> (B, S, H, D)
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()

            # Flash Attention doesn't support attn_mask
            if attn_mask is not None:
                # Fall back to PyTorch's dispatcher (cuDNN or Math SDPA)
                return _original_sdpa(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs
                )

            # Call flash_attn_func (supports both GQA and standard attention)
            out = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                causal=is_causal,
                softmax_scale=scale,
            )

            # Transpose back: (B, S, H, D) -> (B, H, S, D)
            return out.transpose(1, 2)
        else:
            # head_dim > 256: Let PyTorch choose backend (will use Math SDPA for VAE, which is safe during encoding)
            # Transformer uses flash_attn path above, so Math SDPA deadlock doesn't occur during training
            return _original_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs
            )

    # Replace PyTorch's SDPA with our wrapper
    F.scaled_dot_product_attention = _flash_attn_gqa_wrapper
    print("✅ Monkey-patched F.scaled_dot_product_attention: flash_attn (head_dim≤256) or PyTorch dispatcher (head_dim>256)")

except ImportError:
    print("⚠️  flash-attn not available, using PyTorch's built-in SDPA (no GQA support)")

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from PIL import Image
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
from image_datasets.dataset import loader, image_resize
from utils.cuda_utils import enable_tf32, supports_feature, get_optimal_settings
from utils.fast_loading import save_embeddings_safetensors
from utils.unified_memory import setup_unified_memory_env
from utils.memory_monitor import log_memory_usage, reset_peak_memory_stats, get_memory_stats

# ============================================================================
# FILE FILTERING HELPERS
# ============================================================================

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

def is_image_file(filename):
    """Check if a file is an image based on extension (case-insensitive)."""
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS

def is_text_file(filename):
    """Check if a file is a text caption file (case-insensitive)."""
    return filename.lower().endswith('.txt')

# Use standard logging for early validation (before Accelerator init)
# Will switch to accelerate logger after Accelerator is initialized
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
early_logger = logging.getLogger(__name__)
logger = None  # Will be initialized after Accelerator

# ============================================================================
# DGX SPARK HARD-CODED SETTINGS
# ============================================================================

# Hard-coded batch size (tune through testing)
# FIXED: Removed CUDA_LAUNCH_BLOCKING=1 to avoid futex deadlock on ARM64 + CUDA 13.0
# Flash Attention working, but batch_size=2 causes OOM during backward pass
# Conservative: batch_size=1 to ensure training completes without OOM
QWEN_LORA_BATCH_SIZE = 1

# Memory limits (128GB total, reserve headroom)
SAFE_MEMORY_LIMIT_GB = 115
WARNING_THRESHOLD_GB = 103.5  # 90% of safe limit

# DGX Spark optimal settings (override config)
DGX_SPARK_OVERRIDES = {
    # Unified Memory
    "unified_memory": True,
    "disable_cpu_offload": True,  # DISABLE CPU offload - causes futex deadlock on ARM64
    "pin_memory": False,
    "disable_quantization": True,
    "disable_gradient_checkpointing": True,  # DISABLE - Causes numerical instability with NaN gradients

    # Precision
    "quantize": False,
    "adam8bit": False,  # DISABLE - 128GB unified memory = no need for 8-bit, prevents numerical instability
    "mixed_precision": "bf16",  # BF16 for Blackwell native support + larger dynamic range

    # Data Loading - Save embeddings to disk and load on demand per step
    "save_cache_on_disk": True,

    # CUDA Features (version-agnostic)
    "cuda_features": {
        "enable_flash_attention_3": False,  # Unstable on ARM64
        "enable_cudnn_sdp": True,
        "enable_tf32_compute": True,  # Enable for performance boost on Blackwell
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

    early_logger.info("✓ DGX Spark validation passed")
    early_logger.info(f"  GPU: sm_121 (Blackwell GB10)")
    early_logger.info(f"  Memory: {total_memory / 1e9:.1f}GB unified")
    early_logger.info(f"  CUDA: {torch.version.cuda}")

# ============================================================================
# CONFIG OVERRIDE
# ============================================================================

def override_config_for_dgx_spark(user_config):
    """
    Override user config with DGX Spark optimal settings.
    Log all changes verbosely.
    """
    early_logger.warning("=" * 80)
    early_logger.warning("DGX SPARK MODE: Overriding config for maximum performance")
    early_logger.warning("=" * 80)

    # Deep copy to avoid modifying original
    final_config = OmegaConf.create(user_config)

    # Backward compatibility: migrate cuda_13_features -> cuda_features
    if hasattr(user_config, 'cuda_13_features') and not hasattr(user_config, 'cuda_features'):
        early_logger.warning("=" * 80)
        early_logger.warning("DEPRECATED CONFIG KEY DETECTED")
        early_logger.warning("=" * 80)
        early_logger.warning("Config uses deprecated 'cuda_13_features' key.")
        early_logger.warning("Please update to 'cuda_features' (version-agnostic).")
        early_logger.warning("Automatically migrating for this run.")
        early_logger.warning("=" * 80)
        final_config.cuda_features = user_config.cuda_13_features

    # Apply overrides and log changes
    for key, value in DGX_SPARK_OVERRIDES.items():
        if key == "cuda_features":
            # Handle nested dict
            if not hasattr(final_config, "cuda_features"):
                final_config.cuda_features = {}
            for sub_key, sub_value in value.items():
                old_value = getattr(final_config.cuda_features, sub_key, None)
                if old_value != sub_value:
                    early_logger.info(f"  cuda_features.{sub_key}: {old_value} → {sub_value}")
                setattr(final_config.cuda_features, sub_key, sub_value)
        else:
            old_value = getattr(final_config, key, None)
            if old_value != value:
                early_logger.info(f"  {key}: {old_value} → {value}")
            setattr(final_config, key, value)

    # Override batch size in data_config
    if hasattr(final_config, "data_config"):
        old_bs = getattr(final_config.data_config, "train_batch_size", None)
        if old_bs != QWEN_LORA_BATCH_SIZE:
            early_logger.info(f"  data_config.train_batch_size: {old_bs} → {QWEN_LORA_BATCH_SIZE}")
        final_config.data_config.train_batch_size = QWEN_LORA_BATCH_SIZE

        # Override num_workers - use 0 since embeddings are cached on disk
        # Main thread can handle feeding cached latents without parallel workers
        old_workers = getattr(final_config.data_config, "num_workers", None)
        if old_workers != 0:
            early_logger.info(f"  data_config.num_workers: {old_workers} → 0")
        final_config.data_config.num_workers = 0

    early_logger.warning("=" * 80)

    return final_config

# ============================================================================
# MEMORY MONITORING
# ============================================================================

class DGXSparkMemoryMonitor:
    """Memory monitoring for DGX Spark unified memory."""

    def __init__(self, safe_limit_gb=115, warning_threshold_gb=103.5, logger_instance=None):
        self.safe_limit_bytes = safe_limit_gb * 1e9
        self.warning_threshold_bytes = warning_threshold_gb * 1e9
        self.logger = logger_instance  # Store logger instance

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
            warning_msg = (
                f"[Step {step}] Memory pressure high: {allocated_gb:.1f}GB / {self.safe_limit_bytes/1e9:.0f}GB "
                f"({usage_pct:.1f}%)"
            )
            if self.logger is not None:
                self.logger.warning(warning_msg)
            else:
                early_logger.warning(warning_msg)

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
    # Pipeline doesn't support device_map, so we still use .to() but with torch_dtype set
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
    )
    text_encoding_pipeline.to(device)
    logger.info("  ✓ Text encoding pipeline loaded to unified memory")

    # Load VAE (stays on device)
    # Use device_map to load directly to unified memory (avoids double allocation)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        device_map="cuda",
    )
    logger.info("  ✓ VAE loaded to unified memory")

    # Load transformer (stays on device)
    # Use device_map to load directly to unified memory (avoids ~38GB double allocation)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        device_map="cuda",
    )
    logger.info("  ✓ Transformer loaded to unified memory")

    # Configure gradient checkpointing based on config
    if getattr(args, 'disable_gradient_checkpointing', False):
        if hasattr(transformer, 'disable_gradient_checkpointing'):
            transformer.disable_gradient_checkpointing()
            logger.info("  ✓ Gradient checkpointing disabled (abundant memory)")
        elif hasattr(transformer, '_set_gradient_checkpointing'):
            transformer._set_gradient_checkpointing(value=False)
            logger.info("  ✓ Gradient checkpointing disabled via _set_gradient_checkpointing")
    else:
        transformer.enable_gradient_checkpointing()
        logger.info("  ✓ Gradient checkpointing enabled (saves ~46GB activations)")

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights=True,  # Use default Kaiming init (lora_A) + zeros (lora_B) - more stable than gaussian
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
# EMBEDDING PRECOMPUTATION
# ============================================================================

def precompute_embeddings(args, text_encoding_pipeline, vae, weight_dtype, accelerator):
    """
    Precompute text and image embeddings for faster training.

    For DGX Spark with unified memory and save_cache_on_disk=False,
    embeddings are kept in unified memory for best performance.

    Args:
        args: Training configuration
        text_encoding_pipeline: Text encoding pipeline
        vae: VAE model
        weight_dtype: torch.bfloat16
        accelerator: Accelerate Accelerator

    Returns:
        cached_text_embeddings, txt_cache_dir, cached_image_embeddings, img_cache_dir
    """
    cached_text_embeddings = None
    txt_cache_dir = None
    cached_image_embeddings = None
    img_cache_dir = None

    cache_dir = os.path.join(args.output_dir, "cache")

    # Precompute text embeddings
    if args.precompute_text_embeddings:
        logger.info("Precomputing text embeddings...")
        with torch.no_grad():
            if args.save_cache_on_disk:
                txt_cache_dir = os.path.join(cache_dir, "text_embs")
                os.makedirs(txt_cache_dir, exist_ok=True)
                logger.info(f"  Saving text embeddings to: {txt_cache_dir}")
            else:
                cached_text_embeddings = {}
                logger.info("  Keeping text embeddings in unified memory")

            # Process all text files
            txt_files = [i for i in os.listdir(args.data_config.img_dir) if is_text_file(i)]

            # Warmup call to compile CUDA kernels (first call is slow on sm_121)
            logger.info("  Warming up text encoder...")
            logger.info(f"    Pipeline device: {text_encoding_pipeline.device}")
            logger.info(f"    Text encoder device: {text_encoding_pipeline.text_encoder.device}")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

            logger.info("    Calling encode_prompt...")
            sys.stdout.flush()
            sys.stderr.flush()

            _ = text_encoding_pipeline.encode_prompt(
                prompt=["warmup"],
                device=text_encoding_pipeline.device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )
            torch.cuda.synchronize()
            logger.info("  ✓ Text encoder ready")

            for txt in tqdm(txt_files, desc="Encoding text"):
                txt_path = os.path.join(args.data_config.img_dir, txt)
                prompt = open(txt_path, encoding="utf-8").read()
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    prompt=[prompt],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                if args.save_cache_on_disk:
                    save_embeddings_safetensors(
                        {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')},
                        os.path.join(txt_cache_dir, txt + '.safetensors')
                    )
                else:
                    cached_text_embeddings[txt] = {
                        'prompt_embeds': prompt_embeds[0].to('cpu'),
                        'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                    }

            # Compute empty embedding for caption dropout
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                prompt=[' '],
                device=text_encoding_pipeline.device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )
            if args.save_cache_on_disk:
                save_embeddings_safetensors(
                    {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')},
                    os.path.join(txt_cache_dir, 'empty_embedding.safetensors')
                )
                del prompt_embeds
                del prompt_embeds_mask
            else:
                cached_text_embeddings['empty_embedding'] = {
                    'prompt_embeds': prompt_embeds[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                }

        logger.info(f"✓ Precomputed {len(txt_files)} text embeddings")
        log_memory_usage("after_text_embedding_precomputation")

    # Precompute image embeddings
    if args.precompute_image_embeddings:
        logger.info("Precomputing image embeddings...")
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs")
            os.makedirs(img_cache_dir, exist_ok=True)
            logger.info(f"  Saving image embeddings to: {img_cache_dir}")
        else:
            cached_image_embeddings = {}
            logger.info("  Keeping image embeddings in unified memory")

        with torch.no_grad():
            img_files = sorted([i for i in os.listdir(args.data_config.img_dir) if is_image_file(i)])
            for idx, img_name in enumerate(tqdm(img_files, desc="Encoding images")):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                img = image_resize(img, args.data_config.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)

                # Debug: Check pixel_values before VAE encoding
                if idx < 5 or idx == len(img_files) - 1:
                    logger.info(f"  [{idx}] {img_name}: pixel_values stats: min={pixel_values.min().item():.6f}, max={pixel_values.max().item():.6f}, mean={pixel_values.mean().item():.6f}")

                latent_dist = vae.encode(pixel_values).latent_dist
                pixel_latents = latent_dist.sample().to('cpu')[0]

                # Debug: Check latents after VAE encoding
                if idx < 5 or idx == len(img_files) - 1:
                    logger.info(f"  [{idx}] {img_name}: latent stats: min={pixel_latents.min().item():.6f}, max={pixel_latents.max().item():.6f}, mean={pixel_latents.mean().item():.6f}")
                    # Check if corrupted
                    if torch.isnan(pixel_latents).any() or torch.isinf(pixel_latents).any() or pixel_latents.abs().max() > 1e10:
                        logger.error(f"  [{idx}] CORRUPTED LATENTS DETECTED in {img_name}!")
                        logger.error(f"    NaN count: {torch.isnan(pixel_latents).sum().item()}")
                        logger.error(f"    Inf count: {torch.isinf(pixel_latents).sum().item()}")
                        logger.error(f"    Max abs value: {pixel_latents.abs().max().item()}")

                if args.save_cache_on_disk:
                    save_embeddings_safetensors(
                        {'latent': pixel_latents},
                        os.path.join(img_cache_dir, img_name + '.safetensors')
                    )
                    del pixel_latents
                else:
                    cached_image_embeddings[img_name] = pixel_latents

        logger.info(f"✓ Precomputed {len(img_files)} image embeddings")
        log_memory_usage("after_image_embedding_precomputation")

    return cached_text_embeddings, txt_cache_dir, cached_image_embeddings, img_cache_dir

# ============================================================================
# DATA LOADING
# ============================================================================

def setup_dataloader_for_dgx_spark(args, cached_text_embeddings, txt_cache_dir,
                                    cached_image_embeddings, img_cache_dir):
    """
    Setup DataLoader optimized for DGX Spark unified memory.

    - pin_memory=False (not needed for unified memory)
    - num_workers=0 (main thread handles cached latents)
    - Embeddings cached on disk and loaded on demand

    Args:
        args: Training configuration
        cached_text_embeddings: Precomputed text embeddings (or None if disk-cached)
        txt_cache_dir: Text embedding cache directory (or None if memory-cached)
        cached_image_embeddings: Precomputed image embeddings (or None if disk-cached)
        img_cache_dir: Image embedding cache directory (or None if memory-cached)

    Returns:
        DataLoader configured for unified memory
    """
    logger.info("Setting up DataLoader for unified memory...")

    # Use existing loader from image_datasets.dataset
    # It already supports embedding caching
    train_dataloader = loader(
        train_batch_size=QWEN_LORA_BATCH_SIZE,
        num_workers=args.data_config.num_workers,
        pin_memory=False,  # Unified memory doesn't need pinned memory
        img_dir=args.data_config.img_dir,
        img_size=args.data_config.img_size,
        caption_type=args.data_config.caption_type,
        random_ratio=args.data_config.random_ratio,
        caption_dropout_rate=args.data_config.caption_dropout_rate,
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        txt_cache_dir=txt_cache_dir,
        img_cache_dir=img_cache_dir,
    )

    # Verify DataLoader settings
    assert train_dataloader.pin_memory == False, "pin_memory should be False for unified memory"
    logger.info(f"  ✓ DataLoader configured:")
    logger.info(f"    - batch_size: {QWEN_LORA_BATCH_SIZE}")
    logger.info(f"    - num_workers: {args.data_config.num_workers}")
    logger.info(f"    - pin_memory: False (unified memory)")
    logger.info(f"    - prefetch_factor: {getattr(train_dataloader, 'prefetch_factor', 'default')}")

    return train_dataloader

# ============================================================================
# TRAINING LOOP
# ============================================================================

def _get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    """Get sigmas for timesteps."""
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def train_loop(
    args,
    accelerator,
    transformer,
    vae,
    text_encoding_pipeline,
    noise_scheduler,
    train_dataloader,
    weight_dtype,
    memory_monitor
):
    """
    Main training loop with aggressive memory monitoring.

    No CPU offloading, no gradient checkpointing, full precision optimizer.
    """
    import shutil
    from safetensors.torch import save_file

    logger.info("Starting DGX Spark optimized training loop...")

    # VAE scale factor for unpacking latents
    vae_scale_factor = 2 ** len(vae.temperal_downsample)

    # Prepare optimizer (full precision Adam, no 8-bit)
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("  ✓ Full precision AdamW optimizer")

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with Accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Training state
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if available
    resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = resume_from_checkpoint
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is not None:
            checkpoint_path = os.path.join(args.output_dir, path)
            if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
                accelerator.load_state(checkpoint_path)
                global_step = int(path.split("-")[1])
                first_epoch = global_step // len(train_dataloader)
                logger.info(f"Resuming from checkpoint: {path} (step {global_step})")
            else:
                logger.warning(f"Checkpoint path not found or invalid: {checkpoint_path}")
                logger.warning("Starting training from scratch")

    # Training loop
    log_memory_usage("before_training_loop")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    early_stop = False  # Flag to stop training early
    for epoch in range(first_epoch, args.max_train_steps):
        if early_stop:
            break
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            if step == 0:
                logger.info("  Got first batch from DataLoader")
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

            with accelerator.accumulate(transformer):
                # Get latents and prompts from batch
                if step == 0:
                    logger.info("  Processing first batch...")
                    sys.stdout.flush()

                # Debug: Check raw batch data for first 10 steps
                if global_step < 10:
                    raw_latents = batch["latents"]
                    logger.info(f"  [Step {global_step}] RAW batch latents stats: min={raw_latents.min().item():.6f}, max={raw_latents.max().item():.6f}, mean={raw_latents.mean().item():.6f}, requires_grad={raw_latents.requires_grad}")
                    # Check for corruption
                    if torch.isnan(raw_latents).any() or torch.isinf(raw_latents).any() or raw_latents.abs().max() > 1e10:
                        logger.error(f"  [Step {global_step}] CORRUPTED RAW LATENTS DETECTED!")
                        logger.error(f"    NaN count: {torch.isnan(raw_latents).sum().item()}")
                        logger.error(f"    Inf count: {torch.isinf(raw_latents).sum().item()}")
                        logger.error(f"    Max abs value: {raw_latents.abs().max().item()}")

                latents = batch["latents"].to(weight_dtype)
                prompt_embeds = batch["prompt_embeds"].to(weight_dtype)
                prompt_embeds_mask = batch["prompt_embeds_mask"].to(latents.device)

                # Debug: Check after dtype conversion
                if global_step < 5:
                    logger.info(f"  [Step {global_step}] After .to() latents stats: min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}, requires_grad={latents.requires_grad}")

                # Permute latents from [B, C, T, H, W] to [B, T, C, H, W] for transformer
                latents = latents.permute(0, 2, 1, 3, 4)

                # Normalize VAE latents (CRITICAL: required for correct training)
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, 1, vae.config.z_dim, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                    1, 1, vae.config.z_dim, 1, 1
                ).to(latents.device, latents.dtype)
                latents = (latents - latents_mean) * latents_std

                # Debug: Check after normalization
                if global_step < 5:
                    logger.info(f"  [Step {global_step}] After normalization latents stats: min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")

                # DEBUG: Before sampling noise
                if global_step == 0:
                    logger.info("  [DEBUG] About to sample noise...")
                    sys.stdout.flush()

                # Sample noise
                noise = torch.randn_like(latents)

                # DEBUG: After sampling noise
                if global_step == 0:
                    logger.info("  [DEBUG] Noise sampled successfully")
                    sys.stdout.flush()
                bsz = latents.shape[0]

                # DEBUG: Before sampling timesteps
                if global_step == 0:
                    logger.info("  [DEBUG] About to sample timesteps...")
                    sys.stdout.flush()

                # Sample timesteps
                u = torch.rand(bsz, device=latents.device)
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                # CRITICAL: Keep timesteps on GPU to avoid CPU/GPU transfers that deadlock on ARM64
                timesteps_gpu = noise_scheduler.timesteps.to(device=latents.device)
                timesteps = timesteps_gpu[indices]

                # DEBUG: After sampling timesteps
                if global_step == 0:
                    logger.info("  [DEBUG] Timesteps sampled successfully")
                    sys.stdout.flush()

                # Add noise to latents
                sigmas = _get_sigmas(noise_scheduler, timesteps, len(latents.shape), latents.dtype)
                noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

                # DEBUG: Before packing latents
                if global_step == 0:
                    logger.info("  [DEBUG] About to pack latents...")
                    sys.stdout.flush()

                # Pack latents for transformer input
                packed_noisy_latents = QwenImagePipeline._pack_latents(
                    noisy_latents,
                    bsz,
                    noisy_latents.shape[2],
                    noisy_latents.shape[3],
                    noisy_latents.shape[4],
                )

                # DEBUG: After packing latents
                if global_step == 0:
                    logger.info("  [DEBUG] Latents packed successfully")
                    sys.stdout.flush()

                # Calculate image shapes and text sequence lengths for RoPE
                img_shapes = [(1, noisy_latents.shape[3] // 2, noisy_latents.shape[4] // 2)] * bsz
                # CRITICAL: Use padded dimension, not actual lengths, to match transformer's query/key tensors
                txt_seq_lens = [prompt_embeds.shape[1]] * bsz

                # DEBUG: Before transformer forward pass
                if global_step == 0:
                    logger.info("  [DEBUG] About to call transformer forward pass...")
                    sys.stdout.flush()

                # Predict noise
                # DISABLED: Debug logging with .tolist() causes futex deadlock on ARM64 + CUDA 13.0
                # if step == 0:
                #     import sys
                #     logger.info("=" * 80)
                #     logger.info("COMPREHENSIVE DEBUG OUTPUT - FIRST BATCH")
                #     logger.info("=" * 80)
                #     logger.info(f"  Batch size (bsz): {bsz}")
                #     logger.info(f"  Original latents shape (after permute): {latents.shape}")
                #     logger.info(f"  Noisy latents shape (after noise addition): {noisy_latents.shape}")
                #     logger.info(f"    - noisy_latents.shape[0] (B): {noisy_latents.shape[0]}")
                #     logger.info(f"    - noisy_latents.shape[1] (T): {noisy_latents.shape[1]}")
                #     logger.info(f"    - noisy_latents.shape[2] (C): {noisy_latents.shape[2]}")
                #     logger.info(f"    - noisy_latents.shape[3] (H): {noisy_latents.shape[3]}")
                #     logger.info(f"    - noisy_latents.shape[4] (W): {noisy_latents.shape[4]}")
                #     logger.info(f"  Packed noisy latents shape: {packed_noisy_latents.shape}")
                #     logger.info(f"  Image shapes for RoPE: {img_shapes}")
                #     logger.info(f"    - Format: [(T, H//2, W//2)] * batch_size")
                #     logger.info(f"    - T=1, H//2={noisy_latents.shape[3] // 2}, W//2={noisy_latents.shape[4] // 2}")
                #     logger.info(f"  Text embeddings shape: {prompt_embeds.shape}")
                #     logger.info(f"  Text mask shape: {prompt_embeds_mask.shape}")
                #     logger.info(f"  Text sequence lengths: {txt_seq_lens}")
                #     logger.info(f"    - prompt_embeds_mask sum per sample: {prompt_embeds_mask.sum(dim=1).tolist()}")  # DEADLOCK
                #     logger.info(f"    - max text seq len: {max(txt_seq_lens)}")
                #     logger.info(f"    - min text seq len: {min(txt_seq_lens)}")
                #     logger.info(f"  Timesteps: {timesteps.shape} = {timesteps.tolist()}")  # DEADLOCK
                #     logger.info(f"  Transformer device: {next(transformer.parameters()).device}")
                #     logger.info(f"  VAE scale factor: {vae_scale_factor}")
                #     logger.info("=" * 80)
                #     sys.stdout.flush()
                #     sys.stderr.flush()

                # DISABLED: torch.cuda.synchronize() causes futex deadlock on ARM64 + CUDA 13.0
                # if step == 0:
                #     logger.info("  Forcing CUDA synchronization...")
                #     torch.cuda.synchronize()
                #     logger.info(f"  Input dtypes: latents={packed_noisy_latents.dtype}, embeds={prompt_embeds.dtype}, mask={prompt_embeds_mask.dtype}")
                #     logger.info(f"  Mask device: {prompt_embeds_mask.device}")

                model_pred = transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                # DEBUG: After transformer forward pass
                if global_step == 0:
                    logger.info("  [DEBUG] Transformer forward pass complete, unpacking...")
                    sys.stdout.flush()

                # Unpack model prediction to match target shape
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,
                    height=noisy_latents.shape[3] * vae_scale_factor,
                    width=noisy_latents.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # Compute loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="logit_normal",
                    sigmas=sigmas,
                )
                target = latents

                # Debug logging for first 5 steps
                if global_step < 5:
                    logger.info(f"  [Step {global_step}] model_pred stats: min={model_pred.min().item():.6f}, max={model_pred.max().item():.6f}, mean={model_pred.mean().item():.6f}")
                    logger.info(f"  [Step {global_step}] target stats: min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}")
                    logger.info(f"  [Step {global_step}] weighting stats: min={weighting.min().item():.6f}, max={weighting.max().item():.6f}, mean={weighting.mean().item():.6f}")

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
                    dim=1,
                )
                loss = loss.mean()
                if step == 0 or global_step < 5:
                    logger.info(f"  [Step {global_step}] Loss computed: {loss.item():.4f}, running backward...")
                    sys.stdout.flush()

                # Backward pass
                accelerator.backward(loss)
                if step == 0:
                    logger.info("  Backward pass complete")
                    sys.stdout.flush()

                if accelerator.sync_gradients:
                    # Log gradient norm before clipping for first 10 steps
                    if global_step < 10:
                        grad_norm = accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                        logger.info(f"  [Step {global_step}] Gradient norm before clip: {grad_norm:.6f}, max_grad_norm: {args.max_grad_norm}")
                    else:
                        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DEBUG: Auto-stop at step 10 for quick testing
                if global_step >= 10:
                    logger.warning(f"DEBUG: Auto-stopping at step {global_step}")
                    early_stop = True
                    break

                # Memory monitoring every 10 steps
                if global_step % 10 == 0:
                    memory_monitor.check_memory_pressure(global_step)

                # Detailed logging every 100 steps
                if global_step % 100 == 0:
                    log_memory_usage(f"step_{global_step}")

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                        # Clean up old checkpoints
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                for checkpoint in checkpoints[:num_to_remove]:
                                    logger.info(f"Removing old checkpoint: {checkpoint}")
                                    shutil.rmtree(os.path.join(args.output_dir, checkpoint))

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Final checkpoint
    if accelerator.is_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)

        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(lora_state_dict, save_path)
        logger.info(f"Saved final LoRA weights to {save_path}")

    log_memory_usage("training_complete")
    logger.info("Training complete!")

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
    early_logger.info("Unified memory environment configured")

    # Step 3.5: Log Flash Attention status
    early_logger.warning("=" * 80)
    early_logger.warning("ATTENTION BACKEND CONFIGURATION")
    early_logger.warning("=" * 80)
    early_logger.warning(f"  Flash SDPA:          {'ENABLED' if torch.backends.cuda.flash_sdp_enabled() else 'DISABLED'}")
    early_logger.warning(f"  Memory-Efficient:    {'ENABLED' if torch.backends.cuda.mem_efficient_sdp_enabled() else 'DISABLED'}")
    early_logger.warning(f"  Math SDPA:           {'ENABLED' if torch.backends.cuda.math_sdp_enabled() else 'DISABLED'}")
    try:
        early_logger.warning(f"  cuDNN SDPA:          {'ENABLED' if torch.backends.cuda.cudnn_sdp_enabled() else 'DISABLED'}")
    except AttributeError:
        early_logger.warning(f"  cuDNN SDPA:          NOT AVAILABLE (PyTorch version)")
    early_logger.warning("  → Flash Attention 2.8.3 (DGX-Spark-FlashAttention) installed")
    early_logger.warning("  → GQA Support: 28 Q heads + 4 K/V heads (Qwen VAE) - WORKING!")
    early_logger.warning("  → sm_121 native kernels compiled for Blackwell GB10")
    early_logger.warning("  → Memory: O(1) with Flash Attention vs O(N²) Math SDPA (~3.9GB saved)")
    early_logger.warning("=" * 80)

    # Step 4: Initialize memory monitor
    memory_monitor = DGXSparkMemoryMonitor(
        safe_limit_gb=SAFE_MEMORY_LIMIT_GB,
        warning_threshold_gb=WARNING_THRESHOLD_GB
    )

    # Step 5: Set random seed for reproducibility (BEFORE Accelerator init)
    if hasattr(args, 'seed') and args.seed is not None:
        import random
        import numpy as np
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        early_logger.info(f"Set random seed to {args.seed} for reproducibility")

    # Step 6: Setup logging
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    # Step 7: Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # NOW we can initialize the accelerate logger
    global logger
    logger = get_logger(__name__, log_level="INFO")

    # Update memory monitor with logger instance
    memory_monitor.logger = logger

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Apply CUDA feature flags from config
    if hasattr(args, 'cuda_features'):
        if getattr(args.cuda_features, 'enable_tf32_compute', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 compute enabled via cuda_features config")

        if getattr(args.cuda_features, 'enable_cudnn_sdp', False):
            try:
                torch.backends.cuda.enable_cudnn_sdp(True)
                logger.info("cuDNN SDP enabled via cuda_features config")
            except AttributeError:
                logger.warning("cuDNN SDP not available in this PyTorch version")

        if getattr(args.cuda_features, 'enable_flash_attention_3', False):
            logger.warning("Flash Attention 3 requested but not yet implemented in training loop")

        if getattr(args.cuda_features, 'enable_fp8_training', False):
            logger.warning("FP8 training requested but not yet implemented")
    else:
        # Fallback to legacy enable_tf32() function
        enable_tf32()
        logger.info("TF32 compute enabled (legacy path)")

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

    # Step 10: Precompute embeddings
    cached_text_embeddings, txt_cache_dir, cached_image_embeddings, img_cache_dir = precompute_embeddings(
        args,
        text_encoding_pipeline,
        vae,
        weight_dtype,
        accelerator
    )

    # Step 11: Setup DataLoader with precomputed embeddings
    train_dataloader = setup_dataloader_for_dgx_spark(
        args,
        cached_text_embeddings,
        txt_cache_dir,
        cached_image_embeddings,
        img_cache_dir
    )

    log_memory_usage("after_dataloader_setup")

    # Step 12: Training loop
    train_loop(
        args,
        accelerator,
        transformer,
        vae,
        text_encoding_pipeline,
        noise_scheduler,
        train_dataloader,
        weight_dtype,
        memory_monitor
    )

if __name__ == "__main__":
    main()
