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

# CRITICAL: Disable Flash Attention immediately after importing torch
# Flash Attention 3 is unstable on ARM64 + sm_121 (see CLAUDE.md)
# Memory-efficient SDPA also triggers FA sm80 kernels (incompatible with sm_121)
# Must be set before any model loading
torch.backends.cuda.enable_flash_sdp(False)  # Disable Flash Attention
torch.backends.cuda.enable_math_sdp(True)    # Enable math fallback
torch.backends.cuda.enable_mem_efficient_sdp(False)  # DISABLE - triggers FA sm80 kernels on sm_121
try:
    # cuDNN backend is only available on newer PyTorch versions
    torch.backends.cuda.enable_cudnn_sdp(True)  # Enable cuDNN SDPA (stable on ARM64)
except AttributeError:
    pass  # Older PyTorch version
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

    # Data Loading - Keep embeddings in unified memory for best performance
    "save_cache_on_disk": False,

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

    # Apply overrides and log changes
    for key, value in DGX_SPARK_OVERRIDES.items():
        if key == "cuda_13_features":
            # Handle nested dict
            if not hasattr(final_config, "cuda_13_features"):
                final_config.cuda_13_features = {}
            for sub_key, sub_value in value.items():
                old_value = getattr(final_config.cuda_13_features, sub_key, None)
                if old_value != sub_value:
                    early_logger.info(f"  cuda_13_features.{sub_key}: {old_value} → {sub_value}")
                setattr(final_config.cuda_13_features, sub_key, sub_value)
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

        # Override num_workers
        old_workers = getattr(final_config.data_config, "num_workers", None)
        if old_workers != 4:
            early_logger.info(f"  data_config.num_workers: {old_workers} → 4")
        final_config.data_config.num_workers = 4

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

    # Disable gradient checkpointing (abundant memory on DGX Spark)
    if hasattr(transformer, 'disable_gradient_checkpointing'):
        transformer.disable_gradient_checkpointing()
        logger.info("  ✓ Gradient checkpointing disabled (abundant memory)")
    elif hasattr(transformer, '_set_gradient_checkpointing'):
        transformer._set_gradient_checkpointing(value=False)
        logger.info("  ✓ Gradient checkpointing disabled via _set_gradient_checkpointing")

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
            txt_files = [i for i in os.listdir(args.data_config.img_dir) if ".txt" in i]
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
            img_files = [i for i in os.listdir(args.data_config.img_dir)
                        if ".png" in i or ".jpg" in i or ".jpeg" in i or ".JPG" in i or ".PNG" in i]
            for img_name in tqdm(img_files, desc="Encoding images"):
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

                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
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
    - num_workers=4 (lower than discrete GPU)
    - prefetch_factor=4 (moderate prefetch)
    - Embeddings cached in unified memory or on NVMe

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

    for epoch in range(first_epoch, args.max_train_steps):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Get latents and prompts from batch
                latents = batch["latents"].to(weight_dtype)
                prompt_embeds = batch["prompt_embeds"].to(weight_dtype)

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

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timesteps
                u = torch.rand(bsz, device=latents.device)
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise to latents
                sigmas = _get_sigmas(noise_scheduler, timesteps, len(latents.shape), latents.dtype)
                noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

                # Predict noise
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # Compute loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="logit_normal",
                    sigmas=sigmas,
                )
                target = latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
                    dim=1,
                )
                loss = loss.mean()

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

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
    logger.warning("=" * 80)
    logger.warning("ATTENTION BACKEND CONFIGURATION")
    logger.warning("=" * 80)
    logger.warning(f"  Flash SDPA:          {'ENABLED' if torch.backends.cuda.flash_sdp_enabled() else 'DISABLED'}")
    logger.warning(f"  Memory-Efficient:    {'ENABLED' if torch.backends.cuda.mem_efficient_sdp_enabled() else 'DISABLED'}")
    logger.warning(f"  Math SDPA:           {'ENABLED' if torch.backends.cuda.math_sdp_enabled() else 'DISABLED'}")
    try:
        logger.warning(f"  cuDNN SDPA:          {'ENABLED' if torch.backends.cuda.cudnn_sdp_enabled() else 'DISABLED'}")
    except AttributeError:
        logger.warning(f"  cuDNN SDPA:          NOT AVAILABLE (PyTorch version)")
    logger.warning("  → Flash Attention is DISABLED (unstable on ARM64 + sm_121)")
    logger.warning("  → Memory-Efficient SDPA DISABLED (triggers FA sm80 kernels on sm_121)")
    logger.warning("  → Using cuDNN SDPA + Math fallback backends only")
    logger.warning("=" * 80)

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
