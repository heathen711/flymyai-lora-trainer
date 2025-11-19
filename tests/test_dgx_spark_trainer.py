# tests/test_dgx_spark_trainer.py
import pytest
import torch

def test_dgx_spark_detection():
    """Test DGX Spark (sm_121) detection."""
    from train_dgx_spark_qwen_lora import validate_dgx_spark
    # This will fail on non-DGX Spark systems, pass on DGX Spark
    if torch.cuda.get_device_capability() == (12, 1):
        validate_dgx_spark()  # Should not raise
    else:
        with pytest.raises(RuntimeError, match="DGX Spark.*sm_121"):
            validate_dgx_spark()

def test_config_override():
    """Test that config is overridden with DGX Spark settings."""
    from train_dgx_spark_qwen_lora import override_config_for_dgx_spark
    from omegaconf import OmegaConf

    # User config with suboptimal settings
    user_config = OmegaConf.create({
        "unified_memory": False,
        "quantize": True,
        "adam8bit": True,
    })

    final_config = override_config_for_dgx_spark(user_config)

    assert final_config.unified_memory == True
    assert final_config.quantize == False
    assert final_config.adam8bit == False

def test_memory_limit_enforcement():
    """Test that memory limit is enforced."""
    from train_dgx_spark_qwen_lora import DGXSparkMemoryMonitor

    monitor = DGXSparkMemoryMonitor(safe_limit_gb=115)

    # Simulate exceeding limit
    with pytest.raises(RuntimeError, match="Memory limit exceeded"):
        monitor.check_memory_pressure(
            step=100,
            allocated_bytes=116_000_000_000  # 116GB > 115GB limit
        )

def test_zero_movement_model_loading():
    """Test that models stay on device."""
    # This test verifies no .to("cpu") calls
    pass  # Implementation TBD
