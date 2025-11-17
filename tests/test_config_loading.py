import pytest
from omegaconf import OmegaConf


def test_fastsafetensors_config_option():
    """Test that fastsafetensors config options are recognized."""
    config_str = """
    use_fastsafetensors: true
    fastsafetensors_num_threads: 8
    """
    config = OmegaConf.create(config_str)
    assert config.use_fastsafetensors == True
    assert config.fastsafetensors_num_threads == 8
