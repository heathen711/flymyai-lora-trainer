import pytest


def test_fastsafetensors_import():
    """Test that fastsafetensors can be imported."""
    import fastsafetensors
    assert hasattr(fastsafetensors, 'fastsafe_open')
