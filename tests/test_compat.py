# tests/test_compat.py
import pytest

def test_check_requirements():
    """Test requirements checking."""
    from utils.compat import check_requirements
    warnings = check_requirements()
    assert isinstance(warnings, list)

def test_get_deprecation_warnings():
    """Test deprecation warning generation."""
    from utils.compat import get_deprecation_warnings
    warnings = get_deprecation_warnings()
    assert isinstance(warnings, list)
