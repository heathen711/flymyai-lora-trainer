# tests/test_memory_monitor.py
import pytest

def test_get_memory_stats():
    """Test memory stats retrieval."""
    from utils.memory_monitor import get_memory_stats
    stats = get_memory_stats()
    assert "allocated_gb" in stats
    assert "reserved_gb" in stats
    assert isinstance(stats["allocated_gb"], float)

def test_log_memory_usage():
    """Test memory usage logging."""
    from utils.memory_monitor import log_memory_usage
    # Should not raise
    log_memory_usage("test checkpoint")
