"""Tests for the data acquisition module."""

import numpy as np
import pytest

from src.core.data_acquisition import MockEEGDevice


@pytest.fixture
def mock_device():
    """Create a MockEEGDevice instance for testing."""
    return MockEEGDevice(sampling_rate=256, num_channels=8)


def test_mock_device_initialization(mock_device):
    """Test proper initialization of mock device."""
    assert mock_device.sampling_rate == 256
    assert mock_device.num_channels == 8
    assert not mock_device.is_running
    assert mock_device.thread is None


def test_mock_device_connect_disconnect(mock_device):
    """Test connection and disconnection."""
    # Test connect
    mock_device.connect()
    assert mock_device.is_running
    assert mock_device.thread is not None
    assert mock_device.thread.is_alive()
    
    # Test disconnect
    mock_device.disconnect()
    assert not mock_device.is_running
    assert not mock_device.thread.is_alive()


def test_mock_device_data_generation(mock_device):
    """Test data generation and retrieval."""
    mock_device.connect()
    
    # Wait for some data to be generated
    import time
    time.sleep(0.1)
    
    # Get data
    data = mock_device.get_data()
    
    # Verify data shape and characteristics
    assert data is not None
    assert data.shape == (mock_device.num_channels, mock_device.sampling_rate)
    
    # Check that data contains realistic values
    assert np.all(np.abs(data) < 10)  # Reasonable amplitude range
    assert not np.all(data == 0)      # Not all zeros
    
    mock_device.disconnect()
