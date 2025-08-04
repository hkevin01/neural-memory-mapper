"""Tests for the signal processing module."""

import numpy as np
import pytest

from src.core.signal_processor import SignalProcessor


@pytest.fixture
def signal_processor():
    """Create a SignalProcessor instance for testing."""
    return SignalProcessor(sampling_rate=256, window_size=256)


def test_bandpass_filter(signal_processor):
    """Test the bandpass filter functionality."""
    # Generate test signal (10 Hz sine wave)
    t = np.linspace(0, 1, signal_processor.sampling_rate)
    test_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz
    
    # Apply alpha band filter (8-13 Hz)
    filtered = signal_processor.apply_bandpass_filter(test_signal, 8, 13)
    
    # Calculate power before and after filtering
    orig_power = np.sum(test_signal ** 2)
    filt_power = np.sum(filtered ** 2)
    
    # Signal should be preserved (power roughly the same)
    assert 0.5 * orig_power <= filt_power <= 1.5 * orig_power


def test_extract_frequency_bands(signal_processor):
    """Test frequency band extraction."""
    # Generate test signal with multiple frequencies
    t = np.linspace(0, 1, signal_processor.sampling_rate)
    delta = np.sin(2 * np.pi * 2 * t)    # 2 Hz
    theta = np.sin(2 * np.pi * 6 * t)    # 6 Hz
    alpha = np.sin(2 * np.pi * 10 * t)   # 10 Hz
    beta = np.sin(2 * np.pi * 20 * t)    # 20 Hz
    gamma = np.sin(2 * np.pi * 40 * t)   # 40 Hz
    
    test_signal = delta + theta + alpha + beta + gamma
    
    # Extract bands
    bands = signal_processor.extract_frequency_bands(test_signal)
    
    # Check that all expected bands are present
    assert all(band in bands for band in ['delta', 'theta', 'alpha', 'beta', 'gamma'])
    
    # Check that each band contains signal
    assert all(len(band_data) == len(test_signal) for band_data in bands.values())


def test_compute_band_power(signal_processor):
    """Test band power computation."""
    # Generate test signal with known frequency
    t = np.linspace(0, 1, signal_processor.sampling_rate)
    alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha band)
    
    # Compute band powers
    powers = signal_processor.compute_band_power(alpha)
    
    # Alpha power should be highest
    assert powers['alpha'] > powers['theta']
    assert powers['alpha'] > powers['beta']


def test_analyze_memory_state(signal_processor):
    """Test memory state analysis."""
    # Generate test signal
    t = np.linspace(0, 1, signal_processor.sampling_rate)
    theta = 2 * np.sin(2 * np.pi * 6 * t)    # Strong theta
    alpha = np.sin(2 * np.pi * 10 * t)       # Weaker alpha
    
    test_signal = theta + alpha
    powers = signal_processor.compute_band_power(test_signal)
    state = signal_processor.analyze_memory_state(powers)
    
    # Check that all metrics are present
    assert 'memory_formation_strength' in state
    assert 'attention_level' in state
    assert 'theta_power' in state
    assert 'gamma_power' in state
    
    # Memory formation strength should be high (high theta/alpha ratio)
    assert state['memory_formation_strength'] > 1.0
