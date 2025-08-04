"""Tests for artifact removal module."""

import numpy as np
import pytest

from src.core.artifact_removal import ArtifactParams, ArtifactRemoval


@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data for testing."""
    # Create 1 second of data at 256 Hz for 8 channels
    t = np.linspace(0, 1, 256)
    channels = []
    
    # Generate clean signals
    for i in range(8):
        # Base signal: mixture of alpha (10 Hz) and beta (20 Hz)
        signal = (
            np.sin(2 * np.pi * 10 * t) +  # Alpha
            0.5 * np.sin(2 * np.pi * 20 * t)  # Beta
        )
        channels.append(signal)
    
    return np.array(channels)


@pytest.fixture
def artifact_removal():
    """Create ArtifactRemoval instance for testing."""
    channel_names = [f'CH{i}' for i in range(8)]
    return ArtifactRemoval(
        sampling_rate=256,
        channel_names=channel_names,
        params=ArtifactParams()
    )


def test_artifact_detection(artifact_removal, sample_eeg_data):
    """Test artifact detection functionality."""
    # Add some artifacts to the data
    data_with_artifacts = sample_eeg_data.copy()
    data_with_artifacts[0, 100:110] = 150  # Amplitude artifact
    data_with_artifacts[1, 150] = 50       # Gradient artifact
    
    mask, details = artifact_removal.detect_artifacts(data_with_artifacts)
    
    assert isinstance(mask, np.ndarray)
    assert isinstance(details, dict)
    assert 'amplitude' in details
    assert 'gradient' in details
    assert np.any(mask)  # Should detect some artifacts


def test_line_noise_removal(artifact_removal, sample_eeg_data):
    """Test line noise removal."""
    # Add 50 Hz line noise
    t = np.linspace(0, 1, 256)
    noisy_data = sample_eeg_data + 2 * np.sin(2 * np.pi * 50 * t)
    
    cleaned_data = artifact_removal.remove_line_noise(noisy_data)
    
    # Check that the 50 Hz component is reduced
    fft_noisy = np.abs(np.fft.fft(noisy_data[0]))
    fft_cleaned = np.abs(np.fft.fft(cleaned_data[0]))
    
    freq = np.fft.fftfreq(256, 1/256)
    line_freq_idx = np.argmin(np.abs(freq - 50))
    
    assert fft_cleaned[line_freq_idx] < fft_noisy[line_freq_idx]


def test_bandpass_filter(artifact_removal, sample_eeg_data):
    """Test bandpass filtering."""
    # Add low frequency drift and high frequency noise
    t = np.linspace(0, 1, 256)
    noisy_data = (
        sample_eeg_data +
        0.5 * np.sin(2 * np.pi * 0.1 * t) +  # 0.1 Hz drift
        0.1 * np.random.randn(*sample_eeg_data.shape)  # High freq noise
    )
    
    filtered_data = artifact_removal.apply_bandpass(noisy_data)
    
    # Check that the data is within the specified frequency range
    freq = np.fft.fftfreq(256, 1/256)
    fft_filtered = np.abs(np.fft.fft(filtered_data[0]))
    
    # Check that frequencies outside the bandpass range are attenuated
    assert np.all(
        fft_filtered[np.abs(freq) < artifact_removal.params.bandpass_low] <
        np.max(fft_filtered) * 0.1
    )
    assert np.all(
        fft_filtered[np.abs(freq) > artifact_removal.params.bandpass_high] <
        np.max(fft_filtered) * 0.1
    )


def test_full_cleaning_pipeline(artifact_removal, sample_eeg_data):
    """Test the complete signal cleaning pipeline."""
    # Add various types of noise
    t = np.linspace(0, 1, 256)
    noisy_data = sample_eeg_data.copy()
    
    # Add line noise
    noisy_data += 2 * np.sin(2 * np.pi * 50 * t)
    
    # Add amplitude artifacts
    noisy_data[0, 100:110] = 150
    
    # Add some random noise
    noisy_data += 0.1 * np.random.randn(*noisy_data.shape)
    
    # Clean the signal
    cleaned_data, stats = artifact_removal.clean_signal(noisy_data)
    
    assert cleaned_data.shape == noisy_data.shape
    assert isinstance(stats, dict)
    assert 'artifact_percentage' in stats
    assert 'improvement' in stats
    
    # Check that the cleaning reduced the overall noise level
    original_noise = np.std(noisy_data - sample_eeg_data)
    cleaned_noise = np.std(cleaned_data - sample_eeg_data)
    assert cleaned_noise < original_noise
