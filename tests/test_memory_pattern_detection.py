"""Tests for memory pattern detection module."""

import numpy as np
import pytest

from src.core.memory_pattern_detection import (EEGFeatureExtractor,
                                               MemoryPatternDetector,
                                               MemoryPatternNet)


@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data for testing."""
    # Generate 10 samples of 8-channel EEG data
    n_samples = 10
    n_channels = 8
    window_size = 256
    
    # Create sample data with known patterns
    samples = []
    labels = []
    
    for i in range(n_samples):
        t = np.linspace(0, 1, window_size)
        sample = np.zeros((n_channels, window_size))
        
        if i % 2 == 0:
            # Class 0: Strong alpha (memory formation)
            for ch in range(n_channels):
                sample[ch] = (
                    np.sin(2 * np.pi * 10 * t) +  # Alpha
                    0.3 * np.sin(2 * np.pi * 6 * t) +  # Theta
                    0.1 * np.random.randn(window_size)  # Noise
                )
            labels.append(0)
        else:
            # Class 1: Strong theta (active processing)
            for ch in range(n_channels):
                sample[ch] = (
                    0.3 * np.sin(2 * np.pi * 10 * t) +  # Alpha
                    np.sin(2 * np.pi * 6 * t) +  # Theta
                    0.1 * np.random.randn(window_size)  # Noise
                )
            labels.append(1)
        
        samples.append(sample)
    
    return np.array(samples), np.array(labels)


def test_feature_extractor(sample_eeg_data):
    """Test feature extraction functionality."""
    X, _ = sample_eeg_data
    
    extractor = EEGFeatureExtractor(
        sampling_rate=256,
        window_size=256
    )
    
    features = extractor.transform(X)
    
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[0] == len(X)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_memory_pattern_net():
    """Test neural network architecture."""
    net = MemoryPatternNet(input_size=100)
    
    # Test forward pass
    x = torch.randn(10, 100)
    output = net(x)
    
    assert output.shape == (10, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("model_type", ["rf", "nn"])
def test_memory_pattern_detector(sample_eeg_data, model_type):
    """Test memory pattern detection with different models."""
    X, y = sample_eeg_data
    
    detector = MemoryPatternDetector(
        sampling_rate=256,
        window_size=256,
        model_type=model_type
    )
    
    # Train the model
    detector.fit(X, y)
    
    # Test predictions
    predictions = detector.predict(X)
    probabilities = detector.predict_proba(X)
    
    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)
    assert np.all(np.isclose(np.sum(probabilities, axis=1), 1.0))
    
    # Test prediction values
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_model_persistence(sample_eeg_data, tmp_path):
    """Test model saving and loading."""
    X, y = sample_eeg_data
    
    # Train original model
    detector = MemoryPatternDetector(
        sampling_rate=256,
        window_size=256,
        model_type="rf"
    )
    detector.fit(X, y)
    original_predictions = detector.predict(X)
    
    # Save model
    model_path = tmp_path / "model.pkl"
    import joblib
    joblib.dump(detector, model_path)
    
    # Load model
    loaded_detector = joblib.load(model_path)
    loaded_predictions = loaded_detector.predict(X)
    
    # Check predictions match
    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions
    )
