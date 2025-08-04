"""Advanced EEG signal processing pipeline."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from ..utils.logging_config import get_logger
from .artifact_removal import ArtifactParams, ArtifactRemoval
from .memory_pattern_detection import MemoryPatternDetector

logger = get_logger(__name__)


@dataclass
class SignalProcessorConfig:
    """Configuration for EEG signal processing."""
    
    # Sampling parameters
    sampling_rate: float = 256.0
    window_size: int = 256
    overlap: float = 0.5
    
    # Frequency bands (Hz)
    bands: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Artifact removal parameters
    artifact_params: Optional[ArtifactParams] = None
    
    # Memory pattern detection
    detection_model: str = 'rf'  # 'rf' or 'nn'
    
    # Adaptive filtering
    adaptation_rate: float = 0.1
    baseline_duration: float = 60.0  # seconds
    
    def __post_init__(self):
        """Initialize default values."""
        if self.bands is None:
            self.bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        
        if self.artifact_params is None:
            self.artifact_params = ArtifactParams()


class SignalQuality:
    """Monitor and report signal quality metrics."""
    
    def __init__(self):
        """Initialize signal quality monitoring."""
        self.metrics = {
            'snr': [],
            'artifact_rate': [],
            'channel_variance': [],
            'processing_time': []
        }
    
    def update(
        self,
        raw_data: np.ndarray,
        cleaned_data: np.ndarray,
        artifacts_detected: int,
        processing_time: float
    ) -> Dict[str, float]:
        """
        Update signal quality metrics.
        
        Args:
            raw_data: Raw EEG data
            cleaned_data: Processed EEG data
            artifacts_detected: Number of detected artifacts
            processing_time: Processing time in seconds
            
        Returns:
            Current quality metrics
        """
        # Calculate SNR
        noise = raw_data - cleaned_data
        snr = 10 * np.log10(
            np.var(cleaned_data) / np.var(noise)
        )
        
        # Calculate artifact rate
        artifact_rate = artifacts_detected / raw_data.shape[1]
        
        # Calculate channel variance
        channel_variance = np.var(cleaned_data, axis=1).mean()
        
        # Update metrics
        self.metrics['snr'].append(snr)
        self.metrics['artifact_rate'].append(artifact_rate)
        self.metrics['channel_variance'].append(channel_variance)
        self.metrics['processing_time'].append(processing_time)
        
        # Keep only recent history
        max_history = 1000
        for key in self.metrics:
            if len(self.metrics[key]) > max_history:
                self.metrics[key] = self.metrics[key][-max_history:]
        
        return {
            'snr': snr,
            'artifact_rate': artifact_rate,
            'channel_variance': channel_variance,
            'processing_time': processing_time,
            'snr_trend': np.mean(self.metrics['snr'][-10:]),
            'artifact_rate_trend': np.mean(
                self.metrics['artifact_rate'][-10:]
            )
        }


class AdaptiveFilter:
    """Adaptive signal filtering based on user baseline."""
    
    def __init__(
        self,
        config: SignalProcessorConfig,
        channel_names: List[str]
    ):
        """
        Initialize adaptive filter.
        
        Args:
            config: Signal processor configuration
            channel_names: List of EEG channel names
        """
        self.config = config
        self.channel_names = channel_names
        
        # Initialize baseline statistics
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_samples = []
        self.baseline_complete = False
        
        # Calculate required baseline samples
        self.required_samples = int(
            config.baseline_duration * config.sampling_rate
        )
    
    def update_baseline(self, data: np.ndarray) -> None:
        """
        Update baseline statistics with new data.
        
        Args:
            data: New EEG data
        """
        if self.baseline_complete:
            # Adaptive update of baseline
            if self.baseline_mean is not None:
                self.baseline_mean = (
                    (1 - self.config.adaptation_rate) * self.baseline_mean +
                    self.config.adaptation_rate * np.mean(data, axis=1)
                )
                self.baseline_std = (
                    (1 - self.config.adaptation_rate) * self.baseline_std +
                    self.config.adaptation_rate * np.std(data, axis=1)
                )
        else:
            # Collecting initial baseline
            self.baseline_samples.append(data)
            total_samples = sum(
                s.shape[1] for s in self.baseline_samples
            )
            
            if total_samples >= self.required_samples:
                # Compute initial baseline statistics
                all_data = np.hstack(self.baseline_samples)
                self.baseline_mean = np.mean(all_data, axis=1)
                self.baseline_std = np.std(all_data, axis=1)
                self.baseline_complete = True
                logger.info("Baseline calibration completed")
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply adaptive filtering to the data.
        
        Args:
            data: EEG data to filter
            
        Returns:
            Filtered data
        """
        if not self.baseline_complete:
            return data
        
        # Z-score normalization
        normalized = (data - self.baseline_mean[:, np.newaxis]) / (
            self.baseline_std[:, np.newaxis]
        )
        
        # Clip extreme values
        clipped = np.clip(normalized, -5, 5)
        
        # Transform back to original scale
        filtered = (
            clipped * self.baseline_std[:, np.newaxis] +
            self.baseline_mean[:, np.newaxis]
        )
        
        return filtered


class EEGSignalProcessor:
    """Advanced EEG signal processing pipeline."""
    
    def __init__(
        self,
        channel_names: List[str],
        config: Optional[SignalProcessorConfig] = None
    ):
        """
        Initialize the signal processor.
        
        Args:
            channel_names: List of EEG channel names
            config: Signal processor configuration
        """
        self.channel_names = channel_names
        self.config = config or SignalProcessorConfig()
        
        # Initialize components
        self.artifact_removal = ArtifactRemoval(
            sampling_rate=self.config.sampling_rate,
            channel_names=channel_names,
            params=self.config.artifact_params
        )
        
        self.pattern_detector = MemoryPatternDetector(
            sampling_rate=self.config.sampling_rate,
            window_size=self.config.window_size,
            model_type=self.config.detection_model
        )
        
        self.adaptive_filter = AdaptiveFilter(
            config=self.config,
            channel_names=channel_names
        )
        
        self.quality_monitor = SignalQuality()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(
            f"Initialized EEG signal processor with {len(channel_names)} "
            f"channels"
        )
    
    def process_chunk(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a chunk of EEG data.
        
        Args:
            data: EEG data array of shape (n_channels, n_samples)
            
        Returns:
            Tuple of (processed data, quality metrics)
        """
        start_time = time.time()
        
        try:
            # Update baseline and apply adaptive filtering
            self.adaptive_filter.update_baseline(data)
            filtered_data = self.adaptive_filter.apply(data)
            
            # Remove artifacts
            cleaned_data, artifact_stats = self.artifact_removal.clean_signal(
                filtered_data
            )
            
            # Detect memory patterns
            pattern_probs = self.pattern_detector.predict_proba(
                cleaned_data[np.newaxis, :]
            )
            
            # Update quality metrics
            processing_time = time.time() - start_time
            quality_metrics = self.quality_monitor.update(
                data,
                cleaned_data,
                artifact_stats['artifact_samples'],
                processing_time
            )
            
            # Combine results
            results = {
                **quality_metrics,
                'memory_pattern_probability': pattern_probs[0, 1],
                'baseline_complete': self.adaptive_filter.baseline_complete
            }
            
            return cleaned_data, results
            
        except Exception as e:
            logger.error(f"Error processing EEG chunk: {str(e)}")
            raise
    
    def analyze_frequency_bands(
        self, data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Analyze power in different frequency bands.
        
        Args:
            data: EEG data array of shape (n_channels, n_samples)
            
        Returns:
            Dictionary of band powers
        """
        try:
            band_powers = {}
            
            # Calculate frequency spectrum
            freqs = np.fft.fftfreq(
                data.shape[1],
                1/self.config.sampling_rate
            )
            fft_vals = np.abs(np.fft.fft(data))**2
            
            # Calculate power in each band
            for band_name, (low, high) in self.config.bands.items():
                # Find frequencies in band
                mask = (freqs >= low) & (freqs <= high)
                
                # Calculate average power in band for each channel
                band_powers[band_name] = np.mean(
                    fft_vals[:, mask],
                    axis=1
                )
            
            return band_powers
            
        except Exception as e:
            logger.error(f"Error analyzing frequency bands: {str(e)}")
            raise
    
    def get_quality_report(self) -> Dict[str, float]:
        """
        Get a report of signal quality metrics.
        
        Returns:
            Dictionary of quality metrics
        """
        metrics = self.quality_monitor.metrics
        
        return {
            'average_snr': np.mean(metrics['snr']),
            'average_artifact_rate': np.mean(metrics['artifact_rate']),
            'average_processing_time': np.mean(metrics['processing_time']),
            'processing_time_std': np.std(metrics['processing_time']),
            'recent_snr_trend': np.mean(metrics['snr'][-10:]),
            'recent_artifact_trend': np.mean(
                metrics['artifact_rate'][-10:]
            )
        }
    
    def __del__(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


class SignalProcessor:
    """Handles real-time EEG signal processing and frequency band analysis."""
    
    def __init__(self, sampling_rate=256, window_size=256):
        """
        Initialize the signal processor.
        
        Args:
            sampling_rate (int): EEG sampling rate in Hz
            window_size (int): Size of the processing window in samples
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.freqs = np.fft.fftfreq(window_size, 1/sampling_rate)
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """
        Apply a bandpass filter to the EEG data.
        
        Args:
            data (np.ndarray): Raw EEG data
            low_freq (float): Lower cutoff frequency
            high_freq (float): Upper cutoff frequency
            
        Returns:
            np.ndarray: Filtered EEG data
        """
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
        return signal.filtfilt(b, a, data)
    
    def extract_frequency_bands(self, data):
        """
        Extract different frequency bands from EEG data.
        
        Args:
            data (np.ndarray): Raw EEG data
            
        Returns:
            dict: Dictionary containing filtered data for each frequency band
        """
        band_data = {}
        for band_name, (low, high) in self.bands.items():
            band_data[band_name] = self.apply_bandpass_filter(data, low, high)
        return band_data
    
    def compute_band_power(self, data):
        """
        Compute power in different frequency bands.
        
        Args:
            data (np.ndarray): Raw EEG data
            
        Returns:
            dict: Dictionary containing power values for each frequency band
        """
        fft_vals = np.abs(np.fft.fft(data))**2
        
        band_power = {}
        for band_name, (low, high) in self.bands.items():
            # Find frequencies that fall within the band
            mask = (self.freqs >= low) & (self.freqs <= high)
            band_power[band_name] = np.mean(fft_vals[mask])
            
        return band_power
    
    def analyze_memory_state(self, band_powers):
        """
        Analyze the current memory formation state based on band powers.
        
        Args:
            band_powers (dict): Dictionary of band powers
            
        Returns:
            dict: Memory state analysis results
        """
        # Theta/Alpha ratio is an indicator of memory formation
        theta_alpha_ratio = band_powers['theta'] / band_powers['alpha']
        
        # Beta/Theta ratio indicates attention level
        beta_theta_ratio = band_powers['beta'] / band_powers['theta']
        
        return {
            'memory_formation_strength': theta_alpha_ratio,
            'attention_level': beta_theta_ratio,
            'theta_power': band_powers['theta'],
            'gamma_power': band_powers['gamma']  # Associated with active learning
        }
