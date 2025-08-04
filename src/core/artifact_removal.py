"""Artifact removal and signal cleaning for EEG data."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy import signal

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ArtifactParams:
    """Parameters for artifact detection and removal."""
    
    # Thresholds for artifact detection
    amplitude_threshold: float = 100.0  # μV
    gradient_threshold: float = 25.0    # μV/sample
    line_freq: float = 50.0            # Hz
    
    # ICA parameters
    n_components: Optional[int] = None
    random_state: int = 42
    
    # Bandpass filter parameters
    bandpass_low: float = 0.1   # Hz
    bandpass_high: float = 100  # Hz


class ArtifactRemoval:
    """Handles artifact detection and removal from EEG signals."""

    def __init__(
        self,
        sampling_rate: float,
        channel_names: List[str],
        params: Optional[ArtifactParams] = None
    ) -> None:
        """
        Initialize artifact removal processor.

        Args:
            sampling_rate: EEG sampling rate in Hz
            channel_names: List of EEG channel names
            params: Artifact detection and removal parameters
        """
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        self.params = params or ArtifactParams()
        
        # Set up MNE info for creating Raw object
        self.info = mne.create_info(
            ch_names=channel_names,
            sfreq=sampling_rate,
            ch_types=['eeg'] * len(channel_names)
        )
        
        logger.info(
            f"Initialized artifact removal with {len(channel_names)} channels "
            f"at {sampling_rate} Hz"
        )

    def detect_artifacts(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect artifacts in EEG data.

        Args:
            data: EEG data array of shape (n_channels, n_samples)

        Returns:
            Tuple of (artifact mask, artifact details dictionary)
        """
        artifact_mask = np.zeros(data.shape[1], dtype=bool)
        artifact_details = {}

        # Amplitude threshold detection
        amplitude_artifacts = np.any(
            np.abs(data) > self.params.amplitude_threshold,
            axis=0
        )
        artifact_mask |= amplitude_artifacts
        artifact_details['amplitude'] = amplitude_artifacts

        # Gradient (fast changes) detection
        gradients = np.abs(np.diff(data, axis=1))
        gradient_artifacts = np.any(
            gradients > self.params.gradient_threshold,
            axis=0
        )
        artifact_mask[:-1] |= gradient_artifacts
        artifact_details['gradient'] = gradient_artifacts

        logger.debug(
            f"Detected {np.sum(artifact_mask)} samples with artifacts"
        )
        
        return artifact_mask, artifact_details

    def remove_line_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Remove power line noise using notch filter.

        Args:
            data: EEG data array of shape (n_channels, n_samples)

        Returns:
            Cleaned EEG data
        """
        # Create notch filter
        nyquist = self.sampling_rate / 2
        quality_factor = 30.0  # Q-factor for the notch filter
        
        b, a = signal.iirnotch(
            self.params.line_freq,
            quality_factor,
            self.sampling_rate
        )
        
        # Apply filter
        cleaned_data = signal.filtfilt(b, a, data, axis=1)
        
        logger.debug("Applied line noise removal filter")
        return cleaned_data

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to remove DC offset and high-frequency noise.

        Args:
            data: EEG data array of shape (n_channels, n_samples)

        Returns:
            Filtered EEG data
        """
        nyquist = self.sampling_rate / 2
        low = self.params.bandpass_low / nyquist
        high = self.params.bandpass_high / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        
        logger.debug(
            f"Applied bandpass filter ({self.params.bandpass_low}-"
            f"{self.params.bandpass_high} Hz)"
        )
        return filtered_data

    def apply_ica(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Independent Component Analysis for artifact removal.

        Args:
            data: EEG data array of shape (n_channels, n_samples)

        Returns:
            Cleaned EEG data
        """
        # Create MNE Raw object
        raw = mne.io.RawArray(data, self.info)
        
        # Apply ICA
        ica = mne.preprocessing.ICA(
            n_components=self.params.n_components,
            random_state=self.params.random_state
        )
        
        try:
            ica.fit(raw)
            
            # Find components that correlate with EOG (eye movements)
            eog_indices, _ = ica.find_bads_eog(raw)
            
            # Find components that look like ECG (heart beats)
            ecg_indices, _ = ica.find_bads_ecg(raw)
            
            # Combine artifact components
            bad_components = list(set(eog_indices + ecg_indices))
            
            if bad_components:
                logger.info(
                    f"Found {len(bad_components)} artifact components "
                    f"using ICA"
                )
                
                # Remove artifacts
                ica.exclude = bad_components
                cleaned_raw = ica.apply(raw.copy())
                return cleaned_raw.get_data()
            
            return data
            
        except Exception as e:
            logger.error(f"ICA failed: {str(e)}")
            return data

    def clean_signal(
        self, data: np.ndarray, apply_ica: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Clean EEG signal by removing artifacts.

        Args:
            data: EEG data array of shape (n_channels, n_samples)
            apply_ica: Whether to apply ICA-based artifact removal

        Returns:
            Tuple of (cleaned data, artifact details)
        """
        logger.info("Starting signal cleaning process")
        
        # Store original data for comparison
        original_data = data.copy()
        
        # Apply basic filtering
        data = self.apply_bandpass(data)
        data = self.remove_line_noise(data)
        
        # Detect artifacts
        artifact_mask, artifact_details = self.detect_artifacts(data)
        
        # Apply ICA if requested
        if apply_ica:
            data = self.apply_ica(data)
        
        # Calculate cleaning statistics
        stats = {
            'total_samples': len(artifact_mask),
            'artifact_samples': np.sum(artifact_mask),
            'artifact_percentage': np.mean(artifact_mask) * 100,
            'improvement': np.mean(
                np.abs(original_data - data)
            )
        }
        
        logger.info(
            f"Completed signal cleaning. "
            f"Found artifacts in {stats['artifact_percentage']:.2f}% "
            f"of samples"
        )
        
        return data, stats
