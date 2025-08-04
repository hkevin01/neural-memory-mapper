"""
Signal processing module for Neural Memory Mapper.
Handles EEG data processing and frequency analysis.
"""

import mne
import numpy as np
from scipy import signal


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
