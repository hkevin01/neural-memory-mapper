"""
Data acquisition module for Neural Memory Mapper.
Handles EEG data streaming and device integration.
"""

import queue
import threading
import time
from abc import ABC, abstractmethod

import numpy as np


class DataSource(ABC):
    """Abstract base class for EEG data sources."""
    
    @abstractmethod
    def connect(self):
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    def get_data(self):
        """Retrieve data from the source."""
        pass


class MockEEGDevice(DataSource):
    """Mock EEG device for development and testing."""
    
    def __init__(self, sampling_rate=256, num_channels=8):
        """
        Initialize mock EEG device.
        
        Args:
            sampling_rate (int): Sampling rate in Hz
            num_channels (int): Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.is_running = False
        self.data_queue = queue.Queue()
        self.thread = None
    
    def connect(self):
        """Start the mock data generation."""
        self.is_running = True
        self.thread = threading.Thread(target=self._generate_data)
        self.thread.daemon = True
        self.thread.start()
    
    def disconnect(self):
        """Stop the mock data generation."""
        self.is_running = False
        if self.thread:
            self.thread.join()
    
    def get_data(self):
        """
        Get the next chunk of data.
        
        Returns:
            np.ndarray: Generated EEG data
        """
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _generate_data(self):
        """Generate mock EEG data with realistic characteristics."""
        while self.is_running:
            # Generate base signal (pink noise)
            t = np.linspace(0, 1, self.sampling_rate)
            base_signal = np.zeros((self.num_channels, self.sampling_rate))
            
            for ch in range(self.num_channels):
                # Generate pink noise
                white_noise = np.random.normal(0, 1, self.sampling_rate)
                f = np.fft.fftfreq(self.sampling_rate)
                f[0] = np.inf  # Avoid division by zero
                pink_noise = np.real(np.fft.ifft(
                    np.fft.fft(white_noise) / np.sqrt(np.abs(f))
                ))
                
                # Add some simulated brain rhythms
                alpha = 2 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
                theta = 1.5 * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
                beta = np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
                
                base_signal[ch] = (pink_noise + alpha + theta + beta) / 4
            
            self.data_queue.put(base_signal)
            time.sleep(1/self.sampling_rate)
