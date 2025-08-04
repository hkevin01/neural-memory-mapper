"""Machine learning module for memory pattern detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class EEGFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant features from EEG signals for memory pattern detection."""
    
    def __init__(
        self,
        sampling_rate: float,
        window_size: int,
        overlap: float = 0.5
    ):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            window_size: Analysis window size in samples
            overlap: Overlap between windows (0.0-1.0)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
        # Calculate frequency bins
        self.frequencies = np.fft.fftfreq(
            window_size, 1/sampling_rate
        )[:window_size//2]
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def _extract_band_powers(
        self, data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract power in different frequency bands.
        
        Args:
            data: EEG data segment
            
        Returns:
            Dictionary of band powers
        """
        fft_vals = np.abs(np.fft.fft(data))[:len(self.frequencies)]
        powers = {}
        
        for band_name, (low, high) in self.bands.items():
            mask = (self.frequencies >= low) & (self.frequencies <= high)
            powers[band_name] = np.mean(fft_vals[mask])
        
        return powers
    
    def _extract_connectivity(
        self, data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract connectivity features between channels.
        
        Args:
            data: EEG data segment
            
        Returns:
            Dictionary of connectivity metrics
        """
        n_channels = data.shape[0]
        
        # Calculate cross-correlation matrix
        corr_matrix = np.corrcoef(data)
        
        # Calculate coherence matrix
        coherence_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                f, Cxy = signal.coherence(
                    data[i], data[j],
                    fs=self.sampling_rate
                )
                coherence_matrix[i, j] = np.mean(Cxy)
                coherence_matrix[j, i] = coherence_matrix[i, j]
        
        return {
            'correlation': corr_matrix,
            'coherence': coherence_matrix
        }
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform EEG data into feature vectors.
        
        Args:
            X: EEG data array of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Feature matrix
        """
        n_samples = X.shape[0]
        features = []
        
        for i in range(n_samples):
            sample_features = []
            
            # Extract band powers
            powers = self._extract_band_powers(X[i])
            for band in self.bands:
                sample_features.extend(powers[band])
            
            # Extract connectivity features
            connectivity = self._extract_connectivity(X[i])
            
            # Flatten and add upper triangle of connectivity matrices
            corr_features = connectivity['correlation'][
                np.triu_indices_from(
                    connectivity['correlation'], k=1
                )
            ]
            coh_features = connectivity['coherence'][
                np.triu_indices_from(
                    connectivity['coherence'], k=1
                )
            ]
            
            sample_features.extend(corr_features)
            sample_features.extend(coh_features)
            
            features.append(sample_features)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit method (no-op for this transformer)."""
        return self


class MemoryPatternNet(nn.Module):
    """Neural network for memory pattern detection."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2
    ):
        """
        Initialize the network.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class MemoryPatternDetector:
    """Detect memory formation patterns in EEG signals."""
    
    def __init__(
        self,
        sampling_rate: float,
        window_size: int,
        model_type: str = 'rf'
    ):
        """
        Initialize the detector.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            window_size: Analysis window size in samples
            model_type: Type of model to use ('rf' or 'nn')
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.model_type = model_type
        
        # Initialize feature extractor
        self.feature_extractor = EEGFeatureExtractor(
            sampling_rate=sampling_rate,
            window_size=window_size
        )
        
        # Initialize model pipeline
        if model_type == 'rf':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ))
            ])
        else:
            self.model = None  # Neural network model initialized in fit
        
        logger.info(
            f"Initialized MemoryPatternDetector with {model_type} model"
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'MemoryPatternDetector':
        """
        Train the memory pattern detector.
        
        Args:
            X: EEG data array of shape (n_samples, n_channels, n_timepoints)
            y: Labels array
            **kwargs: Additional training parameters
            
        Returns:
            self
        """
        # Extract features
        X_features = self.feature_extractor.transform(X)
        
        if self.model_type == 'rf':
            self.model.fit(X_features, y)
        else:
            # Initialize neural network
            input_size = X_features.shape[1]
            self.model = MemoryPatternNet(
                input_size=input_size,
                **kwargs
            )
            
            # Convert data to tensors
            X_tensor = torch.FloatTensor(X_features)
            y_tensor = torch.LongTensor(y)
            
            # Train the network
            self._train_network(X_tensor, y_tensor, **kwargs)
        
        logger.info("Completed model training")
        return self
    
    def _train_network(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs
    ) -> None:
        """
        Train the neural network model.
        
        Args:
            X: Input tensor
            y: Target tensor
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Simple batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Loss: {total_loss/len(X):.4f}"
                )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict memory formation patterns.
        
        Args:
            X: EEG data array of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Predicted labels
        """
        X_features = self.feature_extractor.transform(X)
        
        if self.model_type == 'rf':
            return self.model.predict(X_features)
        else:
            self.model.eval()
            X_tensor = torch.FloatTensor(X_features)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for memory formation patterns.
        
        Args:
            X: EEG data array of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Predicted probabilities
        """
        X_features = self.feature_extractor.transform(X)
        
        if self.model_type == 'rf':
            return self.model.predict_proba(X_features)
        else:
            self.model.eval()
            X_tensor = torch.FloatTensor(X_features)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                return nn.Softmax(dim=1)(outputs).numpy()
