"""Configuration module for Neural Memory Mapper."""

import os
from pathlib import Path

import yaml


class Config:
    """Configuration handler for Neural Memory Mapper."""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration.
        
        Args:
            config_path (str, optional): Path to config file
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'config',
                'default_config.yml'
            )
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def eeg_config(self):
        """Get EEG configuration."""
        return self.config.get('eeg', {})
    
    @property
    def processing_config(self):
        """Get signal processing configuration."""
        return self.config.get('processing', {})
    
    @property
    def visualization_config(self):
        """Get visualization configuration."""
        return self.config.get('visualization', {})
    
    @property
    def task_config(self):
        """Get memory task configuration."""
        return self.config.get('tasks', {})
