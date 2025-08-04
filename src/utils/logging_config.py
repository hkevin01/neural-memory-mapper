"""Logging configuration for Neural Memory Mapper."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


class LoggerConfig:
    """Configure logging for the application."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        """
        Initialize logging configuration.

        Args:
            log_dir: Directory for log files. Defaults to 'logs' in project root.
        """
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'logs'
            )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging format
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Name of the logger (usually __name__)

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # File handlers
        debug_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'debug.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(self.formatter)

        info_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'info.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(self.formatter)

        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'error.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)

        # Add handlers to logger
        logger.addHandler(debug_handler)
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)

        return logger


# Global logger configuration
logger_config = LoggerConfig()


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Configured logger instance
    """
    return logger_config.get_logger(name)
