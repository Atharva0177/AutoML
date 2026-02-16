"""Logging utilities for AutoML system."""

import logging
import sys
from pathlib import Path
from typing import Optional
import os
from datetime import datetime


class AutoMLLogger:
    """Centralized logging for AutoML system."""

    _instance: Optional["AutoMLLogger"] = None
    _logger: logging.Logger

    def __new__(cls):
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger if not already initialized."""
        if not hasattr(self, '_logger') or self._logger is None:
            self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup logging configuration."""
        # Get log level from environment or use INFO
        log_level = os.getenv("AUTOML_LOG_LEVEL", "INFO").upper()
        
        # Create logger
        self._logger = logging.getLogger("automl")
        self._logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Avoid duplicate handlers
        if self._logger.handlers:
            return
        
        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        self._logger.addHandler(console_handler)
        
        # File handler if log file specified
        log_file = os.getenv("AUTOML_LOG_FILE")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            self._logger.addHandler(file_handler)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        if self._logger is None:
            self._setup_logger()
        return self._logger

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        if self._logger is not None:
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        if self._logger is not None:
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        if self._logger is not None:
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        if self._logger is not None:
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        if self._logger is not None:
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        if self._logger is not None:
            self._logger.exception(msg, *args, **kwargs)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance for a specific module.
    
    Args:
        name: Name for the logger (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"automl.{name}")
    return AutoMLLogger().logger


# Create global logger instance
logger = AutoMLLogger()
