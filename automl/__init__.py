"""
AutoML - Automated Machine Learning System

An intelligent AutoML system that automatically selects, trains, and optimizes
machine learning models for your datasets.
"""

__version__ = "0.1.0"
__author__ = "AutoML Team"
__license__ = "Apache 2.0"

from automl.config.config import Config
from automl.core.automl import AutoML

__all__ = ["AutoML", "Config"]
