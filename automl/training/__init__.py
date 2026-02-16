"""
AutoML Training Module.

This module provides training pipeline components including metrics,
cross-validation, trainers, and model comparison.
"""

from automl.training.metrics import MetricsCalculator
from automl.training.cross_validator import CrossValidator
from automl.training.trainer import Trainer

__all__ = ['MetricsCalculator', 'CrossValidator', 'Trainer']
