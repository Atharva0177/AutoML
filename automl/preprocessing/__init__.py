"""
Preprocessing module for AutoML system.

This module provides data preprocessing functionality including:
- Missing value imputation
- Numerical scaling
- Categorical encoding
- Data splitting
- Pipeline building
"""

from automl.preprocessing.cleaners.missing_handler import MissingValueHandler
from automl.preprocessing.pipeline_builder import PipelineBuilder
from automl.preprocessing.splitters.train_test_splitter import DataSplitter
from automl.preprocessing.transformers.encoders import CategoricalEncoder
from automl.preprocessing.transformers.scalers import NumericalScaler

__all__ = [
    "MissingValueHandler",
    "NumericalScaler",
    "CategoricalEncoder",
    "DataSplitter",
    "PipelineBuilder",
]
