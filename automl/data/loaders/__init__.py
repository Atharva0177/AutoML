"""Data loaders module."""

from automl.data.loaders.base_loader import BaseLoader
from automl.data.loaders.csv_loader import CSVLoader
from automl.data.loaders.parquet_loader import ParquetLoader

__all__ = ["BaseLoader", "CSVLoader", "ParquetLoader"]
