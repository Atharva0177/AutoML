"""Base classes for data loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from automl.utils.exceptions import DataLoadError
from automl.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self):
        """Initialize loader."""
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def load(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            filepath: Path to data file
            **kwargs: Additional loader-specific arguments

        Returns:
            Loaded DataFrame

        Raises:
            DataLoadError: If loading fails
        """
        pass

    @abstractmethod
    def validate_format(self, filepath: Path) -> bool:
        """
        Validate if file format is supported.

        Args:
            filepath: Path to file

        Returns:
            True if format is supported
        """
        pass

    def extract_metadata(self, df: pd.DataFrame, filepath: Path) -> Dict[str, Any]:
        """
        Extract metadata from loaded data.

        Args:
            df: Loaded DataFrame
            filepath: Original file path

        Returns:
            Metadata dictionary
        """
        from automl.utils.helpers import format_bytes, get_memory_usage

        metadata = {
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size if filepath.exists() else 0,
            "file_size_formatted": (
                format_bytes(filepath.stat().st_size) if filepath.exists() else "0 B"
            ),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_bytes": get_memory_usage(df),
            "memory_usage_formatted": format_bytes(get_memory_usage(df)),
        }

        self.metadata = metadata
        return metadata

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"
