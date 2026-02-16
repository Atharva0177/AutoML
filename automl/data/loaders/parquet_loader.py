"""Parquet file loader."""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from automl.data.loaders.base_loader import BaseLoader
from automl.utils.logger import get_logger
from automl.utils.exceptions import DataLoadError

logger = get_logger(__name__)


class ParquetLoader(BaseLoader):
    """Loader for Parquet files."""

    def validate_format(self, filepath: Path) -> bool:
        """
        Validate if file is Parquet format.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file has .parquet extension
        """
        return filepath.suffix.lower() in [".parquet", ".pq"]

    def load(
        self,
        filepath: Path,
        columns: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Parquet file.
        
        Args:
            filepath: Path to Parquet file
            columns: Columns to load (None for all)
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataLoadError: If loading fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise DataLoadError(f"File not found: {filepath}")
        
        if not self.validate_format(filepath):
            raise DataLoadError(f"Invalid file format. Expected Parquet, got {filepath.suffix}")
        
        logger.info(f"Loading Parquet file: {filepath}")
        
        try:
            # Load data
            df = pd.read_parquet(
                filepath,
                columns=columns,
                **kwargs
            )
            
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Extract metadata
            self.extract_metadata(df, filepath)
            
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load Parquet file: {e}")
