"""CSV file loader."""

from pathlib import Path
from typing import Any, Dict, Optional

import chardet
import pandas as pd

from automl.data.loaders.base_loader import BaseLoader
from automl.utils.exceptions import DataLoadError
from automl.utils.logger import get_logger

logger = get_logger(__name__)


class CSVLoader(BaseLoader):
    """Loader for CSV files with automatic encoding detection."""

    def validate_format(self, filepath: Path) -> bool:
        """
        Validate if file is CSV format.

        Args:
            filepath: Path to file

        Returns:
            True if file has .csv extension
        """
        return filepath.suffix.lower() in [".csv", ".txt"]

    def detect_encoding(self, filepath: Path, sample_size: int = 10000) -> str:
        """
        Detect file encoding.

        Args:
            filepath: Path to file
            sample_size: Number of bytes to sample

        Returns:
            Detected encoding (e.g., 'utf-8', 'latin-1')
        """
        try:
            with open(filepath, "rb") as f:
                raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            encoding = result["encoding"]
            confidence = result["confidence"]

            logger.debug(
                f"Detected encoding: {encoding} (confidence: {confidence:.2f})"
            )

            # Default to utf-8 if detection is uncertain
            if confidence < 0.7:
                logger.warning(
                    f"Low encoding confidence ({confidence:.2f}), defaulting to utf-8"
                )
                return "utf-8"

            return encoding if encoding else "utf-8"
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}. Using utf-8")
            return "utf-8"

    def detect_separator(self, filepath: Path, encoding: str, nrows: int = 5) -> str:
        """
        Detect CSV separator.

        Args:
            filepath: Path to file
            encoding: File encoding
            nrows: Number of rows to sample

        Returns:
            Detected separator
        """
        try:
            # Try common separators
            separators = [",", ";", "\t", "|"]
            max_cols = 0
            best_sep = ","

            for sep in separators:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding=encoding, nrows=nrows)
                    if df.shape[1] > max_cols:
                        max_cols = df.shape[1]
                        best_sep = sep
                except:
                    continue

            logger.debug(f"Detected separator: '{best_sep}'")
            return best_sep
        except Exception as e:
            logger.warning(f"Separator detection failed: {e}. Using comma")
            return ","

    def load(
        self,
        filepath: Path,
        encoding: Optional[str] = None,
        separator: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding and separator detection.

        Args:
            filepath: Path to CSV file
            encoding: File encoding (auto-detected if None)
            separator: Column separator (auto-detected if None)
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame

        Raises:
            DataLoadError: If loading fails
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise DataLoadError(f"File not found: {filepath}")

        if not self.validate_format(filepath):
            raise DataLoadError(
                f"Invalid file format. Expected CSV, got {filepath.suffix}"
            )

        logger.info(f"Loading CSV file: {filepath}")

        try:
            # Auto-detect encoding if not provided
            if encoding is None:
                encoding = self.detect_encoding(filepath)

            # Auto-detect separator if not provided
            if separator is None:
                separator = self.detect_separator(filepath, encoding)

            # Load data
            df = pd.read_csv(filepath, encoding=encoding, sep=separator, **kwargs)

            logger.info(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} columns"
            )

            # Extract metadata
            self.extract_metadata(df, filepath)
            self.metadata.update(
                {
                    "encoding": encoding,
                    "separator": separator,
                }
            )

            return df

        except pd.errors.EmptyDataError:
            raise DataLoadError(f"File is empty: {filepath}")
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Failed to parse CSV: {e}")
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV file: {e}")
