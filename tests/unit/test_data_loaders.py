"""Tests for data loaders."""

from pathlib import Path

import pandas as pd
import pytest

from automl.data.loaders import CSVLoader, ParquetLoader
from automl.utils.exceptions import DataLoadError


class TestCSVLoader:
    """Tests for CSV loader."""

    def test_load_csv(self, sample_csv):
        """Test loading a CSV file."""
        loader = CSVLoader()
        df = loader.load(sample_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 5

    def test_validate_format(self):
        """Test format validation."""
        loader = CSVLoader()

        assert loader.validate_format(Path("test.csv")) is True
        assert loader.validate_format(Path("test.txt")) is True
        assert loader.validate_format(Path("test.parquet")) is False

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = CSVLoader()

        with pytest.raises(DataLoadError, match="File not found"):
            loader.load(Path("nonexistent.csv"))

    def test_metadata_extraction(self, sample_csv):
        """Test metadata extraction."""
        loader = CSVLoader()
        df = loader.load(sample_csv)

        assert "n_rows" in loader.metadata
        assert "n_columns" in loader.metadata
        assert loader.metadata["n_rows"] == 100
        assert loader.metadata["n_columns"] == 5


class TestParquetLoader:
    """Tests for Parquet loader."""

    def test_load_parquet(self, sample_parquet):
        """Test loading a Parquet file."""
        loader = ParquetLoader()
        df = loader.load(sample_parquet)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 5

    def test_validate_format(self):
        """Test format validation."""
        loader = ParquetLoader()

        assert loader.validate_format(Path("test.parquet")) is True
        assert loader.validate_format(Path("test.pq")) is True
        assert loader.validate_format(Path("test.csv")) is False

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = ParquetLoader()

        with pytest.raises(DataLoadError, match="File not found"):
            loader.load(Path("nonexistent.parquet"))
