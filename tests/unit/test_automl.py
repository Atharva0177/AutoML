"""Tests for AutoML core functionality."""

from pathlib import Path

import pytest

from automl import AutoML
from automl.utils.exceptions import UnsupportedFormatError


class TestAutoML:
    """Tests for AutoML class."""

    def test_initialization(self):
        """Test AutoML initialization."""
        aml = AutoML()
        assert aml.data is None
        assert aml.target_column is None

    def test_load_csv(self, sample_csv):
        """Test loading CSV data."""
        aml = AutoML()
        df = aml.load_data(sample_csv)

        assert aml.data is not None
        assert len(aml.data) == 100
        assert df is aml.data

    def test_load_parquet(self, sample_parquet):
        """Test loading Parquet data."""
        aml = AutoML()
        df = aml.load_data(sample_parquet)

        assert aml.data is not None
        assert len(aml.data) == 100

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        aml = AutoML()

        with pytest.raises(UnsupportedFormatError):
            aml.load_data(Path("test.xlsx"))

    def test_set_target(self, sample_csv):
        """Test setting target column."""
        aml = AutoML()
        aml.load_data(sample_csv)
        aml.set_target("target")

        assert aml.target_column == "target"

    def test_set_invalid_target(self, sample_csv):
        """Test setting invalid target column."""
        aml = AutoML()
        aml.load_data(sample_csv)

        with pytest.raises(ValueError, match="not found"):
            aml.set_target("nonexistent")

    def test_get_data_info(self, sample_csv):
        """Test getting data information."""
        aml = AutoML()
        aml.load_data(sample_csv, target_column="target")

        info = aml.get_data_info()

        assert info["shape"] == (100, 5)
        assert info["target_column"] == "target"
        assert "quality_score" in info

    def test_save_load_metadata(self, sample_csv, temp_dir):
        """Test saving and loading metadata."""
        aml = AutoML()
        aml.load_data(sample_csv)

        metadata_path = temp_dir / "metadata.json"
        aml.save_metadata(metadata_path)

        assert metadata_path.exists()
