"""Tests for data validators."""

import pytest
import pandas as pd
import numpy as np

from automl.data.validators import DataValidator, SchemaValidator, QualityValidator


class TestDataValidator:
    """Tests for data validator."""

    def test_validate_good_data(self, sample_df):
        """Test validation of good quality data."""
        validator = DataValidator()
        is_valid, issues = validator.validate(sample_df)
        
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_small_data(self):
        """Test validation of too small dataset."""
        validator = DataValidator()
        small_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        is_valid, issues = validator.validate(small_df)
        
        assert is_valid is False
        assert any("Insufficient rows" in issue for issue in issues)

    def test_validate_target(self, sample_df):
        """Test target column validation."""
        validator = DataValidator()
        
        # Valid target
        is_valid, issues = validator.validate_target(sample_df, "target")
        assert is_valid is True
        
        # Invalid target (doesn't exist)
        is_valid, issues = validator.validate_target(sample_df, "nonexistent")
        assert is_valid is False
        assert any("not found" in issue for issue in issues)


class TestSchemaValidator:
    """Tests for schema validator."""

    def test_infer_schema(self, sample_df):
        """Test schema inference."""
        validator = SchemaValidator()
        schema = validator.infer_schema(sample_df)
        
        assert "numeric_1" in schema
        assert "cat_1" in schema
        assert schema["numeric_1"]["inferred_type"] == "numeric"
        assert schema["cat_1"]["inferred_type"] == "categorical"


class TestQualityValidator:
    """Tests for quality validator."""

    def test_generate_quality_report(self, sample_df):
        """Test quality report generation."""
        validator = QualityValidator()
        report = validator.generate_quality_report(sample_df)
        
        assert "overall_score" in report
        assert "dimensions" in report
        assert "missing_values" in report
        assert report["dimensions"]["n_rows"] == 100
        assert report["overall_score"] > 0
