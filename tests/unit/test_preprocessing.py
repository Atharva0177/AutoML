"""
Unit tests for preprocessing module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.preprocessing import (
    CategoricalEncoder,
    DataSplitter,
    MissingValueHandler,
    NumericalScaler,
    PipelineBuilder,
)
from automl.utils.exceptions import ValidationError


class TestMissingValueHandler:
    """Test missing value imputation."""

    def test_mean_imputation(self, sample_df_with_missing):
        """Test mean imputation strategy."""
        handler = MissingValueHandler(strategy="mean")
        df_imputed = handler.fit_transform(sample_df_with_missing)

        assert df_imputed.isnull().sum().sum() == 0
        assert len(handler.imputation_values) > 0

    def test_median_imputation(self, sample_df_with_missing):
        """Test median imputation strategy."""
        handler = MissingValueHandler(strategy="median")
        df_imputed = handler.fit_transform(sample_df_with_missing)

        assert df_imputed.isnull().sum().sum() == 0

    def test_mode_imputation(self, sample_df_with_missing):
        """Test mode imputation strategy."""
        handler = MissingValueHandler(strategy="mode")
        df_imputed = handler.fit_transform(sample_df_with_missing)

        assert df_imputed.isnull().sum().sum() == 0

    def test_constant_imputation(self, sample_df_with_missing):
        """Test constant value imputation."""
        handler = MissingValueHandler(strategy="constant", fill_value=-999)
        df_imputed = handler.fit_transform(sample_df_with_missing)

        assert df_imputed.isnull().sum().sum() == 0
        assert -999 in df_imputed.values

    def test_drop_strategy(self, sample_df_with_missing):
        """Test dropping rows with missing values."""
        handler = MissingValueHandler(strategy="drop", threshold=0.5)
        df_imputed = handler.fit_transform(sample_df_with_missing)

        assert df_imputed.isnull().sum().sum() == 0
        assert len(df_imputed) <= len(sample_df_with_missing)

    def test_transform(self, sample_df_with_missing):
        """Test transform on new data."""
        handler = MissingValueHandler(strategy="mean")
        handler.fit_transform(sample_df_with_missing)

        # Transform new data
        new_df = sample_df_with_missing.copy()
        df_transformed = handler.transform(new_df)

        assert df_transformed.isnull().sum().sum() == 0

    def test_get_summary(self, sample_df_with_missing):
        """Test getting imputation summary."""
        handler = MissingValueHandler(strategy="mean")
        handler.fit_transform(sample_df_with_missing)

        summary = handler.get_imputation_summary()
        assert summary["strategy"] == "mean"
        assert "imputation_values" in summary


class TestNumericalScaler:
    """Test numerical scaling."""

    def test_standard_scaler(self, sample_df):
        """Test standard scaling."""
        scaler = NumericalScaler(method="standard")
        df_scaled = scaler.fit_transform(sample_df)

        # Check that numerical columns are scaled
        for col in scaler.numerical_cols:
            assert abs(df_scaled[col].mean()) < 1e-10  # Mean should be ~0
            assert abs(df_scaled[col].std() - 1.0) < 0.1  # Std should be ~1

    def test_minmax_scaler(self, sample_df):
        """Test min-max scaling."""
        scaler = NumericalScaler(method="minmax", feature_range=(0, 1))
        df_scaled = scaler.fit_transform(sample_df)

        # Check that values are in range [0, 1]
        for col in scaler.numerical_cols:
            assert df_scaled[col].min() >= 0
            assert df_scaled[col].max() <= 1

    def test_robust_scaler(self, sample_df):
        """Test robust scaling."""
        scaler = NumericalScaler(method="robust")
        df_scaled = scaler.fit_transform(sample_df)

        assert len(scaler.numerical_cols) > 0
        assert scaler.scaler is not None

    def test_transform(self, sample_df):
        """Test transform on new data."""
        scaler = NumericalScaler(method="standard")
        scaler.fit_transform(sample_df)

        # Transform new data
        new_df = sample_df.copy()
        df_transformed = scaler.transform(new_df)

        assert df_transformed.shape == new_df.shape

    def test_inverse_transform(self, sample_df):
        """Test inverse transformation."""
        scaler = NumericalScaler(method="standard")
        df_scaled = scaler.fit_transform(sample_df)
        df_original = scaler.inverse_transform(df_scaled)

        # Check that values are close to original
        for col in scaler.numerical_cols:
            assert np.allclose(df_original[col], sample_df[col], rtol=1e-5)

    def test_none_method(self, sample_df):
        """Test no scaling."""
        scaler = NumericalScaler(method="none")
        df_scaled = scaler.fit_transform(sample_df)

        pd.testing.assert_frame_equal(df_scaled, sample_df)

    def test_get_summary(self, sample_df):
        """Test getting scaling summary."""
        scaler = NumericalScaler(method="standard")
        scaler.fit_transform(sample_df)

        summary = scaler.get_scaling_summary()
        assert summary["method"] == "standard"
        assert len(summary["scaling_params"]) > 0


class TestCategoricalEncoder:
    """Test categorical encoding."""

    def test_onehot_encoding(self, sample_df):
        """Test one-hot encoding."""
        encoder = CategoricalEncoder(method="onehot")
        df_encoded = encoder.fit_transform(sample_df)

        # Check that original categorical columns are removed
        for col in encoder.categorical_cols:
            assert col not in df_encoded.columns

        # Check that new columns are created
        assert len(encoder.encoded_columns) > 0

    def test_label_encoding(self, sample_df):
        """Test label encoding."""
        encoder = CategoricalEncoder(method="label")
        df_encoded = encoder.fit_transform(sample_df)

        # Check that categorical columns are now numeric
        for col in encoder.categorical_cols:
            assert pd.api.types.is_numeric_dtype(df_encoded[col])

    def test_ordinal_encoding(self, sample_df):
        """Test ordinal encoding."""
        encoder = CategoricalEncoder(method="ordinal")
        df_encoded = encoder.fit_transform(sample_df)

        # Check that categorical columns are now numeric
        for col in encoder.categorical_cols:
            assert pd.api.types.is_numeric_dtype(df_encoded[col])

    def test_transform(self, sample_df):
        """Test transform on new data."""
        encoder = CategoricalEncoder(method="label")
        encoder.fit_transform(sample_df)

        # Transform new data
        new_df = sample_df.copy()
        df_transformed = encoder.transform(new_df)

        for col in encoder.categorical_cols:
            assert pd.api.types.is_numeric_dtype(df_transformed[col])

    def test_handle_unknown(self, sample_df):
        """Test handling unknown categories."""
        encoder = CategoricalEncoder(method="label", handle_unknown="ignore")
        encoder.fit_transform(sample_df)

        # Create data with unknown category
        new_df = sample_df.copy()
        cat_col = encoder.categorical_cols[0]
        new_df.loc[0, cat_col] = "UNKNOWN_CATEGORY"

        df_transformed = encoder.transform(new_df)
        assert df_transformed[cat_col].iloc[0] == -1  # Unknown mapped to -1

    def test_get_summary(self, sample_df):
        """Test getting encoding summary."""
        encoder = CategoricalEncoder(method="onehot")
        encoder.fit_transform(sample_df)

        summary = encoder.get_encoding_summary()
        assert summary["method"] == "onehot"
        assert len(summary["category_mappings"]) > 0


class TestDataSplitter:
    """Test data splitting."""

    def test_train_test_split(self, sample_df):
        """Test basic train-test split."""
        splitter = DataSplitter(test_size=0.2)
        X = sample_df.drop(columns=["cat_1"])
        y = sample_df["cat_1"]

        X_train, X_test, y_train, y_test = splitter.split(X, y)  # type: ignore[misc]

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_test) == int(len(X) * 0.2)

    def test_train_val_test_split(self, sample_df):
        """Test train-validation-test split."""
        splitter = DataSplitter(test_size=0.2, validation_size=0.2)
        X = sample_df.drop(columns=["cat_1"])
        y = sample_df["cat_1"]

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)  # type: ignore[misc]

        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(X_test) == int(len(X) * 0.2)

    def test_stratified_split(self, sample_df):
        """Test stratified splitting."""
        splitter = DataSplitter(test_size=0.2, stratify=True)
        X = sample_df.drop(columns=["cat_1"])
        y = sample_df["cat_1"]

        X_train, X_test, y_train, y_test = splitter.split(X, y)  # type: ignore[misc]

        # Check that class distributions are similar
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)

        for category in train_dist.index:
            if category in test_dist.index:
                assert abs(train_dist[category] - test_dist[category]) < 0.15

    def test_temporal_split(self, sample_df):
        """Test temporal splitting (no shuffle)."""
        splitter = DataSplitter(test_size=0.2)
        X = sample_df.drop(columns=["cat_1"])
        y = sample_df["cat_1"]

        X_train, X_test, y_train, y_test = splitter.temporal_split(X, y)  # type: ignore[misc]

        assert len(X_train) + len(X_test) == len(X)
        # Check that indices are sequential (no shuffle)
        assert X_train.index.max() < X_test.index.min()

    def test_split_with_target_col(self, sample_df):
        """Test split using target column name."""
        splitter = DataSplitter(test_size=0.2)

        X_train, X_test, y_train, y_test = splitter.split(sample_df, target_col="cat_1")  # type: ignore[misc]

        assert "cat_1" not in X_train.columns
        assert "cat_1" not in X_test.columns

    def test_get_split_summary(self, sample_df):
        """Test getting split summary."""
        splitter = DataSplitter(test_size=0.2)
        X = sample_df.drop(columns=["cat_1"])
        y = sample_df["cat_1"]

        split_data = splitter.split(X, y)
        summary = splitter.get_split_summary(*split_data)

        assert summary["num_splits"] == 2
        assert summary["train_size"] + summary["test_size"] == len(X)

    def test_invalid_sizes(self):
        """Test validation of invalid split sizes."""
        with pytest.raises(ValidationError):
            DataSplitter(test_size=1.5)

        with pytest.raises(ValidationError):
            DataSplitter(test_size=0.5, validation_size=0.6)


class TestPipelineBuilder:
    """Test preprocessing pipeline builder."""

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        pipeline = PipelineBuilder()
        pipeline.add_missing_handler(strategy="mean")
        pipeline.add_encoder(method="label")
        pipeline.add_scaler(method="standard")

        assert len(pipeline) == 3
        assert not pipeline.is_fitted

    def test_pipeline_fit_transform(self, sample_df_with_missing):
        """Test fitting and transforming with pipeline."""
        pipeline = PipelineBuilder()
        pipeline.add_missing_handler(strategy="mean")
        pipeline.add_encoder(method="label")
        pipeline.add_scaler(method="standard")

        df_transformed = pipeline.fit_transform(sample_df_with_missing)

        assert pipeline.is_fitted
        assert df_transformed.isnull().sum().sum() == 0
        assert len(pipeline.feature_names_in) > 0
        assert len(pipeline.feature_names_out) > 0

    def test_pipeline_transform(self, sample_df_with_missing):
        """Test transforming new data with fitted pipeline."""
        pipeline = PipelineBuilder()
        pipeline.add_missing_handler(strategy="mean")
        pipeline.add_encoder(method="label")
        pipeline.add_scaler(method="standard")

        pipeline.fit_transform(sample_df_with_missing)

        # Transform new data
        new_df = sample_df_with_missing.copy()
        df_transformed = pipeline.transform(new_df)

        assert df_transformed.isnull().sum().sum() == 0

    def test_pipeline_save_load(self, sample_df_with_missing, tmp_path):
        """Test saving and loading pipeline."""
        pipeline = PipelineBuilder()
        pipeline.add_missing_handler(strategy="mean")
        pipeline.add_scaler(method="standard")

        pipeline.fit_transform(sample_df_with_missing)

        # Save pipeline
        pipeline_path = tmp_path / "pipeline.pkl"
        pipeline.save(str(pipeline_path))

        assert pipeline_path.exists()
        assert (tmp_path / "pipeline.json").exists()

        # Load pipeline
        loaded_pipeline = PipelineBuilder.load(str(pipeline_path))

        assert loaded_pipeline.is_fitted
        assert len(loaded_pipeline) == len(pipeline)

    def test_pipeline_summary(self, sample_df_with_missing):
        """Test getting pipeline summary."""
        pipeline = PipelineBuilder()
        pipeline.add_missing_handler(strategy="mean")
        pipeline.add_scaler(method="standard")

        pipeline.fit_transform(sample_df_with_missing)

        summary = pipeline.get_pipeline_summary()

        assert summary["num_steps"] == 2
        assert summary["is_fitted"] is True
        assert len(summary["steps"]) == 2

    def test_empty_pipeline(self, sample_df):
        """Test pipeline with no steps."""
        pipeline = PipelineBuilder()
        df_transformed = pipeline.fit_transform(sample_df)

        pd.testing.assert_frame_equal(df_transformed, sample_df)

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        pipeline = PipelineBuilder()
        pipeline.add_missing_handler()

        repr_str = repr(pipeline)
        assert "PipelineBuilder" in repr_str
        assert "missing_handler" in repr_str
        assert "not fitted" in repr_str
