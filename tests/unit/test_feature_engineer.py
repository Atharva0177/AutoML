"""
Tests for Feature Engineering Module
"""

import numpy as np
import pandas as pd
import pytest

from automl.preprocessing.feature_engineering import BinningStrategy, FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100) * 2 + 5,
            "feature3": np.random.randint(1, 10, 100),
            "category": ["A", "B", "C"] * 33 + ["A"],
        }
    )


@pytest.fixture
def simple_data():
    """Create simple data for testing."""
    return pd.DataFrame(
        {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "z": [1, 1, 2, 2, 3]}
    )


class TestFeatureEngineerInitialization:
    """Test FeatureEngineer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        fe = FeatureEngineer()
        assert fe.polynomial_degree is None
        assert fe.interaction_features == []
        assert fe.binning_config == {}
        assert fe.transformations == {}
        assert not fe.include_bias
        assert not fe.interaction_only

    def test_init_with_params(self):
        """Test initialization with parameters."""
        fe = FeatureEngineer(
            polynomial_degree=2,
            interaction_features=[("x", "y")],
            binning_config={"x": {"n_bins": 3}},
            transformations={"x": "log"},
            include_bias=True,
            interaction_only=True,
        )
        assert fe.polynomial_degree == 2
        assert fe.interaction_features == [("x", "y")]
        assert "x" in fe.binning_config
        assert fe.transformations["x"] == "log"
        assert fe.include_bias
        assert fe.interaction_only


class TestMathematicalTransformations:
    """Test mathematical transformations."""

    def test_log_transformation(self, simple_data):
        """Test log transformation."""
        fe = FeatureEngineer(transformations={"x": "log"})
        result = fe.fit_transform(simple_data)

        assert "x_log" in result.columns
        # log1p(abs(x)) for x=[1,2,3,4,5]
        expected = np.log1p([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(np.array(result["x_log"].values), expected)

    def test_sqrt_transformation(self, simple_data):
        """Test sqrt transformation."""
        fe = FeatureEngineer(transformations={"x": "sqrt"})
        result = fe.fit_transform(simple_data)

        assert "x_sqrt" in result.columns
        expected = np.sqrt([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(
            np.array(result["x_sqrt"].values), expected
        )

    def test_square_transformation(self, simple_data):
        """Test square transformation."""
        fe = FeatureEngineer(transformations={"x": "square"})
        result = fe.fit_transform(simple_data)

        assert "x_square" in result.columns
        expected = [1, 4, 9, 16, 25]
        np.testing.assert_array_equal(result["x_square"].values, expected)

    def test_inverse_transformation(self, simple_data):
        """Test inverse transformation."""
        fe = FeatureEngineer(transformations={"x": "inverse"})
        result = fe.fit_transform(simple_data)

        assert "x_inverse" in result.columns
        expected = [1.0, 0.5, 1 / 3, 0.25, 0.2]
        np.testing.assert_array_almost_equal(
            np.array(result["x_inverse"].values), expected
        )

    def test_multiple_transformations(self, simple_data):
        """Test multiple transformations."""
        fe = FeatureEngineer(transformations={"x": "log", "y": "sqrt"})
        result = fe.fit_transform(simple_data)

        assert "x_log" in result.columns
        assert "y_sqrt" in result.columns
        assert result.shape[1] == simple_data.shape[1] + 2


class TestBinning:
    """Test binning/discretization."""

    def test_quantile_binning(self, sample_data):
        """Test quantile binning."""
        fe = FeatureEngineer(
            binning_config={"feature1": {"n_bins": 3, "strategy": "quantile"}}
        )
        result = fe.fit_transform(sample_data)

        assert "feature1_binned" in result.columns
        assert result["feature1_binned"].nunique() <= 3
        assert result["feature1_binned"].dtype == int

    def test_uniform_binning(self, sample_data):
        """Test uniform (equal width) binning."""
        fe = FeatureEngineer(
            binning_config={"feature2": {"n_bins": 5, "strategy": "uniform"}}
        )
        result = fe.fit_transform(sample_data)

        assert "feature2_binned" in result.columns
        assert result["feature2_binned"].nunique() <= 5

    def test_kmeans_binning(self, sample_data):
        """Test k-means binning."""
        fe = FeatureEngineer(
            binning_config={"feature3": {"n_bins": 3, "strategy": "kmeans"}}
        )
        result = fe.fit_transform(sample_data)

        assert "feature3_binned" in result.columns
        assert result["feature3_binned"].nunique() <= 3

    def test_binning_transform_consistency(self, sample_data):
        """Test that transform produces consistent results."""
        fe = FeatureEngineer(
            binning_config={"feature1": {"n_bins": 4, "strategy": "quantile"}}
        )

        # Fit on first half
        train_data = sample_data.iloc[:50]
        fe.fit_transform(train_data)

        # Transform second half
        test_data = sample_data.iloc[50:]
        result = fe.transform(test_data)

        assert "feature1_binned" in result.columns


class TestInteractionFeatures:
    """Test interaction feature creation."""

    def test_single_interaction(self, simple_data):
        """Test creating single interaction feature."""
        fe = FeatureEngineer(interaction_features=[("x", "y")])
        result = fe.fit_transform(simple_data)

        assert "x_x_y" in result.columns
        expected = simple_data["x"] * simple_data["y"]
        np.testing.assert_array_equal(result["x_x_y"].values, expected.values)

    def test_multiple_interactions(self, simple_data):
        """Test creating multiple interaction features."""
        fe = FeatureEngineer(interaction_features=[("x", "y"), ("x", "z"), ("y", "z")])
        result = fe.fit_transform(simple_data)

        assert "x_x_y" in result.columns
        assert "x_x_z" in result.columns
        assert "y_x_z" in result.columns
        assert result.shape[1] == simple_data.shape[1] + 3

    def test_interaction_transform(self, simple_data):
        """Test interaction on transform."""
        fe = FeatureEngineer(interaction_features=[("x", "y")])
        fe.fit_transform(simple_data.iloc[:3])

        result = fe.transform(simple_data.iloc[3:])
        assert "x_x_y" in result.columns


class TestPolynomialFeatures:
    """Test polynomial feature creation."""

    def test_polynomial_degree_2(self, simple_data):
        """Test polynomial features with degree 2."""
        fe = FeatureEngineer(polynomial_degree=2)
        result = fe.fit_transform(simple_data, numerical_cols=["x", "y"])

        # Should have x, y, x^2, xy, y^2 (minus original x, y)
        assert "x^2" in result.columns
        assert "x y" in result.columns
        assert "y^2" in result.columns

    def test_polynomial_degree_3(self, simple_data):
        """Test polynomial features with degree 3."""
        fe = FeatureEngineer(polynomial_degree=3)
        result = fe.fit_transform(simple_data, numerical_cols=["x"])

        assert "x^2" in result.columns
        assert "x^3" in result.columns

    def test_polynomial_with_bias(self, simple_data):
        """Test polynomial features with bias."""
        fe = FeatureEngineer(polynomial_degree=2, include_bias=True)
        result = fe.fit_transform(simple_data, numerical_cols=["x"])

        # Should include constant term
        assert "1" in result.columns

    def test_polynomial_interaction_only(self, simple_data):
        """Test polynomial with interaction_only flag."""
        fe = FeatureEngineer(polynomial_degree=2, interaction_only=True)
        result = fe.fit_transform(simple_data, numerical_cols=["x", "y"])

        # Should have x*y but not x^2 or y^2
        assert "x y" in result.columns
        assert "x^2" not in result.columns
        assert "y^2" not in result.columns

    def test_polynomial_transform(self, simple_data):
        """Test polynomial features on transform."""
        fe = FeatureEngineer(polynomial_degree=2)
        fe.fit_transform(simple_data.iloc[:3], numerical_cols=["x", "y"])

        result = fe.transform(simple_data.iloc[3:])
        assert "x^2" in result.columns
        assert "x y" in result.columns


class TestCombinedFeatures:
    """Test combining multiple feature engineering techniques."""

    def test_all_features_combined(self, sample_data):
        """Test using all feature engineering techniques together."""
        fe = FeatureEngineer(
            polynomial_degree=2,
            interaction_features=[("feature1", "feature2")],
            binning_config={"feature3": {"n_bins": 3}},
            transformations={"feature1": "log"},
        )

        result = fe.fit_transform(sample_data, numerical_cols=["feature1", "feature2"])

        # Check all feature types are present
        assert "feature1_log" in result.columns
        assert "feature3_binned" in result.columns
        assert "feature1_x_feature2" in result.columns
        assert "feature1^2" in result.columns

    def test_combined_transform(self, sample_data):
        """Test transform with combined features."""
        fe = FeatureEngineer(
            polynomial_degree=2,
            interaction_features=[("feature1", "feature2")],
            transformations={"feature1": "sqrt"},
        )

        # Fit on train data
        train_data = sample_data.iloc[:70]
        fe.fit_transform(train_data, numerical_cols=["feature1", "feature2"])

        # Transform test data
        test_data = sample_data.iloc[70:]
        result = fe.transform(test_data)

        assert "feature1_sqrt" in result.columns
        assert "feature1_x_feature2" in result.columns
        assert "feature1^2" in result.columns


class TestFeatureSummary:
    """Test feature summary functionality."""

    def test_feature_summary(self, simple_data):
        """Test get_feature_summary method."""
        fe = FeatureEngineer(
            polynomial_degree=2,
            interaction_features=[("x", "y")],
            binning_config={"z": {"n_bins": 3}},
            transformations={"x": "log"},
        )

        fe.fit_transform(simple_data, numerical_cols=["x", "y"])
        summary = fe.get_feature_summary()

        assert summary["transformations"] == 1
        assert summary["binned_features"] == 1
        assert summary["interactions"] == 1
        assert summary["polynomial_features"] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_transform_before_fit(self, simple_data):
        """Test that transform before fit raises error."""
        fe = FeatureEngineer(polynomial_degree=2)

        with pytest.raises(ValueError, match="must be fitted"):
            fe.transform(simple_data)

    def test_missing_column_in_transformation(self, simple_data):
        """Test handling of missing column in transformations."""
        fe = FeatureEngineer(transformations={"nonexistent": "log"})
        result = fe.fit_transform(simple_data)

        # Should not create the transformation
        assert "nonexistent_log" not in result.columns

    def test_missing_column_in_interaction(self, simple_data):
        """Test handling of missing column in interactions."""
        fe = FeatureEngineer(interaction_features=[("x", "nonexistent")])
        result = fe.fit_transform(simple_data)

        # Should not create the interaction
        assert "x_x_nonexistent" not in result.columns

    def test_zero_in_inverse_transformation(self):
        """Test inverse transformation with zero values."""
        data = pd.DataFrame({"x": [0, 1, 2, 0, 3]})
        fe = FeatureEngineer(transformations={"x": "inverse"})
        result = fe.fit_transform(data)

        # Zeros should remain zero
        assert result.loc[result["x"] == 0, "x_inverse"].iloc[0] == 0

    def test_negative_values_in_sqrt(self):
        """Test sqrt transformation with negative values."""
        data = pd.DataFrame({"x": [-4, -1, 0, 1, 4]})
        fe = FeatureEngineer(transformations={"x": "sqrt"})
        result = fe.fit_transform(data)

        # Should handle negative values (sign * sqrt(abs))
        assert result["x_sqrt"].iloc[0] < 0
        assert result["x_sqrt"].iloc[4] > 0
