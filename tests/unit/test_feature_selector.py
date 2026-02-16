"""
Tests for Feature Selection Module
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.preprocessing.feature_selection import FeatureSelector, SelectionMethod


@pytest.fixture
def classification_data():
    """Create classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def regression_data():
    """Create regression dataset."""
    result = make_regression(
        n_samples=200, n_features=15, n_informative=8, random_state=42
    )
    # Handle both 2-tuple and 3-tuple returns
    X = result[0]
    y = result[1]
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def simple_data():
    """Create simple dataset for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "f1": np.random.randn(100),
            "f2": np.random.randn(100) * 0.1,  # Low variance
            "f3": np.random.randn(100),
            "f4": np.random.randn(100) * 0.1,
            "f5": np.random.randn(100),
        }
    )
    y = pd.Series(X["f1"] * 2 + X["f3"] * 3 + np.random.randn(100) * 0.1, name="target")
    return X, y


class TestFeatureSelectorInitialization:
    """Test FeatureSelector initialization."""

    def test_init_default(self):
        """Test default initialization."""
        fs = FeatureSelector(method="correlation")
        assert fs.method == "correlation"
        assert fs.k_features == 10
        assert fs.task_type == "classification"
        assert fs.correlation_threshold == 0.5

    def test_init_with_params(self):
        """Test initialization with parameters."""
        fs = FeatureSelector(
            method="rfe",
            k_features=5,
            task_type="regression",
            correlation_threshold=0.7,
        )
        assert fs.method == "rfe"
        assert fs.k_features == 5
        assert fs.task_type == "regression"
        assert fs.correlation_threshold == 0.7


class TestCorrelationSelection:
    """Test correlation-based feature selection."""

    def test_correlation_basic(self, simple_data):
        """Test basic correlation selection."""
        X, y = simple_data
        fs = FeatureSelector(method="correlation", correlation_threshold=0.5)
        result = fs.fit_transform(X, y)

        assert fs._is_fitted
        assert len(fs.selected_features_) > 0
        assert len(fs.selected_features_) <= X.shape[1]
        assert all(col in X.columns for col in fs.selected_features_)

    def test_correlation_threshold(self, simple_data):
        """Test correlation threshold."""
        X, y = simple_data
        fs = FeatureSelector(method="correlation", correlation_threshold=0.8)
        result = fs.fit_transform(X, y)

        # High correlation features should be selected
        assert "f1" in fs.selected_features_ or "f3" in fs.selected_features_

    def test_correlation_scores(self, simple_data):
        """Test that correlation scores are computed."""
        X, y = simple_data
        fs = FeatureSelector(method="correlation")
        fs.fit(X, y)

        assert len(fs.feature_scores_) == X.shape[1]
        assert all(0 <= score <= 1 for score in fs.feature_scores_.values())


class TestChi2Selection:
    """Test chi-square based feature selection."""

    def test_chi2_basic(self, classification_data):
        """Test basic chi-square selection."""
        X, y = classification_data
        # Make data non-negative for chi2
        X_positive = X - X.min() + 1

        fs = FeatureSelector(method="chi2", k_features=10)
        result = fs.fit_transform(X_positive, y)

        assert result.shape[1] == 10
        assert len(fs.selected_features_) == 10

    def test_chi2_percentage(self, classification_data):
        """Test chi2 with percentage of features."""
        X, y = classification_data
        X_positive = X - X.min() + 1

        fs = FeatureSelector(method="chi2", k_features=0.5)
        result = fs.fit_transform(X_positive, y)

        expected_features = int(X.shape[1] * 0.5)
        assert result.shape[1] == expected_features


class TestMutualInfoSelection:
    """Test mutual information based feature selection."""

    def test_mutual_info_classification(self, classification_data):
        """Test mutual information for classification."""
        X, y = classification_data
        fs = FeatureSelector(
            method="mutual_info", k_features=8, task_type="classification"
        )
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 8
        assert len(fs.selected_features_) == 8

    def test_mutual_info_regression(self, regression_data):
        """Test mutual information for regression."""
        X, y = regression_data
        fs = FeatureSelector(method="mutual_info", k_features=6, task_type="regression")
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 6
        assert len(fs.selected_features_) == 6


class TestAnovaFSelection:
    """Test ANOVA F-test based feature selection."""

    def test_anova_f_classification(self, classification_data):
        """Test ANOVA F for classification."""
        X, y = classification_data
        fs = FeatureSelector(
            method="anova_f", k_features=12, task_type="classification"
        )
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 12
        assert len(fs.selected_features_) == 12

    def test_anova_f_regression(self, regression_data):
        """Test ANOVA F for regression."""
        X, y = regression_data
        fs = FeatureSelector(method="anova_f", k_features=7, task_type="regression")
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 7


class TestRFESelection:
    """Test recursive feature elimination."""

    def test_rfe_classification(self, classification_data):
        """Test RFE for classification."""
        X, y = classification_data
        fs = FeatureSelector(method="rfe", k_features=10, task_type="classification")
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 10
        assert len(fs.selected_features_) == 10

    def test_rfe_regression(self, regression_data):
        """Test RFE for regression."""
        X, y = regression_data
        fs = FeatureSelector(method="rfe", k_features=8, task_type="regression")
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 8

    def test_rfe_with_custom_estimator(self, classification_data):
        """Test RFE with custom estimator."""
        from sklearn.tree import DecisionTreeClassifier

        X, y = classification_data
        estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
        fs = FeatureSelector(
            method="rfe", k_features=10, task_type="classification", estimator=estimator
        )
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 10


class TestL1Selection:
    """Test L1 regularization based feature selection."""

    def test_l1_classification(self, classification_data):
        """Test L1 for classification."""
        X, y = classification_data
        fs = FeatureSelector(method="l1", k_features=10, task_type="classification")
        result = fs.fit_transform(X, y)

        assert result.shape[1] > 0
        assert len(fs.selected_features_) > 0

    def test_l1_regression(self, regression_data):
        """Test L1 for regression."""
        X, y = regression_data
        fs = FeatureSelector(method="l1", k_features=8, task_type="regression")
        result = fs.fit_transform(X, y)

        assert result.shape[1] > 0
        assert len(fs.selected_features_) > 0


class TestTreeImportanceSelection:
    """Test tree-based importance feature selection."""

    def test_tree_importance_classification(self, classification_data):
        """Test tree importance for classification."""
        X, y = classification_data
        fs = FeatureSelector(
            method="tree_importance", k_features=10, task_type="classification"
        )
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 10
        assert len(fs.selected_features_) == 10

    def test_tree_importance_regression(self, regression_data):
        """Test tree importance for regression."""
        X, y = regression_data
        fs = FeatureSelector(
            method="tree_importance", k_features=7, task_type="regression"
        )
        result = fs.fit_transform(X, y)

        assert result.shape[1] == 7

    def test_tree_importance_scores(self, classification_data):
        """Test that importance scores are computed."""
        X, y = classification_data
        fs = FeatureSelector(
            method="tree_importance", k_features=10, task_type="classification"
        )
        fs.fit(X, y)

        assert len(fs.feature_scores_) == X.shape[1]
        assert all(score >= 0 for score in fs.feature_scores_.values())


class TestFeatureRanking:
    """Test feature ranking functionality."""

    def test_get_feature_ranking(self, simple_data):
        """Test feature ranking."""
        X, y = simple_data
        fs = FeatureSelector(method="correlation")
        fs.fit(X, y)

        ranking = fs.get_feature_ranking()

        assert isinstance(ranking, pd.DataFrame)
        assert len(ranking) == X.shape[1]
        assert "feature" in ranking.columns
        assert "score" in ranking.columns
        assert "selected" in ranking.columns

        # Check that ranking is sorted by score
        assert ranking["score"].is_monotonic_decreasing

    def test_ranking_before_fit_raises_error(self, simple_data):
        """Test that getting ranking before fit raises error."""
        X, y = simple_data
        fs = FeatureSelector(method="correlation")

        with pytest.raises(ValueError, match="must be fitted"):
            fs.get_feature_ranking()


class TestSelectionSummary:
    """Test selection summary functionality."""

    def test_get_selection_summary(self, classification_data):
        """Test selection summary."""
        X, y = classification_data
        fs = FeatureSelector(method="anova_f", k_features=10)
        fs.fit(X, y)

        summary = fs.get_selection_summary()

        assert summary["method"] == "anova_f"
        assert summary["total_features"] == X.shape[1]
        assert summary["selected_features"] == 10
        assert summary["selection_rate"] == 10 / X.shape[1]
        selected_names = summary["selected_feature_names"]
        assert isinstance(selected_names, list) and len(selected_names) == 10


class TestTransformConsistency:
    """Test transform consistency."""

    def test_fit_transform_equals_fit_then_transform(self, classification_data):
        """Test that fit_transform gives same result as fit then transform."""
        X, y = classification_data

        # Method 1: fit_transform
        fs1 = FeatureSelector(method="anova_f", k_features=10)
        result1 = fs1.fit_transform(X, y)

        # Method 2: fit then transform
        fs2 = FeatureSelector(method="anova_f", k_features=10)
        fs2.fit(X, y)
        result2 = fs2.transform(X)

        pd.testing.assert_frame_equal(result1, result2)

    def test_transform_on_new_data(self, classification_data):
        """Test transform on new data."""
        X, y = classification_data

        # Split data
        X_train, y_train = X.iloc[:150], y.iloc[:150]
        X_test = X.iloc[150:]

        # Fit on train
        fs = FeatureSelector(method="anova_f", k_features=10)
        fs.fit(X_train, y_train)

        # Transform test
        result = fs.transform(X_test)

        assert result.shape[1] == 10
        assert list(result.columns) == fs.selected_features_

    def test_transform_before_fit_raises_error(self, classification_data):
        """Test that transform before fit raises error."""
        X, y = classification_data
        fs = FeatureSelector(method="anova_f", k_features=10)

        with pytest.raises(ValueError, match="must be fitted"):
            fs.transform(X)


class TestEdgeCases:
    """Test edge cases."""

    def test_k_features_greater_than_n_features(self, simple_data):
        """Test when k_features > n_features."""
        X, y = simple_data
        fs = FeatureSelector(method="anova_f", k_features=100, task_type="regression")
        result = fs.fit_transform(X, y)

        # Should select all features
        assert result.shape[1] == X.shape[1]

    def test_k_features_as_percentage(self, classification_data):
        """Test k_features as percentage."""
        X, y = classification_data
        fs = FeatureSelector(method="anova_f", k_features=0.3)
        result = fs.fit_transform(X, y)

        expected = int(X.shape[1] * 0.3)
        assert result.shape[1] == expected

    def test_invalid_method_raises_error(self, simple_data):
        """Test that invalid method raises error."""
        X, y = simple_data
        fs = FeatureSelector(method="invalid_method")

        with pytest.raises(ValueError, match="Unknown selection method"):
            fs.fit(X, y)

    def test_single_feature_selection(self, simple_data):
        """Test selection of single feature."""
        X, y = simple_data
        fs = FeatureSelector(method="correlation", k_features=1)
        result = fs.fit_transform(X, y)

        assert result.shape[1] >= 1  # At least one feature selected
