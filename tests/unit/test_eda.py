"""Tests for EDA module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.eda import CorrelationAnalyzer, ProblemDetector, StatisticalProfiler
from automl.eda.problem_detector import ProblemType


class TestStatisticalProfiler:
    """Tests for StatisticalProfiler."""

    def test_generate_profile(self, sample_df):
        """Test profile generation."""
        profiler = StatisticalProfiler()
        profile = profiler.generate_profile(sample_df)

        assert "overview" in profile
        assert "numerical_stats" in profile
        assert "categorical_stats" in profile
        assert "missing_analysis" in profile
        assert "summary" in profile

        # Check overview
        assert profile["overview"]["n_rows"] == len(sample_df)
        assert profile["overview"]["n_columns"] == len(sample_df.columns)

    def test_numerical_stats(self, sample_df):
        """Test numerical statistics."""
        profiler = StatisticalProfiler()
        profile = profiler.generate_profile(sample_df)

        num_stats = profile["numerical_stats"]

        # Check numeric columns are analyzed
        assert "numeric_1" in num_stats
        assert num_stats["numeric_1"]["count"] > 0
        assert "mean" in num_stats["numeric_1"]
        assert "std" in num_stats["numeric_1"]
        assert "min" in num_stats["numeric_1"]
        assert "max" in num_stats["numeric_1"]

    def test_categorical_stats(self, sample_df):
        """Test categorical statistics."""
        profiler = StatisticalProfiler()
        profile = profiler.generate_profile(sample_df)

        cat_stats = profile["categorical_stats"]

        # Check categorical columns are analyzed
        assert "cat_1" in cat_stats
        assert cat_stats["cat_1"]["unique"] > 0
        assert "most_frequent" in cat_stats["cat_1"]

    def test_get_column_stats(self, sample_df):
        """Test getting stats for specific column."""
        profiler = StatisticalProfiler()
        profiler.generate_profile(sample_df)

        num_stats = profiler.get_column_stats("numeric_1")
        assert num_stats is not None
        assert "mean" in num_stats

        cat_stats = profiler.get_column_stats("cat_1")
        assert cat_stats is not None
        assert "unique" in cat_stats


class TestProblemDetector:
    """Tests for ProblemDetector."""

    def test_detect_binary_classification(self):
        """Test binary classification detection."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        detector = ProblemDetector()
        problem_info = detector.detect_problem_type(df, "target")

        assert problem_info["problem_type"] == ProblemType.BINARY_CLASSIFICATION.value
        assert problem_info["n_classes"] == 2
        assert "imbalance_ratio" in problem_info

    def test_detect_multiclass_classification(self):
        """Test multiclass classification detection."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.choice(["A", "B", "C", "D"], 100),
            }
        )

        detector = ProblemDetector()
        problem_info = detector.detect_problem_type(df, "target")

        assert (
            problem_info["problem_type"] == ProblemType.MULTICLASS_CLASSIFICATION.value
        )
        assert problem_info["n_classes"] == 4

    def test_detect_regression(self):
        """Test regression detection."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.uniform(0, 100, 100),
            }
        )

        detector = ProblemDetector()
        problem_info = detector.detect_problem_type(df, "target")

        assert problem_info["problem_type"] == ProblemType.REGRESSION.value
        assert "mean" in problem_info
        assert "std" in problem_info

    def test_infer_target_column(self):
        """Test target column inference."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "label": np.random.choice([0, 1], 100),
            }
        )

        detector = ProblemDetector()
        problem_info = detector.detect_problem_type(df)  # No target specified

        assert problem_info["target_column"] == "label"

    def test_get_suggested_metrics(self):
        """Test suggested metrics."""
        df = pd.DataFrame(
            {
                "feature": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        detector = ProblemDetector()
        detector.detect_problem_type(df, "target")

        metrics = detector.get_suggested_metrics()
        assert len(metrics) > 0
        assert "accuracy" in metrics

    def test_get_suggested_models(self):
        """Test suggested models."""
        df = pd.DataFrame(
            {
                "feature": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        detector = ProblemDetector()
        detector.detect_problem_type(df, "target")

        models = detector.get_suggested_models()
        assert len(models) > 0
        assert any("Classifier" in m for m in models)


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzer."""

    def test_analyze_correlations(self):
        """Test correlation analysis."""
        # Create data with known correlations
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = x1 + np.random.randn(100) * 0.1  # Highly correlated with x1
        x3 = np.random.randn(100)  # Independent

        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "target": x1 + x3,
            }
        )

        analyzer = CorrelationAnalyzer()
        analysis = analyzer.analyze_correlations(df, target_column="target")

        assert "correlation_matrix" in analysis
        assert "high_correlations" in analysis
        assert "target_correlations" in analysis

    def test_high_correlations(self):
        """Test high correlation detection."""
        # Create perfectly correlated features
        df = pd.DataFrame(
            {
                "x1": np.arange(100),
                "x2": np.arange(100) * 2,  # Perfect correlation
                "x3": np.random.randn(100),
            }
        )

        analyzer = CorrelationAnalyzer()
        analysis = analyzer.analyze_correlations(df, threshold=0.7)

        # Should find high correlation between x1 and x2
        assert len(analysis["high_correlations"]) > 0

    def test_multicollinearity_detection(self):
        """Test multicollinearity detection."""
        # Create multiple correlated features
        x1 = np.random.randn(100)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x1 + np.random.randn(100) * 0.1,
                "x3": x1 + np.random.randn(100) * 0.1,
                "x4": np.random.randn(100),
            }
        )

        analyzer = CorrelationAnalyzer()
        analysis = analyzer.analyze_correlations(df, threshold=0.7)

        assert "multicollinearity" in analysis
        multicollinearity = analysis["multicollinearity"]
        assert "detected" in multicollinearity

    def test_target_correlations(self):
        """Test target correlation analysis."""
        df = pd.DataFrame(
            {
                "x1": np.arange(100),
                "x2": np.arange(100) * 2,
                "x3": np.random.randn(100),
                "target": np.arange(100) + np.random.randn(100),
            }
        )

        analyzer = CorrelationAnalyzer()
        analysis = analyzer.analyze_correlations(df, target_column="target")

        target_corrs = analysis["target_correlations"]
        assert "all_correlations" in target_corrs
        assert "top_positive" in target_corrs
        assert "top_negative" in target_corrs

    def test_get_redundant_features(self):
        """Test redundant feature detection."""
        df = pd.DataFrame(
            {
                "x1": np.arange(100),
                "x2": np.arange(100),  # Duplicate of x1
                "x3": np.random.randn(100),
            }
        )

        analyzer = CorrelationAnalyzer()
        analyzer.analyze_correlations(df)

        redundant = analyzer.get_redundant_features(threshold=0.95)
        assert len(redundant) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
