"""
Unit tests for outlier detection.
"""

import pytest
import pandas as pd
import numpy as np
from automl.preprocessing.cleaners.outlier_detector import (
    OutlierDetector,
    OutlierStrategy,
    OutlierAction
)
from automl.utils.exceptions import ValidationError


class TestOutlierDetector:
    """Test cases for OutlierDetector."""
    
    @pytest.fixture
    def sample_data_with_outliers(self):
        """Create sample data with intentional outliers."""
        np.random.seed(42)
        # Normal data
        normal_data = np.random.randn(100) * 10 + 50
        
        # Add outliers
        data = np.concatenate([normal_data, [0, 0, 100, 100, 150]])  # 5 outliers
        
        df = pd.DataFrame({
            'value1': data,
            'value2': np.random.randn(105) * 5 + 25,
            'category': np.random.choice(['A', 'B', 'C'], 105)
        })
        
        return df
    
    @pytest.fixture
    def multivariate_data(self):
        """Create multivariate data for Isolation Forest."""
        np.random.seed(42)
        n_samples = 100
        
        # Normal cluster
        X1 = np.random.randn(n_samples, 2) * 2
        
        # Add some outliers
        outliers = np.array([[10, 10], [-10, -10], [10, -10]])
        
        data = np.vstack([X1, outliers])
        
        df = pd.DataFrame(data, columns=['feature1', 'feature2'])
        return df
    
    def test_init_iqr(self):
        """Test IQR detector initialization."""
        detector = OutlierDetector(strategy='iqr', action='cap', threshold=1.5)
        assert detector.strategy == 'iqr'
        assert detector.action == 'cap'
        assert detector.threshold == 1.5
        assert not detector.is_fitted
    
    def test_init_isolation_forest(self):
        """Test Isolation Forest initialization."""
        detector = OutlierDetector(
            strategy='isolation_forest',
            action='remove',
            contamination=0.1
        )
        assert detector.strategy == 'isolation_forest'
        assert detector.contamination == 0.1
        assert not detector.is_fitted
    
    def test_init_zscore(self):
        """Test Z-score detector initialization."""
        detector = OutlierDetector(
            strategy='zscore',
            action='flag',
            zscore_threshold=3.0
        )
        assert detector.strategy == 'zscore'
        assert detector.threshold == 3.0
    
    def test_iqr_cap_outliers(self, sample_data_with_outliers):
        """Test IQR method with cap action."""
        detector = OutlierDetector(strategy='iqr', action='cap', threshold=1.5)
        
        result = detector.fit_transform(sample_data_with_outliers)
        
        # Check shape preserved
        assert result.shape == sample_data_with_outliers.shape
        
        # Check detector is fitted
        assert detector.is_fitted
        assert len(detector.fitted_columns) > 0
        assert len(detector.bounds) > 0
        
        # Check outliers were capped
        for col in detector.fitted_columns:
            if col in detector.bounds:
                lower, upper = detector.bounds[col]
                assert result[col].min() >= lower or np.isclose(result[col].min(), lower)
                assert result[col].max() <= upper or np.isclose(result[col].max(), upper)
    
    def test_iqr_remove_outliers(self, sample_data_with_outliers):
        """Test IQR method with remove action."""
        detector = OutlierDetector(strategy='iqr', action='remove', threshold=1.5)
        
        rows_before = len(sample_data_with_outliers)
        result = detector.fit_transform(sample_data_with_outliers)
        rows_after = len(result)
        
        # Should have fewer rows
        assert rows_after < rows_before
        
        # Check no outliers remain
        for col in detector.fitted_columns:
            if col in detector.bounds:
                lower, upper = detector.bounds[col]
                assert (result[col] >= lower).all()
                assert (result[col] <= upper).all()
    
    def test_iqr_flag_outliers(self, sample_data_with_outliers):
        """Test IQR method with flag action."""
        detector = OutlierDetector(strategy='iqr', action='flag', threshold=1.5)
        
        result = detector.fit_transform(sample_data_with_outliers)
        
        # Check flag column added
        assert 'is_outlier' in result.columns
        
        # Check flag is binary
        assert result['is_outlier'].dtype == np.int64 or result['is_outlier'].dtype == np.int32
        assert set(result['is_outlier'].unique()).issubset({0, 1})
        
        # Should have some outliers flagged
        assert result['is_outlier'].sum() > 0
    
    def test_zscore_detection(self, sample_data_with_outliers):
        """Test Z-score method."""
        detector = OutlierDetector(
            strategy='zscore',
            action='cap',
            zscore_threshold=3.0
        )
        
        result = detector.fit_transform(sample_data_with_outliers)
        
        assert detector.is_fitted
        assert len(detector.bounds) > 0
        
        # Check outliers were handled
        for col in detector.fitted_columns:
            if col in detector.bounds:
                lower, upper = detector.bounds[col]
                assert result[col].min() >= lower or np.isclose(result[col].min(), lower)
                assert result[col].max() <= upper or np.isclose(result[col].max(), upper)
    
    def test_isolation_forest_detection(self, multivariate_data):
        """Test Isolation Forest method."""
        detector = OutlierDetector(
            strategy='isolation_forest',
            action='flag',
            contamination=0.05
        )
        
        result = detector.fit_transform(multivariate_data)
        
        # Check detector fitted
        assert detector.is_fitted
        assert detector.detector is not None
        
        # Check flag column added
        assert 'is_outlier' in result.columns
        
        # Should detect some outliers
        outlier_count = result['is_outlier'].sum()
        expected_outliers = int(len(multivariate_data) * 0.05)
        
        # Allow some tolerance
        assert outlier_count >= expected_outliers - 2
        assert outlier_count <= expected_outliers + 2
    
    def test_isolation_forest_remove(self, multivariate_data):
        """Test Isolation Forest with remove action."""
        detector = OutlierDetector(
            strategy='isolation_forest',
            action='remove',
            contamination=0.05
        )
        
        rows_before = len(multivariate_data)
        result = detector.fit_transform(multivariate_data)
        rows_after = len(result)
        
        # Should have removed some rows
        assert rows_after < rows_before
    
    def test_transform_consistency(self, sample_data_with_outliers):
        """Test that transform produces consistent results."""
        detector = OutlierDetector(strategy='iqr', action='cap', threshold=1.5)
        
        # Fit on data
        detector.fit_transform(sample_data_with_outliers)
        
        # Create new data with outliers
        new_data = sample_data_with_outliers.copy()
        new_data.loc[0, 'value1'] = 1000  # Add extreme outlier
        
        # Transform should work
        result = detector.transform(new_data)
        
        # Outlier should be capped
        lower, upper = detector.bounds['value1']
        value = result.loc[0, 'value1']
        assert isinstance(value, (int, float, np.number)) and value <= upper  # type: ignore[operator]
    
    def test_transform_before_fit_raises_error(self, sample_data_with_outliers):
        """Test that transform before fit raises error."""
        detector = OutlierDetector(strategy='iqr')
        
        with pytest.raises(ValidationError, match="must be fitted before transform"):
            detector.transform(sample_data_with_outliers)
    
    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises error."""
        detector = OutlierDetector(strategy='iqr')  # type: ignore
        detector.strategy = 'invalid'
        
        data = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
        
        with pytest.raises(ValidationError, match="Unknown strategy"):
            detector.fit_transform(data)
    
    def test_invalid_action_raises_error(self):
        """Test that invalid action raises error."""
        detector = OutlierDetector(strategy='iqr', action='cap')  # type: ignore
        detector.action = 'invalid'
        
        data = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
        
        with pytest.raises(ValidationError, match="Unknown action"):
            detector.fit_transform(data)
    
    def test_no_numerical_columns(self):
        """Test with no numerical columns."""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        detector = OutlierDetector(strategy='iqr', action='cap')
        result = detector.fit_transform(data)
        
        # Should return data unchanged
        pd.testing.assert_frame_equal(result, data)
        assert detector.is_fitted
    
    def test_outlier_summary(self, sample_data_with_outliers):
        """Test outlier summary."""
        detector = OutlierDetector(strategy='iqr', action='cap', threshold=1.5)
        detector.fit_transform(sample_data_with_outliers)
        
        summary = detector.get_outlier_summary()
        
        assert summary['strategy'] == 'iqr'
        assert summary['action'] == 'cap'
        assert summary['threshold'] == 1.5
        assert summary['is_fitted'] is True
        assert len(summary['fitted_columns']) > 0
        assert len(summary['outlier_counts']) > 0
        assert summary['bounds'] is not None
    
    def test_outlier_report(self, sample_data_with_outliers):
        """Test outlier report generation."""
        detector = OutlierDetector(strategy='iqr', action='cap')
        detector.fit_transform(sample_data_with_outliers)
        
        report = detector.get_outlier_report(sample_data_with_outliers)
        
        # Check report structure
        assert isinstance(report, pd.DataFrame)
        assert 'column' in report.columns
        assert 'outlier_count' in report.columns
        assert 'outlier_percentage' in report.columns
        
        # Should have entries for fitted columns
        assert len(report) > 0
    
    def test_different_thresholds(self, sample_data_with_outliers):
        """Test different IQR thresholds."""
        # Stricter threshold (more outliers detected)
        detector_strict = OutlierDetector(strategy='iqr', action='flag', threshold=1.0)
        result_strict = detector_strict.fit_transform(sample_data_with_outliers.copy())
        
        # Lenient threshold (fewer outliers detected)
        detector_lenient = OutlierDetector(strategy='iqr', action='flag', threshold=3.0)
        result_lenient = detector_lenient.fit_transform(sample_data_with_outliers.copy())
        
        # Stricter should detect more outliers
        assert result_strict['is_outlier'].sum() >= result_lenient['is_outlier'].sum()
    
    def test_categorical_columns_ignored(self):
        """Test that categorical columns are ignored."""
        data = pd.DataFrame({
            'num1': [1, 2, 3, 4, 100],
            'num2': [10, 20, 30, 40, 50],
            'cat': ['A', 'B', 'C', 'D', 'E']
        })
        
        detector = OutlierDetector(strategy='iqr', action='cap')
        result = detector.fit_transform(data)
        
        # Categorical column should be unchanged
        pd.testing.assert_series_equal(result['cat'], data['cat'])
        
        # Only numerical columns should be in fitted_columns
        assert 'cat' not in detector.fitted_columns
        assert 'num1' in detector.fitted_columns
        assert 'num2' in detector.fitted_columns
    
    def test_winsorization(self):
        """Test that capping (Winsorization) works correctly."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        })
        
        detector = OutlierDetector(strategy='iqr', action='cap', threshold=1.5)
        result = detector.fit_transform(data)
        
        # Outlier should be capped to upper bound
        lower, upper = detector.bounds['value']
        assert result['value'].max() == upper
        assert result['value'].max() < 100
    
    def test_zscore_with_zero_std(self):
        """Test Z-score method with zero standard deviation."""
        data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 100]
        })
        
        detector = OutlierDetector(strategy='zscore', action='cap', zscore_threshold=3.0)
        result = detector.fit_transform(data)
        
        # Should handle constant column gracefully
        assert 'constant' not in detector.bounds  # Skipped due to std=0
        assert 'variable' in detector.bounds
