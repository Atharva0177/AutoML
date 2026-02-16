"""
Unit tests for advanced imputation methods.
"""

import pytest
import pandas as pd
import numpy as np
from automl.preprocessing.cleaners.advanced_imputation import (
    AdvancedMissingValueHandler,
    AdvancedImputationStrategy
)
from automl.utils.exceptions import ValidationError


class TestAdvancedMissingValueHandler:
    """Test cases for AdvancedMissingValueHandler."""
    
    @pytest.fixture
    def sample_data_numerical(self):
        """Create sample numerical data with missing values."""
        np.random.seed(42)
        data = {
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'feature2': [2.0, np.nan, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            'feature3': [1.5, 3.0, 4.5, np.nan, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0],
            'feature4': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_mixed(self):
        """Create sample mixed data with missing values."""
        np.random.seed(42)
        data = {
            'num1': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            'num2': [10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            'cat1': ['A', 'B', 'A', None, 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', None, 'X', 'Y', 'X', 'Y', 'X']
        }
        return pd.DataFrame(data)
    
    def test_init_knn(self):
        """Test KNN imputer initialization."""
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=3)
        assert handler.strategy == 'knn'
        assert handler.n_neighbors == 3
        assert not handler.is_fitted
    
    def test_init_iterative(self):
        """Test iterative imputer initialization."""
        handler = AdvancedMissingValueHandler(strategy='iterative', max_iter=20)
        assert handler.strategy == 'iterative'
        assert handler.max_iter == 20
        assert not handler.is_fitted
    
    def test_knn_imputation_numerical(self, sample_data_numerical):
        """Test KNN imputation on numerical data."""
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=3)
        
        # Count missing values before
        missing_before = sample_data_numerical.isnull().sum().sum()
        assert missing_before > 0
        
        # Fit and transform
        result = handler.fit_transform(sample_data_numerical)
        
        # Check no missing values after
        missing_after = result.isnull().sum().sum()
        assert missing_after == 0
        
        # Check shape preserved
        assert result.shape == sample_data_numerical.shape
        
        # Check columns preserved
        assert list(result.columns) == list(sample_data_numerical.columns)
        
        # Check handler is fitted
        assert handler.is_fitted
        assert len(handler.fitted_columns) > 0
    
    def test_iterative_imputation_numerical(self, sample_data_numerical):
        """Test iterative imputation on numerical data."""
        handler = AdvancedMissingValueHandler(strategy='iterative', max_iter=10)
        
        # Fit and transform
        result = handler.fit_transform(sample_data_numerical)
        
        # Check no missing values
        assert result.isnull().sum().sum() == 0
        
        # Check shape and columns
        assert result.shape == sample_data_numerical.shape
        assert list(result.columns) == list(sample_data_numerical.columns)
        
        # Check fitted
        assert handler.is_fitted
    
    def test_mixed_data_imputation(self, sample_data_mixed):
        """Test imputation on mixed numerical and categorical data."""
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=2)
        
        # Fit and transform
        result = handler.fit_transform(sample_data_mixed)
        
        # Check no missing values
        assert result.isnull().sum().sum() == 0
        
        # Check categorical columns are still categorical
        assert result['cat1'].dtype == 'object' or str(result['cat1'].dtype) == 'category'
        assert result['cat2'].dtype == 'object' or str(result['cat2'].dtype) == 'category'
        
        # Check categorical values are from original set
        original_cat1 = set(sample_data_mixed['cat1'].dropna().unique())
        result_cat1 = set(result['cat1'].unique())
        assert result_cat1.issubset(original_cat1)
    
    def test_transform_consistency(self, sample_data_numerical):
        """Test that transform produces consistent results."""
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=3)
        
        # Fit on data
        handler.fit_transform(sample_data_numerical)
        
        # Create new data with same pattern
        new_data = sample_data_numerical.copy()
        new_data.iloc[0, 0] = np.nan  # Add a new missing value
        
        # Transform should work
        result = handler.transform(new_data)
        assert result.isnull().sum().sum() == 0
    
    def test_transform_before_fit_raises_error(self, sample_data_numerical):
        """Test that transform before fit raises error."""
        handler = AdvancedMissingValueHandler(strategy='knn')
        
        with pytest.raises(ValidationError, match="must be fitted before transform"):
            handler.transform(sample_data_numerical)
    
    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises error."""
        handler = AdvancedMissingValueHandler(strategy='knn')  # type: ignore
        handler.strategy = 'invalid'
        
        data = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        
        with pytest.raises(ValidationError, match="Unknown strategy"):
            handler.fit_transform(data)
    
    def test_dtype_preservation(self, sample_data_numerical):
        """Test that data types are preserved after imputation."""
        # Create data with specific dtypes
        data = sample_data_numerical.copy()
        data['int_col'] = [1, 2, None, 4, 5, 6, 7, 8, 9, 10]
        data['int_col'] = data['int_col'].astype('Int64')
        
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=3)
        result = handler.fit_transform(data)
        
        # Check float columns are still float
        assert np.issubdtype(result['feature1'].dtype, np.floating)  # type: ignore[arg-type]
        assert np.issubdtype(result['feature2'].dtype, np.floating)  # type: ignore[arg-type]
    
    def test_imputation_summary(self, sample_data_mixed):
        """Test imputation summary."""
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=5)
        handler.fit_transform(sample_data_mixed)
        
        summary = handler.get_imputation_summary()
        
        assert summary['strategy'] == 'knn'
        assert summary['n_neighbors'] == 5
        assert summary['is_fitted'] is True
        assert len(summary['fitted_columns']) > 0
        assert len(summary['categorical_columns_encoded']) > 0
    
    def test_large_missing_percentage(self):
        """Test imputation with large percentage of missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5] + [np.nan] * 15,
            'col2': [10, 20, 30, 40, 50] + [np.nan] * 15,
            'col3': [100, 200, 300, 400, 500] + [np.nan] * 15,
        })
        
        handler = AdvancedMissingValueHandler(strategy='knn', n_neighbors=3)
        result = handler.fit_transform(data)
        
        # Should still complete without error
        assert result.isnull().sum().sum() == 0
        assert result.shape == data.shape
    
    def test_mice_alias(self, sample_data_numerical):
        """Test that 'mice' is an alias for 'iterative'."""
        handler = AdvancedMissingValueHandler(strategy='mice', max_iter=5)
        result = handler.fit_transform(sample_data_numerical)
        
        # Should work same as iterative
        assert result.isnull().sum().sum() == 0
        assert handler.is_fitted
    
    def test_no_missing_values(self):
        """Test imputation when there are no missing values."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'x', 'y']
        })
        
        handler = AdvancedMissingValueHandler(strategy='knn')
        result = handler.fit_transform(data)
        
        # Should return data unchanged (or minimally changed)
        assert result.shape == data.shape
        pd.testing.assert_frame_equal(result, data, check_dtype=False)
    
    def test_column_order_preserved(self, sample_data_mixed):
        """Test that column order is preserved."""
        original_columns = sample_data_mixed.columns.tolist()
        
        handler = AdvancedMissingValueHandler(strategy='knn')
        result = handler.fit_transform(sample_data_mixed)
        
        assert result.columns.tolist() == original_columns
