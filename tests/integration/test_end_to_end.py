"""
End-to-End Integration Tests.

This module tests the complete AutoML pipeline from data loading
to model training and prediction.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from sklearn.datasets import load_iris, load_diabetes, make_classification, make_regression

from automl.pipeline import AutoML


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def classification_csv(temp_dir):
    """Create classification dataset as CSV."""
    # Use iris dataset
    iris = load_iris()  # type: ignore[misc]
    df = pd.DataFrame(
        iris.data,  # type: ignore[union-attr]
        columns=iris.feature_names  # type: ignore[union-attr]
    )
    df['target'] = iris.target  # type: ignore[union-attr]
    
    csv_path = temp_dir / 'classification.csv'
    df.to_csv(csv_path, index=False)
    
    return csv_path, df


@pytest.fixture
def regression_csv(temp_dir):
    """Create regression dataset as CSV."""
    # Use diabetes dataset
    diabetes = load_diabetes()  # type: ignore[misc]
    df = pd.DataFrame(
        diabetes.data,  # type: ignore[union-attr]
        columns=diabetes.feature_names  # type: ignore[union-attr]
    )
    df['target'] = diabetes.target  # type: ignore[union-attr]
    
    csv_path = temp_dir / 'regression.csv'
    df.to_csv(csv_path, index=False)
    
    return csv_path, df


@pytest.fixture
def small_classification_data():
    """Create small classification dataset for quick tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    return df


@pytest.fixture
def small_regression_data():
    """Create small regression dataset for quick tests."""
    X, y = make_regression(  # type: ignore[misc]
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    return df


class TestAutoMLClassification:
    """Test AutoML pipeline with classification tasks."""
    
    def test_basic_classification(self, small_classification_data):
        """Test basic classification pipeline."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_classification_data,
            target_column='target',
            models_to_try=['logistic_regression', 'random_forest']
        )
        
        # Check results structure
        assert 'problem_type' in results
        assert results['problem_type'] == 'classification'
        assert 'best_model' in results
        assert 'model_comparison' in results
        
        # Check best model was selected
        assert results['best_model'] is not None
        
        # Test prediction
        test_data = small_classification_data.drop(columns=['target']).head(10)
        predictions = automl.predict(test_data)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_classification_with_cv(self, small_classification_data):
        """Test classification with cross-validation."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=True,
            cv_folds=3,
            verbose=False
        )
        
        results = automl.fit(
            small_classification_data,
            target_column='target',
            models_to_try=['logistic_regression']
        )
        
        assert results['problem_type'] == 'classification'
        assert results['best_model'] is not None
    
    def test_classification_from_csv(self, classification_csv, temp_dir):
        """Test classification pipeline from CSV file."""
        csv_path, df = classification_csv
        
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            csv_path,
            target_column='target',
            models_to_try=['logistic_regression']
        )
        
        assert results['problem_type'] == 'classification'
        assert results['best_model'] is not None
    
    def test_classification_predict_proba(self, small_classification_data):
        """Test probability predictions for classification."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_classification_data,
            target_column='target',
            models_to_try=['logistic_regression']
        )
        
        # Test probability prediction
        test_data = small_classification_data.drop(columns=['target']).head(10)
        probas = automl.predict_proba(test_data)
        
        assert probas.shape[0] == 10
        assert probas.shape[1] == 2  # Binary classification
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestAutoMLRegression:
    """Test AutoML pipeline with regression tasks."""
    
    def test_basic_regression(self, small_regression_data):
        """Test basic regression pipeline."""
        automl = AutoML(
            problem_type='regression',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_regression_data,
            target_column='target',
            models_to_try=['linear_regression', 'random_forest']
        )
        
        # Check results structure
        assert 'problem_type' in results
        assert results['problem_type'] == 'regression'
        assert 'best_model' in results
        assert 'model_comparison' in results
        
        # Check best model was selected
        assert results['best_model'] is not None
        
        # Test prediction
        test_data = small_regression_data.drop(columns=['target']).head(10)
        predictions = automl.predict(test_data)
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_regression_with_cv(self, small_regression_data):
        """Test regression with cross-validation."""
        automl = AutoML(
            problem_type='regression',
            use_cross_validation=True,
            cv_folds=3,
            verbose=False
        )
        
        results = automl.fit(
            small_regression_data,
            target_column='target',
            models_to_try=['linear_regression']
        )
        
        assert results['problem_type'] == 'regression'
        assert results['best_model'] is not None
    
    def test_regression_from_csv(self, regression_csv):
        """Test regression pipeline from CSV file."""
        csv_path, df = regression_csv
        
        automl = AutoML(
            problem_type='regression',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            csv_path,
            target_column='target',
            models_to_try=['linear_regression']
        )
        
        assert results['problem_type'] == 'regression'
        assert results['best_model'] is not None


class TestAutoMLAutDetection:
    """Test AutoML with automatic problem type detection."""
    
    def test_auto_detect_classification(self, small_classification_data):
        """Test automatic classification detection."""
        automl = AutoML(
            problem_type=None,  # Auto-detect
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_classification_data,
            target_column='target',
            models_to_try=['logistic_regression']
        )
        
        assert results['problem_type'] == 'classification'
    
    def test_auto_detect_regression(self, small_regression_data):
        """Test automatic regression detection."""
        automl = AutoML(
            problem_type=None,  # Auto-detect
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_regression_data,
            target_column='target',
            models_to_try=['linear_regression']
        )
        
        assert results['problem_type'] == 'regression'


class TestAutoMLModelComparison:
    """Test AutoML model comparison features."""
    
    def test_compare_multiple_models(self, small_classification_data):
        """Test comparing multiple models."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_classification_data,
            target_column='target',
            models_to_try=['logistic_regression', 'random_forest', 'xgboost']
        )
        
        # Check that multiple models were trained
        assert 'model_comparison' in results
        assert len(results['model_comparison']['models']) >= 3
        
        # Check rankings
        rankings = results['model_comparison']['rankings']
        assert len(rankings) >= 3
        assert rankings[0]['rank'] == 1
        assert rankings[1]['rank'] == 2
    
    def test_model_ranking_order(self, small_classification_data):
        """Test that models are ranked correctly."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        results = automl.fit(
            small_classification_data,
            target_column='target',
            models_to_try=['logistic_regression', 'random_forest']
        )
        
        rankings = results['model_comparison']['rankings']
        
        # Check scores are in descending order
        scores = [r['score'] for r in rankings]
        assert scores == sorted(scores, reverse=True)


class TestAutoMLErrorHandling:
    """Test AutoML error handling."""
    
    def test_missing_target_column(self, small_classification_data):
        """Test error when target column is missing."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="Target column.*not found"):
            automl.fit(
                small_classification_data,
                target_column='nonexistent_column'
            )
    
    def test_predict_before_fit(self):
        """Test error when predicting before fitting."""
        automl = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            verbose=False
        )
        
        test_data = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
        
        with pytest.raises(RuntimeError, match="hasn't been fitted yet"):
            automl.predict(test_data)


class TestAutoMLReproducibility:
    """Test AutoML reproducibility with random seeds."""
    
    def test_reproducible_results(self, small_classification_data):
        """Test that same random state produces same results."""
        automl1 = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            random_state=42,
            verbose=False
        )
        
        results1 = automl1.fit(
            small_classification_data.copy(),
            target_column='target',
            models_to_try=['logistic_regression']
        )
        
        automl2 = AutoML(
            problem_type='classification',
            use_cross_validation=False,
            random_state=42,
            verbose=False
        )
        
        results2 = automl2.fit(
            small_classification_data.copy(),
            target_column='target',
            models_to_try=['logistic_regression']
        )
        
        # Best model should be the same
        assert results1['best_model'] == results2['best_model']
