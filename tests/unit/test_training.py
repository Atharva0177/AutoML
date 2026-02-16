"""
Tests for training module.

Tests for metrics calculation, cross-validation, and training.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from automl.training import MetricsCalculator, CrossValidator, Trainer
from automl.models import RandomForestModel, LogisticRegressionModel, LinearRegressionModel


# Fixtures

@pytest.fixture
def classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    })
    
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate sample multi-class dataset."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    })
    
    y = pd.Series(np.random.randint(0, 3, n_samples))
    
    return X, y


@pytest.fixture
def regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    })
    
    y = pd.Series(
        X['feature_1'] * 2 + X['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.1
    )
    
    return X, y


# MetricsCalculator Tests

class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    def test_classification_metrics_binary(self, classification_data):
        """Test binary classification metrics."""
        X, y = classification_data
        
        # Create simple predictions
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_classification_metrics(y, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect predictions
    
    def test_classification_metrics_multiclass(self, multiclass_data):
        """Test multi-class classification metrics."""
        X, y = multiclass_data
        
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_classification_metrics(
            y, y_pred, average='weighted'
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert metrics['accuracy'] == 1.0
    
    def test_classification_metrics_with_proba(self, classification_data):
        """Test metrics with probability predictions."""
        X, y = classification_data
        
        y_pred = y.copy()
        y_proba = np.zeros((len(y), 2))
        y_proba[np.arange(len(y)), y] = 1.0  # Perfect probabilities
        
        metrics = MetricsCalculator.calculate_classification_metrics(
            y, y_pred, y_proba
        )
        
        assert 'roc_auc' in metrics
        assert 'log_loss' in metrics
        assert metrics['roc_auc'] == 1.0
        assert metrics['log_loss'] < 0.01  # Very low loss for perfect predictions
    
    def test_regression_metrics(self, regression_data):
        """Test regression metrics."""
        X, y = regression_data
        
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_regression_metrics(y, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'mape' in metrics
        assert 'median_ae' in metrics
        
        assert metrics['mse'] < 1e-10  # Very low for perfect predictions
        assert metrics['r2_score'] == 1.0
    
    def test_get_primary_metric_classification(self, classification_data):
        """Test getting primary metric for classification."""
        X, y = classification_data
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_classification_metrics(y, y_pred)
        primary = MetricsCalculator.get_primary_metric('classification', metrics)
        
        assert primary == metrics['f1_score']
    
    def test_get_primary_metric_regression(self, regression_data):
        """Test getting primary metric for regression."""
        X, y = regression_data
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_regression_metrics(y, y_pred)
        primary = MetricsCalculator.get_primary_metric('regression', metrics)
        
        assert primary == metrics['r2_score']
    
    def test_format_metrics_classification(self, classification_data):
        """Test formatting classification metrics."""
        X, y = classification_data
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_classification_metrics(y, y_pred)
        formatted = MetricsCalculator.format_metrics(metrics, 'classification')
        
        assert 'Accuracy' in formatted
        assert 'Precision' in formatted
        assert 'F1 Score' in formatted
    
    def test_format_metrics_regression(self, regression_data):
        """Test formatting regression metrics."""
        X, y = regression_data
        y_pred = y.copy()
        
        metrics = MetricsCalculator.calculate_regression_metrics(y, y_pred)
        formatted = MetricsCalculator.format_metrics(metrics, 'regression')
        
        assert 'RMSE' in formatted
        assert 'R² Score' in formatted


# CrossValidator Tests

class TestCrossValidator:
    """Tests for CrossValidator class."""
    
    def test_kfold_initialization(self):
        """Test K-Fold initialization."""
        cv = CrossValidator(n_splits=5, stratified=False)
        
        assert cv.n_splits == 5
        assert not cv.stratified
        assert not cv.time_series
    
    def test_stratified_kfold_initialization(self):
        """Test Stratified K-Fold initialization."""
        cv = CrossValidator(n_splits=5, stratified=True)
        
        assert cv.n_splits == 5
        assert cv.stratified
    
    def test_time_series_initialization(self):
        """Test Time Series split initialization."""
        cv = CrossValidator(n_splits=5, time_series=True)
        
        assert cv.n_splits == 5
        assert cv.time_series
        assert not cv.shuffle
    
    def test_cross_validate_classification(self, classification_data):
        """Test cross-validation for classification."""
        X, y = classification_data
        
        model = LogisticRegressionModel()
        cv = CrossValidator(n_splits=3, stratified=True)
        
        results = cv.cross_validate(model, X, y)
        
        assert 'fold_metrics' in results
        assert len(results['fold_metrics']) == 3
        assert 'mean_accuracy' in results
        assert 'mean_f1_score' in results
        assert 'std_accuracy' in results
        assert results['n_splits'] == 3
    
    def test_cross_validate_regression(self, regression_data):
        """Test cross-validation for regression."""
        X, y = regression_data
        
        model = LinearRegressionModel()
        cv = CrossValidator(n_splits=3)
        
        results = cv.cross_validate(model, X, y)
        
        assert 'fold_metrics' in results
        assert len(results['fold_metrics']) == 3
        assert 'mean_rmse' in results
        assert 'mean_r2_score' in results
        assert 'std_rmse' in results
    
    def test_cross_validate_with_predictions(self, classification_data):
        """Test CV with return predictions."""
        X, y = classification_data
        
        model = LogisticRegressionModel()
        cv = CrossValidator(n_splits=3)
        
        results = cv.cross_validate(model, X, y, return_predictions=True)
        
        assert 'predictions' in results
        assert 'true_labels' in results
        assert len(results['predictions']) == len(y)
    
    def test_get_cv_summary(self, classification_data):
        """Test CV summary generation."""
        X, y = classification_data
        
        model = LogisticRegressionModel()
        cv = CrossValidator(n_splits=3)
        
        results = cv.cross_validate(model, X, y)
        summary = cv.get_cv_summary(results, 'classification')
        
        assert 'Cross-Validation Results' in summary
        assert 'Accuracy' in summary
        assert 'F1 Score' in summary


# Trainer Tests

class TestTrainer:
    """Tests for Trainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = Trainer(use_cross_validation=True, cv_folds=5)
        
        assert trainer.use_cross_validation
        assert trainer.cv_folds == 5
        assert trainer.training_history == []
    
    def test_train_without_cv_classification(self, classification_data):
        """Test training without cross-validation."""
        X, y = classification_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LogisticRegressionModel()
        trainer = Trainer(use_cross_validation=False)
        
        results = trainer.train(model, X_train, y_train, X_val, y_val)
        
        assert results['status'] == 'success'
        assert 'train_metrics' in results
        assert 'val_metrics' in results
        assert 'training_time' in results
        assert results['model_name'] == model.name
    
    def test_train_with_cv_classification(self, classification_data):
        """Test training with cross-validation."""
        X, y = classification_data
        
        model = LogisticRegressionModel()
        trainer = Trainer(use_cross_validation=True, cv_folds=3, cv_stratified=True)
        
        results = trainer.train(model, X, y)
        
        assert results['status'] == 'success'
        assert 'cross_validation' in results
        assert results['validation_type'] == 'cross_validation'
        assert model.is_fitted
    
    def test_train_regression(self, regression_data):
        """Test training regression model."""
        X, y = regression_data
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LinearRegressionModel()
        trainer = Trainer()
        
        results = trainer.train(model, X_train, y_train, X_val, y_val)
        
        assert results['status'] == 'success'
        assert 'train_metrics' in results
        assert 'val_metrics' in results
        assert results['val_metrics']['r2_score'] > 0.5
    
    def test_compare_models(self, classification_data):
        """Test model comparison."""
        X, y = classification_data
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        models = [
            LogisticRegressionModel(),
            RandomForestModel(model_type='classification')
        ]
        
        trainer = Trainer()
        results = trainer.compare_models(models, X_train, y_train, X_val, y_val)
        
        assert 'models' in results
        assert len(results['models']) == 2
        assert 'rankings' in results
        assert len(results['rankings']) == 2
        assert results['rankings'][0]['rank'] == 1
    
    def test_get_best_model(self, classification_data):
        """Test getting best model from comparison."""
        X, y = classification_data
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        models = [
            LogisticRegressionModel(),
            RandomForestModel(model_type='classification')
        ]
        
        trainer = Trainer()
        results = trainer.compare_models(models, X_train, y_train, X_val, y_val)
        
        best_model = trainer.get_best_model(results)
        
        assert best_model is not None
        assert best_model in [m.name for m in models]
    
    def test_training_history(self, classification_data):
        """Test training history tracking."""
        X, y = classification_data
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        trainer = Trainer()
        
        # Train multiple models
        model1 = LogisticRegressionModel()
        model2 = RandomForestModel(model_type='classification')
        
        trainer.train(model1, X_train, y_train, X_val, y_val)
        trainer.train(model2, X_train, y_train, X_val, y_val)
        
        assert len(trainer.training_history) == 2
        
        # Get summary
        summary = trainer.get_training_summary()
        assert len(summary) == 2
        assert 'model_name' in summary.columns
