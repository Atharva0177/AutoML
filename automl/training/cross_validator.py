"""
Cross-Validation Module.

This module provides cross-validation functionality for model training
and evaluation.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score
)
import logging

from automl.models.base_model import BaseModel
from automl.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross-validation handler for model evaluation.
    
    Supports K-Fold, Stratified K-Fold, and Time Series cross-validation.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        stratified: bool = False,
        time_series: bool = False
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            stratified: Use stratified splits (for classification)
            time_series: Use time series splits (no shuffle)
        """
        self.n_splits = n_splits
        self.shuffle = shuffle if not time_series else False
        self.random_state = random_state
        self.stratified = stratified
        self.time_series = time_series
        
        # Initialize appropriate splitter
        if time_series:
            self.splitter = TimeSeriesSplit(n_splits=n_splits)
            logger.info(f"Initialized TimeSeriesSplit with {n_splits} splits")
        elif stratified:
            self.splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            logger.info(f"Initialized StratifiedKFold with {n_splits} splits")
        else:
            self.splitter = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            logger.info(f"Initialized KFold with {n_splits} splits")
    
    def cross_validate(
        self,
        model: BaseModel,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        y: Union[pd.Series, NDArray[Any]],
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model instance to evaluate
            X: Features
            y: Target variable
            return_predictions: Whether to return predictions from each fold
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(
            f"Starting {self.n_splits}-fold cross-validation for {model.name}"
        )
        
        # Convert to numpy arrays for sklearn compatibility
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Storage for results
        fold_metrics: List[Dict[str, Any]] = []
        all_predictions: List[NDArray[Any]] = []
        all_true_labels: List[NDArray[Any]] = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.splitter.split(X_array, y_array if self.stratified else None),  # type: ignore[arg-type]
            1
        ):
            logger.info(f"Training fold {fold_idx}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]
            
            # Train model (create a fresh copy for each fold)
            fold_model = model.__class__(
                name=model.name,
                model_type=model.model_type,
                hyperparameters=model.get_params()
            )
            fold_model.fit(X_train, y_train)  # type: ignore[arg-type]
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            
            # Calculate metrics
            if model.model_type == 'classification':
                # Get probabilities if available
                try:
                    y_pred_proba = fold_model.predict_proba(X_val)
                except NotImplementedError:
                    y_pred_proba = None
                
                metrics = MetricsCalculator.calculate_classification_metrics(
                    y_val, y_pred, y_pred_proba  # type: ignore[arg-type]
                )
            else:  # regression
                metrics = MetricsCalculator.calculate_regression_metrics(
                    y_val, y_pred  # type: ignore[arg-type]
                )
            
            metrics['fold'] = fold_idx
            metrics['train_size'] = len(train_idx)
            metrics['val_size'] = len(val_idx)
            fold_metrics.append(metrics)
            
            if return_predictions:
                all_predictions.append(y_pred)
                all_true_labels.append(y_val)  # type: ignore[arg-type]
        
        # Aggregate results
        results = self._aggregate_fold_results(fold_metrics, model.model_type)
        
        if return_predictions:
            results['predictions'] = np.concatenate(all_predictions)
            results['true_labels'] = np.concatenate(all_true_labels)
        
        logger.info(
            f"Cross-validation completed. "
            f"Mean primary metric: {results['mean_primary_metric']:.4f} "
            f"(±{results['std_primary_metric']:.4f})"
        )
        
        return results
    
    def _aggregate_fold_results(
        self,
        fold_metrics: List[Dict[str, Any]],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across folds.
        
        Args:
            fold_metrics: List of metric dictionaries from each fold
            model_type: 'classification' or 'regression'
            
        Returns:
            Aggregated results with mean and std
        """
        results: Dict[str, Any] = {
            'fold_metrics': fold_metrics,
            'n_splits': self.n_splits
        }
        
        # Determine which metrics to aggregate
        if model_type == 'classification':
            metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
            if 'roc_auc' in fold_metrics[0]:
                metric_keys.append('roc_auc')
            if 'log_loss' in fold_metrics[0]:
                metric_keys.append('log_loss')
        else:  # regression
            metric_keys = ['mse', 'rmse', 'mae', 'r2_score', 'mape', 'median_ae']
        
        # Calculate mean and std for each metric
        for key in metric_keys:
            values = [m[key] for m in fold_metrics if key in m]
            if values:
                results[f'mean_{key}'] = float(np.mean(values))
                results[f'std_{key}'] = float(np.std(values))
        
        # Primary metric for comparison
        if model_type == 'classification':
            primary_key = 'f1_score'
        else:
            primary_key = 'r2_score'
        
        results['mean_primary_metric'] = results.get(f'mean_{primary_key}', 0.0)
        results['std_primary_metric'] = results.get(f'std_{primary_key}', 0.0)
        
        return results
    
    def get_cv_summary(
        self,
        cv_results: Dict[str, Any],
        model_type: str
    ) -> str:
        """
        Get a formatted summary of cross-validation results.
        
        Args:
            cv_results: Results from cross_validate()
            model_type: 'classification' or 'regression'
            
        Returns:
            Formatted summary string
        """
        lines = [f"\nCross-Validation Results ({self.n_splits} folds):"]
        lines.append("=" * 50)
        
        if model_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            if 'mean_roc_auc' in cv_results:
                metrics.append('roc_auc')
        else:
            metrics = ['rmse', 'mae', 'r2_score']
        
        for metric in metrics:
            mean_key = f'mean_{metric}'
            std_key = f'std_{metric}'
            if mean_key in cv_results:
                mean_val = cv_results[mean_key]
                std_val = cv_results[std_key]
                metric_name = metric.replace('_', ' ').title()
                lines.append(f"{metric_name:12s}: {mean_val:.4f} (±{std_val:.4f})")
        
        lines.append("=" * 50)
        
        return '\n'.join(lines)
