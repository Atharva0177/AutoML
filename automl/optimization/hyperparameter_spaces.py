"""
Hyperparameter search spaces for different models.

This module defines the hyperparameter search spaces for various
machine learning models, optimized for Optuna's suggest API.
"""

from typing import Dict, Any, Callable
import optuna


def get_hyperparameter_space(model_name: str) -> Dict[str, Callable]:
    """
    Get hyperparameter search space for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary mapping parameter names to Optuna suggest functions
        
    Raises:
        ValueError: If model not found
    """
    spaces = {
        'logistic_regression': _logistic_regression_space,
        'linear_regression': _linear_regression_space,
        'random_forest': _random_forest_space,
        'gradient_boosting': _gradient_boosting_space,
        'xgboost': _xgboost_space,
        'lightgbm': _lightgbm_space,
    }
    
    if model_name not in spaces:
        raise ValueError(f"No hyperparameter space defined for model: {model_name}")
    
    return spaces[model_name]()


def _logistic_regression_space() -> Dict[str, Callable]:
    """Hyperparameter space for Logistic Regression."""
    return {
        'C': lambda trial: trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': lambda trial: trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
        'solver': lambda trial: trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 100, 1000),
        'class_weight': lambda trial: trial.suggest_categorical('class_weight', ['balanced', None]),
    }


def _linear_regression_space() -> Dict[str, Callable]:
    """Hyperparameter space for Linear Regression."""
    return {
        'fit_intercept': lambda trial: trial.suggest_categorical('fit_intercept', [True, False]),
        # Linear regression has very few hyperparameters
        # Could add regularization (Ridge/Lasso) variants later
    }


def _random_forest_space() -> Dict[str, Callable]:
    """Hyperparameter space for Random Forest."""
    return {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': lambda trial: trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': lambda trial: trial.suggest_categorical('bootstrap', [True, False]),
        # Note: class_weight removed as it's only valid for RandomForestClassifier, not Regressor
    }


def _gradient_boosting_space() -> Dict[str, Callable]:
    """Hyperparameter space for Gradient Boosting."""
    return {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': lambda trial: trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }


def _xgboost_space() -> Dict[str, Callable]:
    """Hyperparameter space for XGBoost."""
    return {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': lambda trial: trial.suggest_int('min_child_weight', 1, 10),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': lambda trial: trial.suggest_float('gamma', 0, 5),
        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }


def _lightgbm_space() -> Dict[str, Callable]:
    """Hyperparameter space for LightGBM."""
    return {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
        'num_leaves': lambda trial: trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': lambda trial: trial.suggest_int('min_child_samples', 5, 100),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }


# Simplified search spaces for quick optimization
QUICK_SPACES = {
    'logistic_regression': {
        'C': lambda trial: trial.suggest_float('C', 0.1, 10, log=True),
        'penalty': lambda trial: trial.suggest_categorical('penalty', ['l2', 'none']),
    },
    'linear_regression': {
        'fit_intercept': lambda trial: trial.suggest_categorical('fit_intercept', [True, False]),
    },
    'random_forest': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 300),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 10),
    },
    'gradient_boosting': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 8),
    },
    'xgboost': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.7, 1.0),
    },
    'lightgbm': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
        'num_leaves': lambda trial: trial.suggest_int('num_leaves', 31, 100),
    },
}


def get_quick_hyperparameter_space(model_name: str) -> Dict[str, Callable]:
    """
    Get simplified hyperparameter search space for quick optimization.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary mapping parameter names to Optuna suggest functions
        
    Raises:
        ValueError: If model not found
    """
    if model_name not in QUICK_SPACES:
        raise ValueError(f"No quick hyperparameter space defined for model: {model_name}")
    
    return QUICK_SPACES[model_name]
