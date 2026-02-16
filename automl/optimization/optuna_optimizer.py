"""
Optuna-based Bayesian Hyperparameter Optimization.

This module provides advanced hyperparameter optimization using Optuna's
Bayesian optimization algorithms.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from automl.optimization.hyperparameter_spaces import (
    get_hyperparameter_space,
    get_quick_hyperparameter_space,
)

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna's Bayesian optimization.

    Provides intelligent hyperparameter search using Tree-structured
    Parzen Estimator (TPE) algorithm with early stopping.
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        quick_mode: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (None for no timeout)
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress bar
            quick_mode: Use simplified search spaces for faster optimization
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed logs
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.show_progress_bar = show_progress_bar
        self.quick_mode = quick_mode
        self.random_state = random_state
        self.verbose = verbose

        # Set Optuna logging level
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(
            f"Initialized OptunaOptimizer (trials={n_trials}, "
            f"quick_mode={quick_mode}, random_state={random_state})"
        )

    def optimize(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, NDArray[np.float64]],
        y_train: Union[pd.Series, NDArray[Any]],
        X_val: Union[pd.DataFrame, NDArray[np.float64]],
        y_val: Union[pd.Series, NDArray[Any]],
        metric: str = "auto",
        direction: str = "maximize",
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Optimize hyperparameters for a model.

        Args:
            model: Model instance to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('auto', 'accuracy', 'f1', 'r2', 'mse')
            direction: 'maximize' or 'minimize'

        Returns:
            Tuple of (best_params, study)
        """
        model_name = model.name
        model_type = model.model_type

        # Get hyperparameter space
        if self.quick_mode:
            param_space = get_quick_hyperparameter_space(model_name)
        else:
            param_space = get_hyperparameter_space(model_name)

        # Determine metric and direction
        if metric == "auto":
            if model_type == "classification":
                metric = "f1"
                direction = "maximize"
            else:
                metric = "r2"
                direction = "maximize"

        logger.info(
            f"Starting optimization for {model_name}: "
            f"{self.n_trials} trials, metric={metric}, direction={direction}"
        )

        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna."""
            # Sample hyperparameters
            params = {}
            for param_name, suggest_fn in param_space.items():
                try:
                    params[param_name] = suggest_fn(trial)
                except Exception as e:
                    logger.warning(f"Error sampling {param_name}: {e}")
                    continue

            # Handle special parameter constraints
            params = self._validate_params(params, model_name, trial)

            # Create model with sampled hyperparameters
            try:
                model.set_hyperparameters(**params)
            except Exception as e:
                logger.warning(f"Invalid hyperparameters: {e}")
                # Return poor score for invalid combinations
                return -np.inf if direction == "maximize" else np.inf

            # Train model
            try:
                logger.info(
                    f"Trial {trial.number}: Training {model_name} with {len(params)} hyperparameters..."
                )
                model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Training failed with params {params}: {e}")
                return -np.inf if direction == "maximize" else np.inf

            # Evaluate on validation set
            try:
                predictions = model.predict(X_val)
                score = self._calculate_metric(y_val, predictions, metric, model_type)
                logger.info(f"Trial {trial.number}: {metric}={score:.4f}")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                return -np.inf if direction == "maximize" else np.inf

            return score

        # Create study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        # Optimize
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=self.show_progress_bar,
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")

        best_params = study.best_params
        best_value = study.best_value

        logger.info(
            f"Optimization complete for {model_name}: "
            f"best_{metric}={best_value:.4f}, trials={len(study.trials)}"
        )

        if self.verbose:
            print(f"\nBest hyperparameters for {model_name}:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"Best {metric}: {best_value:.4f}")

        return best_params, study

    def _validate_params(
        self, params: Dict[str, Any], model_name: str, trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Validate and adjust parameters for model-specific constraints.

        Args:
            params: Sampled parameters
            model_name: Name of the model
            trial: Optuna trial object

        Returns:
            Validated parameters
        """
        # Logistic regression penalty-solver compatibility
        if model_name == "logistic_regression":
            penalty = params.get("penalty")
            solver = params.get("solver")

            # l1 penalty only works with liblinear and saga
            if penalty == "l1" and solver not in ["liblinear", "saga"]:
                params["solver"] = "liblinear"

            # elasticnet only works with saga
            if penalty == "elasticnet":
                params["solver"] = "saga"
                # Add l1_ratio for elasticnet
                if "l1_ratio" not in params:
                    params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)

            # none penalty doesn't support liblinear
            if penalty == "none" and solver == "liblinear":
                params["solver"] = "lbfgs"

        # LightGBM num_leaves constraint
        if model_name == "lightgbm":
            # num_leaves should be < 2^max_depth
            max_depth = params.get("max_depth", 10)
            num_leaves = params.get("num_leaves", 31)
            max_num_leaves = 2**max_depth
            if num_leaves >= max_num_leaves:
                params["num_leaves"] = max_num_leaves - 1

        return params

    def _calculate_metric(
        self,
        y_true: Union[pd.Series, NDArray[Any]],
        y_pred: NDArray[Any],
        metric: str,
        model_type: str,
    ) -> float:
        """
        Calculate evaluation metric.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric: Metric name
            model_type: 'classification' or 'regression'

        Returns:
            Metric value
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        if model_type == "classification":
            if metric == "accuracy":
                return float(accuracy_score(y_true, y_pred))
            elif metric in ["f1", "f1_score"]:
                # Handle multiclass
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # For binary classification, detect positive label
                    unique_labels = np.unique(y_true)
                    # Use the second label as positive (typically 'positive', 1, etc.)
                    pos_label = (
                        unique_labels[1] if len(unique_labels) > 1 else unique_labels[0]
                    )
                    return float(
                        f1_score(y_true, y_pred, average="binary", pos_label=pos_label)
                    )
                else:
                    return float(f1_score(y_true, y_pred, average="weighted"))
            else:
                raise ValueError(f"Unknown classification metric: {metric}")
        else:
            if metric == "r2":
                return float(r2_score(y_true, y_pred))
            elif metric == "mse":
                return float(
                    -mean_squared_error(y_true, y_pred)
                )  # Negative for maximization
            elif metric == "mae":
                return float(
                    -mean_absolute_error(y_true, y_pred)
                )  # Negative for maximization
            else:
                raise ValueError(f"Unknown regression metric: {metric}")

    def get_optimization_history(self, study: optuna.Study) -> pd.DataFrame:
        """
        Get optimization history as a DataFrame.

        Args:
            study: Optuna study object

        Returns:
            DataFrame with trial history
        """
        trials_df = study.trials_dataframe()
        return trials_df

    def plot_optimization_history(
        self, study: optuna.Study, save_path: Optional[str] = None
    ) -> None:
        """
        Plot optimization history.

        Args:
            study: Optuna study object
            save_path: Path to save plot (None to display)
        """
        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(study)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Saved optimization history to {save_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed, cannot generate plots")

    def plot_param_importances(
        self, study: optuna.Study, save_path: Optional[str] = None
    ) -> None:
        """
        Plot hyperparameter importances.

        Args:
            study: Optuna study object
            save_path: Path to save plot (None to display)
        """
        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(study)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Saved parameter importances to {save_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed, cannot generate plots")
