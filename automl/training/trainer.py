"""
Trainer Module.

This module provides the main training orchestration for AutoML models.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from automl.models.base_model import BaseModel
from automl.optimization import OptunaOptimizer
from automl.training.cross_validator import CrossValidator
from automl.training.metrics import MetricsCalculator

try:
    from automl.tracking import MLflowTracker

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowTracker = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for training and evaluating models.

    Handles model training, evaluation, and result tracking.
    """

    def __init__(
        self,
        use_cross_validation: bool = False,
        cv_folds: int = 5,
        cv_stratified: bool = False,
        random_state: Optional[int] = 42,
        optimize_hyperparameters: bool = False,
        n_trials: int = 50,
        optimization_timeout: Optional[int] = None,
        mlflow_tracker: Optional[Any] = None,
        log_to_mlflow: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of cross-validation folds
            cv_stratified: Use stratified CV for classification
            random_state: Random seed for reproducibility
            optimize_hyperparameters: Whether to optimize hyperparameters with Optuna
            n_trials: Number of optimization trials (if optimize_hyperparameters=True)
            optimization_timeout: Timeout for optimization in seconds
            mlflow_tracker: Optional MLflowTracker instance for experiment tracking
            log_to_mlflow: Whether to log metrics and models to MLflow
        """
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        self.random_state = random_state
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_trials = n_trials
        self.optimization_timeout = optimization_timeout
        self.mlflow_tracker = mlflow_tracker
        self.log_to_mlflow = (
            log_to_mlflow and MLFLOW_AVAILABLE and mlflow_tracker is not None
        )

        self.training_history: List[Dict[str, Any]] = []

        # Initialize optimizer if needed
        self.optimizer: Optional[OptunaOptimizer] = None
        if optimize_hyperparameters:
            self.optimizer = OptunaOptimizer(
                n_trials=n_trials,
                timeout=optimization_timeout,
                random_state=random_state,
                show_progress_bar=False,  # Keep console clean
                verbose=False,
            )

        logger.info(
            f"Initialized Trainer (CV: {use_cross_validation}, "
            f"folds: {cv_folds if use_cross_validation else 'N/A'}, "
            f"optimize: {optimize_hyperparameters}, "
            f"MLflow: {self.log_to_mlflow})"
        )

    def train(
        self,
        model: BaseModel,
        X_train: Union[pd.DataFrame, NDArray[np.float64]],
        y_train: Union[pd.Series, NDArray[Any]],
        X_val: Optional[Union[pd.DataFrame, NDArray[np.float64]]] = None,
        y_val: Optional[Union[pd.Series, NDArray[Any]]] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train a model and evaluate it.

        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **fit_kwargs: Additional arguments passed to model.fit()

        Returns:
            Dictionary containing training results and metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting training for {model.name}")

        results: Dict[str, Any] = {
            "model_name": model.name,
            "model_type": model.model_type,
            "start_time": start_time.isoformat(),
            "hyperparameters": model.get_params(),
        }

        try:
            # Hyperparameter optimization (if enabled and validation data available)
            if (
                self.optimize_hyperparameters
                and X_val is not None
                and y_val is not None
            ):
                logger.info(f"Optimizing hyperparameters for {model.name}")
                best_params, study = self.optimizer.optimize(  # type: ignore[union-attr]
                    model, X_train, y_train, X_val, y_val
                )
                results["optimized_hyperparameters"] = best_params
                results["optimization_trials"] = len(study.trials)
                results["best_optimization_score"] = study.best_value
                logger.info(
                    f"Hyperparameter optimization complete: {len(study.trials)} trials"
                )

            if self.use_cross_validation:
                # Cross-validation mode
                results.update(self._train_with_cv(model, X_train, y_train))
            else:
                # Simple train-validation split mode
                results.update(
                    self._train_with_validation(
                        model, X_train, y_train, X_val, y_val, **fit_kwargs
                    )
                )

            # Log to MLflow if enabled
            if self.log_to_mlflow:
                self._log_to_mlflow(model, results, X_train)

            results["status"] = "success"

        except Exception as e:
            logger.error(f"Training failed for {model.name}: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        finally:
            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["training_time"] = (end_time - start_time).total_seconds()

            # Store in history
            self.training_history.append(results)

            logger.info(
                f"Training completed for {model.name} in "
                f"{results['training_time']:.2f}s"
            )

        return results

    def _train_with_cv(
        self,
        model: BaseModel,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        y: Union[pd.Series, NDArray[Any]],
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation.

        Args:
            model: Model instance
            X: Features
            y: Target

        Returns:
            CV results dictionary
        """
        logger.info(f"Training with {self.cv_folds}-fold cross-validation")

        # Create cross-validator
        cv = CrossValidator(
            n_splits=self.cv_folds,
            stratified=self.cv_stratified and model.model_type == "classification",
            random_state=self.random_state,
        )

        # Perform cross-validation
        cv_results = cv.cross_validate(model, X, y)

        # Train final model on full dataset
        logger.info("Training final model on full dataset")
        model.fit(X, y)

        return {
            "cross_validation": cv_results,
            "validation_type": "cross_validation",
            "n_folds": self.cv_folds,
        }

    def _train_with_validation(
        self,
        model: BaseModel,
        X_train: Union[pd.DataFrame, NDArray[np.float64]],
        y_train: Union[pd.Series, NDArray[Any]],
        X_val: Optional[Union[pd.DataFrame, NDArray[np.float64]]],
        y_val: Optional[Union[pd.Series, NDArray[Any]]],
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train model with simple train-validation split.

        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            **fit_kwargs: Additional fit arguments

        Returns:
            Training results dictionary
        """
        logger.info("Training with train-validation split")

        # Train model
        model.fit(X_train, y_train, **fit_kwargs)

        # Calculate training metrics
        y_train_pred = model.predict(X_train)

        if model.model_type == "classification":
            try:
                y_train_proba = model.predict_proba(X_train)
            except NotImplementedError:
                y_train_proba = None

            train_metrics = MetricsCalculator.calculate_classification_metrics(
                y_train, y_train_pred, y_train_proba  # type: ignore[arg-type]
            )
        else:
            train_metrics = MetricsCalculator.calculate_regression_metrics(
                y_train, y_train_pred  # type: ignore[arg-type]
            )

        results: Dict[str, Any] = {
            "train_metrics": train_metrics,
            "validation_type": "holdout",
        }

        # Calculate validation metrics if validation set provided
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)

            if model.model_type == "classification":
                try:
                    y_val_proba = model.predict_proba(X_val)
                except NotImplementedError:
                    y_val_proba = None

                val_metrics = MetricsCalculator.calculate_classification_metrics(
                    y_val, y_val_pred, y_val_proba  # type: ignore[arg-type]
                )
            else:
                val_metrics = MetricsCalculator.calculate_regression_metrics(
                    y_val, y_val_pred  # type: ignore[arg-type]
                )

            results["val_metrics"] = val_metrics

            # Log comparison
            if model.model_type == "classification":
                logger.info(
                    f"Train F1: {train_metrics['f1_score']:.4f}, "
                    f"Val F1: {val_metrics['f1_score']:.4f}"
                )
            else:
                logger.info(
                    f"Train R²: {train_metrics['r2_score']:.4f}, "
                    f"Val R²: {val_metrics['r2_score']:.4f}"
                )

        return results

    def compare_models(
        self,
        models: List[BaseModel],
        X_train: Union[pd.DataFrame, NDArray[np.float64]],
        y_train: Union[pd.Series, NDArray[Any]],
        X_val: Optional[Union[pd.DataFrame, NDArray[np.float64]]] = None,
        y_val: Optional[Union[pd.Series, NDArray[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Train and compare multiple models.

        Args:
            models: List of model instances
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {len(models)} models")

        results: Dict[str, Any] = {
            "models": [],
            "model_names": [m.name for m in models],
        }

        # Train each model
        for model in models:
            try:
                model_results = self.train(model, X_train, y_train, X_val, y_val)
                results["models"].append(model_results)
            except Exception as e:
                logger.error(f"Failed to train {model.name}: {e}")
                results["models"].append(
                    {"model_name": model.name, "status": "failed", "error": str(e)}
                )

        # Rank models
        results["rankings"] = self._rank_models(results["models"])

        # Add best score for easy access
        if results["rankings"]:
            results["best_score"] = results["rankings"][0]["score"]
        else:
            results["best_score"] = 0

        return results

    def _rank_models(self, model_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank models based on their performance.

        Args:
            model_results: List of training results

        Returns:
            Sorted list of model rankings
        """
        rankings = []

        for result in model_results:
            if result.get("status") != "success":
                continue

            # Extract primary metric
            if "cross_validation" in result:
                score = result["cross_validation"].get("mean_primary_metric", 0)
            elif "val_metrics" in result:
                model_type = result.get("model_type", "classification")
                score = MetricsCalculator.get_primary_metric(
                    model_type, result["val_metrics"]
                )
            else:
                # Use train metrics if no validation
                model_type = result.get("model_type", "classification")
                score = MetricsCalculator.get_primary_metric(
                    model_type, result.get("train_metrics", {})
                )

            rankings.append(
                {
                    "model_name": result["model_name"],
                    "score": score,
                    "training_time": result.get("training_time", 0),
                }
            )

        # Sort by score (descending)
        rankings.sort(key=lambda x: x["score"], reverse=True)

        # Add ranks
        for i, ranking in enumerate(rankings, 1):
            ranking["rank"] = i

        return rankings

    def get_best_model(self, comparison_results: Dict[str, Any]) -> Optional[str]:
        """
        Get the name of the best performing model.

        Args:
            comparison_results: Results from compare_models()

        Returns:
            Name of best model, or None if no successful models
        """
        rankings = comparison_results.get("rankings", [])
        if rankings:
            return rankings[0]["model_name"]
        return None

    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary of all training runs.

        Returns:
            DataFrame with training history
        """
        if not self.training_history:
            return pd.DataFrame()

        summary_data = []
        for result in self.training_history:
            row = {
                "model_name": result.get("model_name"),
                "status": result.get("status"),
                "training_time": result.get("training_time"),
            }

            # Add metrics
            if "val_metrics" in result:
                metrics = result["val_metrics"]
                if "f1_score" in metrics:
                    row["f1_score"] = metrics["f1_score"]
                    row["accuracy"] = metrics.get("accuracy")
                elif "r2_score" in metrics:
                    row["r2_score"] = metrics["r2_score"]
                    row["rmse"] = metrics.get("rmse")
            elif "cross_validation" in result:
                cv = result["cross_validation"]
                row["cv_score"] = cv.get("mean_primary_metric")
                row["cv_std"] = cv.get("std_primary_metric")

            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def _log_to_mlflow(
        self,
        model: BaseModel,
        results: Dict[str, Any],
        X_train: Union[pd.DataFrame, NDArray[np.float64]],
    ):
        """
        Log training results to MLflow.

        Args:
            model: Trained model
            results: Training results dict
            X_train: Training features (for input example)
        """
        if not self.mlflow_tracker:
            return

        try:
            # Log hyperparameters
            params = {
                "model_type": results.get("model_type"),
                "use_cv": self.use_cross_validation,
                "cv_folds": self.cv_folds if self.use_cross_validation else None,
                "optimized": self.optimize_hyperparameters,
            }

            # Add model hyperparameters (prefix with model name to avoid conflicts)
            model_params = results.get("hyperparameters", {})
            model_name = results.get("model_name", "unknown")
            for k, v in model_params.items():
                params[f"{model_name}_{k}"] = v

            self.mlflow_tracker.log_params(params)

            # Log optimization metrics if available
            if "optimization_trials" in results:
                self.mlflow_tracker.log_metric(
                    "optimization_trials", results["optimization_trials"]
                )
                self.mlflow_tracker.log_metric(
                    "best_optimization_score", results.get("best_optimization_score", 0)
                )

            # Log training metrics
            metrics = {}

            # Skip non-scalar metrics like confusion_matrix and per_class_metrics
            scalar_metrics_only = lambda d: {
                k: v for k, v in d.items() 
                if k not in ['confusion_matrix', 'per_class_metrics'] 
                and not isinstance(v, (list, dict, np.ndarray))
            }

            if "train_metrics" in results:
                for k, v in scalar_metrics_only(results["train_metrics"]).items():
                    metrics[f"train_{k}"] = v

            if "val_metrics" in results:
                for k, v in scalar_metrics_only(results["val_metrics"]).items():
                    metrics[f"val_{k}"] = v

            if "cross_validation" in results:
                cv = results["cross_validation"]
                metrics["cv_mean_score"] = cv.get("mean_primary_metric", 0)
                metrics["cv_std_score"] = cv.get("std_primary_metric", 0)

            if "training_time" in results:
                metrics["training_time_seconds"] = results["training_time"]

            self.mlflow_tracker.log_metrics(metrics)

            # Log model
            try:
                # Get input example (first 5 rows)
                input_example = None
                if isinstance(X_train, pd.DataFrame):
                    input_example = X_train.head(5)
                elif isinstance(X_train, np.ndarray) and X_train.shape[0] >= 5:
                    input_example = X_train[:5]

                self.mlflow_tracker.log_model(
                    model.model, artifact_path=model.name, input_example=input_example
                )
            except Exception as e:
                logger.warning(f"Could not log model to MLflow: {e}")

            # Set tags
            self.mlflow_tracker.set_tags(
                {"model_name": model.name, "status": results.get("status", "unknown")}
            )

            logger.debug(f"Logged training results to MLflow for {model.name}")

        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")
