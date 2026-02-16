"""
MLflow integration for experiment tracking, model versioning, and artifact storage.

This module provides a wrapper around MLflow to track experiments, log metrics,
parameters, artifacts, and register models.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment tracking."""

    experiment_name: str
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    enable_autolog: bool = False
    log_models: bool = True
    log_artifacts: bool = True
    log_system_metrics: bool = True


class MLflowTracker:
    """
    MLflow experiment tracker for AutoML pipeline.

    Handles experiment creation, run management, metric/parameter logging,
    model registration, and artifact storage.
    """

    def __init__(
        self,
        experiment_name: str = "automl_experiments",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        enable_autolog: bool = False,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (None for local)
            artifact_location: Location to store artifacts
            enable_autolog: Enable automatic logging for supported libraries
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or self._get_default_tracking_uri()
        self.artifact_location = artifact_location
        self.enable_autolog = enable_autolog

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.experiment_id = self._setup_experiment()

        # MLflow client for advanced operations
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Enable autologging if requested
        if enable_autolog:
            self._enable_autologging()

        # Current run context
        self.current_run = None
        self.current_run_id = None

        logger.info(
            f"Initialized MLflowTracker (experiment: {experiment_name}, uri: {self.tracking_uri})"
        )

    def _get_default_tracking_uri(self) -> str:
        """Get default tracking URI (local file store)."""
        default_path = Path("mlruns").absolute()
        return f"file:///{default_path}"

    def _setup_experiment(self) -> str:
        """Create or get existing experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    self.experiment_name, artifact_location=self.artifact_location
                )
                logger.info(
                    f"Created new experiment: {self.experiment_name} (ID: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})"
                )

            return experiment_id

        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise

    def _enable_autologging(self):
        """Enable automatic logging for supported libraries."""
        try:
            # Enable autologging for scikit-learn
            mlflow.sklearn.autolog(  # type: ignore
                log_input_examples=False, log_model_signatures=True, log_models=True
            )

            # Enable autologging for XGBoost
            mlflow.xgboost.autolog(  # type: ignore
                log_input_examples=False, log_model_signatures=True, log_models=True
            )

            # Enable autologging for LightGBM
            mlflow.lightgbm.autolog(  # type: ignore
                log_input_examples=False, log_model_signatures=True, log_models=True
            )

            logger.info("Enabled MLflow autologging")

        except Exception as e:
            logger.warning(f"Failed to enable autologging: {e}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Dictionary of tags for the run
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        try:
            self.current_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags,
                nested=nested,
            )
            self.current_run_id = self.current_run.info.run_id

            logger.info(
                f"Started MLflow run: {run_name or 'unnamed'} (ID: {self.current_run_id})"
            )

            return self.current_run_id

        except Exception as e:
            logger.error(f"Error starting run: {e}")
            raise

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            if self.current_run is not None:
                mlflow.end_run(status=status)
                logger.info(
                    f"Ended MLflow run (ID: {self.current_run_id}, status: {status})"
                )
                self.current_run = None
                self.current_run_id = None
        except Exception as e:
            logger.error(f"Error ending run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameters
        """
        try:
            # MLflow has limits on param values, so convert to strings
            sanitized_params = {
                str(k): str(v)[:250] if v is not None else "None"
                for k, v in params.items()
            }
            mlflow.log_params(sanitized_params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        try:
            mlflow.log_param(key, str(value)[:250] if value is not None else "None")
        except Exception as e:
            logger.error(f"Error logging parameter {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric name to value
            step: Step number for the metrics
        """
        try:
            # Filter out None and infinite values
            sanitized_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if v is not None and np.isfinite(v)
            }
            mlflow.log_metrics(sanitized_metrics, step=step)
            logger.debug(f"Logged {len(sanitized_metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        try:
            if value is not None and np.isfinite(value):
                mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            logger.error(f"Error logging metric {key}: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ):
        """
        Log a model to MLflow.

        Args:
            model: The model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: If provided, register model with this name
            signature: Model signature
            input_example: Input example for the model
        """
        try:
            # Determine model type and use appropriate logging function
            model_type = type(model).__module__

            # Build kwargs conditionally to avoid None values
            kwargs: Dict[str, Any] = {"registered_model_name": registered_model_name}
            if signature is not None:
                kwargs["signature"] = signature
            if input_example is not None:
                kwargs["input_example"] = input_example

            if "sklearn" in model_type:
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)  # type: ignore
            elif "xgboost" in model_type:
                mlflow.xgboost.log_model(model, artifact_path, **kwargs)  # type: ignore
            elif "lightgbm" in model_type:
                mlflow.lightgbm.log_model(model, artifact_path, **kwargs)  # type: ignore
            else:
                # Fallback to pickle
                mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)  # type: ignore

            logger.info(f"Logged model to {artifact_path}")

        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to local file
            artifact_path: Subdirectory within artifacts directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Subdirectory within artifacts directory
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Name for the artifact file
        """
        try:
            mlflow.log_dict(dictionary, filename)
            logger.debug(f"Logged dictionary as: {filename}")
        except Exception as e:
            logger.error(f"Error logging dictionary: {e}")

    def log_figure(self, figure, filename: str):
        """
        Log a matplotlib/plotly figure.

        Args:
            figure: Figure object
            filename: Name for the artifact file
        """
        try:
            mlflow.log_figure(figure, filename)
            logger.debug(f"Logged figure as: {filename}")
        except Exception as e:
            logger.error(f"Error logging figure: {e}")

    def log_text(self, text: str, filename: str):
        """
        Log text content as an artifact.

        Args:
            text: Text content
            filename: Name for the artifact file
        """
        try:
            mlflow.log_text(text, filename)
            logger.debug(f"Logged text as: {filename}")
        except Exception as e:
            logger.error(f"Error logging text: {e}")

    def set_tag(self, key: str, value: Any):
        """Set a tag on the current run."""
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Error setting tag {key}: {e}")

    def set_tags(self, tags: Dict[str, Any]):
        """Set multiple tags on the current run."""
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Error setting tags: {e}")

    def get_run(self, run_id: Optional[str] = None) -> Optional[Any]:
        """
        Get a run by ID.

        Args:
            run_id: Run ID (uses current run if None)

        Returns:
            Run object or None
        """
        try:
            run_id = run_id or self.current_run_id
            if run_id:
                return self.client.get_run(run_id)
            return None
        except Exception as e:
            logger.error(f"Error getting run: {e}")
            return None

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> pd.DataFrame:
        """
        Search for runs in the current experiment.

        Args:
            filter_string: Filter query string
            order_by: List of order by clauses
            max_results: Maximum number of results

        Returns:
            DataFrame of runs
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
            )
            return runs  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return pd.DataFrame()

    def get_best_run(self, metric: str, ascending: bool = False) -> Optional[Any]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: If True, minimize metric; if False, maximize

        Returns:
            Best run or None
        """
        try:
            runs = self.search_runs(
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1,
            )

            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                return self.client.get_run(run_id)

            return None
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            return None

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Any:
        """
        Register a model in the MLflow Model Registry.

        Args:
            model_uri: URI of the model (e.g., "runs:/<run_id>/model")
            name: Name for the registered model
            tags: Tags for the model version
            description: Description of the model

        Returns:
            ModelVersion object
        """
        try:
            # Register the model
            model_version = mlflow.register_model(model_uri, name)

            # Set description if provided
            if description:
                self.client.update_model_version(
                    name=name, version=model_version.version, description=description
                )

            # Set tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=name, version=model_version.version, key=key, value=value
                    )

            logger.info(f"Registered model: {name} (version: {model_version.version})")

            return model_version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None

    def transition_model_stage(
        self,
        name: str,
        version: Union[int, str],
        stage: str,
        archive_existing: bool = False,
    ):
        """
        Transition a model version to a new stage.

        Args:
            name: Registered model name
            version: Model version number
            stage: New stage (Staging, Production, Archived)
            archive_existing: Archive existing versions in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=str(version),
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(f"Transitioned model {name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")

    def delete_run(self, run_id: str):
        """Delete a run."""
        try:
            self.client.delete_run(run_id)
            logger.info(f"Deleted run: {run_id}")
        except Exception as e:
            logger.error(f"Error deleting run: {e}")

    def delete_experiment(self, experiment_id: Optional[str] = None):
        """Delete an experiment."""
        try:
            exp_id = experiment_id or self.experiment_id
            self.client.delete_experiment(exp_id)
            logger.info(f"Deleted experiment: {exp_id}")
        except Exception as e:
            logger.error(f"Error deleting experiment: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.end_run(status=status)
