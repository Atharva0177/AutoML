"""
Metrics Calculation Module.

This module provides comprehensive metric calculation for both
classification and regression tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # Classification metrics; Regression metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculator for classification and regression metrics.

    Provides methods to compute comprehensive metrics for model evaluation.
    """

    @staticmethod
    def calculate_classification_metrics(
        y_true: Union[NDArray[Any], List[Any]],
        y_pred: Union[NDArray[Any], List[Any]],
        y_pred_proba: Optional[NDArray[np.float64]] = None,
        average: str = "weighted",
        labels: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            average: Averaging strategy for multi-class ('micro', 'macro', 'weighted')
            labels: List of labels to include in metrics

        Returns:
            Dictionary containing all classification metrics
        """
        metrics: Dict[str, Any] = {}

        # Convert to numpy arrays if needed
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))

        # Handle binary vs multi-class
        unique_labels = np.unique(y_true_arr)
        n_classes = len(unique_labels)

        # Precision, Recall, F1
        if n_classes == 2:
            # Binary classification
            metrics["precision"] = float(
                precision_score(
                    y_true_arr, y_pred_arr, average="binary", zero_division=0
                )
            )
            metrics["recall"] = float(
                recall_score(y_true_arr, y_pred_arr, average="binary", zero_division=0)
            )
            metrics["f1_score"] = float(
                f1_score(y_true_arr, y_pred_arr, average="binary", zero_division=0)
            )
        else:
            # Multi-class classification
            metrics["precision"] = float(
                precision_score(
                    y_true_arr, y_pred_arr, average=average, zero_division=0
                )
            )
            metrics["recall"] = float(
                recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)
            )
            metrics["f1_score"] = float(
                f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)
            )

        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    # Binary: use probabilities of positive class
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true_arr, y_pred_proba[:, 1])
                    )
                else:
                    # Multi-class: use one-vs-rest
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true_arr, y_pred_proba, multi_class="ovr", average=average
                        )
                    )

                # Log loss
                metrics["log_loss"] = float(log_loss(y_true_arr, y_pred_proba))
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC/Log Loss: {e}")

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(
            y_true_arr, y_pred_arr, labels=labels
        ).tolist()

        # Per-class metrics
        if labels is None:
            labels = unique_labels.tolist()

        report = classification_report(
            y_true_arr, y_pred_arr, labels=labels, output_dict=True, zero_division=0
        )
        metrics["per_class_metrics"] = report

        logger.info(
            f"Classification metrics: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_score']:.4f}"
        )

        return metrics

    @staticmethod
    def calculate_regression_metrics(
        y_true: Union[NDArray[Any], List[Any]], y_pred: Union[NDArray[Any], List[Any]]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary containing all regression metrics
        """
        # Convert to numpy arrays
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        metrics: Dict[str, float] = {}

        # Mean Squared Error and Root Mean Squared Error
        mse = mean_squared_error(y_true_arr, y_pred_arr)
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(np.sqrt(mse))

        # Mean Absolute Error
        metrics["mae"] = float(mean_absolute_error(y_true_arr, y_pred_arr))

        # R² Score
        metrics["r2_score"] = float(r2_score(y_true_arr, y_pred_arr))

        # Mean Absolute Percentage Error
        try:
            metrics["mape"] = float(
                mean_absolute_percentage_error(y_true_arr, y_pred_arr)
            )
        except Exception:
            # MAPE can fail if y_true contains zeros
            metrics["mape"] = float("nan")

        # Median Absolute Error
        metrics["median_ae"] = float(median_absolute_error(y_true_arr, y_pred_arr))

        # Adjusted R² (requires number of features)
        # Will be calculated separately if needed

        logger.info(
            f"Regression metrics: RMSE={metrics['rmse']:.4f}, "
            f"R²={metrics['r2_score']:.4f}"
        )

        return metrics

    @staticmethod
    def get_primary_metric(model_type: str, metrics: Dict[str, Any]) -> float:
        """
        Get the primary metric for model comparison.

        Args:
            model_type: 'classification' or 'regression'
            metrics: Dictionary of calculated metrics

        Returns:
            Primary metric value
        """
        if model_type == "classification":
            return float(metrics.get("f1_score", metrics.get("accuracy", 0.0)))
        else:  # regression
            return float(metrics.get("r2_score", 0.0))

    @staticmethod
    def format_metrics(
        metrics: Dict[str, Any], model_type: str, decimals: int = 4
    ) -> str:
        """
        Format metrics as a readable string.

        Args:
            metrics: Dictionary of metrics
            model_type: 'classification' or 'regression'
            decimals: Number of decimal places

        Returns:
            Formatted string
        """
        if model_type == "classification":
            lines = [
                f"Accuracy:  {metrics.get('accuracy', 0):.{decimals}f}",
                f"Precision: {metrics.get('precision', 0):.{decimals}f}",
                f"Recall:    {metrics.get('recall', 0):.{decimals}f}",
                f"F1 Score:  {metrics.get('f1_score', 0):.{decimals}f}",
            ]
            if "roc_auc" in metrics:
                lines.append(f"ROC-AUC:   {metrics['roc_auc']:.{decimals}f}")
        else:  # regression
            lines = [
                f"RMSE:      {metrics.get('rmse', 0):.{decimals}f}",
                f"MAE:       {metrics.get('mae', 0):.{decimals}f}",
                f"R² Score:  {metrics.get('r2_score', 0):.{decimals}f}",
                f"MAPE:      {metrics.get('mape', 0):.{decimals}f}",
            ]

        return "\n".join(lines)
