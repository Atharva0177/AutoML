"""Problem type detection for machine learning tasks."""

from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from automl.utils.logger import get_logger

logger = get_logger(__name__)


class ProblemType(Enum):
    """Enumeration of machine learning problem types."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"


class ProblemDetector:
    """Automatically detect the type of machine learning problem."""

    def __init__(self, classification_threshold: int = 20):
        """
        Initialize problem detector.

        Args:
            classification_threshold: Max unique values for classification
        """
        self.classification_threshold = classification_threshold
        self.problem_info: Optional[Dict[str, Any]] = None

    def detect_problem_type(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect the machine learning problem type.

        Args:
            df: Input DataFrame
            target_column: Name of target column (if known)

        Returns:
            Dictionary with problem type information
        """
        logger.info("Detecting problem type...")

        if target_column is None:
            # Try to infer target column
            target_column = self._infer_target_column(df)
            logger.info(f"Inferred target column: {target_column}")

        if target_column is None or target_column not in df.columns:
            problem_info = {
                "problem_type": ProblemType.UNKNOWN.value,
                "target_column": None,
                "recommendation": "Please specify a target column",
            }
            self.problem_info = problem_info
            return problem_info

        target = df[target_column]
        problem_type = self._determine_problem_type(target)

        problem_info = {
            "problem_type": problem_type.value,
            "target_column": target_column,
            "target_dtype": str(target.dtype),
            "n_samples": len(df),
            "n_unique_values": int(target.nunique()),
            "missing_target": int(target.isna().sum()),
            "missing_target_percentage": float(target.isna().sum() / len(target) * 100),
        }

        # Add problem-specific information
        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            problem_info.update(self._analyze_binary_classification(target))
        elif problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            problem_info.update(self._analyze_multiclass_classification(target))
        elif problem_type == ProblemType.REGRESSION:
            problem_info.update(self._analyze_regression(target))

        # Add recommendations
        problem_info["recommendations"] = self._generate_recommendations(problem_info)

        self.problem_info = problem_info
        logger.info(f"Problem type detected: {problem_type.value}")
        return problem_info

    def _infer_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to infer the target column from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Inferred target column name or None
        """
        # Common target column names
        common_targets = [
            "target",
            "label",
            "class",
            "y",
            "output",
            "prediction",
            "outcome",
            "result",
            "value",
        ]

        # Check for exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in common_targets:
                return col

        # Check for partial matches
        for col in df.columns:
            col_lower = col.lower()
            for target in common_targets:
                if target in col_lower:
                    return col

        # If last column is likely a target (not ID-like, not too many unique values)
        last_col = df.columns[-1]
        if df[last_col].nunique() < len(df):
            return last_col

        return None

    def _determine_problem_type(self, target: pd.Series) -> ProblemType:
        """
        Determine problem type based on target column.

        Args:
            target: Target column

        Returns:
            Detected problem type
        """
        n_unique = target.nunique()

        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(target)

        # Binary classification
        if n_unique == 2:
            return ProblemType.BINARY_CLASSIFICATION

        # Multiclass classification
        if n_unique <= self.classification_threshold and not is_numeric:
            return ProblemType.MULTICLASS_CLASSIFICATION

        # Check if numeric with discrete values (could be classification)
        if is_numeric and n_unique <= self.classification_threshold:
            # Check if values are integers
            if target.dropna().apply(lambda x: x == int(x)).all():
                return ProblemType.MULTICLASS_CLASSIFICATION
            else:
                return ProblemType.REGRESSION

        # Regression for continuous numeric
        if is_numeric:
            return ProblemType.REGRESSION

        # High cardinality categorical
        if n_unique > self.classification_threshold:
            return (
                ProblemType.REGRESSION
                if is_numeric
                else ProblemType.MULTICLASS_CLASSIFICATION
            )

        return ProblemType.UNKNOWN

    def _analyze_binary_classification(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze binary classification problem."""
        value_counts = target.value_counts()
        classes = value_counts.index.tolist()

        return {
            "n_classes": 2,
            "classes": [str(c) for c in classes],
            "class_distribution": {str(k): int(v) for k, v in value_counts.items()},
            "is_balanced": self._check_balance(value_counts),
            "majority_class": str(value_counts.index[0]),
            "minority_class": (
                str(value_counts.index[1]) if len(value_counts) > 1 else None
            ),
            "imbalance_ratio": (
                float(value_counts.iloc[0] / value_counts.iloc[1])
                if len(value_counts) > 1
                else 1.0
            ),
        }

    def _analyze_multiclass_classification(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze multiclass classification problem."""
        value_counts = target.value_counts()
        classes = value_counts.index.tolist()

        return {
            "n_classes": len(classes),
            "classes": [str(c) for c in classes[:10]],  # Limit to first 10
            "class_distribution": {
                str(k): int(v) for k, v in value_counts.head(10).items()
            },
            "is_balanced": self._check_balance(value_counts),
            "majority_class": str(value_counts.index[0]),
            "minority_class": str(value_counts.index[-1]),
            "imbalance_ratio": float(value_counts.iloc[0] / value_counts.iloc[-1]),
        }

    def _analyze_regression(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression problem."""
        target_clean = target.dropna()

        return {
            "min": float(target_clean.min()) if len(target_clean) > 0 else None,
            "max": float(target_clean.max()) if len(target_clean) > 0 else None,
            "mean": float(target_clean.mean()) if len(target_clean) > 0 else None,
            "median": float(target_clean.median()) if len(target_clean) > 0 else None,
            "std": float(target_clean.std()) if len(target_clean) > 0 else None,
            "skewness": float(target_clean.skew()) if len(target_clean) > 0 else None,  # type: ignore[arg-type]
            "kurtosis": float(target_clean.kurtosis()) if len(target_clean) > 0 else None,  # type: ignore[arg-type]
            "is_positive_only": (
                bool((target_clean > 0).all()) if len(target_clean) > 0 else False
            ),
            "has_zeros": (
                bool((target_clean == 0).any()) if len(target_clean) > 0 else False
            ),
        }

    def _check_balance(self, value_counts: pd.Series, threshold: float = 0.3) -> bool:
        """
        Check if classes are balanced.

        Args:
            value_counts: Series with class counts
            threshold: Imbalance threshold (e.g., 0.3 means 30% difference)

        Returns:
            True if balanced, False otherwise
        """
        if len(value_counts) < 2:
            return True

        min_count = value_counts.min()
        max_count = value_counts.max()

        ratio = min_count / max_count
        return ratio >= (1 - threshold)

    def _generate_recommendations(self, problem_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on problem type."""
        recommendations = []

        problem_type = problem_info["problem_type"]

        # Missing target values
        if problem_info.get("missing_target", 0) > 0:
            recommendations.append(
                f"Target column has {problem_info['missing_target']} missing values. "
                "Consider removing or imputing these rows."
            )

        # Imbalanced classification
        if "is_balanced" in problem_info and not problem_info["is_balanced"]:
            imbalance_ratio = problem_info.get("imbalance_ratio", 1.0)
            recommendations.append(
                f"Classes are imbalanced (ratio: {imbalance_ratio:.2f}). "
                "Consider using stratified sampling, class weights, or resampling techniques."
            )

        # High cardinality multiclass
        if problem_type == "multiclass_classification":
            n_classes = problem_info.get("n_classes", 0)
            if n_classes > 10:
                recommendations.append(
                    f"High number of classes ({n_classes}). "
                    "Consider grouping rare classes or using hierarchical classification."
                )

        # Regression with skewness
        if problem_type == "regression":
            skewness = problem_info.get("skewness")
            if skewness and abs(skewness) > 1.0:
                recommendations.append(
                    f"Target variable is skewed (skewness: {skewness:.2f}). "
                    "Consider log transformation or Box-Cox transformation."
                )

        # Small dataset
        n_samples = problem_info.get("n_samples", 0)
        if n_samples < 100:
            recommendations.append(
                f"Small dataset ({n_samples} samples). "
                "Use cross-validation and be cautious of overfitting."
            )

        return recommendations

    def get_suggested_metrics(self) -> List[str]:
        """
        Get suggested evaluation metrics for the detected problem type.

        Returns:
            List of metric names
        """
        if self.problem_info is None:
            return []

        problem_type = self.problem_info["problem_type"]

        metrics_map = {
            "binary_classification": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "log_loss",
            ],
            "multiclass_classification": [
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "log_loss",
                "confusion_matrix",
            ],
            "regression": ["mse", "rmse", "mae", "r2_score", "mape"],
        }

        return metrics_map.get(problem_type, ["accuracy"])

    def get_suggested_models(self) -> List[str]:
        """
        Get suggested models for the detected problem type.

        Returns:
            List of model names
        """
        if self.problem_info is None:
            return []

        problem_type = self.problem_info["problem_type"]
        n_samples = self.problem_info.get("n_samples", 0)

        # Base models for each problem type
        models_map = {
            "binary_classification": [
                "LogisticRegression",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "LGBMClassifier",
            ],
            "multiclass_classification": [
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "LGBMClassifier",
            ],
            "regression": [
                "LinearRegression",
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "XGBRegressor",
                "LGBMRegressor",
            ],
        }

        models = models_map.get(problem_type, [])

        # Filter based on dataset size
        if n_samples < 1000:
            # Prefer simpler models for small datasets
            models = [m for m in models if "Forest" not in m or "Gradient" not in m][:3]

        return models
