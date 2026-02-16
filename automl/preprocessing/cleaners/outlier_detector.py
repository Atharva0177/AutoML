"""
Outlier detection and handling.

This module provides methods for detecting and handling outliers:
- IQR (Interquartile Range): Statistical method using quartiles
- Isolation Forest: ML-based anomaly detection
- Standard deviation: Multiple of standard deviations from mean
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from automl.utils.exceptions import ValidationError
from automl.utils.logger import get_logger

logger = get_logger(__name__)

OutlierStrategy = Literal["iqr", "isolation_forest", "zscore"]
OutlierAction = Literal["remove", "cap", "flag"]


class OutlierDetector:
    """
    Detect and handle outliers in datasets.

    Strategies:
    - IQR: Uses Interquartile Range (Q3 - Q1) to detect outliers
    - Isolation Forest: ML-based anomaly detection
    - Z-Score: Standard deviation-based method

    Actions:
    - Remove: Drop rows containing outliers
    - Cap: Cap outliers to threshold values (Winsorization)
    - Flag: Add binary column indicating outlier status
    """

    def __init__(
        self,
        strategy: OutlierStrategy = "iqr",
        action: OutlierAction = "cap",
        threshold: float = 1.5,
        contamination: float = 0.1,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ):
        """
        Initialize the outlier detector.

        Args:
            strategy: Detection strategy ('iqr', 'isolation_forest', 'zscore')
            action: How to handle outliers ('remove', 'cap', 'flag')
            threshold: IQR multiplier (default 1.5) or z-score threshold (default 3.0)
            contamination: Expected proportion of outliers (for Isolation Forest)
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for the detector
        """
        self.strategy = strategy
        self.action = action
        self.threshold = (
            threshold if strategy != "zscore" else kwargs.get("zscore_threshold", 3.0)
        )
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs

        self.detector: Optional[IsolationForest] = None
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.fitted_columns: List[str] = []
        self.outlier_counts: Dict[str, int] = {}
        self.is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit the detector and transform the data.

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names to check

        Returns:
            Transformed DataFrame with outliers handled
        """
        logger.info(f"Fitting outlier detector with strategy: {self.strategy}")

        df_copy = df.copy()

        # Auto-detect numerical columns if not provided
        if numerical_cols is None:
            numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in df_copy.columns]

        if not numerical_cols:
            logger.warning("No numerical columns found for outlier detection")
            self.is_fitted = True
            return df_copy

        self.fitted_columns = numerical_cols

        # Detect outliers based on strategy
        if self.strategy == "iqr":
            df_result = self._fit_transform_iqr(df_copy, numerical_cols)
        elif self.strategy == "isolation_forest":
            df_result = self._fit_transform_isolation_forest(df_copy, numerical_cols)
        elif self.strategy == "zscore":
            df_result = self._fit_transform_zscore(df_copy, numerical_cols)
        else:
            raise ValidationError(f"Unknown strategy: {self.strategy}")

        self.is_fitted = True
        logger.info(f"Outlier detection complete. Shape: {df_result.shape}")

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted detector.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValidationError("Detector must be fitted before transform")

        df_copy = df.copy()

        if self.strategy == "iqr" or self.strategy == "zscore":
            # Apply bounds from fit
            df_result = self._apply_bounds(df_copy, self.fitted_columns)
        elif self.strategy == "isolation_forest":
            if self.detector is None:
                raise ValidationError("Isolation Forest detector not fitted")
            df_result = self._apply_isolation_forest(df_copy, self.fitted_columns)
        else:
            df_result = df_copy

        return df_result

    def _fit_transform_iqr(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        logger.info(f"Applying IQR method with threshold={self.threshold}")

        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in numerical_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define bounds
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR

            self.bounds[col] = (lower_bound, upper_bound)

            # Identify outliers
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = col_outliers.sum()
            self.outlier_counts[col] = outlier_count

            if outlier_count > 0:
                logger.info(
                    f"  {col}: {outlier_count} outliers "
                    f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

            outlier_mask |= col_outliers

        total_outliers = outlier_mask.sum()
        logger.info(
            f"Total rows with outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)"
        )

        # Apply action
        df_result = self._apply_action(df, outlier_mask, numerical_cols)

        return df_result

    def _fit_transform_zscore(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Detect outliers using Z-score method."""
        logger.info(f"Applying Z-score method with threshold={self.threshold}")

        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in numerical_cols:
            # Calculate mean and std
            mean = df[col].mean()
            std = df[col].std()

            if std == 0:
                logger.warning(f"  {col}: std=0, skipping")
                continue

            # Define bounds based on z-score
            lower_bound = mean - self.threshold * std
            upper_bound = mean + self.threshold * std

            self.bounds[col] = (lower_bound, upper_bound)

            # Identify outliers
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = col_outliers.sum()
            self.outlier_counts[col] = outlier_count

            if outlier_count > 0:
                logger.info(
                    f"  {col}: {outlier_count} outliers "
                    f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

            outlier_mask |= col_outliers

        total_outliers = outlier_mask.sum()
        logger.info(
            f"Total rows with outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)"
        )

        # Apply action
        df_result = self._apply_action(df, outlier_mask, numerical_cols)

        return df_result

    def _fit_transform_isolation_forest(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Detect outliers using Isolation Forest."""
        logger.info(
            f"Applying Isolation Forest with contamination={self.contamination}"
        )

        # Extract numerical features
        X = df[numerical_cols].values

        # Initialize and fit Isolation Forest
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.kwargs,
        )

        # Predict (-1 for outliers, 1 for inliers)
        predictions = self.detector.fit_predict(X)
        outlier_mask = predictions == -1

        outlier_count = outlier_mask.sum()
        logger.info(
            f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)"
        )

        # Store outlier counts per column (approximate)
        for col in numerical_cols:
            self.outlier_counts[col] = outlier_count  # Rough estimate

        # Apply action
        df_result = self._apply_action(
            df, pd.Series(outlier_mask, index=df.index), numerical_cols
        )

        return df_result

    def _apply_action(
        self, df: pd.DataFrame, outlier_mask: pd.Series, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply the specified action to handle outliers."""
        if self.action == "remove":
            # Remove rows with outliers
            rows_before = len(df)
            df_result = df[~outlier_mask].copy()
            rows_removed = rows_before - len(df_result)
            logger.info(f"Removed {rows_removed} rows containing outliers")

        elif self.action == "cap":
            # Cap values to bounds (Winsorization)
            df_result = df.copy()
            capped_count = 0

            for col in numerical_cols:
                if col not in self.bounds:
                    continue

                lower_bound, upper_bound = self.bounds[col]

                # Cap lower outliers
                lower_outliers = df_result[col] < lower_bound
                df_result.loc[lower_outliers, col] = lower_bound

                # Cap upper outliers
                upper_outliers = df_result[col] > upper_bound
                df_result.loc[upper_outliers, col] = upper_bound

                col_capped = lower_outliers.sum() + upper_outliers.sum()
                if col_capped > 0:
                    capped_count += col_capped
                    logger.info(f"  Capped {col_capped} values in {col}")

            logger.info(f"Total values capped: {capped_count}")

        elif self.action == "flag":
            # Add flag column indicating outliers
            df_result = df.copy()
            df_result["is_outlier"] = outlier_mask.astype(int)
            logger.info(f"Added 'is_outlier' flag column")

        else:
            raise ValidationError(f"Unknown action: {self.action}")

        return df_result

    def _apply_bounds(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply fitted bounds to new data."""
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in numerical_cols:
            if col not in self.bounds:
                continue

            lower_bound, upper_bound = self.bounds[col]
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask |= col_outliers

        df_result = self._apply_action(df, outlier_mask, numerical_cols)
        return df_result

    def _apply_isolation_forest(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply fitted Isolation Forest to new data."""
        if self.detector is None:
            raise ValidationError("Isolation Forest not fitted")

        X = df[numerical_cols].values
        predictions = self.detector.predict(X)
        outlier_mask = predictions == -1

        df_result = self._apply_action(
            df, pd.Series(outlier_mask, index=df.index), numerical_cols
        )
        return df_result

    def get_outlier_summary(self) -> Dict[str, Any]:
        """Get summary of outlier detection."""
        return {
            "strategy": self.strategy,
            "action": self.action,
            "threshold": self.threshold,
            "contamination": (
                self.contamination if self.strategy == "isolation_forest" else None
            ),
            "fitted_columns": self.fitted_columns,
            "outlier_counts": self.outlier_counts,
            "bounds": self.bounds if self.bounds else None,
            "is_fitted": self.is_fitted,
        }

    def get_outlier_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a detailed outlier report for a dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with outlier statistics per column
        """
        if not self.is_fitted:
            raise ValidationError("Detector must be fitted before generating report")

        report_data = []

        for col in self.fitted_columns:
            if col not in df.columns:
                continue

            col_data = {
                "column": col,
                "total_values": len(df),
                "outlier_count": self.outlier_counts.get(col, 0),
                "outlier_percentage": self.outlier_counts.get(col, 0) / len(df) * 100,
            }

            if col in self.bounds:
                lower, upper = self.bounds[col]
                col_data["lower_bound"] = lower
                col_data["upper_bound"] = upper
                col_data["min_value"] = df[col].min()
                col_data["max_value"] = df[col].max()

            report_data.append(col_data)

        return pd.DataFrame(report_data)
