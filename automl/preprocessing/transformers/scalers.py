"""
Numerical scaling transformations.

This module provides various scaling strategies for numerical features.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from automl.utils.exceptions import ValidationError
from automl.utils.logger import get_logger

logger = get_logger(__name__)

ScalingMethod = Literal["standard", "minmax", "robust", "maxabs", "none"]


class NumericalScaler:
    """
    Scale numerical features using various scaling methods.

    Supports:
    - StandardScaler: Standardize features by removing mean and scaling to unit variance
    - MinMaxScaler: Scale features to a given range (default [0, 1])
    - RobustScaler: Scale using statistics robust to outliers (median, IQR)
    - MaxAbsScaler: Scale by maximum absolute value
    """

    def __init__(
        self,
        method: ScalingMethod = "standard",
        feature_range: Tuple[int, int] = (0, 1),
    ):
        """
        Initialize the numerical scaler.

        Args:
            method: Scaling method to use
            feature_range: Range for MinMaxScaler (min, max)
        """
        self.method = method
        self.feature_range = feature_range
        self.scaler: Optional[object] = None
        self.numerical_cols: List[str] = []
        self.scaling_params: Dict[str, Dict[str, float]] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit the scaler and transform the data.

        Args:
            df: Input DataFrame
            columns: List of columns to scale (if None, scale all numerical)

        Returns:
            Transformed DataFrame with scaled values
        """
        if self.method == "none":
            logger.info("Scaling method is 'none', returning original data")
            return df.copy()

        logger.info(f"Fitting numerical scaler with method: {self.method}")

        df_copy = df.copy()

        # Determine columns to scale
        if columns is None:
            self.numerical_cols = df_copy.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        else:
            self.numerical_cols = columns

        if not self.numerical_cols:
            logger.warning("No numerical columns found to scale")
            return df_copy

        # Initialize scaler based on method
        self.scaler = self._get_scaler()

        # Fit and transform
        df_copy[self.numerical_cols] = self.scaler.fit_transform(df_copy[self.numerical_cols])  # type: ignore[union-attr]

        # Store scaling parameters
        self._store_scaling_params()

        logger.info(
            f"Scaled {len(self.numerical_cols)} numerical columns using {self.method}"
        )
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scaler.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if self.scaler is None and self.method != "none":
            raise ValidationError("Scaler must be fitted before transform")

        if self.method == "none":
            return df.copy()

        df_copy = df.copy()

        # Only transform columns that were fitted
        cols_to_transform = [
            col for col in self.numerical_cols if col in df_copy.columns
        ]

        if cols_to_transform:
            df_copy[cols_to_transform] = self.scaler.transform(df_copy[cols_to_transform])  # type: ignore[union-attr]

        return df_copy

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.

        Args:
            df: Scaled DataFrame

        Returns:
            DataFrame in original scale
        """
        if self.scaler is None and self.method != "none":
            raise ValidationError("Scaler must be fitted before inverse_transform")

        if self.method == "none":
            return df.copy()

        df_copy = df.copy()

        # Only inverse transform columns that were fitted
        cols_to_transform = [
            col for col in self.numerical_cols if col in df_copy.columns
        ]

        if cols_to_transform:
            df_copy[cols_to_transform] = self.scaler.inverse_transform(df_copy[cols_to_transform])  # type: ignore[union-attr]

        return df_copy

    def _get_scaler(self) -> object:
        """Get the appropriate scaler based on method."""
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == "robust":
            return RobustScaler()
        elif self.method == "maxabs":
            return MaxAbsScaler()
        else:
            raise ValidationError(f"Unknown scaling method: {self.method}")

    def _store_scaling_params(self) -> None:
        """Store scaling parameters for each column."""
        if self.scaler is None:
            return

        for i, col in enumerate(self.numerical_cols):
            params: Dict[str, float] = {}

            if self.method == "standard":
                scaler = self.scaler  # type: ignore[assignment]
                params["mean"] = float(scaler.mean_[i])  # type: ignore[attr-defined]
                params["std"] = float(scaler.scale_[i])  # type: ignore[attr-defined]
            elif self.method == "minmax":
                scaler = self.scaler  # type: ignore[assignment]
                params["min"] = float(scaler.data_min_[i])  # type: ignore[attr-defined]
                params["max"] = float(scaler.data_max_[i])  # type: ignore[attr-defined]
                params["range_min"] = self.feature_range[0]
                params["range_max"] = self.feature_range[1]
            elif self.method == "robust":
                scaler = self.scaler  # type: ignore[assignment]
                params["center"] = float(scaler.center_[i])  # type: ignore[attr-defined]
                params["scale"] = float(scaler.scale_[i])  # type: ignore[attr-defined]
            elif self.method == "maxabs":
                scaler = self.scaler  # type: ignore[assignment]
                params["max_abs"] = float(scaler.max_abs_[i])  # type: ignore[attr-defined]

            self.scaling_params[col] = params

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of scaling performed."""
        return {
            "method": self.method,
            "feature_range": self.feature_range,
            "numerical_cols": self.numerical_cols,
            "scaling_params": self.scaling_params,
        }
