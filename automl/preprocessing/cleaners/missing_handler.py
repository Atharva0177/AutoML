"""
Missing value imputation handler.

This module provides various strategies for handling missing values in datasets.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from automl.utils.exceptions import ValidationError
from automl.utils.logger import get_logger

logger = get_logger(__name__)

ImputationStrategy = Literal["mean", "median", "mode", "constant", "drop"]


class MissingValueHandler:
    """
    Handle missing values in datasets using various imputation strategies.

    Supports:
    - Mean imputation (numerical columns)
    - Median imputation (numerical columns)
    - Mode imputation (categorical columns)
    - Constant value imputation
    - Drop rows/columns with missing values
    """

    def __init__(
        self,
        strategy: ImputationStrategy = "mean",
        fill_value: Optional[Union[int, float, str]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the missing value handler.

        Args:
            strategy: Imputation strategy to use
            fill_value: Value to use for constant strategy
            threshold: Threshold for dropping columns (if > threshold missing, drop)
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.threshold = threshold
        self.numerical_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.columns_dropped: List[str] = []
        self.imputation_values: Dict[str, Union[int, float, str]] = {}
        self.fitted_numerical_cols: List[str] = []
        self.fitted_categorical_cols: List[str] = []
        self.is_fitted = False  # Track if fit_transform has been called

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit the imputer and transform the data.

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names

        Returns:
            Transformed DataFrame with imputed values
        """
        logger.info(f"Fitting missing value handler with strategy: {self.strategy}")

        df_copy = df.copy()

        # Auto-detect column types if not provided
        if numerical_cols is None or categorical_cols is None:
            numerical_cols, categorical_cols = self._detect_column_types(df_copy)

        # Log missing value statistics
        self._log_missing_stats(df_copy)

        # Handle based on strategy
        if self.strategy == "drop":
            df_copy = self._drop_missing(df_copy)
        else:
            # Impute numerical columns
            if numerical_cols:
                df_copy = self._impute_numerical(df_copy, numerical_cols)

            # Impute categorical columns
            if categorical_cols:
                df_copy = self._impute_categorical(df_copy, categorical_cols)

        self.is_fitted = True  # Mark as fitted
        logger.info(f"Missing value imputation complete. Shape: {df_copy.shape}")
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted imputers.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValidationError("Handler must be fitted before transform")

        df_copy = df.copy()

        # Drop columns that were dropped during fit
        if self.columns_dropped:
            df_copy = df_copy.drop(columns=self.columns_dropped, errors="ignore")

        if self.strategy == "drop":
            df_copy = df_copy.dropna()
            return df_copy

        # Transform numerical columns
        if self.numerical_imputer is not None and self.fitted_numerical_cols:
            # Only transform columns that were fitted
            cols_to_transform = [
                col for col in self.fitted_numerical_cols if col in df_copy.columns
            ]
            if cols_to_transform:
                df_copy[cols_to_transform] = self.numerical_imputer.transform(
                    df_copy[cols_to_transform]
                )

        # Transform categorical columns
        if self.categorical_imputer is not None and self.fitted_categorical_cols:
            # Only transform columns that were fitted
            cols_to_transform = [
                col for col in self.fitted_categorical_cols if col in df_copy.columns
            ]
            if cols_to_transform:
                df_copy[cols_to_transform] = self.categorical_imputer.transform(
                    df_copy[cols_to_transform]
                )

        return df_copy

    def _detect_column_types(self, df: pd.DataFrame) -> tuple[List[str], List[str]]:
        """Detect numerical and categorical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.debug(
            f"Detected {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns"
        )
        return numerical_cols, categorical_cols

    def _log_missing_stats(self, df: pd.DataFrame) -> None:
        """Log missing value statistics."""
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if len(missing_cols) > 0:
            logger.info(f"Found missing values in {len(missing_cols)} columns:")
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                logger.info(f"  - {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("No missing values found")

    def _drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows or columns with missing values based on threshold."""
        # Drop columns with missing values above threshold
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > self.threshold].index.tolist()

        if cols_to_drop:
            logger.info(
                f"Dropping {len(cols_to_drop)} columns with >{self.threshold*100}% missing"
            )
            self.columns_dropped = cols_to_drop
            df = df.drop(columns=cols_to_drop)

        # Drop remaining rows with any missing values
        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)

        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with missing values")

        return df

    def _impute_numerical(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Impute numerical columns."""
        if not numerical_cols:
            return df

        # Filter to columns that exist and have missing values
        numerical_cols = [
            col
            for col in numerical_cols
            if col in df.columns and df[col].isnull().any()
        ]

        if not numerical_cols:
            return df

        # Store fitted columns
        self.fitted_numerical_cols = numerical_cols

        if self.strategy == "constant":
            fill_val = self.fill_value if self.fill_value is not None else 0
            self.numerical_imputer = SimpleImputer(
                strategy="constant", fill_value=fill_val
            )
        elif self.strategy in ["mean", "median"]:
            self.numerical_imputer = SimpleImputer(strategy=self.strategy)
        else:
            # For mode, use most_frequent
            self.numerical_imputer = SimpleImputer(strategy="most_frequent")

        # Fit and transform
        df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])

        # Store imputation values
        if hasattr(self.numerical_imputer, "statistics_"):
            for col, val in zip(numerical_cols, self.numerical_imputer.statistics_):
                self.imputation_values[col] = float(val) if not np.isnan(val) else 0.0

        logger.info(
            f"Imputed {len(numerical_cols)} numerical columns using {self.strategy}"
        )
        return df

    def _impute_categorical(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Impute categorical columns."""
        if not categorical_cols:
            return df

        # Filter to columns that exist and have missing values
        categorical_cols = [
            col
            for col in categorical_cols
            if col in df.columns and df[col].isnull().any()
        ]

        if not categorical_cols:
            return df

        # Store fitted columns
        self.fitted_categorical_cols = categorical_cols

        if self.strategy == "constant":
            fill_val = self.fill_value if self.fill_value is not None else "missing"
            self.categorical_imputer = SimpleImputer(
                strategy="constant", fill_value=fill_val
            )
        else:
            # For categorical, always use most_frequent (mode)
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")

        # Fit and transform
        df[categorical_cols] = self.categorical_imputer.fit_transform(
            df[categorical_cols]
        )

        # Store imputation values
        if hasattr(self.categorical_imputer, "statistics_"):
            for col, val in zip(categorical_cols, self.categorical_imputer.statistics_):
                self.imputation_values[col] = str(val)

        logger.info(f"Imputed {len(categorical_cols)} categorical columns")
        return df

    def get_imputation_summary(self) -> Dict[str, Any]:
        """Get summary of imputation performed."""
        return {
            "strategy": self.strategy,
            "imputation_values": self.imputation_values,
            "columns_dropped": self.columns_dropped,
            "fill_value": self.fill_value,
            "threshold": self.threshold,
        }
