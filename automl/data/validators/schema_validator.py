"""Data validation utilities."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from automl.config.config import config
from automl.utils.exceptions import DataValidationError, InsufficientDataError
from automl.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates loaded data for quality and completeness."""

    def __init__(self):
        """Initialize validator with configuration."""
        self.min_rows = config.get("data.min_rows", 10)
        self.min_columns = config.get("data.min_columns", 2)
        self.max_missing_ratio = config.get("data.validation.max_missing_ratio", 0.9)

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Comprehensive data validation.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues = []

        # Check dimensions
        dim_issues = self._validate_dimensions(df)
        issues.extend(dim_issues)

        # Check for empty data
        if df.empty:
            issues.append("DataFrame is empty")

        # Check for excessive missing values
        missing_issues = self._validate_missing_values(df)
        issues.extend(missing_issues)

        # Check for duplicate rows
        dup_issues = self._validate_duplicates(df)
        issues.extend(dup_issues)

        # Check for constant columns
        const_issues = self._validate_constant_columns(df)
        issues.extend(const_issues)

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Data validation found {len(issues)} issue(s)")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Data validation passed")

        return is_valid, issues

    def _validate_dimensions(self, df: pd.DataFrame) -> List[str]:
        """Validate DataFrame dimensions."""
        issues = []

        if len(df) < self.min_rows:
            issues.append(f"Insufficient rows: {len(df)} (minimum: {self.min_rows})")

        if len(df.columns) < self.min_columns:
            issues.append(
                f"Insufficient columns: {len(df.columns)} (minimum: {self.min_columns})"
            )

        return issues

    def _validate_missing_values(self, df: pd.DataFrame) -> List[str]:
        """Validate missing values."""
        issues = []

        # Overall missing ratio
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0

        if missing_ratio > self.max_missing_ratio:
            issues.append(
                f"Excessive missing values: {missing_ratio:.1%} "
                f"(maximum: {self.max_missing_ratio:.1%})"
            )

        # Columns with all missing values
        all_missing = df.columns[df.isnull().all()].tolist()
        if all_missing:
            issues.append(f"Columns with all missing values: {', '.join(all_missing)}")

        return issues

    def _validate_duplicates(self, df: pd.DataFrame) -> List[str]:
        """Check for duplicate rows."""
        issues = []

        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            dup_ratio = n_duplicates / len(df)
            issues.append(f"Found {n_duplicates} duplicate rows ({dup_ratio:.1%})")

        return issues

    def _validate_constant_columns(self, df: pd.DataFrame) -> List[str]:
        """Check for columns with constant values."""
        issues = []

        constant_cols = []
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)

        if constant_cols:
            issues.append(f"Columns with constant values: {', '.join(constant_cols)}")

        return issues

    def validate_target(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate target column.

        Args:
            df: DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check if column exists
        if target_column not in df.columns:
            issues.append(f"Target column '{target_column}' not found in data")
            return False, issues

        target = df[target_column]

        # Check for all missing
        if target.isnull().all():
            issues.append(f"Target column '{target_column}' has all missing values")

        # Check for constant values
        if target.nunique(dropna=True) <= 1:
            issues.append(f"Target column '{target_column}' has constant values")

        # Check for excessive missing
        missing_ratio = target.isnull().mean()
        if missing_ratio > 0.5:
            issues.append(
                f"Target column has {missing_ratio:.1%} missing values "
                f"(more than 50%)"
            )

        is_valid = len(issues) == 0
        return is_valid, issues


class SchemaValidator:
    """Validates data schema and types."""

    def infer_schema(self, df: pd.DataFrame) -> Dict:
        """
        Infer schema from DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Schema dictionary
        """
        schema = {}

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "inferred_type": self._infer_column_type(df[col]),
                "nullable": df[col].isnull().any(),
                "unique_count": df[col].nunique(),
                "sample_values": df[col].dropna().head(5).tolist(),
            }
            schema[col] = col_info

        return schema

    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer semantic type of column.

        Args:
            series: Pandas Series

        Returns:
            Inferred type: 'numeric', 'categorical', 'datetime', 'text', 'boolean'
        """
        # Boolean
        if series.dtype == bool or set(series.dropna().unique()).issubset({0, 1}):
            return "boolean"

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Try to parse as datetime
        if series.dtype == object:
            try:
                pd.to_datetime(series.dropna().head(100))
                return "datetime"
            except:
                pass

        # Categorical vs Text
        if series.dtype == object or series.dtype.name == "category":
            unique_ratio = series.nunique() / len(series)
            # If less than 50% unique values, likely categorical
            if unique_ratio < 0.5:
                return "categorical"
            else:
                return "text"

        return "unknown"
