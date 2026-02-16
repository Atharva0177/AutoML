"""
Train-test data splitting functionality.

This module provides various strategies for splitting data into train/test/validation sets.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split

from automl.utils.exceptions import ValidationError
from automl.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Split data into train, test, and optionally validation sets.

    Supports:
    - Random splitting
    - Stratified splitting (for classification)
    - Time-based splitting (for time series)
    """

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        stratify: bool = False,
        shuffle: bool = True,
    ):
        """
        Initialize the data splitter.

        Args:
            test_size: Proportion of data to use for testing (0-1)
            validation_size: Proportion of data to use for validation (0-1)
            random_state: Random seed for reproducibility
            stratify: Whether to perform stratified splitting
            shuffle: Whether to shuffle data before splitting
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify = stratify
        self.shuffle = shuffle

        # Validate sizes
        if not 0 < test_size < 1:
            raise ValidationError(f"test_size must be between 0 and 1, got {test_size}")
        if not 0 <= validation_size < 1:
            raise ValidationError(
                f"validation_size must be between 0 and 1, got {validation_size}"
            )
        if test_size + validation_size >= 1:
            raise ValidationError(
                f"test_size ({test_size}) + validation_size ({validation_size}) must be < 1"
            )

    def _get_stratify_column(self, y: pd.Series) -> Optional[pd.Series]:
        """
        Determine if stratification is feasible for the given target.

        Args:
            y: Target Series

        Returns:
            y if stratification is possible, None otherwise
        """
        # Check if target is continuous (regression)
        # If more than 20 unique values or if values are floats, likely regression
        n_unique = y.nunique()

        # If target is numeric and has many unique values, it's likely regression
        if pd.api.types.is_numeric_dtype(y):
            # Check if values are continuous (have decimals)
            unique_values = y.unique()
            has_decimals = any(
                isinstance(val, (float, np.floating)) and val != int(val)
                for val in unique_values
                if not pd.isna(val)
            )

            if has_decimals or n_unique > 20:
                logger.info(
                    f"Disabling stratification: target appears to be continuous "
                    f"({n_unique} unique values, decimals={has_decimals})"
                )
                return None

        # Check class distribution for stratified splitting
        value_counts = y.value_counts()
        min_samples = value_counts.min()

        # Need at least 2 samples per class for stratified splitting
        if min_samples < 2:
            classes_with_one = value_counts[value_counts < 2].index.tolist()
            logger.warning(
                f"Disabling stratification: {len(classes_with_one)} classes have fewer than 2 samples. "
                f"Classes: {classes_with_one[:5]}{'...' if len(classes_with_one) > 5 else ''}"
            )
            return None

        # Check if we have enough samples for the split ratios
        # For stratified split, each class needs at least ceil(n_splits) samples
        n_classes_insufficient = sum(value_counts < max(2, int(1 / self.test_size)))
        if n_classes_insufficient > 0:
            logger.warning(
                f"Disabling stratification: {n_classes_insufficient} classes have insufficient samples "
                f"for test_size={self.test_size}"
            )
            return None

        logger.info(
            f"Stratification enabled: {n_unique} classes, min samples per class: {min_samples}"
        )
        return y

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_col: Optional[str] = None,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
        ],
    ]:
        """
        Split data into train/test or train/validation/test sets.

        Args:
            X: Feature DataFrame
            y: Target Series (optional, can be extracted from X using target_col)
            target_col: Name of target column in X (if y not provided)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or
            (X_train, X_val, X_test, y_train, y_val, y_test) if validation_size > 0
        """
        logger.info(
            f"Splitting data: test_size={self.test_size}, "
            f"validation_size={self.validation_size}, stratify={self.stratify}"
        )

        # Extract target if needed
        if y is None and target_col is not None:
            if target_col not in X.columns:
                raise ValidationError(f"Target column '{target_col}' not found in data")
            y = X[target_col].copy()
            X = X.drop(columns=[target_col])
        elif y is None:
            raise ValidationError("Either y or target_col must be provided")

        # Ensure indices match
        if not X.index.equals(y.index):
            logger.warning("X and y indices don't match, resetting indices")
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        # Determine stratification (check if it's feasible)
        stratify_col = self._get_stratify_column(y) if self.stratify else None

        # Split into train+val and test
        if self.validation_size > 0:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = sklearn_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_col,
                shuffle=self.shuffle,
            )

            # Calculate validation size relative to train+val
            val_size_adjusted = self.validation_size / (1 - self.test_size)

            # Second split: train vs val (check stratification again for the subset)
            stratify_col_temp = (
                self._get_stratify_column(y_temp) if self.stratify else None
            )
            X_train, X_val, y_train, y_val = sklearn_split(
                X_temp,
                y_temp,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=stratify_col_temp,
                shuffle=self.shuffle,
            )

            logger.info(
                f"Split complete: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Simple train/test split
            X_train, X_test, y_train, y_test = sklearn_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_col,
                shuffle=self.shuffle,
            )

            logger.info(f"Split complete: train={len(X_train)}, test={len(X_test)}")

            return X_train, X_test, y_train, y_test

    def temporal_split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
        ],
    ]:
        """
        Split data based on temporal ordering (no shuffling).

        Args:
            X: Feature DataFrame
            y: Target Series
            target_col: Name of target column in X
            date_col: Name of date column to sort by (if None, use index)

        Returns:
            Tuple of split data
        """
        logger.info("Performing temporal split (no shuffling)")

        # Extract target if needed
        if y is None and target_col is not None:
            y = X[target_col].copy()
            X = X.drop(columns=[target_col])
        elif y is None:
            raise ValidationError("Either y or target_col must be provided")

        # Sort by date if specified
        if date_col is not None:
            if date_col not in X.columns:
                raise ValidationError(f"Date column '{date_col}' not found in data")
            X = X.sort_values(by=date_col).reset_index(drop=True)
            y = y.loc[X.index].reset_index(drop=True)

        # Calculate split indices
        n = len(X)
        test_idx = int(n * (1 - self.test_size))

        if self.validation_size > 0:
            val_idx = int(n * (1 - self.test_size - self.validation_size))

            X_train = X.iloc[:val_idx].copy()
            X_val = X.iloc[val_idx:test_idx].copy()
            X_test = X.iloc[test_idx:].copy()

            y_train = y.iloc[:val_idx].copy()
            y_val = y.iloc[val_idx:test_idx].copy()
            y_test = y.iloc[test_idx:].copy()

            logger.info(
                f"Temporal split complete: train={len(X_train)}, "
                f"val={len(X_val)}, test={len(X_test)}"
            )

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train = X.iloc[:test_idx].copy()
            X_test = X.iloc[test_idx:].copy()

            y_train = y.iloc[:test_idx].copy()
            y_test = y.iloc[test_idx:].copy()

            logger.info(
                f"Temporal split complete: train={len(X_train)}, test={len(X_test)}"
            )

            return X_train, X_test, y_train, y_test

    def get_split_summary(self, *split_data) -> Dict[str, Any]:
        """
        Get summary of the split performed.

        Args:
            *split_data: The split data returned from split() or temporal_split()

        Returns:
            Dictionary with split statistics
        """
        if len(split_data) == 4:
            X_train, X_test, y_train, y_test = split_data
            return {
                "num_splits": 2,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_ratio": len(X_train) / (len(X_train) + len(X_test)),
                "test_ratio": len(X_test) / (len(X_train) + len(X_test)),
                "num_features": X_train.shape[1],
                "target_distribution_train": (
                    y_train.value_counts().to_dict()
                    if hasattr(y_train.dtype, "categories") or y_train.nunique() < 20
                    else None
                ),
                "target_distribution_test": (
                    y_test.value_counts().to_dict()
                    if hasattr(y_test.dtype, "categories") or y_test.nunique() < 20
                    else None
                ),
            }
        elif len(split_data) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = split_data
            total = len(X_train) + len(X_val) + len(X_test)
            return {
                "num_splits": 3,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "train_ratio": len(X_train) / total,
                "val_ratio": len(X_val) / total,
                "test_ratio": len(X_test) / total,
                "num_features": X_train.shape[1],
                "target_distribution_train": (
                    y_train.value_counts().to_dict()
                    if hasattr(y_train.dtype, "categories") or y_train.nunique() < 20
                    else None
                ),
                "target_distribution_val": (
                    y_val.value_counts().to_dict()
                    if hasattr(y_val.dtype, "categories") or y_val.nunique() < 20
                    else None
                ),
                "target_distribution_test": (
                    y_test.value_counts().to_dict()
                    if hasattr(y_test.dtype, "categories") or y_test.nunique() < 20
                    else None
                ),
            }
        else:
            raise ValidationError(
                f"Expected 4 or 6 split components, got {len(split_data)}"
            )
