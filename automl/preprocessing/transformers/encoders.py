"""
Categorical encoding transformations.

This module provides various encoding strategies for categorical features.
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from automl.utils.exceptions import ValidationError
from automl.utils.logger import get_logger

logger = get_logger(__name__)

EncodingMethod = Literal["onehot", "label", "ordinal", "none"]


class CategoricalEncoder:
    """
    Encode categorical features using various encoding methods.

    Supports:
    - OneHot: Create binary columns for each category
    - Label: Convert categories to integer labels
    - Ordinal: Map categories to ordered integers
    """

    def __init__(
        self,
        method: EncodingMethod = "onehot",
        handle_unknown: Literal["error", "ignore"] = "ignore",
        max_categories: int = 50,
    ):
        """
        Initialize the categorical encoder.

        Args:
            method: Encoding method to use
            handle_unknown: How to handle unknown categories during transform
            max_categories: Maximum unique categories for onehot encoding
        """
        self.method = method
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self.encoders: Dict[str, object] = {}
        self.categorical_cols: List[str] = []
        self.encoded_columns: List[str] = []
        self.category_mappings: Dict[str, Dict] = {}

    @staticmethod
    def _sanitize_feature_name(name: str) -> str:
        """
        Sanitize feature names to be compatible with all ML libraries.

        LightGBM and some other libraries don't support certain special characters
        in feature names. This method replaces problematic characters with underscores.

        Args:
            name: Original feature name

        Returns:
            Sanitized feature name safe for all ML libraries
        """
        import re

        # Replace special characters with underscores
        # Keep only alphanumeric, underscore, hyphen, and space
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\s]", "_", str(name))
        # Replace multiple underscores/spaces with single underscore
        sanitized = re.sub(r"[_\s]+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        return sanitized

    @staticmethod
    def _make_unique_column_names(names: List[str]) -> List[str]:
        """
        Make column names unique by appending indices to duplicates.

        Args:
            names: List of potentially duplicate names

        Returns:
            List of unique names
        """
        seen: Dict[str, int] = {}
        unique_names: List[str] = []

        for name in names:
            if name in seen:
                # Append counter to make unique
                seen[name] += 1
                unique_name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
                unique_name = name
            unique_names.append(unique_name)

        return unique_names

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.

        Args:
            df: Input DataFrame
            columns: List of columns to encode (if None, encode all categorical)

        Returns:
            Transformed DataFrame with encoded values
        """
        if self.method == "none":
            logger.info("Encoding method is 'none', returning original data")
            return df.copy()

        logger.info(f"Fitting categorical encoder with method: {self.method}")

        df_copy = df.copy()

        # Determine columns to encode
        if columns is None:
            self.categorical_cols = df_copy.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            self.categorical_cols = columns

        if not self.categorical_cols:
            logger.warning("No categorical columns found to encode")
            return df_copy

        # Check for high cardinality columns
        self._check_cardinality(df_copy)

        # Apply encoding based on method
        if self.method == "onehot":
            df_copy = self._fit_transform_onehot(df_copy)
        elif self.method == "label":
            df_copy = self._fit_transform_label(df_copy)
        elif self.method == "ordinal":
            df_copy = self._fit_transform_ordinal(df_copy)

        logger.info(
            f"Encoded {len(self.categorical_cols)} categorical columns using {self.method}"
        )
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.encoders and self.method != "none":
            raise ValidationError("Encoder must be fitted before transform")

        if self.method == "none":
            return df.copy()

        df_copy = df.copy()

        if self.method == "onehot":
            df_copy = self._transform_onehot(df_copy)
        elif self.method == "label":
            df_copy = self._transform_label(df_copy)
        elif self.method == "ordinal":
            df_copy = self._transform_ordinal(df_copy)

        return df_copy

    def _check_cardinality(self, df: pd.DataFrame) -> None:
        """Check for high cardinality columns and warn."""
        for col in self.categorical_cols:
            n_unique = df[col].nunique()
            if n_unique > self.max_categories:
                logger.warning(
                    f"Column '{col}' has {n_unique} unique values (>{self.max_categories}). "
                    f"Consider using a different encoding method or reducing cardinality."
                )

    def _fit_transform_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform using OneHotEncoder."""
        for col in self.categorical_cols:
            # Sanitize the column name itself (for LightGBM compatibility)
            sanitized_col_name = self._sanitize_feature_name(col)

            # Create encoder for this column
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown,  # type: ignore[arg-type]
                dtype=np.int8,
            )

            # Fit and transform
            encoded = encoder.fit_transform(df[[col]])

            # Create column names
            categories = encoder.categories_[0]  # type: ignore[attr-defined]
            # Sanitize category names to remove special characters (LightGBM compatibility)
            sanitized_cats = [self._sanitize_feature_name(str(cat)) for cat in categories]  # type: ignore[union-attr]
            new_cols = [f"{sanitized_col_name}_{cat}" for cat in sanitized_cats]

            # Handle duplicate column names (e.g., when encoding text data)
            new_cols = self._make_unique_column_names(new_cols)

            # Add encoded columns to dataframe
            encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)

            # Store encoder and column names
            self.encoders[col] = encoder
            self.encoded_columns.extend(new_cols)
            # Map original categories to sanitized feature names
            self.category_mappings[col] = {
                cat: new_col for cat, new_col in zip(categories, new_cols)
            }  # type: ignore[union-attr]

        # Drop original categorical columns
        df = df.drop(columns=self.categorical_cols)

        return df

    def _transform_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted OneHotEncoder."""
        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            # Sanitize the column name itself (for LightGBM compatibility)
            sanitized_col_name = self._sanitize_feature_name(col)

            encoder = self.encoders[col]  # type: ignore[assignment]

            # Transform
            encoded = encoder.transform(df[[col]])  # type: ignore[attr-defined]

            # Create column names (must match the ones from fit_transform)
            categories = encoder.categories_[0]  # type: ignore[attr-defined]
            # Sanitize category names to remove special characters (LightGBM compatibility)
            sanitized_cats = [self._sanitize_feature_name(str(cat)) for cat in categories]  # type: ignore[union-attr]
            new_cols = [f"{sanitized_col_name}_{cat}" for cat in sanitized_cats]

            # Handle duplicate column names (same logic as fit_transform)
            new_cols = self._make_unique_column_names(new_cols)

            # Add encoded columns to dataframe
            encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)

        # Drop original categorical columns
        df = df.drop(
            columns=[col for col in self.categorical_cols if col in df.columns]
        )

        return df

    def _fit_transform_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform using LabelEncoder."""
        for col in self.categorical_cols:
            encoder = LabelEncoder()

            # Fit and transform
            df[col] = encoder.fit_transform(df[col].astype(str))

            # Store encoder and mapping
            self.encoders[col] = encoder
            self.category_mappings[col] = {
                str(cat): int(idx) for idx, cat in enumerate(encoder.classes_)
            }

        return df

    def _transform_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted LabelEncoder."""
        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            encoder = self.encoders[col]  # type: ignore[assignment]

            # Handle unknown categories
            if self.handle_unknown == "ignore":
                # Map unknown categories to -1
                valid_mask = df[col].astype(str).isin(encoder.classes_)  # type: ignore[attr-defined]
                result = np.full(len(df), -1, dtype=np.int64)
                result[valid_mask] = encoder.transform(df.loc[valid_mask, col].astype(str))  # type: ignore[attr-defined]
                df[col] = result
            else:
                df[col] = encoder.transform(df[col].astype(str))  # type: ignore[attr-defined]

        return df

    def _fit_transform_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform using OrdinalEncoder."""
        for col in self.categorical_cols:
            # OrdinalEncoder uses 'use_encoded_value' instead of 'ignore'
            handle_unknown_param = (
                "use_encoded_value"
                if self.handle_unknown == "ignore"
                else self.handle_unknown
            )
            encoder = OrdinalEncoder(
                handle_unknown=handle_unknown_param,  # type: ignore[arg-type]
                unknown_value=-1 if self.handle_unknown == "ignore" else np.nan,
            )

            # Fit and transform
            df[col] = encoder.fit_transform(df[[col]])

            # Store encoder and mapping
            self.encoders[col] = encoder
            categories = encoder.categories_[0]  # type: ignore[attr-defined]
            self.category_mappings[col] = {
                str(cat): int(idx) for idx, cat in enumerate(categories)  # type: ignore[arg-type]
            }

        return df

    def _transform_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted OrdinalEncoder."""
        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            encoder = self.encoders[col]  # type: ignore[assignment]
            df[col] = encoder.transform(df[[col]])  # type: ignore[attr-defined]

        return df

    def get_encoding_summary(self) -> Dict[str, Any]:
        """Get summary of encoding performed."""
        return {
            "method": self.method,
            "categorical_cols": self.categorical_cols,
            "encoded_columns": self.encoded_columns,
            "category_mappings": self.category_mappings,
            "handle_unknown": self.handle_unknown,
            "max_categories": self.max_categories,
        }
