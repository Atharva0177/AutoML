"""
Advanced missing value imputation methods.

This module provides sophisticated imputation strategies including:
- KNN Imputation: Uses k-nearest neighbors to impute missing values
- Iterative Imputation: Models each feature with missing values as a function of other features
"""

from typing import Any, Dict, List, Literal, Optional, Union
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# Import experimental module BEFORE importing IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer

from automl.utils.logger import get_logger
from automl.utils.exceptions import ValidationError

logger = get_logger(__name__)

AdvancedImputationStrategy = Literal["knn", "iterative", "mice"]


class AdvancedMissingValueHandler:
    """
    Handle missing values using advanced machine learning-based imputation.
    
    Strategies:
    - KNN: Uses k-nearest neighbors based on feature similarity
    - Iterative (MICE): Multivariate Imputation by Chained Equations
    
    These methods are more sophisticated than simple mean/median/mode imputation
    and can capture relationships between features.
    """
    
    def __init__(
        self,
        strategy: AdvancedImputationStrategy = "knn",
        n_neighbors: int = 5,
        max_iter: int = 10,
        random_state: Optional[int] = 42,
        **kwargs: Any
    ):
        """
        Initialize the advanced missing value handler.
        
        Args:
            strategy: Imputation strategy ('knn' or 'iterative')
            n_neighbors: Number of neighbors for KNN imputation
            max_iter: Maximum iterations for iterative imputation
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for the imputers
        """
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.imputer: Optional[Union[KNNImputer, IterativeImputer]] = None
        self.fitted_columns: List[str] = []
        self.original_dtypes: Dict[str, Any] = {}
        self.categorical_mappings: Dict[str, Dict[Any, int]] = {}
        self.reverse_mappings: Dict[str, Dict[int, Any]] = {}
        self.is_fitted = False
        
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
        logger.info(f"Fitting advanced imputation with strategy: {self.strategy}")
        
        df_copy = df.copy()
        
        # Auto-detect column types if not provided
        if numerical_cols is None or categorical_cols is None:
            numerical_cols, categorical_cols = self._detect_column_types(df_copy)
        
        # Log missing value statistics
        self._log_missing_stats(df_copy)
        
        # Store original dtypes
        self.original_dtypes = {col: df_copy[col].dtype for col in df_copy.columns}
        
        # Prepare data for imputation
        df_prepared, self.fitted_columns = self._prepare_data(
            df_copy, numerical_cols, categorical_cols
        )
        
        # If no columns need imputation, return original data
        if not self.fitted_columns:
            logger.info("No missing values found. Returning data unchanged.")
            self.is_fitted = True
            return df_copy
        
        # Create and fit imputer
        if self.strategy == "knn":
            self.imputer = KNNImputer(
                n_neighbors=self.n_neighbors,
                weights='uniform',
                **self.kwargs
            )
        elif self.strategy in ["iterative", "mice"]:
            self.imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self.kwargs
            )
        else:
            raise ValidationError(f"Unknown strategy: {self.strategy}")
        
        # Fit and transform
        logger.info(f"Applying {self.strategy} imputation to {len(self.fitted_columns)} columns")
        imputed_values = self.imputer.fit_transform(df_prepared[self.fitted_columns])
        df_prepared[self.fitted_columns] = imputed_values
        
        # Restore to original format
        df_result = self._restore_data(df_prepared, df_copy.columns)
        
        self.is_fitted = True
        logger.info(f"Advanced imputation complete. Shape: {df_result.shape}")
        
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted imputer.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValidationError("Handler must be fitted before transform")
        
        df_copy = df.copy()
        
        # Prepare data
        df_prepared, _ = self._prepare_data_transform(df_copy)
        
        # Transform
        if self.imputer is not None and self.fitted_columns:
            # Only transform columns that were fitted
            cols_to_transform = [col for col in self.fitted_columns if col in df_prepared.columns]
            if cols_to_transform:
                imputed_values = self.imputer.transform(df_prepared[cols_to_transform])
                df_prepared[cols_to_transform] = imputed_values
        
        # Restore to original format
        df_result = self._restore_data(df_prepared, df_copy.columns)
        
        return df_result
    
    def _detect_column_types(self, df: pd.DataFrame) -> tuple[List[str], List[str]]:
        """Detect numerical and categorical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        logger.debug(
            f"Detected {len(numerical_cols)} numerical and "
            f"{len(categorical_cols)} categorical columns"
        )
        return numerical_cols, categorical_cols
    
    def _log_missing_stats(self, df: pd.DataFrame) -> None:
        """Log missing value statistics."""
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            total_missing = missing_counts.sum()
            total_cells = df.shape[0] * df.shape[1]
            logger.info(
                f"Found {total_missing} missing values ({total_missing/total_cells*100:.2f}%) "
                f"across {len(missing_cols)} columns"
            )
            for col, count in missing_cols.head(10).items():
                pct = (count / len(df)) * 100
                logger.info(f"  - {col}: {count} ({pct:.2f}%)")
            if len(missing_cols) > 10:
                logger.info(f"  ... and {len(missing_cols) - 10} more columns")
        else:
            logger.info("No missing values found")
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for imputation by encoding categorical variables.
        
        Args:
            df: Input DataFrame
            numerical_cols: Numerical column names
            categorical_cols: Categorical column names
            
        Returns:
            Prepared DataFrame and list of columns to impute
        """
        df_prepared = df.copy()
        columns_to_impute = []
        
        # Add numerical columns with missing values
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                columns_to_impute.append(col)
        
        # Encode categorical columns with missing values
        for col in categorical_cols:
            if col not in df.columns or not df[col].isnull().any():
                continue
            
            # Create label encoding for categorical column
            unique_values = df[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            reverse_mapping = {idx: val for val, idx in mapping.items()}
            
            self.categorical_mappings[col] = mapping
            self.reverse_mappings[col] = reverse_mapping
            
            # Apply encoding (NaN stays as NaN for imputation)
            df_prepared[col] = df[col].map(mapping)
            columns_to_impute.append(col)
        
        logger.debug(f"Prepared {len(columns_to_impute)} columns for imputation")
        
        return df_prepared, columns_to_impute
    
    def _prepare_data_transform(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, List[str]]:
        """Prepare new data for transformation using fitted mappings."""
        df_prepared = df.copy()
        
        # Apply categorical encodings
        for col, mapping in self.categorical_mappings.items():
            if col in df_prepared.columns:
                # Handle unseen categories by assigning them to -1
                df_prepared[col] = df_prepared[col].map(lambda x: mapping.get(x, -1) if pd.notna(x) else np.nan)
        
        return df_prepared, self.fitted_columns
    
    def _restore_data(
        self,
        df_prepared: pd.DataFrame,
        original_columns: pd.Index
    ) -> pd.DataFrame:
        """
        Restore data to original format by decoding categorical variables.
        
        Args:
            df_prepared: Prepared DataFrame with imputed values
            original_columns: Original column order
            
        Returns:
            DataFrame in original format
        """
        df_result = df_prepared.copy()
        
        # Decode categorical columns
        for col, reverse_mapping in self.reverse_mappings.items():
            if col in df_result.columns:
                # Round to nearest integer for categorical encoding
                df_result[col] = df_result[col].round().astype(int)
                # Map back to original categories
                df_result[col] = df_result[col].map(reverse_mapping)
        
        # Restore original dtypes where possible
        for col, dtype in self.original_dtypes.items():
            if col in df_result.columns:
                try:
                    if dtype in [np.int64, np.int32, np.int16, np.int8]:
                        df_result[col] = df_result[col].round().astype(dtype)
                    elif col not in self.categorical_mappings:
                        df_result[col] = df_result[col].astype(dtype)
                except (ValueError, TypeError):
                    logger.warning(f"Could not restore dtype for column {col}")
        
        # Restore original column order
        df_result = df_result[original_columns]
        
        return df_result
    
    def get_imputation_summary(self) -> Dict[str, Any]:
        """Get summary of imputation performed."""
        return {
            "strategy": self.strategy,
            "n_neighbors": self.n_neighbors if self.strategy == "knn" else None,
            "max_iter": self.max_iter if self.strategy == "iterative" else None,
            "fitted_columns": self.fitted_columns,
            "categorical_columns_encoded": list(self.categorical_mappings.keys()),
            "is_fitted": self.is_fitted,
        }
