"""
Feature Engineering Module

This module provides advanced feature engineering capabilities including:
- Polynomial features
- Interaction terms
- Binning/discretization
- Mathematical transformations
"""

from typing import Dict, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BinningStrategy(str, Enum):
    """Binning strategies for feature discretization."""
    UNIFORM = "uniform"  # Equal width bins
    QUANTILE = "quantile"  # Equal frequency bins
    KMEANS = "kmeans"  # K-means clustering


class FeatureEngineer:
    """
    Feature engineering class for creating new features from existing ones.
    
    Supports:
    - Polynomial features (degree 2, 3, etc.)
    - Interaction terms between specific features
    - Binning/discretization
    - Mathematical transformations (log, sqrt, square, inverse)
    
    Parameters
    ----------
    polynomial_degree : int, optional
        Degree for polynomial features (default: None, no polynomial features)
    interaction_features : List[Tuple[str, str]], optional
        List of feature pairs to create interactions (default: None)
    binning_config : Dict[str, Dict], optional
        Configuration for binning: {column_name: {'n_bins': int, 'strategy': str}}
    transformations : Dict[str, str], optional
        Mathematical transformations: {column_name: 'log'|'sqrt'|'square'|'inverse'}
    include_bias : bool, optional
        Include bias column in polynomial features (default: False)
    interaction_only : bool, optional
        For polynomial features, only create interaction terms (default: False)
    
    Attributes
    ----------
    poly_features_ : PolynomialFeatures
        Fitted polynomial features transformer
    feature_names_ : List[str]
        Names of engineered features
    binning_transformers_ : Dict[str, KBinsDiscretizer]
        Fitted binning transformers for each column
    """
    
    def __init__(
        self,
        polynomial_degree: Optional[int] = None,
        interaction_features: Optional[List[Tuple[str, str]]] = None,
        binning_config: Optional[Dict[str, Dict]] = None,
        transformations: Optional[Dict[str, str]] = None,
        include_bias: bool = False,
        interaction_only: bool = False
    ):
        self.polynomial_degree = polynomial_degree
        self.interaction_features = interaction_features or []
        self.binning_config = binning_config or {}
        self.transformations = transformations or {}
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        
        # Fitted attributes
        self.poly_features_: Optional[PolynomialFeatures] = None
        self.feature_names_: List[str] = []
        self.binning_transformers_: Dict[str, KBinsDiscretizer] = {}
        self.poly_input_columns_: List[str] = []
        self._is_fitted = False
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit the feature engineer and transform the data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        numerical_cols : List[str], optional
            Columns to use for polynomial features (default: all numeric columns)
        
        Returns
        -------
        pd.DataFrame
            Transformed data with engineered features
        """
        result_df = df.copy()
        
        # Determine numerical columns if not provided
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Apply mathematical transformations
        if self.transformations:
            result_df = self._apply_transformations(result_df)
        
        # 2. Apply binning
        if self.binning_config:
            result_df = self._fit_transform_binning(result_df)
        
        # 3. Create interaction features
        if self.interaction_features:
            result_df = self._create_interactions(result_df)
        
        # 4. Create polynomial features
        if self.polynomial_degree and self.polynomial_degree > 1:
            result_df = self._fit_transform_polynomial(result_df, numerical_cols)
        
        self._is_fitted = True
        return result_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineer.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        
        Returns
        -------
        pd.DataFrame
            Transformed data with engineered features
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Call fit_transform first.")
        
        result_df = df.copy()
        
        # 1. Apply mathematical transformations
        if self.transformations:
            result_df = self._apply_transformations(result_df)
        
        # 2. Apply binning
        if self.binning_transformers_:
            result_df = self._transform_binning(result_df)
        
        # 3. Create interaction features
        if self.interaction_features:
            result_df = self._create_interactions(result_df)
        
        # 4. Create polynomial features
        if self.poly_features_ is not None:
            result_df = self._transform_polynomial(result_df)
        
        return result_df
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mathematical transformations to specified columns."""
        result_df = df.copy()
        
        for col, transform_type in self.transformations.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe, skipping transformation")
                continue
            
            new_col_name = f"{col}_{transform_type}"
            
            if transform_type == 'log':
                # Add small constant to avoid log(0)
                result_df[new_col_name] = np.log1p(np.abs(df[col]))
            elif transform_type == 'sqrt':
                # Use absolute value then restore sign
                result_df[new_col_name] = np.sign(df[col]) * np.sqrt(np.abs(df[col]))
            elif transform_type == 'square':
                result_df[new_col_name] = df[col] ** 2
            elif transform_type == 'inverse':
                # Avoid division by zero
                result_df[new_col_name] = np.where(
                    df[col] != 0,
                    1 / df[col],
                    0
                )
            else:
                logger.warning(f"Unknown transformation type: {transform_type}")
        
        return result_df
    
    def _fit_transform_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply binning to specified columns."""
        result_df = df.copy()
        
        for col, config in self.binning_config.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe, skipping binning")
                continue
            
            n_bins = config.get('n_bins', 5)
            strategy = config.get('strategy', 'quantile')
            encode = config.get('encode', 'ordinal')  # 'ordinal' or 'onehot'
            
            # Create and fit discretizer
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode=encode,
                strategy=strategy
            )
            
            # Reshape for sklearn
            col_data = df[[col]].values
            binned_data = discretizer.fit_transform(col_data)
            
            # Store the fitted transformer
            self.binning_transformers_[col] = discretizer
            
            # Add binned column
            if encode == 'ordinal':
                result_df[f"{col}_binned"] = binned_data.astype(int)
            else:  # onehot
                # Create multiple binary columns
                for i in range(n_bins):
                    result_df[f"{col}_bin_{i}"] = binned_data[:, i]
        
        return result_df
    
    def _transform_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted binning to new data."""
        result_df = df.copy()
        
        for col, discretizer in self.binning_transformers_.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe, skipping binning")
                continue
            
            col_data = df[[col]].values
            binned_data = discretizer.transform(col_data)
            
            # Convert to dense if sparse
            if hasattr(binned_data, 'toarray'):
                binned_data = binned_data.toarray()  # type: ignore[union-attr]
            
            # Add binned column
            if discretizer.encode == 'ordinal':  # type: ignore[attr-defined]
                result_df[f"{col}_binned"] = binned_data.astype(int).flatten()  # type: ignore[union-attr]
            else:  # onehot
                # Create multiple binary columns
                n_bins_actual = discretizer.n_bins  # type: ignore[attr-defined]
                for i in range(n_bins_actual):
                    result_df[f"{col}_bin_{i}"] = binned_data[:, i]  # type: ignore[index]
        
        return result_df
    
    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between specified column pairs."""
        result_df = df.copy()
        
        for col1, col2 in self.interaction_features:
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"Columns '{col1}' or '{col2}' not found, skipping interaction")
                continue
            
            interaction_col = f"{col1}_x_{col2}"
            result_df[interaction_col] = df[col1] * df[col2]
        
        return result_df
    
    def _fit_transform_polynomial(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Fit and create polynomial features."""
        # Filter to only existing numerical columns
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No numerical columns available for polynomial features")
            return df
        
        self.poly_input_columns_ = available_cols
        
        # Create polynomial features transformer
        # Use degree=2 if polynomial_degree is None
        degree = self.polynomial_degree if self.polynomial_degree is not None else 2
        self.poly_features_ = PolynomialFeatures(
            degree=degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        # Fit and transform
        poly_data = self.poly_features_.fit_transform(df[available_cols])
        
        # Get feature names
        poly_feature_names = self.poly_features_.get_feature_names_out(available_cols)
        
        # Create dataframe with polynomial features
        poly_df = pd.DataFrame(
            poly_data,
            columns=poly_feature_names,
            index=df.index
        )
        
        # Remove original columns to avoid duplication
        poly_df = poly_df.drop(columns=available_cols, errors='ignore')
        
        # Concatenate with original dataframe
        result_df = pd.concat([df, poly_df], axis=1)
        
        return result_df
    
    def _transform_polynomial(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted polynomial features to new data."""
        if not self.poly_input_columns_:
            return df
        
        # Check if all required columns exist
        available_cols = [col for col in self.poly_input_columns_ if col in df.columns]
        
        if not available_cols or self.poly_features_ is None:
            logger.warning("No numerical columns available for polynomial features")
            return df
        
        # Transform
        poly_data = self.poly_features_.transform(df[available_cols])
        
        # Get feature names
        poly_feature_names = self.poly_features_.get_feature_names_out(available_cols)
        
        # Convert sparse to dense if needed
        if hasattr(poly_data, 'toarray'):
            poly_data = poly_data.toarray()  # type: ignore[union-attr]
        
        # Create dataframe with polynomial features
        poly_df = pd.DataFrame(
            poly_data,  # type: ignore[arg-type]
            columns=poly_feature_names,
            index=df.index
        )
        
        # Remove original columns to avoid duplication
        poly_df = poly_df.drop(columns=available_cols, errors='ignore')
        
        # Concatenate with original dataframe
        result_df = pd.concat([df, poly_df], axis=1)
        
        return result_df
    
    def get_feature_summary(self) -> Dict[str, int]:
        """
        Get summary of engineered features.
        
        Returns
        -------
        Dict[str, int]
            Dictionary with counts of different feature types
        """
        summary = {
            'transformations': len(self.transformations),
            'binned_features': len(self.binning_transformers_),
            'interactions': len(self.interaction_features),
            'polynomial_features': 0
        }
        
        if self.poly_features_ is not None:
            n_poly = self.poly_features_.n_output_features_
            n_original = len(self.poly_input_columns_)
            summary['polynomial_features'] = n_poly - n_original
        
        return summary
