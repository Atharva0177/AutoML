"""
Preprocessing pipeline builder.

This module provides a builder for creating and managing preprocessing pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pickle
import json
from pathlib import Path

from automl.preprocessing.cleaners.missing_handler import MissingValueHandler
from automl.preprocessing.transformers.scalers import NumericalScaler
from automl.preprocessing.transformers.encoders import CategoricalEncoder
from automl.utils.logger import get_logger
from automl.utils.exceptions import ValidationError

logger = get_logger(__name__)


class PipelineBuilder:
    """
    Build and manage preprocessing pipelines.
    
    Allows chaining of preprocessing steps:
    1. Missing value imputation
    2. Categorical encoding
    3. Numerical scaling
    """
    
    def __init__(self):
        """Initialize the pipeline builder."""
        self.steps: List[Tuple[str, Any]] = []
        self.missing_handler: Optional[MissingValueHandler] = None
        self.encoder: Optional[CategoricalEncoder] = None
        self.scaler: Optional[NumericalScaler] = None
        self.is_fitted: bool = False
        self.feature_names_in: List[str] = []
        self.feature_names_out: List[str] = []
        
    def add_missing_handler(
        self,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        threshold: float = 0.5,
    ) -> "PipelineBuilder":
        """
        Add missing value handler to pipeline.
        
        Args:
            strategy: Imputation strategy
            fill_value: Fill value for constant strategy
            threshold: Threshold for dropping columns
            
        Returns:
            Self for chaining
        """
        self.missing_handler = MissingValueHandler(
            strategy=strategy,  # type: ignore[arg-type]
            fill_value=fill_value,
            threshold=threshold,
        )
        self.steps.append(("missing_handler", self.missing_handler))
        logger.debug(f"Added missing value handler with strategy: {strategy}")
        return self
    
    def add_encoder(
        self,
        method: str = "onehot",
        handle_unknown: str = "ignore",
        max_categories: int = 50,
    ) -> "PipelineBuilder":
        """
        Add categorical encoder to pipeline.
        
        Args:
            method: Encoding method
            handle_unknown: How to handle unknown categories
            max_categories: Maximum categories for onehot
            
        Returns:
            Self for chaining
        """
        self.encoder = CategoricalEncoder(
            method=method,  # type: ignore[arg-type]
            handle_unknown=handle_unknown,  # type: ignore[arg-type]
            max_categories=max_categories,
        )
        self.steps.append(("encoder", self.encoder))
        logger.debug(f"Added categorical encoder with method: {method}")
        return self
    
    def add_scaler(
        self,
        method: str = "standard",
        feature_range: Tuple[int, int] = (0, 1),
    ) -> "PipelineBuilder":
        """
        Add numerical scaler to pipeline.
        
        Args:
            method: Scaling method
            feature_range: Range for minmax scaler
            
        Returns:
            Self for chaining
        """
        self.scaler = NumericalScaler(
            method=method,  # type: ignore[arg-type]
            feature_range=feature_range,
        )
        self.steps.append(("scaler", self.scaler))
        logger.debug(f"Added numerical scaler with method: {method}")
        return self
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit pipeline and transform data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info(f"Fitting preprocessing pipeline with {len(self.steps)} steps")
        
        if not self.steps:
            logger.warning("No preprocessing steps added to pipeline")
            return df.copy()
        
        self.feature_names_in = df.columns.tolist()
        df_transformed = df.copy()
        
        # Apply each step
        for step_name, transformer in self.steps:
            logger.debug(f"Applying step: {step_name}")
            df_transformed = transformer.fit_transform(df_transformed)
        
        self.feature_names_out = df_transformed.columns.tolist()
        self.is_fitted = True
        
        logger.info(
            f"Pipeline fit complete. Features: {len(self.feature_names_in)} -> "
            f"{len(self.feature_names_out)}"
        )
        
        return df_transformed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValidationError("Pipeline must be fitted before transform")
        
        logger.debug(f"Transforming data using fitted pipeline ({len(self.steps)} steps)")
        
        df_transformed = df.copy()
        
        # Apply each step
        for step_name, transformer in self.steps:
            df_transformed = transformer.transform(df_transformed)
        
        return df_transformed
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath: Path to save pipeline
        """
        if not self.is_fitted:
            raise ValidationError("Pipeline must be fitted before saving")
        
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline object
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        # Save metadata as JSON
        metadata = self.get_pipeline_summary()
        metadata_path = filepath_obj.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pipeline saved to {filepath}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load(filepath: str) -> "PipelineBuilder":
        """
        Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to load pipeline from
            
        Returns:
            Loaded pipeline
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        if not isinstance(pipeline, PipelineBuilder):
            raise ValidationError(f"Loaded object is not a PipelineBuilder: {type(pipeline)}")
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline configuration.
        
        Returns:
            Dictionary with pipeline details
        """
        step_summaries = []
        
        for step_name, transformer in self.steps:
            if hasattr(transformer, f'get_{step_name.replace("_", "")}_summary'):
                summary_method = getattr(transformer, f'get_{step_name.replace("_", "")}_summary')
                step_summaries.append({
                    "name": step_name,
                    "config": summary_method(),
                })
            else:
                step_summaries.append({
                    "name": step_name,
                    "type": type(transformer).__name__,
                })
        
        return {
            "num_steps": len(self.steps),
            "steps": step_summaries,
            "is_fitted": self.is_fitted,
            "feature_names_in": self.feature_names_in,
            "feature_names_out": self.feature_names_out,
            "num_features_in": len(self.feature_names_in),
            "num_features_out": len(self.feature_names_out),
        }
    
    def __repr__(self) -> str:
        """String representation of pipeline."""
        step_names = [name for name, _ in self.steps]
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"PipelineBuilder(steps={step_names}, {fitted_str})"
    
    def __len__(self) -> int:
        """Number of steps in pipeline."""
        return len(self.steps)
