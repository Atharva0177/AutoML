"""Statistical profiling and analysis for datasets."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from automl.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalProfiler:
    """Generate comprehensive statistical profiles for datasets."""

    def __init__(self):
        """Initialize the statistical profiler."""
        self.profile: Optional[Dict[str, Any]] = None

    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive statistical profile for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing statistical profile
        """
        logger.info("Generating statistical profile...")
        
        profile = {
            "overview": self._generate_overview(df),
            "numerical_stats": self._analyze_numerical(df),
            "categorical_stats": self._analyze_categorical(df),
            "missing_analysis": self._analyze_missing(df),
            "summary": self._generate_summary(df),
        }
        
        self.profile = profile
        logger.info("Statistical profile generated successfully")
        return profile

    def _generate_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overview statistics."""
        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numerical": len(df.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(df.select_dtypes(include=["object", "category"]).columns),
            "n_datetime": len(df.select_dtypes(include=["datetime64"]).columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100),
        }

    def _analyze_numerical(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numerical_cols:
            series = df[col]
            stats[col] = {
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "missing_percentage": float(series.isna().sum() / len(series) * 100),
                "unique": int(series.nunique()),
                "mean": float(series.mean()) if not series.isna().all() else None,
                "std": float(series.std()) if not series.isna().all() else None,
                "min": float(series.min()) if not series.isna().all() else None,
                "25%": float(series.quantile(0.25)) if not series.isna().all() else None,
                "50%": float(series.median()) if not series.isna().all() else None,
                "75%": float(series.quantile(0.75)) if not series.isna().all() else None,
                "max": float(series.max()) if not series.isna().all() else None,
                "skewness": float(series.skew()) if not series.isna().all() else None,  # type: ignore[arg-type]
                "kurtosis": float(series.kurtosis()) if not series.isna().all() else None,  # type: ignore[arg-type]
                "zeros": int((series == 0).sum()),
                "zeros_percentage": float((series == 0).sum() / len(series) * 100),
                "outliers": self._count_outliers(series),
            }
        
        return stats

    def _analyze_categorical(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze categorical columns."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        stats = {}
        
        for col in categorical_cols:
            series = df[col]
            value_counts = series.value_counts()
            
            stats[col] = {
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "missing_percentage": float(series.isna().sum() / len(series) * 100),
                "unique": int(series.nunique()),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "most_frequent_percentage": float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0.0,
                "top_5_values": value_counts.head(5).to_dict(),
                "is_high_cardinality": series.nunique() > len(series) * 0.5,
                "cardinality_ratio": float(series.nunique() / len(series)),
            }
        
        return stats

    def _analyze_missing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values across the dataset."""
        missing_counts = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = int(missing_counts.sum())
        
        columns_with_missing = missing_counts[missing_counts > 0].to_dict()
        
        return {
            "total_missing": total_missing,
            "total_cells": total_cells,
            "missing_percentage": float(total_missing / total_cells * 100) if total_cells > 0 else 0.0,
            "columns_with_missing": len(columns_with_missing),
            "missing_by_column": {k: int(v) for k, v in columns_with_missing.items()},
            "completely_missing_columns": list(df.columns[df.isna().all()]),
        }

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary insights."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        
        insights = []
        
        # Check for high missing values
        missing_pct = (df.isna().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            insights.append(f"{len(high_missing)} column(s) with >50% missing values: {', '.join(high_missing[:3])}")
        
        # Check for high cardinality
        high_card = []
        for col in cat_cols:
            if df[col].nunique() > len(df) * 0.8:
                high_card.append(col)
        if high_card:
            insights.append(f"{len(high_card)} high cardinality column(s): {', '.join(high_card[:3])}")
        
        # Check for potential ID columns
        potential_ids = []
        for col in df.columns:
            if df[col].nunique() == len(df):
                potential_ids.append(col)
        if potential_ids:
            insights.append(f"{len(potential_ids)} potential ID column(s): {', '.join(potential_ids[:3])}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            insights.append(f"{len(constant_cols)} constant column(s): {', '.join(constant_cols)}")
        
        return {
            "insights": insights,
            "data_quality_issues": len(insights),
        }

    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        if series.isna().all() or len(series) < 4:
            return 0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
        return outliers

    def get_column_stats(self, column: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific column.
        
        Args:
            column: Column name
            
        Returns:
            Statistics dictionary or None if not found
        """
        if self.profile is None:
            return None
        
        if column in self.profile.get("numerical_stats", {}):
            return self.profile["numerical_stats"][column]
        elif column in self.profile.get("categorical_stats", {}):
            return self.profile["categorical_stats"][column]
        
        return None

    def get_insights(self) -> List[str]:
        """
        Get data quality insights.
        
        Returns:
            List of insight strings
        """
        if self.profile is None:
            return []
        
        return self.profile.get("summary", {}).get("insights", [])
