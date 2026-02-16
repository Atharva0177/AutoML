"""Quality validation for datasets."""

from typing import Dict, List

import numpy as np
import pandas as pd

from automl.utils.logger import get_logger

logger = get_logger(__name__)


class QualityValidator:
    """Validates data quality and generates quality report."""

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Quality report dictionary
        """
        report = {
            "overall_score": 0.0,
            "dimensions": self._analyze_dimensions(df),
            "missing_values": self._analyze_missing_values(df),
            "duplicates": self._analyze_duplicates(df),
            "outliers": self._analyze_outliers(df),
            "data_types": self._analyze_data_types(df),
            "recommendations": [],
        }

        # Calculate overall quality score
        report["overall_score"] = self._calculate_quality_score(report)

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        logger.info(f"Data quality score: {report['overall_score']:.2f}/100")

        return report

    def _analyze_dimensions(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset dimensions."""
        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()

        missing_by_column = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_by_column[col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(df) * 100),
                }

        return {
            "total_missing": int(missing_cells),
            "percentage": float(missing_cells / total_cells * 100)
            if total_cells > 0
            else 0,
            "by_column": missing_by_column,
        }

    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate rows."""
        n_duplicates = df.duplicated().sum()

        return {
            "count": int(n_duplicates),
            "percentage": float(n_duplicates / len(df) * 100) if len(df) > 0 else 0,
        }

    def _analyze_outliers(self, df: pd.DataFrame) -> Dict:
        """Analyze outliers in numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_by_column = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > 0:
                outliers_by_column[col] = {
                    "count": int(outliers),
                    "percentage": float(outliers / len(df) * 100),
                }

        total_outliers = sum(v["count"] for v in outliers_by_column.values())

        return {
            "total_outlier_values": total_outliers,
            "by_column": outliers_by_column,
        }

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict:
        """Analyze data type distribution."""
        type_counts = {
            "numeric": len(df.select_dtypes(include=[np.number]).columns),
            "categorical": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime": len(df.select_dtypes(include=["datetime64"]).columns),
            "boolean": len(df.select_dtypes(include=["bool"]).columns),
        }

        return type_counts

    def _calculate_quality_score(self, report: Dict) -> float:
        """
        Calculate overall quality score (0-100).

        Args:
            report: Quality report

        Returns:
            Quality score
        """
        score = 100.0

        # Deduct for missing values
        missing_pct = report["missing_values"]["percentage"]
        score -= min(missing_pct, 30)  # Max deduction: 30 points

        # Deduct for duplicates
        dup_pct = report["duplicates"]["percentage"]
        score -= min(dup_pct, 20)  # Max deduction: 20 points

        # Deduct for small dataset
        n_rows = report["dimensions"]["n_rows"]
        if n_rows < 100:
            score -= 20
        elif n_rows < 1000:
            score -= 10

        return float(max(0, score))

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []

        # Missing values
        missing_pct = report["missing_values"]["percentage"]
        if missing_pct > 20:
            recommendations.append(
                f"High percentage of missing values ({missing_pct:.1f}%). "
                "Consider imputation or removing affected columns."
            )

        # Duplicates
        dup_pct = report["duplicates"]["percentage"]
        if dup_pct > 5:
            recommendations.append(
                f"Found {dup_pct:.1f}% duplicate rows. "
                "Consider removing duplicates before training."
            )

        # Small dataset
        n_rows = report["dimensions"]["n_rows"]
        if n_rows < 100:
            recommendations.append(
                "Very small dataset (<100 rows). "
                "Results may not be reliable. Consider collecting more data."
            )
        elif n_rows < 1000:
            recommendations.append(
                "Small dataset (<1000 rows). "
                "Use cross-validation carefully to avoid overfitting."
            )

        # Outliers
        if report["outliers"]["by_column"]:
            outlier_cols = list(report["outliers"]["by_column"].keys())
            recommendations.append(
                f"Outliers detected in {len(outlier_cols)} column(s): {', '.join(outlier_cols[:3])}. "
                "Consider outlier treatment based on domain knowledge."
            )

        return recommendations
