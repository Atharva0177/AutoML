"""Correlation analysis for features."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from automl.utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """Analyze correlations between features."""

    def __init__(self):
        """Initialize correlation analyzer."""
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.analysis: Optional[Dict[str, Any]] = None

    def analyze_correlations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        method: str = "pearson",
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Analyze correlations in the dataset.

        Args:
            df: Input DataFrame
            target_column: Target column name (optional)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Threshold for identifying high correlations

        Returns:
            Dictionary with correlation analysis
        """
        logger.info(f"Analyzing correlations using {method} method...")

        # Get numerical columns
        numerical_df = df.select_dtypes(include=[np.number])

        if len(numerical_df.columns) < 2:
            logger.warning("Insufficient numerical columns for correlation analysis")
            return {
                "correlation_matrix": None,
                "high_correlations": [],
                "target_correlations": {},
                "message": "Insufficient numerical columns for correlation analysis",
            }

        # Calculate correlation matrix
        try:
            corr_matrix = numerical_df.corr(method=method)  # type: ignore[arg-type]
            self.correlation_matrix = corr_matrix
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {
                "correlation_matrix": None,
                "high_correlations": [],
                "target_correlations": {},
                "error": str(e),
            }

        # Find high correlations
        high_corrs = self._find_high_correlations(corr_matrix, threshold)

        # Analyze target correlations if specified
        target_corrs = {}
        if target_column and target_column in corr_matrix.columns:
            target_corrs = self._analyze_target_correlations(corr_matrix, target_column)

        # Identify multicollinearity
        multicollinearity = self._detect_multicollinearity(high_corrs)

        analysis = {
            "method": method,
            "n_features": len(corr_matrix.columns),
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_corrs,
            "target_correlations": target_corrs,
            "multicollinearity": multicollinearity,
            "recommendations": self._generate_recommendations(
                high_corrs, multicollinearity
            ),
        }

        self.analysis = analysis
        logger.info(f"Found {len(high_corrs)} high correlations")
        return analysis

    def _find_high_correlations(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Find pairs of features with high correlation."""
        high_corrs = []

        # Get upper triangle (avoid duplicates)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:  # type: ignore[arg-type,operator]
                    high_corrs.append(
                        {
                            "feature_1": col1,
                            "feature_2": col2,
                            "correlation": float(corr_value),  # type: ignore[arg-type]
                            "abs_correlation": float(abs(corr_value)),  # type: ignore[arg-type]
                        }
                    )

        # Sort by absolute correlation
        high_corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return high_corrs

    def _analyze_target_correlations(
        self,
        corr_matrix: pd.DataFrame,
        target_column: str,
    ) -> Dict[str, Any]:
        """Analyze correlations with target variable."""
        if target_column not in corr_matrix.columns:
            return {}

        target_corrs = corr_matrix[target_column].drop(target_column)
        target_corrs_sorted = target_corrs.abs().sort_values(ascending=False)

        return {
            "all_correlations": {k: float(v) for k, v in target_corrs.items()},
            "top_positive": self._get_top_correlations(
                target_corrs, n=5, positive=True
            ),
            "top_negative": self._get_top_correlations(
                target_corrs, n=5, positive=False
            ),
            "top_absolute": {
                k: float(target_corrs[k]) for k in target_corrs_sorted.head(10).index
            },
            "weak_correlations": [k for k, v in target_corrs.items() if abs(v) < 0.1],
        }

    def _get_top_correlations(
        self,
        correlations: pd.Series,
        n: int = 5,
        positive: bool = True,
    ) -> Dict[str, float]:
        """Get top N positive or negative correlations."""
        if positive:
            top = correlations.nlargest(n)
        else:
            top = correlations.nsmallest(n)

        return {str(k): float(v) for k, v in top.items()}  # type: ignore[arg-type]

    def _detect_multicollinearity(
        self,
        high_corrs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect multicollinearity issues."""
        if not high_corrs:
            return {
                "detected": False,
                "high_correlation_groups": [],
                "features_to_review": [],
            }

        # Group features with multiple high correlations
        feature_corr_counts: Dict[str, int] = {}
        for corr in high_corrs:
            f1, f2 = corr["feature_1"], corr["feature_2"]
            feature_corr_counts[f1] = feature_corr_counts.get(f1, 0) + 1
            feature_corr_counts[f2] = feature_corr_counts.get(f2, 0) + 1

        # Features with multiple high correlations
        features_to_review = [
            {"feature": k, "n_high_correlations": v}
            for k, v in feature_corr_counts.items()
            if v >= 2
        ]
        features_to_review.sort(key=lambda x: int(x["n_high_correlations"]), reverse=True)

        return {
            "detected": len(features_to_review) > 0,
            "n_highly_correlated_pairs": len(high_corrs),
            "features_to_review": features_to_review,
            "severity": self._get_multicollinearity_severity(high_corrs),
        }

    def _get_multicollinearity_severity(
        self,
        high_corrs: List[Dict[str, Any]],
    ) -> str:
        """Determine severity of multicollinearity."""
        if not high_corrs:
            return "none"

        max_corr = max([corr["abs_correlation"] for corr in high_corrs])
        n_corrs = len(high_corrs)

        if max_corr > 0.95 or n_corrs > 10:
            return "severe"
        elif max_corr > 0.85 or n_corrs > 5:
            return "moderate"
        else:
            return "mild"

    def _generate_recommendations(
        self,
        high_corrs: List[Dict[str, Any]],
        multicollinearity: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []

        if not high_corrs:
            recommendations.append(
                "No high correlations detected. Features appear to be independent."
            )
            return recommendations

        # Multicollinearity recommendations
        if multicollinearity["detected"]:
            severity = multicollinearity["severity"]
            n_pairs = multicollinearity["n_highly_correlated_pairs"]

            if severity == "severe":
                recommendations.append(
                    f"Severe multicollinearity detected ({n_pairs} pairs). "
                    "Consider using PCA, removing redundant features, or using regularization."
                )
            elif severity == "moderate":
                recommendations.append(
                    f"Moderate multicollinearity detected ({n_pairs} pairs). "
                    "Review highly correlated features and consider feature selection."
                )
            else:
                recommendations.append(
                    f"Mild multicollinearity detected ({n_pairs} pairs). "
                    "Monitor model performance but may not require immediate action."
                )

        # Specific feature recommendations
        features_to_review = multicollinearity.get("features_to_review", [])
        if features_to_review:
            top_features = ", ".join([f["feature"] for f in features_to_review[:3]])
            recommendations.append(
                f"Features with multiple high correlations: {top_features}. "
                "Consider removing or engineering these features."
            )

        return recommendations

    def get_feature_importance_proxy(
        self, target_column: str
    ) -> Optional[Dict[str, float]]:
        """
        Get a proxy for feature importance based on target correlation.

        Args:
            target_column: Target column name

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.analysis is None or "target_correlations" not in self.analysis:
            return None

        target_corrs = self.analysis["target_correlations"].get("all_correlations", {})

        # Use absolute correlation as importance proxy
        importance = {k: abs(v) for k, v in target_corrs.items()}

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def get_redundant_features(self, threshold: float = 0.95) -> List[Tuple[str, str]]:
        """
        Get pairs of potentially redundant features.

        Args:
            threshold: Correlation threshold for redundancy

        Returns:
            List of feature pairs that may be redundant
        """
        if self.analysis is None:
            return []

        high_corrs = self.analysis.get("high_correlations", [])
        redundant = [
            (corr["feature_1"], corr["feature_2"])
            for corr in high_corrs
            if corr["abs_correlation"] >= threshold
        ]

        return redundant
