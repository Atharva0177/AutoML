"""
Feature Selection Module

This module provides advanced feature selection capabilities including:
- Filter methods (correlation, chi-square, mutual information, ANOVA F-test)
- Wrapper methods (recursive feature elimination)
- Embedded methods (L1 regularization, tree-based importance)
"""

import logging
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


class SelectionMethod(str, Enum):
    """Feature selection methods."""

    # Filter methods
    CORRELATION = "correlation"
    CHI2 = "chi2"
    MUTUAL_INFO = "mutual_info"
    ANOVA_F = "anova_f"

    # Wrapper methods
    RFE = "rfe"

    # Embedded methods
    L1_REGULARIZATION = "l1"
    TREE_IMPORTANCE = "tree_importance"


class FeatureSelector:
    """
    Feature selection class for selecting most relevant features.

    Supports:
    - Filter methods: correlation, chi-square, mutual information, ANOVA F-test
    - Wrapper methods: recursive feature elimination (RFE)
    - Embedded methods: L1 regularization (LASSO), tree-based importance

    Parameters
    ----------
    method : str
        Selection method ('correlation', 'chi2', 'mutual_info', 'anova_f', 'rfe', 'l1', 'tree_importance')
    k_features : int or float, optional
        Number of features to select. If float (0-1), percentage of features (default: 10 or 0.5)
    task_type : str, optional
        'classification' or 'regression' (default: 'classification')
    correlation_threshold : float, optional
        For correlation method, minimum correlation threshold (default: 0.5)
    estimator : object, optional
        For RFE or embedded methods, estimator to use (default: None, uses RandomForest)

    Attributes
    ----------
    selected_features_ : List[str]
        Names of selected features
    feature_scores_ : Dict[str, float]
        Feature importance/relevance scores
    selector_ : object
        Fitted sklearn selector object
    """

    def __init__(
        self,
        method: str,
        k_features: Union[int, float] = 10,
        task_type: Literal["classification", "regression"] = "classification",
        correlation_threshold: float = 0.5,
        estimator=None,
    ):
        self.method = method
        self.k_features = k_features
        self.task_type = task_type
        self.correlation_threshold = correlation_threshold
        self.estimator = estimator

        # Fitted attributes
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
        self.selector_ = None
        self._is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None
    ) -> "FeatureSelector":
        """
        Fit the feature selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        feature_names : List[str], optional
            Feature names (default: X.columns)

        Returns
        -------
        self
            Fitted feature selector
        """
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                # Convert integer indices to strings for consistency
                feature_names = [str(i) for i in range(X.shape[1])]

        # Convert k_features percentage to absolute number
        k_features = self._convert_k_features(X.shape[1])

        # Select method
        if self.method == SelectionMethod.CORRELATION:
            self._fit_correlation(X, y, feature_names)
        elif self.method == SelectionMethod.CHI2:
            self._fit_chi2(X, y, feature_names, k_features)
        elif self.method == SelectionMethod.MUTUAL_INFO:
            self._fit_mutual_info(X, y, feature_names, k_features)
        elif self.method == SelectionMethod.ANOVA_F:
            self._fit_anova_f(X, y, feature_names, k_features)
        elif self.method == SelectionMethod.RFE:
            self._fit_rfe(X, y, feature_names, k_features)
        elif self.method == SelectionMethod.L1_REGULARIZATION:
            self._fit_l1(X, y, feature_names)
        elif self.method == SelectionMethod.TREE_IMPORTANCE:
            self._fit_tree_importance(X, y, feature_names, k_features)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            Transformed data with selected features
        """
        if not self._is_fitted:
            raise ValueError(
                "FeatureSelector must be fitted before transform. Call fit first."
            )

        return X[self.selected_features_]

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        feature_names : List[str], optional
            Feature names (default: X.columns)

        Returns
        -------
        pd.DataFrame
            Transformed data with selected features
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _convert_k_features(self, n_features: int) -> int:
        """Convert k_features to absolute number."""
        if isinstance(self.k_features, float) and 0 < self.k_features < 1:
            return max(1, int(n_features * self.k_features))
        return min(int(self.k_features), n_features)

    def _fit_correlation(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]
    ) -> None:
        """Fit correlation-based selection."""
        # Calculate correlation with target
        correlations = {}
        for col in feature_names:
            if col in X.columns:
                corr = abs(X[col].corr(y))
                correlations[col] = corr if not np.isnan(corr) else 0.0

        self.feature_scores_ = correlations

        # Select features above threshold
        self.selected_features_ = [
            col
            for col, corr in correlations.items()
            if corr >= self.correlation_threshold
        ]

        # If no features meet threshold, select top k
        if not self.selected_features_:
            k = self._convert_k_features(len(feature_names))
            sorted_features = sorted(
                correlations.items(), key=lambda x: x[1], reverse=True
            )
            self.selected_features_ = [col for col, _ in sorted_features[:k]]

    def _fit_chi2(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], k_features: int
    ) -> None:
        """Fit chi-square based selection."""
        # Ensure non-negative values for chi2
        X_positive = X.copy()
        X_positive = X_positive - X_positive.min() + 1e-10

        self.selector_ = SelectKBest(score_func=chi2, k=k_features)
        self.selector_.fit(X_positive, y)

        # Get selected features
        mask = self.selector_.get_support()
        self.selected_features_ = [
            feature_names[i] for i, selected in enumerate(mask) if selected
        ]

        # Get scores
        scores = self.selector_.scores_
        self.feature_scores_ = {feature_names[i]: float(scores[i]) for i in range(len(feature_names))}  # type: ignore[index,arg-type]

    def _fit_mutual_info(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], k_features: int
    ) -> None:
        """Fit mutual information based selection."""
        if self.task_type == "classification":
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression

        self.selector_ = SelectKBest(score_func=score_func, k=k_features)
        self.selector_.fit(X, y)

        # Get selected features
        mask = self.selector_.get_support()
        self.selected_features_ = [
            feature_names[i] for i, selected in enumerate(mask) if selected
        ]

        # Get scores
        scores = self.selector_.scores_
        self.feature_scores_ = {feature_names[i]: float(scores[i]) for i in range(len(feature_names))}  # type: ignore[index,arg-type]

    def _fit_anova_f(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], k_features: int
    ) -> None:
        """Fit ANOVA F-test based selection."""
        if self.task_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression

        self.selector_ = SelectKBest(score_func=score_func, k=k_features)
        self.selector_.fit(X, y)

        # Get selected features
        mask = self.selector_.get_support()
        self.selected_features_ = [
            feature_names[i] for i, selected in enumerate(mask) if selected
        ]

        # Get scores
        scores = self.selector_.scores_
        self.feature_scores_ = {feature_names[i]: float(scores[i]) for i in range(len(feature_names))}  # type: ignore[index,arg-type]

    def _fit_rfe(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], k_features: int
    ) -> None:
        """Fit recursive feature elimination."""
        # Use provided estimator or default
        if self.estimator is None:
            if self.task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            estimator = self.estimator

        self.selector_ = RFE(estimator=estimator, n_features_to_select=k_features)
        self.selector_.fit(X, y)

        # Get selected features
        mask = self.selector_.get_support()
        self.selected_features_ = [
            feature_names[i] for i, selected in enumerate(mask) if selected
        ]

        # Get ranking as scores (inverse, so higher is better)
        ranking = self.selector_.ranking_
        self.feature_scores_ = {
            feature_names[i]: 1.0 / ranking[i] for i in range(len(feature_names))
        }

    def _fit_l1(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> None:
        """Fit L1 (LASSO) regularization based selection."""
        if self.task_type == "classification":
            estimator = LogisticRegression(
                penalty="l1", solver="liblinear", random_state=42, max_iter=1000
            )
        else:
            estimator = LassoCV(cv=5, random_state=42, max_iter=1000)

        estimator.fit(X, y)

        # Get non-zero coefficients
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = coef[0]  # For multiclass, use first class
        else:
            coef = np.zeros(X.shape[1])

        # Store scores
        self.feature_scores_ = {
            feature_names[i]: abs(coef[i]) for i in range(len(feature_names))
        }

        # Select features with non-zero coefficients
        self.selected_features_ = [
            feature_names[i] for i in range(len(feature_names)) if abs(coef[i]) > 1e-10
        ]

        # If all features are zero, select top k
        if not self.selected_features_:
            k = self._convert_k_features(len(feature_names))
            sorted_features = sorted(
                self.feature_scores_.items(), key=lambda x: abs(x[1]), reverse=True
            )
            self.selected_features_ = [col for col, _ in sorted_features[:k]]

    def _fit_tree_importance(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], k_features: int
    ) -> None:
        """Fit tree-based feature importance selection."""
        # Use provided estimator or default
        if self.estimator is None:
            if self.task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            estimator = self.estimator

        # Fit and get importances
        estimator.fit(X, y)
        importances = estimator.feature_importances_

        # Store scores
        self.feature_scores_ = {
            feature_names[i]: importances[i] for i in range(len(feature_names))
        }

        # Select top k features
        sorted_features = sorted(
            self.feature_scores_.items(), key=lambda x: x[1], reverse=True
        )
        self.selected_features_ = [col for col, _ in sorted_features[:k_features]]

    def get_feature_ranking(self) -> pd.DataFrame:
        """
        Get ranking of all features by importance.

        Returns
        -------
        pd.DataFrame
            DataFrame with features and their scores, sorted by score
        """
        if not self._is_fitted:
            raise ValueError("FeatureSelector must be fitted before getting rankings")

        ranking_df = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "score": score,
                    "selected": feature in self.selected_features_,
                }
                for feature, score in self.feature_scores_.items()
            ]
        )

        return ranking_df.sort_values("score", ascending=False).reset_index(drop=True)

    def get_selection_summary(self) -> Dict[str, Union[str, int, float, List[str]]]:
        """
        Get summary of feature selection.

        Returns
        -------
        Dict
            Summary with total features, selected features, and method
        """
        return {
            "method": self.method,
            "total_features": len(self.feature_scores_),
            "selected_features": len(self.selected_features_),
            "selection_rate": (
                len(self.selected_features_) / len(self.feature_scores_)
                if self.feature_scores_
                else 0
            ),
            "selected_feature_names": self.selected_features_,
        }
