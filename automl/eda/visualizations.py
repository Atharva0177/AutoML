"""Visualization generation for EDA."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from automl.utils.helpers import ensure_dir
from automl.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


class VisualizationGenerator:
    """Generate various visualizations for EDA."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.generated_plots: List[str] = []

    def generate_all_visualizations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        Generate all standard visualizations.

        Args:
            df: Input DataFrame
            target_column: Target column name (optional)
            correlation_matrix: Pre-computed correlation matrix (optional)

        Returns:
            Dictionary mapping plot types to file paths
        """
        logger.info("Generating visualizations...")
        ensure_dir(self.output_dir)

        plots = {}

        # Missing values heatmap
        try:
            plots["missing_values"] = self.plot_missing_values(df)
        except Exception as e:
            logger.warning(f"Failed to generate missing values plot: {e}")

        # Correlation heatmap
        if (
            correlation_matrix is not None
            or len(df.select_dtypes(include=[np.number]).columns) > 1
        ):
            try:
                plots["correlation"] = self.plot_correlation_heatmap(
                    df, correlation_matrix
                )
            except Exception as e:
                logger.warning(f"Failed to generate correlation heatmap: {e}")

        # Distribution plots for numerical features
        try:
            plots["distributions"] = self.plot_distributions(df)
        except Exception as e:
            logger.warning(f"Failed to generate distribution plots: {e}")

        # Target distribution
        if target_column and target_column in df.columns:
            try:
                plots["target_distribution"] = self.plot_target_distribution(
                    df, target_column
                )
            except Exception as e:
                logger.warning(f"Failed to generate target distribution: {e}")

        # Box plots for outlier detection
        try:
            plots["boxplots"] = self.plot_boxplots(df)
        except Exception as e:
            logger.warning(f"Failed to generate box plots: {e}")

        self.generated_plots = list(plots.values())
        logger.info(f"Generated {len(plots)} visualizations")
        return plots

    def plot_missing_values(self, df: pd.DataFrame) -> str:
        """
        Plot missing values heatmap.

        Args:
            df: Input DataFrame

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate missing values
        missing = df.isna()

        if missing.sum().sum() == 0:
            # No missing values
            ax.text(
                0.5,
                0.5,
                "No Missing Values Detected",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.axis("off")
        else:
            # Plot heatmap of missing values
            sns.heatmap(missing, yticklabels=False, cbar=True, cmap="viridis", ax=ax)
            ax.set_title("Missing Values Heatmap", fontsize=14, fontweight="bold")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")

        plt.tight_layout()
        filepath = self.output_dir / "missing_values.png"
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Plot correlation heatmap.

        Args:
            df: Input DataFrame
            correlation_matrix: Pre-computed correlation matrix

        Returns:
            Path to saved plot
        """
        if correlation_matrix is None:
            numerical_df = df.select_dtypes(include=[np.number])
            if len(numerical_df.columns) < 2:
                logger.warning("Not enough numerical columns for correlation plot")
                return ""
            correlation_matrix = numerical_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Plot heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=len(correlation_matrix.columns)
            <= 10,  # Annotate if not too many columns
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt=".2f",
            ax=ax,
        )

        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = self.output_dir / "correlation_heatmap.png"
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_distributions(self, df: pd.DataFrame, max_plots: int = 12) -> str:
        """
        Plot distributions of numerical features.

        Args:
            df: Input DataFrame
            max_plots: Maximum number of plots to generate

        Returns:
            Path to saved plot
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numerical_cols:
            logger.warning("No numerical columns for distribution plots")
            return ""

        # Limit number of plots
        numerical_cols = numerical_cols[:max_plots]
        n_cols = min(3, len(numerical_cols))
        n_rows = int(np.ceil(len(numerical_cols) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            data = df[col].dropna()

            if len(data) > 0:
                # Plot histogram with KDE
                ax.hist(data, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
                ax2 = ax.twinx()
                data.plot(kind="kde", ax=ax2, color="red", linewidth=2)
                ax2.set_ylabel("")
                ax2.set_yticks([])

                ax.set_title(f"{col}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()

        filepath = self.output_dir / "distributions.png"
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_target_distribution(self, df: pd.DataFrame, target_column: str) -> str:
        """
        Plot target variable distribution.

        Args:
            df: Input DataFrame
            target_column: Target column name

        Returns:
            Path to saved plot
        """
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found")
            return ""

        target = df[target_column].dropna()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine if categorical or numerical
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
            # Numerical target - histogram + KDE
            ax.hist(target, bins=30, alpha=0.7, color="coral", edgecolor="black")
            ax2 = ax.twinx()
            target.plot(kind="kde", ax=ax2, color="darkred", linewidth=2)
            ax2.set_ylabel("")
            ax2.set_yticks([])
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        else:
            # Categorical target - bar plot
            value_counts = target.value_counts()
            ax.bar(range(len(value_counts)), value_counts.values, color="coral", edgecolor="black")  # type: ignore[arg-type]
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha="right")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")

            # Add value labels on bars
            for i, v in enumerate(value_counts.values):
                ax.text(i, v, str(v), ha="center", va="bottom")

        ax.set_title(
            f"Target Distribution: {target_column}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.output_dir / "target_distribution.png"
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_boxplots(self, df: pd.DataFrame, max_plots: int = 12) -> str:
        """
        Plot boxplots for outlier detection.

        Args:
            df: Input DataFrame
            max_plots: Maximum number of plots

        Returns:
            Path to saved plot
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numerical_cols:
            logger.warning("No numerical columns for boxplots")
            return ""

        numerical_cols = numerical_cols[:max_plots]
        n_cols = min(3, len(numerical_cols))
        n_rows = int(np.ceil(len(numerical_cols) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            data = df[col].dropna()

            if len(data) > 0:
                bp = ax.boxplot(data, vert=True, patch_artist=True)
                bp["boxes"][0].set_facecolor("lightblue")
                bp["boxes"][0].set_edgecolor("blue")
                bp["medians"][0].set_color("red")
                bp["medians"][0].set_linewidth(2)

                ax.set_title(f"{col}", fontsize=12, fontweight="bold")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3, axis="y")

        # Hide unused subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Box Plots (Outlier Detection)", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        filepath = self.output_dir / "boxplots.png"
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def plot_feature_target_relationship(
        self,
        df: pd.DataFrame,
        target_column: str,
        top_n: int = 10,
    ) -> str:
        """
        Plot relationships between top features and target.

        Args:
            df: Input DataFrame
            target_column: Target column name
            top_n: Number of top features to plot

        Returns:
            Path to saved plot
        """
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found")
            return ""

        # Get numerical columns (excluding target)
        numerical_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col != target_column
        ]

        if not numerical_cols:
            logger.warning("No numerical features for relationship plots")
            return ""

        # Calculate correlations with target
        correlations = {}
        for col in numerical_cols:
            corr = df[col].corr(df[target_column])
            if not np.isnan(corr):
                correlations[col] = abs(corr)

        # Get top N correlated features
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        top_feature_names = [f[0] for f in top_features]

        if not top_feature_names:
            return ""

        # Create scatter plots
        n_cols = min(3, len(top_feature_names))
        n_rows = int(np.ceil(len(top_feature_names) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, feature in enumerate(top_feature_names):
            ax = axes[idx]
            ax.scatter(df[feature], df[target_column], alpha=0.5, s=20)
            ax.set_xlabel(feature)
            ax.set_ylabel(target_column)
            ax.set_title(f"Corr: {correlations[feature]:.3f}", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(top_feature_names), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Top Features vs {target_column}", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        filepath = self.output_dir / "feature_target_relationships.png"
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def clear_plots(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close("all")
        self.generated_plots = []
