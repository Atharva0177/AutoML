"""
Example: Using the Model Recommendation Engine

This example demonstrates how to use the intelligent model recommendation
system to automatically select the best models for your dataset.
"""

import numpy as np
import pandas as pd

from automl.models.recommender import ModelRecommender
from automl.pipeline.automl import AutoML

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Model Recommendation Engine - Example")
print("=" * 80)

# Example 1: Small Dataset - Classification
print("\n" + "=" * 80)
print("EXAMPLE 1: Small Dataset (200 samples) - Classification")
print("=" * 80)

# Create a small classification dataset
small_df = pd.DataFrame(
    np.random.randn(200, 8), columns=[f"feature_{i}" for i in range(8)]
)
small_df["target"] = np.random.randint(0, 2, 200)

# Initialize recommender
recommender = ModelRecommender(verbose=True)

# Analyze dataset and get recommendations
print("\nAnalyzing dataset characteristics...")
chars = recommender.analyze_dataset(
    small_df, target_column="target", problem_type="classification"
)

print(f"\nDataset Characteristics:")
print(f"  Samples: {chars.n_samples}")
print(f"  Features: {chars.n_features}")
print(f"  Size Category: {chars.size_category}")
print(f"  Has Missing Values: {chars.has_missing}")
print(f"  Is Imbalanced: {chars.is_imbalanced}")

# Get top 3 recommendations
print("\nGetting model recommendations...")
recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

print(f"\nTop 3 Recommended Models:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec.model_name.upper()}")
    print(f"   Score: {rec.score:.1f}/100")
    print(f"   Confidence: {rec.confidence}")
    print(f"   Expected Performance: {rec.expected_performance}")
    print(f"   Reasons:")
    for reason in rec.justification:
        print(f"     • {reason}")
    if rec.warnings:
        print(f"   Warnings:")
        for warning in rec.warnings:
            print(f"     ⚠ {warning}")

# Example 2: Large Dataset - Regression
print("\n\n" + "=" * 80)
print("EXAMPLE 2: Large Dataset (50,000 samples) - Regression")
print("=" * 80)

# Create a large regression dataset
large_df = pd.DataFrame(
    np.random.randn(50000, 25), columns=[f"feature_{i}" for i in range(25)]
)
large_df["target"] = np.random.randn(50000)

# Analyze and recommend
print("\nAnalyzing dataset characteristics...")
chars = recommender.analyze_dataset(
    large_df, target_column="target", problem_type="regression"
)

print(f"\nDataset Characteristics:")
print(f"  Samples: {chars.n_samples}")
print(f"  Features: {chars.n_features}")
print(f"  Size Category: {chars.size_category}")

recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

print(f"\nTop 3 Recommended Models:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec.model_name.upper()}")
    print(f"   Score: {rec.score:.1f}/100")
    print(f"   Expected Performance: {rec.expected_performance}")

# Example 3: Integration with AutoML Pipeline
print("\n\n" + "=" * 80)
print("EXAMPLE 3: Automatic Model Recommendation in AutoML Pipeline")
print("=" * 80)

# Create a medium-sized dataset
medium_df = pd.DataFrame(
    np.random.randn(5000, 15), columns=[f"feature_{i}" for i in range(15)]
)
medium_df["target"] = np.random.randint(0, 3, 5000)

print("\nTraining AutoML with automatic model recommendation...")
print("(The pipeline will analyze the dataset and train only the top 3 models)")

# Initialize AutoML - it automatically uses the recommender
automl = AutoML(
    problem_type="classification", use_cross_validation=False, verbose=False
)

# Fit - AutoML will automatically recommend and train top models
results = automl.fit(medium_df, target_column="target")

print(f"\nAutoML automatically selected and trained:")
if "model_comparison" in results and "rankings" in results["model_comparison"]:
    for result in results["model_comparison"]["rankings"]:
        print(f"  • {result['model_name']}: {result['score']:.4f}")

    print(f"\nBest Model: {results['best_model']}")
    print(f"Best Score: {results['model_comparison']['rankings'][0]['score']:.4f}")

# Example 4: Imbalanced Dataset
print("\n\n" + "=" * 80)
print("EXAMPLE 4: Imbalanced Dataset (90% majority class)")
print("=" * 80)

# Create imbalanced dataset
imbalanced_df = pd.DataFrame(
    np.random.randn(1000, 10), columns=[f"feature_{i}" for i in range(10)]
)
# 90% class 0, 10% class 1
imbalanced_df["target"] = np.concatenate(
    [np.zeros(900, dtype=int), np.ones(100, dtype=int)]
)

# Analyze
chars = recommender.analyze_dataset(
    imbalanced_df, target_column="target", problem_type="classification"
)

print(f"\nDataset Characteristics:")
print(f"  Samples: {chars.n_samples}")
print(f"  Is Imbalanced: {chars.is_imbalanced}")
print(f"  Number of Classes: {chars.n_classes}")

recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

print(f"\nRecommendations for imbalanced data:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec.model_name.upper()} - Score: {rec.score:.1f}/100")
    # Models that handle imbalanced data well should be recommended
    if rec.warnings:
        print(f"   Warnings:")
        for warning in rec.warnings:
            print(f"     ⚠ {warning}")

# Example 5: Dataset with Missing Values
print("\n\n" + "=" * 80)
print("EXAMPLE 5: Dataset with Missing Values")
print("=" * 80)

# Create dataset with missing values
missing_df = pd.DataFrame(
    np.random.randn(500, 8), columns=[f"feature_{i}" for i in range(8)]
)
# Add missing values (20% in first column)
missing_df.iloc[::5, 0] = np.nan
missing_df["target"] = np.random.randint(0, 2, 500)

# Analyze
chars = recommender.analyze_dataset(
    missing_df, target_column="target", problem_type="classification"
)

print(f"\nDataset Characteristics:")
print(f"  Has Missing Values: {chars.has_missing}")
print(f"  Missing Percentage: {chars.missing_percentage:.1f}%")

recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

print(f"\nTop recommendations (models that handle missing values):")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec.model_name} - Score: {rec.score:.1f}/100")

print("\n" + "=" * 80)
print("Key Takeaways:")
print("=" * 80)
print("""
The Model Recommendation Engine helps you:

1. Save Time: Automatically identifies best models without manual trial & error
2. Better Performance: Matches model capabilities to dataset characteristics
3. Avoid Pitfalls: Warns about potential issues (small dataset, imbalance, etc.)
4. Make Informed Decisions: Provides clear justifications for each recommendation

The recommender considers:
- Dataset size (small, medium, large)
- Number of features
- Missing values
- Class imbalance (for classification)
- High cardinality features
- Sparsity
- Problem type (classification vs regression)

Try it with your own datasets!
""")
