"""
Example demonstrating advanced imputation methods.

This example shows how to use KNN and Iterative (MICE) imputation
for handling missing values in datasets.
"""

import numpy as np
import pandas as pd

from automl.preprocessing.cleaners.advanced_imputation import (
    AdvancedMissingValueHandler,
)


def create_sample_data_with_missing():
    """Create sample data with missing values."""
    np.random.seed(42)

    # Create correlated data
    n_samples = 100
    feature1 = np.random.randn(n_samples)
    feature2 = 2 * feature1 + np.random.randn(n_samples) * 0.5
    feature3 = -1.5 * feature1 + np.random.randn(n_samples) * 0.3

    data = pd.DataFrame(
        {
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "category": np.random.choice(["A", "B", "C"], n_samples),
        }
    )

    # Introduce missing values (20%)
    missing_mask = np.random.rand(n_samples, 3) < 0.2
    data.loc[missing_mask[:, 0], "feature1"] = np.nan
    data.loc[missing_mask[:, 1], "feature2"] = np.nan
    data.loc[missing_mask[:, 2], "feature3"] = np.nan

    # Add categorical missing values
    data.loc[np.random.rand(n_samples) < 0.15, "category"] = None

    return data


def example_knn_imputation():
    """Example 1: KNN Imputation."""
    print("=" * 70)
    print("Example 1: KNN Imputation")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_missing()
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    print(
        f"\nTotal missing: {data.isnull().sum().sum()} "
        f"({data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100:.1f}%)"
    )

    # Apply KNN imputation
    handler = AdvancedMissingValueHandler(strategy="knn", n_neighbors=5)

    imputed_data = handler.fit_transform(data)

    print(f"\nAfter KNN imputation:")
    print(f"Missing values: {imputed_data.isnull().sum().sum()}")
    print(f"Shape: {imputed_data.shape}")
    print(f"\nFirst 5 rows:")
    print(imputed_data.head())

    # Show summary
    summary = handler.get_imputation_summary()
    print(f"\nImputation summary:")
    print(f"  Strategy: {summary['strategy']}")
    print(f"  N neighbors: {summary['n_neighbors']}")
    print(f"  Fitted columns: {len(summary['fitted_columns'])}")
    print(
        f"  Categorical columns encoded: {len(summary['categorical_columns_encoded'])}"
    )


def example_iterative_imputation():
    """Example 2: Iterative (MICE) Imputation."""
    print("\n" + "=" * 70)
    print("Example 2: Iterative (MICE) Imputation")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_missing()
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")

    # Apply iterative imputation
    handler = AdvancedMissingValueHandler(
        strategy="iterative", max_iter=10, random_state=42
    )

    imputed_data = handler.fit_transform(data)

    print(f"\nAfter Iterative imputation:")
    print(f"Missing values: {imputed_data.isnull().sum().sum()}")
    print(f"Shape: {imputed_data.shape}")
    print(f"\nFirst 5 rows:")
    print(imputed_data.head())


def example_comparison():
    """Example 3: Compare KNN vs Iterative imputation."""
    print("\n" + "=" * 70)
    print("Example 3: Comparing KNN vs Iterative Imputation")
    print("=" * 70)

    # Create same data for both methods
    np.random.seed(123)
    data = create_sample_data_with_missing()

    print(f"\nOriginal data statistics:")
    print(data.describe())

    # KNN imputation
    knn_handler = AdvancedMissingValueHandler(strategy="knn", n_neighbors=5)
    knn_result = knn_handler.fit_transform(data.copy())

    # Iterative imputation
    iter_handler = AdvancedMissingValueHandler(strategy="iterative", max_iter=10)
    iter_result = iter_handler.fit_transform(data.copy())

    print(f"\nKNN imputed statistics:")
    print(knn_result.describe())

    print(f"\nIterative imputed statistics:")
    print(iter_result.describe())

    print(f"\nDifference in means:")
    print(knn_result.mean() - iter_result.mean())


def example_transform_new_data():
    """Example 4: Transform new data with fitted imputer."""
    print("\n" + "=" * 70)
    print("Example 4: Transform New Data")
    print("=" * 70)

    # Fit on training data
    train_data = create_sample_data_with_missing()
    handler = AdvancedMissingValueHandler(strategy="knn", n_neighbors=5)
    handler.fit_transform(train_data)

    print("Fitted imputer on training data")

    # Create new test data with missing values
    np.random.seed(999)
    test_data = create_sample_data_with_missing()

    print(f"\nTest data missing values:\n{test_data.isnull().sum()}")

    # Transform test data
    transformed_test = handler.transform(test_data)

    print(f"\nTransformed test data missing values:")
    print(f"Total missing: {transformed_test.isnull().sum().sum()}")
    print(f"\nTest data shape: {transformed_test.shape}")


def example_real_world_usage():
    """Example 5: Real-world usage in preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("Example 5: Real-World Usage")
    print("=" * 70)

    # Load or create realistic data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 200),
            "income": np.random.randint(20000, 150000, 200),
            "credit_score": np.random.randint(300, 850, 200),
            "education": np.random.choice(
                ["High School", "Bachelor", "Master", "PhD"], 200
            ),
            "employment": np.random.choice(
                ["Employed", "Self-Employed", "Unemployed"], 200
            ),
        }
    )

    # Add realistic missing patterns (MCAR - Missing Completely At Random)
    data.loc[np.random.choice(200, 30, replace=False).tolist(), "age"] = np.nan
    data.loc[np.random.choice(200, 25, replace=False).tolist(), "income"] = np.nan
    data.loc[np.random.choice(200, 20, replace=False).tolist(), "credit_score"] = np.nan
    data.loc[np.random.choice(200, 15, replace=False).tolist(), "education"] = None

    print(f"Dataset shape: {data.shape}")
    print(f"\nMissing values by column:")
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            pct = missing_count / len(data) * 100
            print(f"  {col}: {missing_count} ({pct:.1f}%)")

    # Use KNN imputation for better accuracy
    print(f"\nApplying KNN imputation with k=7...")
    handler = AdvancedMissingValueHandler(strategy="knn", n_neighbors=7)

    imputed_data = handler.fit_transform(data)

    print(f"\nImputation complete!")
    print(f"  Missing values remaining: {imputed_data.isnull().sum().sum()}")
    print(f"\nData types preserved:")
    for col in imputed_data.columns:
        print(f"  {col}: {imputed_data[col].dtype}")

    print(f"\nSample of imputed data:")
    print(imputed_data.head(10))


if __name__ == "__main__":
    # Run examples
    example_knn_imputation()
    example_iterative_imputation()
    example_comparison()
    example_transform_new_data()
    example_real_world_usage()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
