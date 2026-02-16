"""
Example demonstrating the preprocessing pipeline functionality.

This example shows how to:
1. Handle missing values
2. Encode categorical features
3. Scale numerical features
4. Split data into train/test sets
5. Build and use preprocessing pipelines
"""

import numpy as np
import pandas as pd
from pathlib import Path

from automl.preprocessing import (
    MissingValueHandler,
    NumericalScaler,
    CategoricalEncoder,
    DataSplitter,
    PipelineBuilder,
)


def create_sample_data():
    """Create a sample dataset with various data types and missing values."""
    np.random.seed(42)

    n_samples = 200

    data = {
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.normal(50000, 20000, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "employment_length": np.random.randint(0, 30, n_samples),
        "education": np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"], n_samples
        ),
        "occupation": np.random.choice(
            ["Engineer", "Teacher", "Doctor", "Artist", "Manager"], n_samples
        ),
        "marital_status": np.random.choice(
            ["Single", "Married", "Divorced"], n_samples
        ),
        "loan_approved": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    }

    df = pd.DataFrame(data)

    # Introduce missing values (10% for numerical, 5% for categorical)
    for col in ["age", "income", "credit_score"]:
        missing_idx = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
        df.loc[missing_idx.tolist(), col] = np.nan  # type: ignore[index]

    for col in ["education", "occupation"]:
        missing_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
        df.loc[missing_idx.tolist(), col] = np.nan  # type: ignore[index]

    return df


def example_1_missing_values():
    """Example 1: Handling missing values."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Missing Value Imputation")
    print("=" * 80)

    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")

    # Strategy 1: Mean imputation (numerical) and mode (categorical)
    print("\n--- Strategy 1: Mean/Mode Imputation ---")
    handler = MissingValueHandler(strategy="mean")
    df_imputed = handler.fit_transform(df.drop(columns=["loan_approved"]))

    print(f"After imputation: {df_imputed.isnull().sum().sum()} missing values")
    summary = handler.get_imputation_summary()
    print(f"Imputation values: {summary['imputation_values']}")

    # Strategy 2: Median imputation
    print("\n--- Strategy 2: Median Imputation ---")
    handler_median = MissingValueHandler(strategy="median")
    df_median = handler_median.fit_transform(df.drop(columns=["loan_approved"]))
    print(f"After median imputation: {df_median.isnull().sum().sum()} missing values")

    # Strategy 3: Drop rows with missing values
    print("\n--- Strategy 3: Drop Missing ---")
    handler_drop = MissingValueHandler(strategy="drop", threshold=0.5)
    df_dropped = handler_drop.fit_transform(df.drop(columns=["loan_approved"]))
    print(f"Shape after dropping: {df_dropped.shape} (original: {df.shape})")


def example_2_encoding():
    """Example 2: Categorical encoding."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Categorical Encoding")
    print("=" * 80)

    df = create_sample_data()
    df = df.dropna()  # Remove missing for this example

    print(
        f"\nCategorical columns: {df.select_dtypes(include='object').columns.tolist()}"
    )
    print(f"Unique values per category:")
    for col in df.select_dtypes(include="object").columns:
        print(f"  {col}: {df[col].nunique()} categories")

    # Strategy 1: One-Hot Encoding
    print("\n--- Strategy 1: One-Hot Encoding ---")
    encoder_oh = CategoricalEncoder(method="onehot")
    df_onehot = encoder_oh.fit_transform(df.drop(columns=["loan_approved"]))
    print(
        f"Shape after one-hot: {df_onehot.shape} (original: {df.drop(columns=['loan_approved']).shape})"
    )
    print(f"New columns created: {len(encoder_oh.encoded_columns)}")

    # Strategy 2: Label Encoding
    print("\n--- Strategy 2: Label Encoding ---")
    encoder_label = CategoricalEncoder(method="label")
    df_label = encoder_label.fit_transform(df.drop(columns=["loan_approved"]))
    print(f"Shape after label encoding: {df_label.shape}")
    print(f"Category mappings:")
    for col, mapping in encoder_label.category_mappings.items():
        print(f"  {col}: {mapping}")

    # Strategy 3: Ordinal Encoding
    print("\n--- Strategy 3: Ordinal Encoding ---")
    encoder_ordinal = CategoricalEncoder(method="ordinal")
    df_ordinal = encoder_ordinal.fit_transform(df.drop(columns=["loan_approved"]))
    print(f"Shape after ordinal encoding: {df_ordinal.shape}")


def example_3_scaling():
    """Example 3: Numerical scaling."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Numerical Scaling")
    print("=" * 80)

    df = create_sample_data()
    df = df.dropna()

    numerical_cols = ["age", "income", "credit_score", "employment_length"]
    print(f"\nNumerical columns statistics (before scaling):")
    print(df[numerical_cols].describe())

    # Strategy 1: Standard Scaling (z-score normalization)
    print("\n--- Strategy 1: Standard Scaling ---")
    scaler_std = NumericalScaler(method="standard")
    df_std = scaler_std.fit_transform(df[numerical_cols])
    print("Statistics after standard scaling:")
    print(df_std.describe())

    # Strategy 2: Min-Max Scaling
    print("\n--- Strategy 2: Min-Max Scaling [0, 1] ---")
    scaler_mm = NumericalScaler(method="minmax", feature_range=(0, 1))
    df_mm = scaler_mm.fit_transform(df[numerical_cols])
    print("Statistics after min-max scaling:")
    print(df_mm.describe())

    # Strategy 3: Robust Scaling (robust to outliers)
    print("\n--- Strategy 3: Robust Scaling ---")
    scaler_robust = NumericalScaler(method="robust")
    df_robust = scaler_robust.fit_transform(df[numerical_cols])
    print("Statistics after robust scaling:")
    print(df_robust.describe())


def example_4_splitting():
    """Example 4: Train-test splitting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Data Splitting")
    print("=" * 80)

    df = create_sample_data()
    df = df.dropna()

    X = df.drop(columns=["loan_approved"])
    y = df["loan_approved"]

    print(f"\nTotal samples: {len(df)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Strategy 1: Simple train-test split
    print("\n--- Strategy 1: Simple Train-Test Split (80/20) ---")
    splitter = DataSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)  # type: ignore[misc]

    summary = splitter.get_split_summary(X_train, X_test, y_train, y_test)
    print(f"Train size: {summary['train_size']} ({summary['train_ratio']:.1%})")
    print(f"Test size: {summary['test_size']} ({summary['test_ratio']:.1%})")
    print(f"Train class distribution: {summary['target_distribution_train']}")
    print(f"Test class distribution: {summary['target_distribution_test']}")

    # Strategy 2: Train-validation-test split
    print("\n--- Strategy 2: Train-Val-Test Split (60/20/20) ---")
    splitter_3way = DataSplitter(test_size=0.2, validation_size=0.2, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter_3way.split(X, y)  # type: ignore[misc]

    summary = splitter_3way.get_split_summary(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    print(f"Train size: {summary['train_size']} ({summary['train_ratio']:.1%})")
    print(f"Val size: {summary['val_size']} ({summary['val_ratio']:.1%})")
    print(f"Test size: {summary['test_size']} ({summary['test_ratio']:.1%})")

    # Strategy 3: Stratified split (preserves class distribution)
    print("\n--- Strategy 3: Stratified Split ---")
    splitter_strat = DataSplitter(test_size=0.2, stratify=True, random_state=42)
    X_train, X_test, y_train, y_test = splitter_strat.split(X, y)  # type: ignore[misc]

    summary = splitter_strat.get_split_summary(X_train, X_test, y_train, y_test)
    print(f"Train distribution: {summary['target_distribution_train']}")
    print(f"Test distribution: {summary['target_distribution_test']}")


def example_5_pipeline():
    """Example 5: Complete preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Preprocessing Pipeline")
    print("=" * 80)

    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Split target before preprocessing
    X = df.drop(columns=["loan_approved"])
    y = df["loan_approved"]

    # Build preprocessing pipeline
    print("\n--- Building Preprocessing Pipeline ---")
    pipeline = PipelineBuilder()
    pipeline.add_missing_handler(strategy="mean")
    pipeline.add_encoder(method="label")
    pipeline.add_scaler(method="standard")

    print(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")

    # Fit and transform training data
    print("\n--- Fitting and Transforming Data ---")
    X_transformed = pipeline.fit_transform(X)

    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Missing values after pipeline: {X_transformed.isnull().sum().sum()}")
    print(f"\nPipeline summary:")
    summary = pipeline.get_pipeline_summary()
    for key, value in summary.items():
        if key != "steps":
            print(f"  {key}: {value}")

    # Save pipeline
    print("\n--- Saving Pipeline ---")
    output_dir = Path("outputs/preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_dir / "pipeline.pkl"
    pipeline.save(str(pipeline_path))
    print(f"Pipeline saved to {pipeline_path}")

    # Load and use pipeline on new data
    print("\n--- Loading and Using Pipeline ---")
    loaded_pipeline = PipelineBuilder.load(str(pipeline_path))

    # Create new sample data
    new_data = create_sample_data().head(10).drop(columns=["loan_approved"])
    print(f"New data shape: {new_data.shape}")

    # Transform new data
    new_transformed = loaded_pipeline.transform(new_data)
    print(f"Transformed new data shape: {new_transformed.shape}")
    print(f"First row of transformed data:\n{new_transformed.iloc[0]}")


def main():
    """Run all preprocessing examples."""
    print("\n" + "=" * 80)
    print(" " * 20 + "PREPROCESSING MODULE EXAMPLES")
    print("=" * 80)

    example_1_missing_values()
    example_2_encoding()
    example_3_scaling()
    example_4_splitting()
    example_5_pipeline()

    print("\n" + "=" * 80)
    print(" " * 25 + "ALL EXAMPLES COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
