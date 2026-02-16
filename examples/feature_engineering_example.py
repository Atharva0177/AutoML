"""
Feature Engineering Example

This example demonstrates how to use the FeatureEngineer class to create
new features from existing ones using various techniques.
"""

import numpy as np
import pandas as pd
from automl.preprocessing.feature_engineering import FeatureEngineer


def create_sample_data():
    """Create sample dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 200),
            "income": np.random.randint(20000, 150000, 200),
            "credit_score": np.random.randint(300, 850, 200),
            "loan_amount": np.random.randint(5000, 500000, 200),
            "employment_years": np.random.randint(0, 40, 200),
        }
    )


def example_1_mathematical_transformations():
    """Example 1: Mathematical transformations."""
    print("=" * 80)
    print("Example 1: Mathematical Transformations")
    print("=" * 80)

    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # Apply various mathematical transformations
    fe = FeatureEngineer(
        transformations={
            "income": "log",  # Log transform for skewed income
            "loan_amount": "sqrt",  # Sqrt for loan amount
            "age": "square",  # Square of age
            "credit_score": "inverse",  # Inverse of credit score
        }
    )

    result = fe.fit_transform(df)
    print(f"\nTransformed data shape: {result.shape}")
    print(f"New columns: {[col for col in result.columns if col not in df.columns]}")
    print("\nSample of transformed features:")
    print(result[["income", "income_log", "age", "age_square"]].head())


def example_2_binning():
    """Example 2: Feature binning/discretization."""
    print("\n" + "=" * 80)
    print("Example 2: Feature Binning")
    print("=" * 80)

    df = create_sample_data()

    # Bin features into discrete categories
    fe = FeatureEngineer(
        binning_config={
            "age": {"n_bins": 5, "strategy": "quantile"},  # Age quintiles
            "income": {"n_bins": 3, "strategy": "uniform"},  # Income terciles
            "credit_score": {
                "n_bins": 4,
                "strategy": "kmeans",
            },  # Credit score quartiles
        }
    )

    result = fe.fit_transform(df)
    print(f"\nOriginal continuous features binned into discrete categories")
    print(f"New columns: {[col for col in result.columns if 'binned' in col]}")

    print("\nAge distribution across bins:")
    print(result.groupby("age_binned")["age"].agg(["min", "max", "count"]))

    print("\nIncome distribution across bins:")
    print(result.groupby("income_binned")["income"].agg(["min", "max", "count"]))


def example_3_interaction_features():
    """Example 3: Interaction features."""
    print("\n" + "=" * 80)
    print("Example 3: Interaction Features")
    print("=" * 80)

    df = create_sample_data()

    # Create interaction features between related variables
    fe = FeatureEngineer(
        interaction_features=[
            ("income", "employment_years"),  # Income * experience
            ("loan_amount", "credit_score"),  # Loan * creditworthiness
            ("age", "income"),  # Age * income
        ]
    )

    result = fe.fit_transform(df)
    print(f"\nCreated {len(fe.interaction_features)} interaction features")
    print(f"New columns: {[col for col in result.columns if '_x_' in col]}")

    print("\nSample interaction features:")
    print(result[["income", "employment_years", "income_x_employment_years"]].head())


def example_4_polynomial_features():
    """Example 4: Polynomial features."""
    print("\n" + "=" * 80)
    print("Example 4: Polynomial Features")
    print("=" * 80)

    df = create_sample_data()

    # Create polynomial features
    fe = FeatureEngineer(polynomial_degree=2, include_bias=False)

    result = fe.fit_transform(df, numerical_cols=["age", "income", "credit_score"])

    print(f"\nOriginal features: {df.shape[1]}")
    print(f"After polynomial features: {result.shape[1]}")
    print(
        f"New polynomial columns: {[col for col in result.columns if '^' in col or ' ' in col][:10]}..."
    )

    # Get feature summary
    summary = fe.get_feature_summary()
    print(f"\nFeature engineering summary:")
    for feature_type, count in summary.items():
        print(f"  {feature_type}: {count}")


def example_5_interaction_only_polynomial():
    """Example 5: Polynomial with interaction only."""
    print("\n" + "=" * 80)
    print("Example 5: Polynomial Features (Interactions Only)")
    print("=" * 80)

    df = create_sample_data()

    # Create only interaction terms, no squared terms
    fe = FeatureEngineer(polynomial_degree=2, interaction_only=True)

    result = fe.fit_transform(df, numerical_cols=["age", "income", "credit_score"])

    poly_cols = [col for col in result.columns if " " in col]
    print(f"\nCreated {len(poly_cols)} interaction features")
    print(f"Interaction columns: {poly_cols[:5]}...")

    # Verify no squared terms
    squared_cols = [col for col in result.columns if "^2" in col]
    print(f"Squared terms: {len(squared_cols)} (should be 0)")


def example_6_combined_engineering():
    """Example 6: Combining all feature engineering techniques."""
    print("\n" + "=" * 80)
    print("Example 6: Combined Feature Engineering")
    print("=" * 80)

    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")

    # Apply all techniques together
    fe = FeatureEngineer(
        polynomial_degree=2,
        interaction_only=True,
        interaction_features=[("age", "employment_years"), ("income", "credit_score")],
        binning_config={"age": {"n_bins": 5, "strategy": "quantile"}},
        transformations={"income": "log", "loan_amount": "sqrt"},
    )

    result = fe.fit_transform(
        df, numerical_cols=["income", "credit_score", "loan_amount"]
    )

    print(f"Transformed data shape: {result.shape}")
    print(f"Features added: {result.shape[1] - df.shape[1]}")

    summary = fe.get_feature_summary()
    print(f"\nFeature engineering summary:")
    for feature_type, count in summary.items():
        print(f"  {feature_type}: {count}")

    print(f"\nSample of engineered features:")
    eng_cols = [col for col in result.columns if col not in df.columns][:5]
    print(result[eng_cols].head())


def example_7_train_test_consistency():
    """Example 7: Ensuring consistency between train and test."""
    print("\n" + "=" * 80)
    print("Example 7: Train/Test Consistency")
    print("=" * 80)

    df = create_sample_data()

    # Split into train and test
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]

    print(f"Train size: {train_df.shape}")
    print(f"Test size: {test_df.shape}")

    # Fit on train data
    fe = FeatureEngineer(
        polynomial_degree=2,
        binning_config={"age": {"n_bins": 4, "strategy": "quantile"}},
        transformations={"income": "log"},
    )

    train_result = fe.fit_transform(train_df, numerical_cols=["age", "income"])
    print(f"\nTrain data after feature engineering: {train_result.shape}")

    # Transform test data using same fitted engineer
    test_result = fe.transform(test_df)
    print(f"Test data after feature engineering: {test_result.shape}")

    # Verify same columns
    assert list(train_result.columns) == list(test_result.columns), "Column mismatch!"
    print(f"\n✓ Train and test have same features: {train_result.shape[1]} columns")


def example_8_real_world_pipeline():
    """Example 8: Real-world feature engineering pipeline."""
    print("\n" + "=" * 80)
    print("Example 8: Real-World Feature Engineering Pipeline")
    print("=" * 80)

    # Create more realistic dataset
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "customer_age": np.random.randint(20, 70, 300),
            "annual_income": np.random.lognormal(10, 1, 300),
            "credit_utilization": np.random.uniform(0, 1, 300),
            "num_accounts": np.random.poisson(3, 300),
            "account_age_months": np.random.randint(1, 120, 300),
            "payment_history_score": np.random.randint(0, 100, 300),
        }
    )

    print(f"Dataset: {df.shape}")
    print(f"\nSample data:")
    print(df.head())

    # Design feature engineering strategy
    fe = FeatureEngineer(
        # Log transform for skewed income
        transformations={"annual_income": "log", "account_age_months": "sqrt"},
        # Create domain-specific interactions
        interaction_features=[
            ("annual_income", "credit_utilization"),  # Income * utilization
            ("customer_age", "account_age_months"),  # Customer age * account age
            ("num_accounts", "payment_history_score"),  # Accounts * payment history
        ],
        # Bin customer age into life stages
        binning_config={
            "customer_age": {
                "n_bins": 4,
                "strategy": "uniform",
            },  # Young, middle, senior, elderly
            "credit_utilization": {
                "n_bins": 5,
                "strategy": "quantile",
            },  # Very low to very high
        },
        # Add polynomial features for key metrics
        polynomial_degree=2,
        interaction_only=True,
    )

    result = fe.fit_transform(
        df, numerical_cols=["payment_history_score", "num_accounts"]
    )

    print(f"\n{'Feature Engineering Results':^80}")
    print(f"{'-' * 80}")
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {result.shape[1]}")
    print(f"Features added: {result.shape[1] - df.shape[1]}")

    summary = fe.get_feature_summary()
    print(f"\nBreakdown:")
    print(f"  Mathematical transformations: {summary['transformations']}")
    print(f"  Binned features: {summary['binned_features']}")
    print(f"  Interaction features: {summary['interactions']}")
    print(f"  Polynomial features: {summary['polynomial_features']}")

    print(f"\nNew feature columns:")
    new_cols = [col for col in result.columns if col not in df.columns]
    for i, col in enumerate(new_cols, 1):
        print(f"  {i:2d}. {col}")


if __name__ == "__main__":
    example_1_mathematical_transformations()
    example_2_binning()
    example_3_interaction_features()
    example_4_polynomial_features()
    example_5_interaction_only_polynomial()
    example_6_combined_engineering()
    example_7_train_test_consistency()
    example_8_real_world_pipeline()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
