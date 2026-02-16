"""
AutoML End-to-End Example.

This example demonstrates the complete AutoML pipeline from
data loading to model training and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from automl.pipeline import AutoML


def example_1_basic_classification():
    """Example 1: Basic classification with AutoML."""
    print("=" * 80)
    print("Example 1: Basic Classification Pipeline")
    print("=" * 80)

    # Load iris dataset
    iris = load_iris()  # type: ignore[misc]
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore[union-attr]
    df["species"] = iris.target  # type: ignore[union-attr]

    print(f"\nDataset: Iris")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(iris.feature_names)}")  # type: ignore
    print(f"Classes: {len(np.unique(iris.target))}")  # type: ignore

    # Create AutoML instance
    automl = AutoML(
        problem_type="classification",
        use_cross_validation=False,
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        verbose=True,
    )

    # Train models
    results = automl.fit(
        data=df,
        target_column="species",
        models_to_try=["logistic_regression", "random_forest", "xgboost"],
    )

    # Make predictions on test data
    test_data = df.drop(columns=["species"]).head(10)
    predictions = automl.predict(test_data)
    probabilities = automl.predict_proba(test_data)

    print("\n" + "=" * 80)
    print("Predictions (first 10 samples):")
    print("=" * 80)
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Class {pred}, Probabilities: {proba}")

    return automl, results


def example_2_classification_with_cv():
    """Example 2: Classification with cross-validation."""
    print("\n\n" + "=" * 80)
    print("Example 2: Classification with Cross-Validation")
    print("=" * 80)

    # Load iris dataset
    iris = load_iris()  # type: ignore[misc]
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore[union-attr]
    df["species"] = iris.target  # type: ignore[union-attr]

    # Create AutoML instance with CV
    automl = AutoML(
        problem_type="classification",
        use_cross_validation=True,
        cv_folds=5,
        test_size=0.2,
        random_state=42,
        verbose=True,
    )

    # Train models
    results = automl.fit(
        data=df,
        target_column="species",
        models_to_try=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
    )

    return automl, results


def example_3_regression():
    """Example 3: Regression pipeline."""
    print("\n\n" + "=" * 80)
    print("Example 3: Regression Pipeline")
    print("=" * 80)

    # Load diabetes dataset
    diabetes = load_diabetes()  # type: ignore[misc]
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)  # type: ignore[union-attr]
    df["target"] = diabetes.target  # type: ignore[union-attr]

    print(f"\nDataset: Diabetes")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(diabetes.feature_names)}")  # type: ignore

    # Create AutoML instance
    automl = AutoML(
        problem_type="regression",
        use_cross_validation=False,
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        verbose=True,
    )

    # Train models
    results = automl.fit(
        data=df,
        target_column="target",
        models_to_try=["linear_regression", "random_forest", "xgboost", "lightgbm"],
    )

    # Make predictions
    test_data = df.drop(columns=["target"]).head(10)
    predictions = automl.predict(test_data)

    print("\n" + "=" * 80)
    print("Predictions (first 10 samples):")
    print("=" * 80)
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred:.2f}")

    return automl, results


def example_4_auto_detection():
    """Example 4: Automatic problem type detection."""
    print("\n\n" + "=" * 80)
    print("Example 4: Automatic Problem Type Detection")
    print("=" * 80)

    # Create a binary classification dataset
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y

    print(f"\nDataset: Synthetic Binary Classification")
    print(f"Samples: {len(df)}")
    print(f"Features: 20")

    # Create AutoML with auto-detection
    automl = AutoML(
        problem_type=None,  # Auto-detect
        use_cross_validation=True,
        cv_folds=5,
        random_state=42,
        verbose=True,
    )

    # Train models
    results = automl.fit(data=df, target_column="target")

    print(f"\nDetected problem type: {results['problem_type']}")

    return automl, results


def example_5_csv_workflow():
    """Example 5: Complete workflow from CSV file."""
    print("\n\n" + "=" * 80)
    print("Example 5: Complete Workflow from CSV File")
    print("=" * 80)

    # Create a CSV file
    iris = load_iris()  # type: ignore[misc]
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore[union-attr]
    df["species"] = iris.target  # type: ignore[union-attr]

    csv_path = "iris_dataset.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nCreated CSV file: {csv_path}")

    # Create AutoML instance
    automl = AutoML(
        use_cross_validation=True, cv_folds=5, random_state=42, verbose=True
    )

    # Train from CSV
    results = automl.fit(
        data=csv_path,
        target_column="species",
        models_to_try=["logistic_regression", "random_forest", "xgboost"],
    )

    # Save the model
    automl.save("saved_models/iris_model")
    print("\nModel saved to: saved_models/iris_model")

    # Clean up
    import os

    os.remove(csv_path)

    return automl, results


def example_6_model_comparison():
    """Example 6: Detailed model comparison."""
    print("\n\n" + "=" * 80)
    print("Example 6: Detailed Model Comparison")
    print("=" * 80)

    # Load iris dataset
    iris = load_iris()  # type: ignore[misc]
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore[union-attr]
    df["species"] = iris.target  # type: ignore[union-attr]

    # Create AutoML instance
    automl = AutoML(
        problem_type="classification",
        use_cross_validation=True,
        cv_folds=5,
        random_state=42,
        verbose=True,
    )

    # Train all available models
    results = automl.fit(
        data=df,
        target_column="species",
        # models_to_try=None means all models
    )

    # Display detailed comparison
    print("\n" + "=" * 80)
    print("Detailed Model Comparison")
    print("=" * 80)

    rankings = results["model_comparison"]["rankings"]

    print(f"\n{'Rank':<6} {'Model':<30} {'Score':<12} {'Time (s)':<10}")
    print("-" * 70)

    for ranking in rankings:
        print(
            f"{ranking['rank']:<6} "
            f"{ranking['model_name']:<30} "
            f"{ranking['score']:<12.4f} "
            f"{ranking.get('training_time', 0):<10.2f}"
        )

    return automl, results


def main():
    """Run all examples."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "AutoML - End-to-End Examples" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Run examples
    try:
        # Example 1: Basic classification
        automl1, results1 = example_1_basic_classification()

        # Example 2: Classification with CV
        automl2, results2 = example_2_classification_with_cv()

        # Example 3: Regression
        automl3, results3 = example_3_regression()

        # Example 4: Auto-detection
        automl4, results4 = example_4_auto_detection()

        # Example 5: CSV workflow
        automl5, results5 = example_5_csv_workflow()

        # Example 6: Model comparison
        automl6, results6 = example_6_model_comparison()

        print("\n\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
