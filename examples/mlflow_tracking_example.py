"""
Example: Using MLflow Experiment Tracking with AutoML

This example demonstrates how to use MLflow for experiment tracking,
model logging, and model registry with the AutoML pipeline.
"""

import numpy as np
import pandas as pd

from automl.pipeline.automl import AutoML
from automl.tracking import MLflowTracker

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("MLflow Experiment Tracking Example")
print("=" * 80)

# Example 1: AutoML with MLflow Integration
print("\n" + "=" * 80)
print("EXAMPLE 1: AutoML with Built-in MLflow Tracking")
print("=" * 80)

# Create a sample dataset
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
df["target"] = y

print(f"\nDataset: {len(df)} samples, {len(df.columns)-1} features")

# Initialize AutoML with MLflow enabled
print("\nInitializing AutoML with MLflow tracking...")
automl = AutoML(
    problem_type="classification",
    cv=False,
    verbose=True,
    optimize_hyperparameters=True,
    n_trials=10,  # Small number for demo
    enable_mlflow=True,
    experiment_name="automl_demo",
    mlflow_tracking_uri=None,  # Uses local ./mlruns directory
)

# Train models (MLflow will automatically track everything)
print("\nTraining models with MLflow tracking...")
results = automl.fit(df, target_column="target")

print(f"\nBest Model: {results['best_model']}")
print(f"Best Score: {results['best_score']:.4f}")

print("\n✅ MLflow tracked all experiments to ./mlruns/")
print("   Run 'mlflow ui' to view the dashboard at http://localhost:5000")

# Example 2: Manual MLflow Tracking
print("\n\n" + "=" * 80)
print("EXAMPLE 2: Manual MLflow Tracking")
print("=" * 80)

from automl.models import ModelRegistry
from automl.training import Trainer

# Initialize MLflow tracker
print("\nInitializing MLflow tracker...")
mlflow_tracker = MLflowTracker(
    experiment_name="manual_tracking_demo",
    enable_autolog=True,  # Enable automatic logging for sklearn, xgboost, lightgbm
)

# Start a run
print("Starting MLflow run...")
run_id = mlflow_tracker.start_run(
    run_name="manual_training_example", tags={"dataset": "synthetic", "purpose": "demo"}
)

try:
    # Log parameters
    mlflow_tracker.log_params(
        {"n_samples": len(df), "n_features": 20, "problem_type": "classification"}
    )

    # Create and train a model
    print("\nTraining Random Forest model...")
    registry = ModelRegistry()
    model = registry.get_model("random_forest", problem_type="classification")

    # Simple train-test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    from automl.training.metrics import MetricsCalculator

    calculator = MetricsCalculator()
    y_pred = model.predict(X_test)
    metrics = calculator.calculate_classification_metrics(y_test, y_pred)

    # Log metrics
    mlflow_tracker.log_metrics(
        {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
    )

    # Log model
    mlflow_tracker.log_model(
        model.model, artifact_path="model", registered_model_name="rf_classifier_demo"
    )

    print(f"\n✅ Logged model with accuracy: {metrics['accuracy']:.4f}")
    print(f"   Run ID: {run_id}")

finally:
    # End run
    mlflow_tracker.end_run()

# Example 3: Comparing Multiple Runs
print("\n\n" + "=" * 80)
print("EXAMPLE 3: Comparing Multiple Runs")
print("=" * 80)

mlflow_tracker = MLflowTracker(experiment_name="model_comparison", enable_autolog=False)

models_to_compare = ["logistic_regression", "random_forest", "xgboost"]

print(f"\nTraining and comparing {len(models_to_compare)} models...")

for model_name in models_to_compare:
    print(f"\n  Training {model_name}...")

    # Start run
    run_id = mlflow_tracker.start_run(
        run_name=f"{model_name}_comparison", tags={"model_type": model_name}
    )

    try:
        # Get model
        model = registry.get_model(model_name, problem_type="classification")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = calculator.calculate_classification_metrics(y_test, y_pred)

        # Log
        mlflow_tracker.log_param("model_name", model_name)
        mlflow_tracker.log_metrics(metrics)
        mlflow_tracker.log_model(model.model, artifact_path="model")

        print(f"    F1 Score: {metrics['f1_score']:.4f}")

    finally:
        mlflow_tracker.end_run()

print("\n✅ Comparison complete!")

# Search for best run
print("\nFinding best model...")
best_run = mlflow_tracker.get_best_run(metric="f1_score", ascending=False)

if best_run:
    print(f"\n🏆 Best Model:")
    print(f"   Run ID: {best_run.info.run_id}")
    print(f"   Model: {best_run.data.tags.get('model_type', 'unknown')}")
    print(f"   F1 Score: {best_run.data.metrics.get('f1_score', 0):.4f}")

# Example 4: Model Registry
print("\n\n" + "=" * 80)
print("EXAMPLE 4: Model Registry")
print("=" * 80)

print("\nRegistering best model...")

if best_run:
    model_uri = f"runs:/{best_run.info.run_id}/model"

    model_version = mlflow_tracker.register_model(
        model_uri=model_uri,
        name="best_classifier",
        description="Best performing classifier from comparison",
        tags={"dataset": "synthetic", "task": "classification"},
    )

    if model_version:
        print(f"\n✅ Registered model: best_classifier")
        print(f"   Version: {model_version.version}")

        # Transition to production
        print("\nTransitioning to Production stage...")
        mlflow_tracker.transition_model_stage(
            name="best_classifier",
            version=model_version.version,
            stage="Production",
            archive_existing=True,
        )

        print("✅ Model promoted to Production!")

print("\n" + "=" * 80)
print("Examples Complete!")
print("=" * 80)
print("\nTo view all experiments and models:")
print("  1. Run: mlflow ui")
print("  2. Open: http://localhost:5000")
print("\nYou can:")
print("  • Compare runs across experiments")
print("  • View model artifacts and parameters")
print("  • Manage model versions and stages")
print("  • Download models for deployment")
