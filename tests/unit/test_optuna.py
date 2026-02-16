"""Test Bayesian hyperparameter optimization with Optuna."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from automl.pipeline.automl import AutoML

# Create a classification dataset
X, y = make_classification(
    n_samples=300, n_features=10, n_informative=5, n_classes=2, random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df["target"] = y

print("=" * 80)
print("Testing AutoML with Bayesian Hyperparameter Optimization (Optuna)")
print("=" * 80)

# Test with hyperparameter optimization
automl = AutoML(
    problem_type="classification",
    use_cross_validation=False,
    verbose=True,
    optimize_hyperparameters=True,
    n_trials=20,  # Reduced for faster testing
)

print("\nTraining with hyperparameter optimization...")
print("This will optimize hyperparameters for each model using Bayesian optimization\n")

results = automl.fit(
    df,
    target_column="target",
    models_to_try=["random_forest", "xgboost"],  # Test with 2 models
)

print("\n" + "=" * 80)
print("Optimization Results:")
print("=" * 80)

for model_result in results["model_comparison"]["models"]:
    print(f"\nModel: {model_result['model_name']}")
    if "optimized_hyperparameters" in model_result:
        print(
            f"  Optimization trials: {model_result.get('optimization_trials', 'N/A')}"
        )
        best_score = model_result.get("best_optimization_score", "N/A")
        if isinstance(best_score, (int, float)):
            print(f"  Best optimization score: {best_score:.4f}")
        else:
            print(f"  Best optimization score: {best_score}")
        print(f"  Optimized hyperparameters:")
        for param, value in model_result["optimized_hyperparameters"].items():
            print(f"    {param}: {value}")
    score = model_result.get("score", "N/A")
    if isinstance(score, (int, float)):
        print(f"  Final validation score: {score:.4f}")
    else:
        print(f"  Final validation score: {score}")

print("\n" + "=" * 80)
print(f"Best model: {automl.best_model.name}")
print("=" * 80)
