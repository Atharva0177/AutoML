"""
Training Pipeline Example.

This example demonstrates:
1. Training a single model with train-validation split
2. Training with cross-validation
3. Comparing multiple models
4. Using metrics calculation
5. Analyzing training results
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from automl.models import (
    GradientBoostingModel,
    LightGBMModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from automl.training import CrossValidator, MetricsCalculator, Trainer

print("=" * 80)
print("AutoML Training Pipeline Examples")
print("=" * 80)


# =============================================================================
# Example 1: Simple Training with Train-Validation Split
# =============================================================================
print("\n" + "=" * 80)
print("Example 1: Simple Training with Train-Validation Split")
print("=" * 80)

# Generate classification dataset
X_class, y_class = make_classification(
    n_samples=500, n_features=20, n_informative=15, n_classes=2, random_state=42
)

X_class_df = pd.DataFrame(
    X_class, columns=[f"feature_{i}" for i in range(X_class.shape[1])]
)
y_class_series = pd.Series(y_class, name="target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_class_df, y_class_series, test_size=0.2, random_state=42, stratify=y_class_series
)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nDataset splits:")
print(f"  Training: {len(X_train_sub)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# Create and train model
print("\n--- Training Random Forest ---")
model = RandomForestModel(
    model_type="classification", hyperparameters={"n_estimators": 100, "max_depth": 10}
)

trainer = Trainer(use_cross_validation=False)
results = trainer.train(model, X_train_sub, y_train_sub, X_val, y_val)

# Display results
print(f"\nTraining completed in {results['training_time']:.2f} seconds")
print("\n--- Training Metrics ---")
print(
    MetricsCalculator.format_metrics(
        results["train_metrics"], "classification", decimals=4
    )
)
print("\n--- Validation Metrics ---")
print(
    MetricsCalculator.format_metrics(
        results["val_metrics"], "classification", decimals=4
    )
)

# Test set evaluation
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)
test_metrics = MetricsCalculator.calculate_classification_metrics(
    y_test, y_test_pred, y_test_proba
)
print("\n--- Test Metrics ---")
print(MetricsCalculator.format_metrics(test_metrics, "classification", decimals=4))


# =============================================================================
# Example 2: Training with Cross-Validation
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: Training with Cross-Validation")
print("=" * 80)

# Use full training set for CV
print("\n--- Training XGBoost with 5-Fold Stratified CV ---")
xgb_model = XGBoostModel(
    model_type="classification",
    hyperparameters={"n_estimators": 50, "learning_rate": 0.1, "max_depth": 5},
)

cv_trainer = Trainer(use_cross_validation=True, cv_folds=5, cv_stratified=True)

cv_results = cv_trainer.train(xgb_model, X_train, y_train)

# Display CV results
cv_data = cv_results["cross_validation"]
print(f"\nCross-validation completed in {cv_results['training_time']:.2f} seconds")
print(
    f"\nMean Accuracy: {cv_data['mean_accuracy']:.4f} (±{cv_data['std_accuracy']:.4f})"
)
print(f"Mean F1 Score: {cv_data['mean_f1_score']:.4f} (±{cv_data['std_f1_score']:.4f})")
print(
    f"Mean Precision: {cv_data['mean_precision']:.4f} (±{cv_data['std_precision']:.4f})"
)
print(f"Mean Recall: {cv_data['mean_recall']:.4f} (±{cv_data['std_recall']:.4f})")

# Show per-fold results
print("\n--- Per-Fold Results ---")
for fold_metrics in cv_data["fold_metrics"]:
    print(
        f"Fold {fold_metrics['fold']}: "
        f"F1={fold_metrics['f1_score']:.4f}, "
        f"Acc={fold_metrics['accuracy']:.4f}"
    )


# =============================================================================
# Example 3: Comparing Multiple Models
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: Comparing Multiple Classification Models")
print("=" * 80)

# Define models to compare
models_to_compare = [
    LogisticRegressionModel(),
    RandomForestModel(
        model_type="classification", hyperparameters={"n_estimators": 100}
    ),
    GradientBoostingModel(
        model_type="classification", hyperparameters={"n_estimators": 100}
    ),
    XGBoostModel(model_type="classification", hyperparameters={"n_estimators": 100}),
    LightGBMModel(model_type="classification", hyperparameters={"n_estimators": 100}),
]

print(f"\nComparing {len(models_to_compare)} models:")
for m in models_to_compare:
    print(f"  - {m.name}")

# Compare models
comparison_trainer = Trainer(use_cross_validation=False)
comparison_results = comparison_trainer.compare_models(
    models_to_compare, X_train_sub, y_train_sub, X_val, y_val
)

# Display rankings
print("\n--- Model Rankings ---")
print(f"{'Rank':<6} {'Model':<20} {'Score':<10} {'Time (s)':<10}")
print("-" * 50)
for ranking in comparison_results["rankings"]:
    print(
        f"{ranking['rank']:<6} "
        f"{ranking['model_name']:<20} "
        f"{ranking['score']:<10.4f} "
        f"{ranking['training_time']:<10.2f}"
    )

best_model_name = comparison_trainer.get_best_model(comparison_results)
print(f"\nBest model: {best_model_name}")

# Show training summary
print("\n--- Training Summary ---")
summary_df = comparison_trainer.get_training_summary()
print(summary_df.to_string(index=False))


# =============================================================================
# Example 4: Regression Model Training
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Regression Model Training")
print("=" * 80)

# Generate regression dataset
X_reg, y_reg = make_regression(  # type: ignore[assignment]
    n_samples=500, n_features=20, n_informative=15, noise=10.0, random_state=42
)

X_reg_df = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(X_reg.shape[1])])
y_reg_series = pd.Series(y_reg, name="target")

# Split data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_df, y_reg_series, test_size=0.2, random_state=42
)
X_train_r_sub, X_val_r, y_train_r_sub, y_val_r = train_test_split(
    X_train_r, y_train_r, test_size=0.2, random_state=42
)

print(f"\nDataset: {len(X_train_r_sub)} training, {len(X_val_r)} validation samples")

# Compare regression models
regression_models = [
    LinearRegressionModel(),
    RandomForestModel(model_type="regression", hyperparameters={"n_estimators": 100}),
    XGBoostModel(model_type="regression", hyperparameters={"n_estimators": 100}),
]

print(f"\nComparing {len(regression_models)} regression models:")
for m in regression_models:
    print(f"  - {m.name}")

reg_trainer = Trainer()
reg_results = reg_trainer.compare_models(
    regression_models, X_train_r_sub, y_train_r_sub, X_val_r, y_val_r
)

# Display results
print("\n--- Regression Model Rankings ---")
print(f"{'Rank':<6} {'Model':<20} {'R² Score':<12} {'Time (s)':<10}")
print("-" * 52)
for ranking in reg_results["rankings"]:
    print(
        f"{ranking['rank']:<6} "
        f"{ranking['model_name']:<20} "
        f"{ranking['score']:<12.4f} "
        f"{ranking['training_time']:<10.2f}"
    )

# Show detailed metrics for best model
best_reg_model = reg_trainer.get_best_model(reg_results)
print(f"\nBest regression model: {best_reg_model}")

# Find best model results
for model_result in reg_results["models"]:
    if model_result["model_name"] == best_reg_model:
        print("\n--- Best Model Validation Metrics ---")
        print(
            MetricsCalculator.format_metrics(
                model_result["val_metrics"], "regression", decimals=4
            )
        )
        break


# =============================================================================
# Example 5: Cross-Validation Comparison
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Cross-Validation Model Comparison")
print("=" * 80)

print("\nComparing models using 5-fold stratified cross-validation")

cv_comparison_models = [
    LogisticRegressionModel(),
    RandomForestModel(
        model_type="classification",
        hyperparameters={"n_estimators": 50, "max_depth": 10},
    ),
    XGBoostModel(model_type="classification", hyperparameters={"n_estimators": 50}),
]

cv_comparison_trainer = Trainer(
    use_cross_validation=True, cv_folds=5, cv_stratified=True
)

cv_comp_results = cv_comparison_trainer.compare_models(
    cv_comparison_models, X_train, y_train
)

# Display CV comparison results
print("\n--- Cross-Validation Comparison ---")
print(f"{'Rank':<6} {'Model':<22} {'Mean Score':<12} {'Std':<10} {'Time (s)':<10}")
print("-" * 66)
for ranking in cv_comp_results["rankings"]:
    # Find corresponding model results to get std
    for model_res in cv_comp_results["models"]:
        if model_res["model_name"] == ranking["model_name"]:
            cv_res = model_res.get("cross_validation", {})
            std = cv_res.get("std_primary_metric", 0)
            break

    print(
        f"{ranking['rank']:<6} "
        f"{ranking['model_name']:<22} "
        f"{ranking['score']:<12.4f} "
        f"{std:<10.4f} "
        f"{ranking['training_time']:<10.2f}"
    )


# =============================================================================
# Example 6: Manual Cross-Validation
# =============================================================================
print("\n" + "=" * 80)
print("Example 6: Manual Cross-Validation with Custom Settings")
print("=" * 80)

print("\n--- Testing Different CV Strategies ---")

# Strategy 1: Standard K-Fold
print("\n1. Standard 5-Fold CV (no stratification):")
cv_standard = CrossValidator(n_splits=5, stratified=False)
model_cv = RandomForestModel(model_type="classification")
results_standard = cv_standard.cross_validate(model_cv, X_train, y_train)
print(cv_standard.get_cv_summary(results_standard, "classification"))

# Strategy 2: Stratified K-Fold
print("\n2. Stratified 5-Fold CV:")
cv_stratified = CrossValidator(n_splits=5, stratified=True)
model_cv2 = RandomForestModel(model_type="classification")
results_stratified = cv_stratified.cross_validate(model_cv2, X_train, y_train)
print(cv_stratified.get_cv_summary(results_stratified, "classification"))

# Strategy 3: 10-Fold CV
print("\n3. Stratified 10-Fold CV:")
cv_10fold = CrossValidator(n_splits=10, stratified=True)
model_cv3 = RandomForestModel(model_type="classification")
results_10fold = cv_10fold.cross_validate(model_cv3, X_train, y_train)
print(cv_10fold.get_cv_summary(results_10fold, "classification"))


print("\n" + "=" * 80)
print("Training Pipeline Examples Completed!")
print("=" * 80)
