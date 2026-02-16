"""
Model Framework Example.

This example demonstrates:
1. Using the model registry to discover available models
2. Creating and training different model types
3. Comparing classification models
4. Comparing regression models
5. Saving and loading models
6. Feature importance extraction
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from automl.models import (
    GradientBoostingModel,
    LightGBMModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    ModelRegistry,
    RandomForestModel,
    XGBoostModel,
)

print("=" * 80)
print("AutoML Model Framework Examples")
print("=" * 80)


# =============================================================================
# Example 1: Model Registry - Discover Available Models
# =============================================================================
print("\n" + "=" * 80)
print("Example 1: Model Registry - Discover Available Models")
print("=" * 80)

# List all available models
all_models = ModelRegistry.list_models()
print(f"\nTotal available models: {len(all_models)}")
print(f"Models: {', '.join(all_models)}")

# Filter by classification models
classification_models = ModelRegistry.list_models(model_type="classification")
print(f"\nClassification models: {', '.join(classification_models)}")

# Filter by regression models
regression_models = ModelRegistry.list_models(model_type="regression")
print(f"Regression models: {', '.join(regression_models)}")

# Get detailed model information
print("\n--- Model Details: XGBoost ---")
xgb_info = ModelRegistry.get_model_info("xgboost")
print(f"Type: {xgb_info['type']}")
print(f"Description: {xgb_info['description']}")
print(f"Supports multiclass: {xgb_info['supports_multiclass']}")
print(f"Requires GPU: {xgb_info['requires_gpu']}")


# =============================================================================
# Example 2: Classification Model Comparison
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: Classification Model Comparison")
print("=" * 80)

# Generate synthetic classification dataset
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42,
)

# Convert to DataFrame for better feature names
X_class_df = pd.DataFrame(
    X_class, columns=[f"feature_{i}" for i in range(X_class.shape[1])]
)
y_class_series = pd.Series(y_class, name="target")

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class_df, y_class_series, test_size=0.2, random_state=42, stratify=y_class_series
)

print(
    f"\nDataset: {X_train_c.shape[0]} training samples, {X_test_c.shape[0]} test samples"
)
print(f"Features: {X_train_c.shape[1]}")
print(f"Classes: {sorted(y_class_series.unique())}")

# Define classification models to compare
classification_model_configs = [
    ("logistic_regression", LogisticRegressionModel, {}),
    ("random_forest", RandomForestModel, {"hyperparameters": {"n_estimators": 100}}),
    (
        "gradient_boosting",
        GradientBoostingModel,
        {"hyperparameters": {"n_estimators": 100}},
    ),
    ("xgboost", XGBoostModel, {"hyperparameters": {"n_estimators": 100}}),
    ("lightgbm", LightGBMModel, {"hyperparameters": {"n_estimators": 100}}),
]

print("\n--- Training and Evaluating Classification Models ---")
classification_results = []

for model_name, model_class, kwargs in classification_model_configs:
    print(f"\nTraining {model_name}...")

    # Create and train model
    model = model_class(model_type="classification", **kwargs)
    model.fit(X_train_c, y_train_c)

    # Make predictions
    y_pred = model.predict(X_test_c)
    y_pred_proba = model.predict_proba(X_test_c)

    # Calculate metrics
    accuracy = accuracy_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred, average="weighted")

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    classification_results.append(
        {"model": model_name, "accuracy": accuracy, "f1_score": f1}
    )

# Display results table
print("\n--- Classification Results Summary ---")
results_df = pd.DataFrame(classification_results)
results_df = results_df.sort_values("accuracy", ascending=False)
print(results_df.to_string(index=False))


# =============================================================================
# Example 3: Regression Model Comparison
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: Regression Model Comparison")
print("=" * 80)

# Generate synthetic regression dataset
X_reg, y_reg = make_regression(  # type: ignore[assignment]
    n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=42
)

# Convert to DataFrame
X_reg_df = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(X_reg.shape[1])])
y_reg_series = pd.Series(y_reg, name="target")

# Split data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_df, y_reg_series, test_size=0.2, random_state=42
)

print(
    f"\nDataset: {X_train_r.shape[0]} training samples, {X_test_r.shape[0]} test samples"
)
print(f"Features: {X_train_r.shape[1]}")

# Define regression models to compare
regression_model_configs = [
    ("linear_regression", LinearRegressionModel, {}),
    ("random_forest", RandomForestModel, {"hyperparameters": {"n_estimators": 100}}),
    (
        "gradient_boosting",
        GradientBoostingModel,
        {"hyperparameters": {"n_estimators": 100}},
    ),
    ("xgboost", XGBoostModel, {"hyperparameters": {"n_estimators": 100}}),
    ("lightgbm", LightGBMModel, {"hyperparameters": {"n_estimators": 100}}),
]

print("\n--- Training and Evaluating Regression Models ---")
regression_results = []

for model_name, model_class, kwargs in regression_model_configs:
    print(f"\nTraining {model_name}...")

    # Create and train model
    model = model_class(model_type="regression", **kwargs)
    model.fit(X_train_r, y_train_r)

    # Make predictions
    y_pred = model.predict(X_test_r)

    # Calculate metrics
    mse = mean_squared_error(y_test_r, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_pred)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")

    regression_results.append({"model": model_name, "rmse": rmse, "r2_score": r2})

# Display results table
print("\n--- Regression Results Summary ---")
results_df = pd.DataFrame(regression_results)
results_df = results_df.sort_values("r2_score", ascending=False)
print(results_df.to_string(index=False))


# =============================================================================
# Example 4: Feature Importance Analysis
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Feature Importance Analysis")
print("=" * 80)

# Train a Random Forest model
print("\nTraining Random Forest for feature importance analysis...")
rf_model = RandomForestModel(model_type="classification")
rf_model.fit(X_train_c, y_train_c)

# Get feature importance
importance = rf_model.get_feature_importance()

# Sort by importance
if importance is not None:
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

if importance is not None:
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Top 10 Most Important Features ---")
    for i, (feature, score) in enumerate(sorted_importance[:10], 1):
        print(f"{i:2d}. {feature:15s}: {score:.4f}")


# =============================================================================
# Example 5: Model Persistence (Save/Load)
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Model Persistence (Save/Load)")
print("=" * 80)

# Train a model
print("\nTraining XGBoost model...")
xgb_model = XGBoostModel(
    model_type="classification",
    hyperparameters={"n_estimators": 50, "learning_rate": 0.1},
)
xgb_model.fit(X_train_c, y_train_c)

# Save model
save_path = "saved_models/xgboost_example"
xgb_model.save(save_path)
print(f"Model saved to: {save_path}.pkl and {save_path}.json")

# Get predictions from original model
original_predictions = xgb_model.predict(X_test_c)
original_accuracy = accuracy_score(y_test_c, original_predictions)
print(f"\nOriginal model accuracy: {original_accuracy:.4f}")

# Load model
print("\nLoading model from disk...")
loaded_model = XGBoostModel(model_type="classification")
loaded_model.load(save_path)

# Get predictions from loaded model
loaded_predictions = loaded_model.predict(X_test_c)
loaded_accuracy = accuracy_score(y_test_c, loaded_predictions)
print(f"Loaded model accuracy: {loaded_accuracy:.4f}")

# Verify predictions match
predictions_match = np.array_equal(original_predictions, loaded_predictions)
print(f"\nPredictions match: {predictions_match}")


# =============================================================================
# Example 6: Using Model Registry to Create Models
# =============================================================================
print("\n" + "=" * 80)
print("Example 6: Using Model Registry to Create Models")
print("=" * 80)

print("\nCreating models dynamically from registry...")

# Create models by name
models = {
    "LogReg": ModelRegistry.create_model("logistic_regression"),
    "RF": ModelRegistry.create_model(
        "random_forest",
        model_type="classification",
        hyperparameters={"n_estimators": 50},
    ),
    "XGB": ModelRegistry.create_model(
        "xgboost", model_type="classification", hyperparameters={"n_estimators": 50}
    ),
}

print("\n--- Training Models from Registry ---")
for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    metadata = model.get_metadata()
    print(
        f"  Metadata: {metadata['metadata']['n_features']} features, "
        f"{metadata['metadata']['n_samples']} training samples"
    )


# =============================================================================
# Example 7: Custom Hyperparameters
# =============================================================================
print("\n" + "=" * 80)
print("Example 7: Custom Hyperparameters and Parameter Updates")
print("=" * 80)

# Create model with custom hyperparameters
custom_params = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.7,
}

print("\nCreating XGBoost with custom parameters:")
print(f"  {custom_params}")

xgb_custom = XGBoostModel(model_type="classification", hyperparameters=custom_params)

# Get current parameters
params = xgb_custom.get_params()
print(f"\nCurrent n_estimators: {params['n_estimators']}")

# Update parameters
xgb_custom.set_params(n_estimators=300)
updated_params = xgb_custom.get_params()
print(f"Updated n_estimators: {updated_params['n_estimators']}")

# Train with updated parameters
xgb_custom.fit(X_train_c, y_train_c)
y_pred = xgb_custom.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred)
print(f"\nModel accuracy: {acc:.4f}")


print("\n" + "=" * 80)
print("Examples completed!")
print("=" * 80)
