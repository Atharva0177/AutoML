"""
Deep Learning MLP Example.

Demonstrates how to use the Multi-Layer Perceptron (MLP) models for
classification and regression tasks.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

try:
    from automl.models.deep_learning import MLPClassifier, MLPRegressor
    from automl.models.deep_learning.device_manager import DeviceManager
    PYTORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch not installed. Install it with:")
    print("   pip install torch")
    PYTORCH_AVAILABLE = False


def example_1_simple_classification():
    """Example 1: Simple binary classification with MLP."""
    print("\n" + "=" * 70)
    print("Example 1: Binary Classification with MLP")
    print("=" * 70)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Create and train model
    print("\nTraining MLP Classifier...")
    model = MLPClassifier(
        name='binary_mlp',
        hidden_layers=[128, 64, 32],  # 3 hidden layers
        activation='relu',
        dropout_rate=0.2,
        use_batch_norm=True,
        learning_rate=0.001,
        batch_size=32,
        max_epochs=50,
        early_stopping_patience=10,
        use_gpu=True,  # Will use GPU if available
        random_state=42
    )
    
    model.fit(X_train, y_train, X_test, y_test, verbose=True)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"Model trained for {model.metadata['epochs_trained']} epochs")
    print(f"Model size: {model.get_model_size()['total_params']:,} parameters")
    
    # Show training history
    print("\nTraining History (last 5 epochs):")
    for i in range(max(0, len(model.history['train_loss']) - 5), len(model.history['train_loss'])):
        print(
            f"  Epoch {i+1}: "
            f"train_loss={model.history['train_loss'][i]:.4f}, "
            f"train_acc={model.history['train_metric'][i]:.4f}, "
            f"val_loss={model.history['val_loss'][i]:.4f}, "
            f"val_acc={model.history['val_metric'][i]:.4f}"
        )


def example_2_multiclass_classification():
    """Example 2: Multi-class classification."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-class Classification (5 classes)")
    print("=" * 70)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=800,
        n_features=15,
        n_classes=5,
        n_informative=12,
        n_redundant=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"Classes: {len(np.unique(y))}")
    
    # Create and train model with different architecture
    print("\nTraining Deep MLP (5 hidden layers)...")
    model = MLPClassifier(
        name='multiclass_mlp',
        hidden_layers=[256, 128, 64, 32, 16],  # Deep network
        activation='relu',
        dropout_rate=0.3,
        use_batch_norm=True,
        learning_rate=0.001,
        batch_size=16,
        max_epochs=100,
        early_stopping_patience=15,
        optimizer_name='adam',
        use_gpu=True,
        random_state=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"Model architecture: {model.hidden_layers}")
    print(f"Total parameters: {model.get_model_size()['total_params']:,}")
    print(f"Model size: {model.get_model_size()['size_mb']:.2f} MB")


def example_3_regression():
    """Example 3: Regression with MLP."""
    print("\n" + "=" * 70)
    print("Example 3: Regression with MLP")
    print("=" * 70)
    
    # Generate dataset
    result = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    X = result[0]
    y = result[1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Create and train model
    print("\nTraining MLP Regressor...")
    model = MLPRegressor(
        name='regression_mlp',
        hidden_layers=[128, 64, 32],
        activation='relu',
        dropout_rate=0.2,
        use_batch_norm=True,
        learning_rate=0.001,
        batch_size=32,
        max_epochs=50,
        early_stopping_patience=10,
        optimizer_name='adam',
        use_gpu=True,
        random_state=42
    )
    
    model.fit(X_train, y_train, X_test, y_test, verbose=True)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n✅ Test R² Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Model trained for {model.metadata['epochs_trained']} epochs")


def example_4_pandas_integration():
    """Example 4: Using pandas DataFrames."""
    print("\n" + "=" * 70)
    print("Example 4: Integration with Pandas DataFrames")
    print("=" * 70)
    
    # Create DataFrame
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples)
    })
    
    # Target: loan approval (binary)
    df['loan_approved'] = (
        (df['credit_score'] > 650) & 
        (df['debt_ratio'] < 0.5) & 
        (df['income'] > 50000)
    ).astype(int)
    
    print("\nDataset:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Loan approval rate: {df['loan_approved'].mean():.2%}")
    
    # Split data
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    print("\nTraining MLP with DataFrame input...")
    model = MLPClassifier(
        hidden_layers=[64, 32],
        max_epochs=30,
        early_stopping_patience=5,
        use_gpu=False,  # CPU for this small dataset
        random_state=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"Feature names: {model.feature_names}")


def example_5_device_management():
    """Example 5: GPU/CPU device management."""
    print("\n" + "=" * 70)
    print("Example 5: Device Management (GPU/CPU)")
    print("=" * 70)
    
    # Check available devices
    dm = DeviceManager(prefer_gpu=True)
    
    print(f"\nCurrent device: {dm.device}")
    print(f"Is GPU: {dm.is_gpu()}")
    print(f"Available devices: {DeviceManager.get_available_devices()}")
    print(f"GPU count: {DeviceManager.get_device_count()}")
    
    # Get memory info
    memory_info = dm.get_memory_info()
    print(f"\nMemory info:")
    for key, value in memory_info.items():
        if value is not None:
            if 'gb' in key.lower():
                print(f"  {key}: {value:.2f} GB")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: N/A")
    
    # Train model on specific device
    print("\nTraining model on detected device...")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    model = MLPClassifier(
        hidden_layers=[32],
        max_epochs=10,
        use_gpu=dm.is_gpu(),
        random_state=42
    )
    
    model.fit(X, y, verbose=False)
    
    print(f"✅ Model trained on: {model.device}")


def example_6_model_saving_loading(tmp_dir="/tmp"):
    """Example 6: Saving and loading models."""
    print("\n" + "=" * 70)
    print("Example 6: Saving and Loading Models")
    print("=" * 70)
    
    # Train a model
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("\nTraining model...")
    model = MLPClassifier(
        name='saved_model',
        hidden_layers=[64, 32],
        max_epochs=20,
        use_gpu=False,
        random_state=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Make predictions before saving
    y_pred_before = model.predict(X_test)
    accuracy_before = accuracy_score(y_test, y_pred_before)
    
    print(f"Accuracy before saving: {accuracy_before:.4f}")
    
    # Save model
    model_path = f"{tmp_dir}/mlp_classifier_example.pth"
    model.save_model(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Load model
    print("\nLoading model...")
    loaded_model = MLPClassifier(use_gpu=False)
    loaded_model.load_model(model_path)
    
    # Make predictions with loaded model
    y_pred_after = loaded_model.predict(X_test)
    accuracy_after = accuracy_score(y_test, y_pred_after)
    
    print(f"Accuracy after loading: {accuracy_after:.4f}")
    print(f"Models match: {np.array_equal(y_pred_before, y_pred_after)}")
    print(f"Loaded model name: {loaded_model.name}")
    print(f"Loaded model epochs: {loaded_model.metadata['epochs_trained']}")


def main():
    """Run all examples."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("\n" + "=" * 70)
    print("PyTorch Deep Learning Examples for AutoML")
    print("=" * 70)
    
    try:
        example_1_simple_classification()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {e}")
    
    try:
        example_2_multiclass_classification()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {e}")
    
    try:
        example_3_regression()
    except Exception as e:
        print(f"\n❌ Example 3 failed: {e}")
    
    try:
        example_4_pandas_integration()
    except Exception as e:
        print(f"\n❌ Example 4 failed: {e}")
    
    try:
        example_5_device_management()
    except Exception as e:
        print(f"\n❌ Example 5 failed: {e}")
    
    try:
        example_6_model_saving_loading()
    except Exception as e:
        print(f"\n❌ Example 6 failed: {e}")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
