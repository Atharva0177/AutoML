"""
Tests for Deep Learning Models.

Tests MLP classifier and regressor, device management, and data loaders.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

torch_available = True
try:
    import torch
    from automl.models.deep_learning import MLPClassifier, MLPRegressor
    from automl.models.deep_learning.device_manager import DeviceManager
    from automl.models.deep_learning.tabular_dataset import (
        TabularDataset,
        create_dataloaders,
        normalize_data
    )
except ImportError:
    torch_available = False


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestDeviceManager:
    """Tests for DeviceManager."""
    
    def test_device_detection(self):
        """Test device detection."""
        dm = DeviceManager(prefer_gpu=True)
        assert dm.device is not None
        assert dm.device.type in ['cuda', 'mps', 'cpu']
    
    def test_cpu_fallback(self):
        """Test CPU fallback."""
        dm = DeviceManager(prefer_gpu=False)
        assert dm.device.type == 'cpu'
    
    def test_is_gpu(self):
        """Test GPU detection."""
        dm = DeviceManager(prefer_gpu=True)
        is_gpu = dm.is_gpu()
        assert isinstance(is_gpu, bool)
    
    def test_get_available_devices(self):
        """Test getting available devices."""
        devices = DeviceManager.get_available_devices()
        assert 'cpu' in devices
        assert isinstance(devices, list)
    
    def test_to_device(self):
        """Test moving tensor to device."""
        dm = DeviceManager()
        tensor = torch.randn(10, 5)
        tensor_on_device = dm.to_device(tensor)
        assert tensor_on_device.device.type == dm.device.type


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestTabularDataset:
    """Tests for TabularDataset."""
    
    def test_classification_dataset(self):
        """Test creating classification dataset."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        dataset = TabularDataset(X, y, task_type='classification')
        
        assert len(dataset) == 100
        assert dataset.get_feature_dim() == 10
        assert dataset.get_num_classes() == 2
    
    def test_regression_dataset(self):
        """Test creating regression dataset."""
        result = make_regression(n_samples=100, n_features=10, random_state=42)
        X = result[0]
        y = result[1]
        dataset = TabularDataset(X, y, task_type='regression')
        
        assert len(dataset) == 100
        assert dataset.get_feature_dim() == 10
        assert dataset.get_num_classes() is None
    
    def test_dataframe_input(self):
        """Test dataset with DataFrame input."""
        df = pd.DataFrame(np.random.randn(50, 5), columns=['A', 'B', 'C', 'D', 'E'])
        y = np.random.randint(0, 2, 50)
        
        dataset = TabularDataset(df, y, task_type='classification')
        
        assert len(dataset) == 50
        assert dataset.feature_names == ['A', 'B', 'C', 'D', 'E']
    
    def test_getitem(self):
        """Test getting items from dataset."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        dataset = TabularDataset(X, y, task_type='classification')
        
        X_sample, y_sample = dataset[0]
        
        assert isinstance(X_sample, torch.Tensor)
        assert isinstance(y_sample, torch.Tensor)
        assert X_sample.shape == (5,)


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestDataLoaders:
    """Tests for DataLoader utilities."""
    
    def test_create_dataloaders(self):
        """Test creating data loaders."""
        X_train, y_train = make_classification(n_samples=100, n_features=10, random_state=42)
        X_val, y_val = make_classification(n_samples=30, n_features=10, random_state=43)
        
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val,
            task_type='classification',
            batch_size=16
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) == 7  # 100/16 = 6.25, rounded up
    
    def test_normalize_data(self):
        """Test data normalization."""
        X = np.random.randn(100, 5)
        X_normalized, stats = normalize_data(X)
        
        assert X_normalized.shape == X.shape
        assert 'mean' in stats
        assert 'std' in stats
        assert len(stats['mean']) == 5
        
        # Check normalization (should be approximately mean=0, std=1)
        assert np.abs(np.mean(X_normalized)) < 0.1
        assert np.abs(np.std(X_normalized) - 1.0) < 0.1


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestMLPClassifier:
    """Tests for MLPClassifier."""
    
    def test_initialization(self):
        """Test MLP classifier initialization."""
        model = MLPClassifier(
            name='test_mlp',
            hidden_layers=[64, 32],
            max_epochs=10
        )
        
        assert model.name == 'test_mlp'
        assert model.model_type == 'classification'
        assert model.hidden_layers == [64, 32]
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        # Create dataset
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = MLPClassifier(
            hidden_layers=[32, 16],
            max_epochs=20,
            batch_size=16,
            early_stopping_patience=5,
            use_gpu=False,  # Use CPU for testing
            random_state=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        assert model.is_fitted
        assert len(model.history['train_loss']) > 0
        
        # Make predictions
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert predictions.min() >= 0
        assert predictions.max() < 3  # 3 classes: 0, 1, 2
    
    def test_predict_proba(self):
        """Test probability predictions."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = MLPClassifier(
            hidden_layers=[32],
            max_epochs=10,
            use_gpu=False,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Between 0 and 1
    
    def test_with_validation_set(self):
        """Test training with validation set."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        model = MLPClassifier(
            hidden_layers=[32],
            max_epochs=30,
            early_stopping_patience=5,
            use_gpu=False,
            random_state=42
        )
        
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        assert 'val_loss' in model.history
        assert len(model.history['val_loss']) > 0
    
    def test_get_model_size(self):
        """Test model size calculation."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        model = MLPClassifier(hidden_layers=[64, 32], use_gpu=False, random_state=42)
        model.fit(X, y, verbose=False)
        
        size_info = model.get_model_size()
        
        assert 'total_params' in size_info
        assert 'trainable_params' in size_info
        assert 'size_mb' in size_info
        assert size_info['total_params'] > 0


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestMLPRegressor:
    """Tests for MLPRegressor."""
    
    def test_initialization(self):
        """Test MLP regressor initialization."""
        model = MLPRegressor(
            name='test_mlp_reg',
            hidden_layers=[128, 64, 32],
            max_epochs=50
        )
        
        assert model.name == 'test_mlp_reg'
        assert model.model_type == 'regression'
        assert model.hidden_layers == [128, 64, 32]
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        # Create dataset
        result = make_regression(
            n_samples=200,
            n_features=10,
            noise=10,
            random_state=42
        )
        X = result[0]
        y = result[1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = MLPRegressor(
            hidden_layers=[64, 32],
            max_epochs=20,
            batch_size=16,
            early_stopping_patience=5,
            use_gpu=False,  # Use CPU for testing
            random_state=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        assert model.is_fitted
        assert len(model.history['train_loss']) > 0
        
        # Make predictions
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert np.all(np.isfinite(predictions))  # All predictions should be finite
    
    def test_prediction_accuracy(self):
        """Test that model learns (predictions improve)."""
        result = make_regression(n_samples=200, n_features=5, noise=1, random_state=42)
        X = result[0]
        y = result[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = MLPRegressor(
            hidden_layers=[64, 32],
            max_epochs=50,
            use_gpu=False,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
        
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
        
        # Model should perform better than baseline
        assert mse < baseline_mse
    
    def test_with_dataframe(self):
        """Test training with pandas DataFrame."""
        result = make_regression(n_samples=100, n_features=5, random_state=42)
        X = result[0]
        y = result[1]
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y_series = pd.Series(y, name='target')
        
        model = MLPRegressor(
            hidden_layers=[32],
            max_epochs=10,
            use_gpu=False,
            random_state=42
        )
        
        model.fit(X_df, y_series, verbose=False)
        
        assert model.feature_names == ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # Predictions should work with DataFrame too
        X_test_df = pd.DataFrame(X[:10], columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        predictions = model.predict(X_test_df)
        
        assert predictions.shape == (10,)


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestModelSavingLoading:
    """Tests for model saving and loading."""
    
    def test_save_load_classifier(self, tmp_path):
        """Test saving and loading classifier."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Train model
        model = MLPClassifier(
            hidden_layers=[32],
            max_epochs=10,
            use_gpu=False,
            random_state=42
        )
        model.fit(X, y, verbose=False)
        
        # Make predictions before saving
        predictions_before = model.predict(X[:10])
        
        # Save model
        model_path = tmp_path / "mlp_classifier.pth"
        model.save_model(model_path)
        
        # Load model
        loaded_model = MLPClassifier(use_gpu=False)
        loaded_model.load_model(model_path)
        
        # Make predictions with loaded model
        predictions_after = loaded_model.predict(X[:10])
        
        # Predictions should be identical
        np.testing.assert_array_equal(predictions_before, predictions_after)
    
    def test_save_load_regressor(self, tmp_path):
        """Test saving and loading regressor."""
        result = make_regression(n_samples=100, n_features=10, random_state=42)
        X = result[0]
        y = result[1]
        
        # Train model
        model = MLPRegressor(
            hidden_layers=[64, 32],
            max_epochs=15,
            use_gpu=False,
            random_state=42
        )
        model.fit(X, y, verbose=False)
        
        # Make predictions before saving
        predictions_before = model.predict(X[:10])
        
        # Save model
        model_path = tmp_path / "mlp_regressor.pth"
        model.save_model(model_path)
        
        # Load model
        loaded_model = MLPRegressor(use_gpu=False)
        loaded_model.load_model(model_path)
        
        # Make predictions with loaded model
        predictions_after = loaded_model.predict(X[:10])
        
        # Predictions should be very close (floating point precision)
        np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
