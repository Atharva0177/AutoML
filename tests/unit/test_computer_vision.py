"""
Unit tests for computer vision models.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from automl.models.deep_learning.vision import (
    VISION_AVAILABLE,
    CNNClassifier,
    ImageDataset,
    MediumCNN,
    SimpleCNN,
    create_image_dataloaders,
)

if not VISION_AVAILABLE:
    pytest.skip("Vision dependencies not available", allow_module_level=True)

import torch

from automl.models.deep_learning.vision.transforms import (
    denormalize_image,
    get_train_transforms,
    get_val_transforms,
)


class TestCNNArchitectures:
    """Test custom CNN architectures."""

    def test_simple_cnn_forward(self):
        """Test SimpleCNN forward pass."""
        model = SimpleCNN(num_classes=5, input_channels=3)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        assert output.shape == (2, 5)

    def test_simple_cnn_custom_channels(self):
        """Test SimpleCNN with custom input channels."""
        model = SimpleCNN(num_classes=10, input_channels=1)
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 10)

    def test_medium_cnn_forward(self):
        """Test MediumCNN forward pass."""
        model = MediumCNN(num_classes=7, input_channels=3)
        x = torch.randn(3, 3, 64, 64)
        output = model(x)

        assert output.shape == (3, 7)

    def test_medium_cnn_dropout(self):
        """Test MediumCNN with different dropout rates."""
        model = MediumCNN(num_classes=5, dropout_rate=0.3)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        assert output.shape == (2, 5)


class TestImageTransforms:
    """Test image transformation pipelines."""

    def test_train_transforms_output_shape(self):
        """Test train transforms output shape."""
        transform = get_train_transforms(image_size=(64, 64))

        # Create dummy PIL image
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        transformed = transform(img)
        assert transformed.shape == (3, 64, 64)  # type: ignore[union-attr]
        assert isinstance(transformed, torch.Tensor)

    def test_val_transforms_no_augmentation(self):
        """Test validation transforms without augmentation."""
        transform = get_val_transforms(image_size=(128, 128))

        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))

        transformed = transform(img)
        assert transformed.shape == (3, 128, 128)  # type: ignore[union-attr]

    def test_denormalize_image(self):
        """Test image denormalization."""
        # Create normalized tensor
        tensor = torch.randn(3, 64, 64)

        # Denormalize
        denormalized = denormalize_image(tensor)

        assert denormalized.shape == tensor.shape
        # Denormalization reverses ImageNet normalization, so values may be outside [0,1]


class TestImageDataset:
    """Test image dataset classes."""

    def test_image_dataset_from_arrays(self):
        """Test ImageDataset with numpy arrays."""
        # Create synthetic images
        images = np.random.rand(10, 32, 32, 3).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 2, 2, 1, 0, 2, 1])

        transform = get_train_transforms(image_size=(32, 32))
        dataset = ImageDataset(
            images=images,  # type: ignore[arg-type]
            labels=labels,
            transform=transform,
            load_from_path=False,
        )

        assert len(dataset) == 10

        img, label = dataset[0]
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)

    def test_image_dataset_different_sizes(self):
        """Test ImageDataset with different image sizes."""
        images = np.random.rand(5, 64, 64, 3).astype(np.float32)
        labels = np.array([0, 1, 2, 0, 1])

        transform = get_val_transforms(image_size=(128, 128))
        dataset = ImageDataset(images, labels, transform, load_from_path=False)  # type: ignore[arg-type]

        img, _ = dataset[0]
        assert img.shape == (3, 128, 128)  # Resized

    def test_create_dataloaders(self):
        """Test dataloader creation."""
        train_images = np.random.rand(20, 32, 32, 3).astype(np.float32)
        train_labels = np.random.randint(0, 3, 20)
        val_images = np.random.rand(10, 32, 32, 3).astype(np.float32)
        val_labels = np.random.randint(0, 3, 10)

        train_transform = get_train_transforms(image_size=(32, 32))
        val_transform = get_val_transforms(image_size=(32, 32))

        train_loader, val_loader = create_image_dataloaders(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=8,
            load_from_path=False,
        )

        assert len(train_loader) == 3  # 20 / 8 = 2.5 -> 3 batches
        assert val_loader is not None
        if val_loader is not None:
            assert len(val_loader) == 2  # 10 / 8 = 1.25 -> 2 batches

        # Check batch
        for batch_X, batch_y in train_loader:
            assert batch_X.shape[0] <= 8
            assert batch_X.shape[1] == 3  # RGB channels
            assert batch_X.shape[2] == 32  # Height
            assert batch_X.shape[3] == 32  # Width
            assert batch_y.shape[0] <= 8
            break


class TestCNNClassifier:
    """Test CNNClassifier model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = np.random.rand(50, 32, 32, 3).astype(np.float32)
        y_train = np.random.randint(0, 3, 50)
        X_val = np.random.rand(20, 32, 32, 3).astype(np.float32)
        y_val = np.random.randint(0, 3, 20)
        return X_train, y_train, X_val, y_val

    def test_simple_cnn_initialization(self):
        """Test CNNClassifier initialization with SimpleCNN."""
        model = CNNClassifier(
            architecture="simple", image_size=(32, 32), batch_size=8, max_epochs=2
        )

        assert model.architecture == "simple"
        assert model.image_size == (32, 32)
        assert not model.is_fitted

    def test_medium_cnn_initialization(self):
        """Test CNNClassifier initialization with MediumCNN."""
        model = CNNClassifier(
            architecture="medium", learning_rate=0.001, weight_decay=0.0001
        )

        assert model.architecture == "medium"
        assert model.learning_rate == 0.001
        assert model.weight_decay == 0.0001

    def test_fit_simple_cnn(self, sample_data):
        """Test training SimpleCNN."""
        X_train, y_train, X_val, y_val = sample_data

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=2,
            early_stopping_patience=5,
            use_gpu=False,  # Use CPU for tests
            random_state=42,
        )

        model.fit(X_train, y_train, X_val, y_val, load_from_path=False, verbose=False)

        assert model.is_fitted
        assert model.num_classes == 3
        assert len(model.history["train_loss"]) == 2
        assert len(model.history["val_loss"]) == 2

    def test_fit_medium_cnn(self, sample_data):
        """Test training MediumCNN with optimizations."""
        X_train, y_train, X_val, y_val = sample_data

        model = CNNClassifier(
            architecture="medium",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=3,
            lr_scheduler="plateau",
            lr_patience=2,
            gradient_clip_value=1.0,
            weight_decay=0.0001,
            use_gpu=False,
            random_state=42,
        )

        model.fit(X_train, y_train, X_val, y_val, load_from_path=False, verbose=False)

        assert model.is_fitted
        assert "best_val_loss" in model.metadata

    def test_predict(self, sample_data):
        """Test prediction."""
        X_train, y_train, X_val, y_val = sample_data

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=1,
            use_gpu=False,
        )

        model.fit(X_train, y_train, load_from_path=False, verbose=False)
        predictions = model.predict(X_val, load_from_path=False)

        assert predictions.shape == (20,)
        assert predictions.dtype == np.int64
        assert predictions.min() >= 0
        assert predictions.max() < 3

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, y_train, X_val, y_val = sample_data

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=1,
            use_gpu=False,
        )

        model.fit(X_train, y_train, load_from_path=False, verbose=False)
        probas = model.predict_proba(X_val, load_from_path=False)

        assert probas.shape == (20, 3)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)
        assert probas.min() >= 0
        assert probas.max() <= 1

    def test_save_and_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X_train, y_train, X_val, y_val = sample_data

        # Train model
        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
            random_state=42,
        )

        model.fit(X_train, y_train, X_val, y_val, load_from_path=False, verbose=False)
        original_predictions = model.predict(X_val, load_from_path=False)

        # Save model
        save_path = tmp_path / "test_cnn.pkl"
        model.save_model(str(save_path))

        assert save_path.exists()

        # Load model
        loaded_model = CNNClassifier()
        loaded_model.load_model(str(save_path))

        assert loaded_model.is_fitted
        assert loaded_model.num_classes == 3
        assert loaded_model.architecture == "simple"

        # Test predictions match
        loaded_predictions = loaded_model.predict(X_val, load_from_path=False)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_early_stopping(self, sample_data):
        """Test early stopping functionality."""
        X_train, y_train, X_val, y_val = sample_data

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=50,
            early_stopping_patience=3,
            use_gpu=False,
        )

        model.fit(X_train, y_train, X_val, y_val, load_from_path=False, verbose=False)

        # Should stop before max_epochs due to early stopping
        assert model.metadata["epochs_trained"] < 50

    def test_without_validation(self, sample_data):
        """Test training without validation set."""
        X_train, y_train, _, _ = sample_data

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(X_train, y_train, load_from_path=False, verbose=False)

        assert model.is_fitted
        assert len(model.history["train_loss"]) == 2
        assert len(model.history["val_loss"]) == 0

    def test_class_names(self, sample_data):
        """Test with class names."""
        X_train, y_train, X_val, y_val = sample_data
        class_names = ["cat", "dog", "bird"]

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            batch_size=8,
            max_epochs=1,
            use_gpu=False,
        )

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            class_names=class_names,
            load_from_path=False,
            verbose=False,
        )

        assert model.class_names == class_names

    def test_resnet18_architecture(self):
        """Test ResNet18 architecture."""
        model = CNNClassifier(
            architecture="resnet18",
            pretrained=False,
            image_size=(224, 224),
            batch_size=8,
            max_epochs=1,
            use_gpu=False,
        )

        # Create balanced labels to ensure all classes are present
        X_train = np.random.rand(12, 224, 224, 3).astype(np.float32)
        y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])  # Balanced 4 classes

        model.fit(X_train, y_train, load_from_path=False, verbose=False)

        assert model.is_fitted
        assert model.num_classes == 4

    def test_different_optimizers(self, sample_data):
        """Test different optimizers."""
        X_train, y_train, _, _ = sample_data

        for optimizer in ["adam", "adamw", "sgd"]:
            model = CNNClassifier(
                architecture="simple",
                image_size=(32, 32),
                optimizer_name=optimizer,
                batch_size=8,
                max_epochs=1,
                use_gpu=False,
            )

            model.fit(X_train, y_train, load_from_path=False, verbose=False)
            assert model.is_fitted

    def test_different_lr_schedulers(self, sample_data):
        """Test different learning rate schedulers."""
        X_train, y_train, X_val, y_val = sample_data

        for scheduler in ["plateau", "step", "cosine", "none"]:
            model = CNNClassifier(
                architecture="simple",
                image_size=(32, 32),
                lr_scheduler=scheduler,
                batch_size=8,
                max_epochs=3,
                use_gpu=False,
            )

            model.fit(
                X_train, y_train, X_val, y_val, load_from_path=False, verbose=False
            )
            assert model.is_fitted

    def test_gradient_clipping(self, sample_data):
        """Test gradient clipping."""
        X_train, y_train, _, _ = sample_data

        model = CNNClassifier(
            architecture="simple",
            image_size=(32, 32),
            gradient_clip_value=0.5,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(X_train, y_train, load_from_path=False, verbose=False)

        assert model.is_fitted
        assert model.gradient_clip_value == 0.5


class TestCNNIntegration:
    """Integration tests for CNN models."""

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete CNN workflow."""
        # Generate data
        np.random.seed(42)
        X_train = np.random.rand(30, 64, 64, 3).astype(np.float32)
        y_train = np.random.randint(0, 3, 30)
        X_val = np.random.rand(10, 64, 64, 3).astype(np.float32)
        y_val = np.random.randint(0, 3, 10)
        X_test = np.random.rand(5, 64, 64, 3).astype(np.float32)

        # Train model
        model = CNNClassifier(
            architecture="medium",
            image_size=(64, 64),
            batch_size=8,
            max_epochs=3,
            learning_rate=0.001,
            weight_decay=0.0001,
            lr_scheduler="plateau",
            augment_data=True,
            use_gpu=False,
            random_state=42,
        )

        model.fit(X_train, y_train, X_val, y_val, load_from_path=False, verbose=False)

        # Predict
        predictions = model.predict(X_test, load_from_path=False)
        probas = model.predict_proba(X_test, load_from_path=False)

        # Save
        save_path = tmp_path / "cnn_model.pkl"
        model.save_model(str(save_path))

        # Load
        loaded_model = CNNClassifier()
        loaded_model.load_model(str(save_path))

        # Predict with loaded model
        loaded_predictions = loaded_model.predict(X_test, load_from_path=False)

        # Verify
        np.testing.assert_array_equal(predictions, loaded_predictions)
        assert (
            model.history["train_loss"][-1] < model.history["train_loss"][0]
        )  # Loss decreased
