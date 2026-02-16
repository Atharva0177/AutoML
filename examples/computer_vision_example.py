"""
Computer Vision Examples with CNNClassifier.

Demonstrates image classification using custom and pretrained CNNs.
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)

# Example 1: Training SimpleCNN on synthetic data
print("=" * 80)
print("Example 1: Training SimpleCNN on Synthetic Data")
print("=" * 80)

from automl.models.deep_learning.vision import CNNClassifier

# Generate synthetic image data (32x32 RGB images)
np.random.seed(42)
n_samples = 200
X_train = np.random.rand(n_samples, 32, 32, 3).astype(np.float32)
y_train = np.random.randint(0, 5, n_samples)  # 5 classes

X_val = np.random.rand(50, 32, 32, 3).astype(np.float32)
y_val = np.random.randint(0, 5, 50)

# Train simple CNN
cnn = CNNClassifier(
    architecture="simple",
    image_size=(32, 32),
    learning_rate=0.001,
    batch_size=16,
    max_epochs=10,
    early_stopping_patience=5,
    augment_data=True,
    use_gpu=True,
)

cnn.fit(
    X_train,
    y_train,
    X_val,
    y_val,
    load_from_path=False,  # We're using arrays, not file paths
    verbose=True,
)

# Predictions
predictions = cnn.predict(X_val, load_from_path=False)
probabilities = cnn.predict_proba(X_val, load_from_path=False)

print(f"\nPredictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Sample predictions: {predictions[:5]}")
print(f"Sample probabilities (first image): {probabilities[0]}")

# Accuracy
accuracy = np.mean(predictions == y_val)
print(f"\nValidation Accuracy: {accuracy:.4f}")

print(f"\nModel trained for {cnn.metadata['epochs_trained']} epochs")
print(f"Best validation loss: {cnn.metadata['best_val_loss']:.4f}")


# Example 2: Medium CNN with More Capacity
print("\n" + "=" * 80)
print("Example 2: Medium CNN Architecture")
print("=" * 80)

medium_cnn = CNNClassifier(
    architecture="medium",
    image_size=(64, 64),
    learning_rate=0.0005,
    batch_size=16,
    max_epochs=10,
    weight_decay=0.0001,
    lr_scheduler="plateau",
    lr_patience=3,
    lr_factor=0.5,
    gradient_clip_value=1.0,
    use_gpu=True,
)

# Generate larger images
X_train_64 = np.random.rand(n_samples, 64, 64, 3).astype(np.float32)
X_val_64 = np.random.rand(50, 64, 64, 3).astype(np.float32)

medium_cnn.fit(X_train_64, y_train, X_val_64, y_val, load_from_path=False, verbose=True)

print(f"\nMedium CNN trained for {medium_cnn.metadata['epochs_trained']} epochs")


# Example 3: Transfer Learning with ResNet18
print("\n" + "=" * 80)
print("Example 3: Transfer Learning with ResNet18")
print("=" * 80)

# Note: Setting pretrained=True will download ImageNet weights
resnet_cnn = CNNClassifier(
    architecture="resnet18",
    pretrained=False,  # Set to True to use ImageNet weights
    freeze_backbone=False,  # Set to True to freeze pretrained layers
    image_size=(224, 224),
    learning_rate=0.001,
    batch_size=8,
    max_epochs=5,
    use_gpu=True,
)

# Generate ImageNet-sized images
X_train_224 = np.random.rand(100, 224, 224, 3).astype(np.float32)
y_train_small = np.random.randint(0, 5, 100)
X_val_224 = np.random.rand(30, 224, 224, 3).astype(np.float32)
y_val_small = np.random.randint(0, 5, 30)

resnet_cnn.fit(
    X_train_224,
    y_train_small,
    X_val_224,
    y_val_small,
    load_from_path=False,
    verbose=True,
)

print(f"\nResNet18 trained for {resnet_cnn.metadata['epochs_trained']} epochs")


# Example 4: Loading Images from File Paths
print("\n" + "=" * 80)
print("Example 4: Loading Images from File Paths (Conceptual)")
print("=" * 80)

print("""
To load images from file paths, organize your data as:

dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    val/
        class1/
            ...
        class2/
            ...

Then use:

from automl.models.deep_learning.vision import ImageFolderDataset

# Or provide list of file paths
train_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
train_labels = [0, 1, 2, ...]

cnn = CNNClassifier(architecture='resnet50', pretrained=True)
cnn.fit(
    train_paths, train_labels,
    val_paths, val_labels,
    load_from_path=True,  # Load from file paths
    class_names=['cat', 'dog', 'bird']
)
""")


# Example 5: Saving and Loading Models
print("\n" + "=" * 80)
print("Example 5: Saving and Loading Models")
print("=" * 80)

# Save the model
save_path = Path("saved_models/cnn_example.pkl")
save_path.parent.mkdir(parents=True, exist_ok=True)
cnn.save_model(str(save_path))
print(f"\nModel saved to: {save_path}")

# Load the model
loaded_cnn = CNNClassifier()
loaded_cnn.load_model(str(save_path))
print(f"Model loaded successfully")
print(f"Architecture: {loaded_cnn.architecture}")
print(f"Number of classes: {loaded_cnn.num_classes}")
print(f"Image size: {loaded_cnn.image_size}")

# Make predictions with loaded model
loaded_predictions = loaded_cnn.predict(X_val, load_from_path=False)
print(f"\nPredictions match: {np.all(loaded_predictions == predictions)}")


# Example 6: Model Comparison
print("\n" + "=" * 80)
print("Example 6: Architecture Comparison")
print("=" * 80)

architectures = ["simple", "medium"]
results = {}

X_train_comp = np.random.rand(150, 64, 64, 3).astype(np.float32)
y_train_comp = np.random.randint(0, 3, 150)
X_val_comp = np.random.rand(50, 64, 64, 3).astype(np.float32)
y_val_comp = np.random.randint(0, 3, 50)

for arch in architectures:
    print(f"\nTraining {arch.upper()} architecture...")

    model = CNNClassifier(
        architecture=arch,
        image_size=(64, 64),
        learning_rate=0.001,
        batch_size=16,
        max_epochs=8,
        early_stopping_patience=4,
        verbose=False,
        use_gpu=True,
    )

    model.fit(
        X_train_comp,
        y_train_comp,
        X_val_comp,
        y_val_comp,
        load_from_path=False,
        verbose=False,
    )

    # Evaluate
    predictions = model.predict(X_val_comp, load_from_path=False)
    accuracy = np.mean(predictions == y_val_comp)

    results[arch] = {
        "epochs": model.metadata["epochs_trained"],
        "best_val_loss": model.metadata["best_val_loss"],
        "accuracy": accuracy,
    }

    print(
        f"{arch.upper()} - Epochs: {results[arch]['epochs']}, "
        f"Val Loss: {results[arch]['best_val_loss']:.4f}, "
        f"Accuracy: {results[arch]['accuracy']:.4f}"
    )

print("\n" + "=" * 80)
print("Comparison Summary")
print("=" * 80)
for arch, metrics in results.items():
    print(
        f"{arch.upper():10s} | "
        f"Epochs: {metrics['epochs']:2d} | "
        f"Loss: {metrics['best_val_loss']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f}"
    )


# Example 7: All Optimization Features Combined
print("\n" + "=" * 80)
print("Example 7: Full Optimization Stack")
print("=" * 80)

optimized_cnn = CNNClassifier(
    architecture="medium",
    image_size=(64, 64),
    learning_rate=0.001,
    batch_size=16,
    max_epochs=15,
    early_stopping_patience=5,
    # Regularization
    weight_decay=0.0001,
    augment_data=True,
    # Learning rate scheduling
    lr_scheduler="plateau",
    lr_patience=3,
    lr_factor=0.5,
    # Gradient clipping
    gradient_clip_value=1.0,
    # Optimizer
    optimizer_name="adamw",
    use_gpu=True,
)

optimized_cnn.fit(
    X_train_comp,
    y_train_comp,
    X_val_comp,
    y_val_comp,
    load_from_path=False,
    verbose=True,
)

print(f"\nOptimized model configuration:")
print(f"  - Architecture: {optimized_cnn.architecture}")
print(f"  - Optimizer: {optimized_cnn.optimizer_name}")
print(f"  - Weight decay: {optimized_cnn.weight_decay}")
print(f"  - LR scheduler: {optimized_cnn.lr_scheduler}")
print(f"  - Gradient clipping: {optimized_cnn.gradient_clip_value}")
print(f"  - Data augmentation: {optimized_cnn.augment_data}")
print(f"\nTraining results:")
print(f"  - Epochs trained: {optimized_cnn.metadata['epochs_trained']}")
print(f"  - Best val loss: {optimized_cnn.metadata['best_val_loss']:.4f}")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("=" * 80)
