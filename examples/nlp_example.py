"""
NLP Text Classification Examples.

Demonstrates text classification using RNN, LSTM, GRU, CNN, and Attention models.
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)

# Example 1: Basic LSTM Text Classification
print("=" * 80)
print("Example 1: LSTM Text Classification")
print("=" * 80)

from automl.models.deep_learning.nlp import TextClassifier

# Sample movie reviews (positive/negative sentiment)
train_texts = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible film, waste of time and money.",
    "Great acting and amazing storyline. Highly recommended!",
    "Boring and predictable. Would not watch again.",
    "One of the best movies I've ever seen. Brilliant!",
    "Awful movie with bad acting.",
    "Excellent cinematography and compelling characters.",
    "Disappointed. The plot made no sense.",
    "Masterpiece! A must-watch for everyone.",
    "Horrible experience. Left the theater early.",
] * 10  # Repeat for more data

train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10  # 1=positive, 0=negative

val_texts = [
    "Amazing movie with great performances!",
    "Waste of my time, very disappointing.",
    "Loved it! Will watch again.",
    "Not worth watching.",
]

val_labels = [1, 0, 1, 0]

# Train LSTM model
lstm_model = TextClassifier(
    architecture="lstm",
    embedding_dim=50,
    hidden_dim=64,
    num_layers=1,
    bidirectional=True,
    max_length=50,
    learning_rate=0.001,
    batch_size=16,
    max_epochs=10,
    early_stopping_patience=5,
    use_gpu=True,
)

lstm_model.fit(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    class_names=["negative", "positive"],
    verbose=True,
)

# Predict on new texts
test_texts = ["This film is absolutely wonderful!", "Terrible movie, don't watch it."]

predictions = lstm_model.predict(test_texts)
probabilities = lstm_model.predict_proba(test_texts)

print(f"\nTest Predictions:")
for i, text in enumerate(test_texts):
    if lstm_model.class_names is not None:
        pred = lstm_model.class_names[predictions[i]]
    else:
        pred = str(predictions[i])
    conf = probabilities[i][predictions[i]]
    print(f'  Text: "{text}"')
    print(f"  Predicted: {pred} (confidence: {conf:.3f})")

print(f"\nModel Info:")
print(f"  - Vocabulary size: {lstm_model.vocab_size}")
print(f"  - Epochs trained: {lstm_model.metadata['epochs_trained']}")
print(f"  - Best val loss: {lstm_model.metadata['best_val_loss']:.4f}")


# Example 2: GRU Model
print("\n" + "=" * 80)
print("Example 2: GRU Text Classification")
print("=" * 80)

gru_model = TextClassifier(
    architecture="gru",
    embedding_dim=50,
    hidden_dim=64,
    num_layers=2,
    bidirectional=True,
    dropout=0.5,
    max_length=50,
    batch_size=16,
    max_epochs=10,
    use_gpu=True,
)

gru_model.fit(train_texts, train_labels, val_texts, val_labels, verbose=False)

print(f"GRU Model trained for {gru_model.metadata['epochs_trained']} epochs")
print(f"Best validation loss: {gru_model.metadata['best_val_loss']:.4f}")


# Example 3: TextCNN
print("\n" + "=" * 80)
print("Example 3: CNN for Text Classification")
print("=" * 80)

cnn_model = TextClassifier(
    architecture="cnn",
    embedding_dim=50,
    hidden_dim=100,  # num_filters
    max_length=50,
    batch_size=16,
    max_epochs=10,
    use_gpu=True,
)

cnn_model.fit(train_texts, train_labels, val_texts, val_labels, verbose=False)

print(f"TextCNN trained for {cnn_model.metadata['epochs_trained']} epochs")
print(f"Best validation loss: {cnn_model.metadata['best_val_loss']:.4f}")


# Example 4: Attention-based Model
print("\n" + "=" * 80)
print("Example 4: Attention-based RNN")
print("=" * 80)

attention_model = TextClassifier(
    architecture="attention",
    embedding_dim=50,
    hidden_dim=64,
    num_layers=1,
    bidirectional=True,
    max_length=50,
    batch_size=16,
    max_epochs=10,
    use_gpu=True,
)

attention_model.fit(train_texts, train_labels, val_texts, val_labels, verbose=False)

print(
    f"Attention model trained for {attention_model.metadata['epochs_trained']} epochs"
)
print(f"Best validation loss: {attention_model.metadata['best_val_loss']:.4f}")


# Example 5: Multi-class Classification
print("\n" + "=" * 80)
print("Example 5: Multi-class Topic Classification")
print("=" * 80)

# Sample news articles
topics_texts = [
    "The stock market reached a new high today.",
    "Scientists discover new planet in distant galaxy.",
    "The football team won the championship game.",
    "New smartphone model released with advanced features.",
    "Government announces new economic policy.",
    "Research shows breakthrough in cancer treatment.",
    "Basketball player scores 50 points in final game.",
    "Latest laptop features powerful processor.",
] * 15  # Repeat for more data

# 0=finance, 1=science, 2=sports, 3=technology
topics_labels = [0, 1, 2, 3, 0, 1, 2, 3] * 15

val_topics_texts = [
    "Major tech company reports record profits.",
    "New study on climate change published.",
    "Tennis player wins grand slam tournament.",
    "Revolutionary AI model announced.",
]

val_topics_labels = [0, 1, 2, 3]

multiclass_model = TextClassifier(
    architecture="lstm",
    embedding_dim=50,
    hidden_dim=64,
    num_layers=2,
    bidirectional=True,
    max_length=30,
    batch_size=16,
    max_epochs=15,
    use_gpu=True,
)

multiclass_model.fit(
    topics_texts,
    topics_labels,
    val_topics_texts,
    val_topics_labels,
    class_names=["Finance", "Science", "Sports", "Technology"],
    verbose=False,
)

# Test predictions
test_topics = [
    "New cryptocurrency reaches all-time high value.",
    "Astronomers observe distant supernova explosion.",
]

predictions = multiclass_model.predict(test_topics)
probabilities = multiclass_model.predict_proba(test_topics)

print(f"\nMulti-class Predictions:")
for i, text in enumerate(test_topics):
    if multiclass_model.class_names is not None:
        pred = multiclass_model.class_names[predictions[i]]
        all_probs = dict(zip(multiclass_model.class_names, probabilities[i]))
    else:
        pred = str(predictions[i])
        all_probs = {}
    conf = probabilities[i][predictions[i]]
    print(f'  Text: "{text}"')
    print(f"  Predicted: {pred} (confidence: {conf:.3f})")
    if all_probs:
        print(f"  All probabilities: {all_probs}")


# Example 6: Model Comparison
print("\n" + "=" * 80)
print("Example 6: Architecture Comparison")
print("=" * 80)

architectures = ["lstm", "gru", "cnn", "attention"]
results = {}

for arch in architectures:
    print(f"\nTraining {arch.upper()}...")

    model = TextClassifier(
        architecture=arch,
        embedding_dim=50,
        hidden_dim=64,
        num_layers=1,
        bidirectional=True if arch != "cnn" else False,
        max_length=50,
        batch_size=16,
        max_epochs=10,
        use_gpu=True,
    )

    model.fit(train_texts, train_labels, val_texts, val_labels, verbose=False)

    # Evaluate
    predictions = model.predict(val_texts)
    accuracy = np.mean(predictions == np.array(val_labels))

    results[arch] = {
        "epochs": model.metadata["epochs_trained"],
        "best_val_loss": model.metadata["best_val_loss"],
        "accuracy": accuracy,
    }

print("\n" + "=" * 80)
print("Architecture Comparison Results")
print("=" * 80)
print(f"{'Architecture':<15} {'Epochs':<10} {'Val Loss':<12} {'Accuracy':<10}")
print("-" * 80)
for arch, metrics in results.items():
    print(
        f"{arch.upper():<15} "
        f"{metrics['epochs']:<10} "
        f"{metrics['best_val_loss']:<12.4f} "
        f"{metrics['accuracy']:<10.4f}"
    )


# Example 7: Save and Load Model
print("\n" + "=" * 80)
print("Example 7: Model Persistence")
print("=" * 80)

# Save model
save_path = Path("saved_models/text_lstm.pkl")
save_path.parent.mkdir(parents=True, exist_ok=True)
lstm_model.save_model(str(save_path))
print(f"Model saved to: {save_path}")

# Make predictions before loading
original_predictions = lstm_model.predict(test_texts)

# Load model
loaded_model = TextClassifier()
loaded_model.load_model(str(save_path))
print(f"Model loaded successfully")
print(f"  - Architecture: {loaded_model.architecture}")
print(f"  - Vocabulary size: {loaded_model.vocab_size}")
print(f"  - Classes: {loaded_model.class_names}")

# Test loaded model
loaded_predictions = loaded_model.predict(test_texts)
print(
    f"\nLoaded model predictions match: {np.all(loaded_predictions == original_predictions)}"
)


# Example 8: Optimization Features
print("\n" + "=" * 80)
print("Example 8: Advanced Training Features")
print("=" * 80)

optimized_model = TextClassifier(
    architecture="lstm",
    embedding_dim=50,
    hidden_dim=64,
    num_layers=2,
    bidirectional=True,
    max_length=50,
    # Optimizer settings
    optimizer_name="adamw",
    learning_rate=0.001,
    weight_decay=0.0001,
    # Learning rate scheduling
    lr_scheduler="plateau",
    lr_patience=3,
    lr_factor=0.5,
    # Gradient clipping
    gradient_clip_value=1.0,
    # Regularization
    dropout=0.5,
    # Training settings
    batch_size=16,
    max_epochs=20,
    early_stopping_patience=5,
    use_gpu=True,
)

optimized_model.fit(train_texts, train_labels, val_texts, val_labels, verbose=True)

print(f"\nOptimized model configuration:")
print(f"  - Architecture: {optimized_model.architecture}")
print(f"  - Optimizer: {optimized_model.optimizer_name}")
print(f"  - Weight decay: {optimized_model.weight_decay}")
print(f"  - LR scheduler: {optimized_model.lr_scheduler}")
print(f"  - Gradient clipping: {optimized_model.gradient_clip_value}")
print(f"\nTraining results:")
print(f"  - Epochs trained: {optimized_model.metadata['epochs_trained']}")
print(f"  - Best val loss: {optimized_model.metadata['best_val_loss']:.4f}")
print(f"  - Vocabulary size: {optimized_model.vocab_size}")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("=" * 80)
