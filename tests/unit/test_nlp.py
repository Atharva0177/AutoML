"""
Unit tests for NLP models.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from automl.models.deep_learning.nlp import (
    NLP_AVAILABLE,
    AttentionTextRNN,
    TextClassifier,
    TextCNN,
    TextDataset,
    TextRNN,
    Tokenizer,
    create_text_dataloaders,
)

if not NLP_AVAILABLE:
    pytest.skip("NLP dependencies not available", allow_module_level=True)

import torch


class TestTokenizer:
    """Test tokenizer functionality."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = Tokenizer(max_vocab_size=1000, min_freq=2)

        assert tokenizer.max_vocab_size == 1000
        assert tokenizer.min_freq == 2
        assert not tokenizer.is_fitted

    def test_clean_text(self):
        """Test text cleaning."""
        tokenizer = Tokenizer(lower=True)

        text = "  Hello   WORLD!  "
        cleaned = tokenizer.clean_text(text)

        assert cleaned == "hello world!"

    def test_tokenize(self):
        """Test tokenization."""
        tokenizer = Tokenizer()

        text = "This is a test."
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "this" in tokens or "This" in tokens

    def test_fit_vocabulary(self):
        """Test vocabulary building."""
        tokenizer = Tokenizer(max_vocab_size=20, min_freq=1)

        texts = ["the quick brown fox", "the lazy dog", "quick fox jumps"]

        tokenizer.fit(texts)

        assert tokenizer.is_fitted
        assert "the" in tokenizer.word2idx
        assert "quick" in tokenizer.word2idx
        assert tokenizer.vocab_size <= 20

    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = Tokenizer(max_vocab_size=100)

        texts = ["hello world", "goodbye world"]
        tokenizer.fit(texts)

        # Encode
        encoded = tokenizer.encode("hello world", max_length=10)

        assert isinstance(encoded, list)
        assert len(encoded) == 10  # Padded

        # Decode
        decoded = tokenizer.decode(encoded)

        assert isinstance(decoded, str)
        assert "hello" in decoded or "world" in decoded

    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = Tokenizer()

        texts = ["hello", "world", "test"]
        tokenizer.fit(texts)

        encoded = tokenizer.encode_batch(texts, max_length=5)

        assert encoded.shape == (3, 5)
        assert encoded.dtype == np.int64

    def test_save_load_state(self):
        """Test tokenizer state saving/loading."""
        tokenizer = Tokenizer()
        tokenizer.fit(["hello world", "test data"])

        # Save state
        state = tokenizer.get_state()

        # Load into new tokenizer
        new_tokenizer = Tokenizer()
        new_tokenizer.load_state(state)

        assert new_tokenizer.is_fitted
        assert new_tokenizer.vocab_size == tokenizer.vocab_size
        assert new_tokenizer.word2idx == tokenizer.word2idx


class TestTextDataset:
    """Test text dataset functionality."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        tokenizer = Tokenizer()
        texts = ["hello world", "test data"]
        labels = [0, 1]

        tokenizer.fit(texts)

        dataset = TextDataset(
            texts=texts, labels=labels, tokenizer=tokenizer, max_length=10
        )

        assert len(dataset) == 2

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        tokenizer = Tokenizer()
        texts = ["hello world", "test data"]
        labels = [0, 1]

        tokenizer.fit(texts)

        dataset = TextDataset(texts, labels, tokenizer, max_length=10)

        text_tensor, label = dataset[0]

        assert isinstance(text_tensor, torch.Tensor)
        assert text_tensor.shape == (10,)
        assert isinstance(label, int)

    def test_create_dataloaders(self):
        """Test dataloader creation."""
        tokenizer = Tokenizer()
        train_texts = ["text one", "text two", "text three"] * 5
        train_labels = [0, 1, 2] * 5

        tokenizer.fit(train_texts)

        train_loader, val_loader = create_text_dataloaders(
            train_texts=train_texts,
            train_labels=train_labels,
            tokenizer=tokenizer,
            batch_size=4,
        )

        assert len(train_loader) > 0

        # Check batch
        for batch_X, batch_y in train_loader:
            assert batch_X.dim() == 2  # (batch_size, seq_len)
            assert batch_y.dim() == 1  # (batch_size,)
            break


class TestRNNModels:
    """Test RNN model architectures."""

    def test_text_rnn_lstm_forward(self):
        """Test TextRNN LSTM forward pass."""
        model = TextRNN(
            vocab_size=100,
            embedding_dim=50,
            hidden_dim=64,
            num_classes=2,
            rnn_type="lstm",
        )

        x = torch.randint(0, 100, (4, 20))  # (batch_size, seq_len)
        output = model(x)

        assert output.shape == (4, 2)

    def test_text_rnn_gru_forward(self):
        """Test TextRNN GRU forward pass."""
        model = TextRNN(
            vocab_size=100,
            embedding_dim=50,
            hidden_dim=64,
            num_classes=3,
            rnn_type="gru",
            bidirectional=True,
        )

        x = torch.randint(0, 100, (2, 15))
        output = model(x)

        assert output.shape == (2, 3)

    def test_text_cnn_forward(self):
        """Test TextCNN forward pass."""
        model = TextCNN(
            vocab_size=100,
            embedding_dim=50,
            num_filters=100,
            filter_sizes=(3, 4, 5),
            num_classes=2,
        )

        x = torch.randint(0, 100, (3, 25))
        output = model(x)

        assert output.shape == (3, 2)

    def test_attention_rnn_forward(self):
        """Test AttentionTextRNN forward pass."""
        model = AttentionTextRNN(
            vocab_size=100,
            embedding_dim=50,
            hidden_dim=64,
            num_classes=4,
            bidirectional=True,
        )

        x = torch.randint(0, 100, (5, 30))
        output = model(x)

        assert output.shape == (5, 4)


class TestTextClassifier:
    """Test TextClassifier model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)

        texts = [
            "this is positive",
            "this is negative",
            "good movie",
            "bad movie",
            "excellent film",
            "terrible experience",
        ] * 10

        labels = [1, 0, 1, 0, 1, 0] * 10

        val_texts = ["great show", "awful show", "nice film", "poor quality"]
        val_labels = [1, 0, 1, 0]

        return texts, labels, val_texts, val_labels

    def test_lstm_initialization(self):
        """Test LSTM classifier initialization."""
        model = TextClassifier(
            architecture="lstm", embedding_dim=50, hidden_dim=64, max_length=20
        )

        assert model.architecture == "lstm"
        assert model.embedding_dim == 50
        assert not model.is_fitted

    def test_fit_lstm(self, sample_data):
        """Test training LSTM classifier."""
        texts, labels, val_texts, val_labels = sample_data

        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            max_length=20,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
            random_state=42,
        )

        model.fit(texts, labels, val_texts, val_labels, verbose=False)

        assert model.is_fitted
        assert model.num_classes == 2
        assert model.vocab_size is not None and model.vocab_size > 0
        assert len(model.history["train_loss"]) == 2

    def test_fit_gru(self, sample_data):
        """Test training GRU classifier."""
        texts, labels, _, _ = sample_data

        model = TextClassifier(
            architecture="gru",
            embedding_dim=30,
            hidden_dim=32,
            num_layers=1,
            max_length=20,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)

        assert model.is_fitted
        assert model.architecture == "gru"

    def test_fit_cnn(self, sample_data):
        """Test training TextCNN classifier."""
        texts, labels, _, _ = sample_data

        model = TextClassifier(
            architecture="cnn",
            embedding_dim=30,
            hidden_dim=50,
            max_length=20,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)

        assert model.is_fitted
        assert model.architecture == "cnn"

    def test_fit_attention(self, sample_data):
        """Test training attention model."""
        texts, labels, _, _ = sample_data

        model = TextClassifier(
            architecture="attention",
            embedding_dim=30,
            hidden_dim=32,
            max_length=20,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)

        assert model.is_fitted
        assert model.architecture == "attention"

    def test_predict(self, sample_data):
        """Test prediction."""
        texts, labels, val_texts, _ = sample_data

        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            max_length=20,
            batch_size=8,
            max_epochs=1,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)
        predictions = model.predict(val_texts)

        assert predictions.shape == (4,)
        assert predictions.dtype == np.int64
        assert predictions.min() >= 0
        assert predictions.max() < 2

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        texts, labels, val_texts, _ = sample_data

        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            max_length=20,
            batch_size=8,
            max_epochs=1,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)
        probas = model.predict_proba(val_texts)

        assert probas.shape == (4, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)
        assert probas.min() >= 0
        assert probas.max() <= 1

    def test_save_and_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        texts, labels, val_texts, _ = sample_data

        # Train model
        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            max_length=20,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
            random_state=42,
        )

        model.fit(texts, labels, verbose=False)
        original_predictions = model.predict(val_texts)

        # Save model
        save_path = tmp_path / "test_nlp.pkl"
        model.save_model(str(save_path))

        assert save_path.exists()

        # Load model
        loaded_model = TextClassifier()
        loaded_model.load_model(str(save_path))

        assert loaded_model.is_fitted
        assert loaded_model.num_classes == 2
        assert loaded_model.architecture == "lstm"
        assert loaded_model.vocab_size == model.vocab_size

        # Test predictions match
        loaded_predictions = loaded_model.predict(val_texts)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_early_stopping(self, sample_data):
        """Test early stopping functionality."""
        texts, labels, val_texts, val_labels = sample_data

        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            max_length=20,
            batch_size=8,
            max_epochs=50,
            early_stopping_patience=3,
            use_gpu=False,
        )

        model.fit(texts, labels, val_texts, val_labels, verbose=False)

        # Should stop before max_epochs
        assert model.metadata["epochs_trained"] < 50

    def test_multiclass_classification(self):
        """Test multi-class classification."""
        texts = ["class zero", "class one", "class two"] * 10
        labels = [0, 1, 2] * 10

        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            max_length=10,
            batch_size=6,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)

        assert model.is_fitted
        assert model.num_classes == 3

        predictions = model.predict(["class zero", "class one"])
        assert predictions.shape == (2,)

    def test_different_optimizers(self, sample_data):
        """Test different optimizers."""
        texts, labels, _, _ = sample_data

        for optimizer in ["adam", "adamw", "sgd"]:
            model = TextClassifier(
                architecture="lstm",
                embedding_dim=30,
                hidden_dim=32,
                optimizer_name=optimizer,
                max_length=20,
                batch_size=8,
                max_epochs=1,
                use_gpu=False,
            )

            model.fit(texts, labels, verbose=False)
            assert model.is_fitted

    def test_different_schedulers(self, sample_data):
        """Test different learning rate schedulers."""
        texts, labels, val_texts, val_labels = sample_data

        for scheduler in ["plateau", "step", "cosine", "none"]:
            model = TextClassifier(
                architecture="lstm",
                embedding_dim=30,
                hidden_dim=32,
                lr_scheduler=scheduler,
                max_length=20,
                batch_size=8,
                max_epochs=3,
                use_gpu=False,
            )

            model.fit(texts, labels, val_texts, val_labels, verbose=False)
            assert model.is_fitted

    def test_gradient_clipping(self, sample_data):
        """Test gradient clipping."""
        texts, labels, _, _ = sample_data

        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            gradient_clip_value=0.5,
            max_length=20,
            batch_size=8,
            max_epochs=2,
            use_gpu=False,
        )

        model.fit(texts, labels, verbose=False)

        assert model.is_fitted
        assert model.gradient_clip_value == 0.5


class TestNLPIntegration:
    """Integration tests for NLP models."""

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete NLP workflow."""
        # Generate data
        texts = [
            "positive sentiment",
            "negative sentiment",
            "happy mood",
            "sad mood",
        ] * 10

        labels = [1, 0, 1, 0] * 10

        val_texts = ["joyful", "disappointed", "excited", "upset"]
        val_labels = [1, 0, 1, 0]

        test_texts = ["cheerful", "unhappy"]

        # Train model
        model = TextClassifier(
            architecture="lstm",
            embedding_dim=30,
            hidden_dim=32,
            num_layers=1,
            bidirectional=True,
            max_length=15,
            batch_size=8,
            max_epochs=5,
            learning_rate=0.001,
            weight_decay=0.0001,
            lr_scheduler="plateau",
            use_gpu=False,
            random_state=42,
        )

        model.fit(texts, labels, val_texts, val_labels, verbose=False)

        # Predict
        predictions = model.predict(test_texts)
        probas = model.predict_proba(test_texts)

        # Save
        save_path = tmp_path / "nlp_model.pkl"
        model.save_model(str(save_path))

        # Load
        loaded_model = TextClassifier()
        loaded_model.load_model(str(save_path))

        # Predict with loaded model
        loaded_predictions = loaded_model.predict(test_texts)

        # Verify
        np.testing.assert_array_equal(predictions, loaded_predictions)
        assert model.history["train_loss"][-1] < model.history["train_loss"][0]
