"""
Tests for models module.

Tests for BaseModel, ModelRegistry, and all traditional ML models.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.models import (
    BaseModel,
    GradientBoostingModel,
    LightGBMModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    ModelRegistry,
    RandomForestModel,
    XGBoostModel,
)

# Fixtures


@pytest.fixture
def classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        }
    )

    # Create classification target
    y = pd.Series(np.random.randint(0, 3, n_samples))

    return X, y


@pytest.fixture
def regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        }
    )

    # Create regression target
    y = pd.Series(
        X["feature_1"] * 2 + X["feature_2"] * 0.5 + np.random.randn(n_samples) * 0.1
    )

    return X, y


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


# Model Registry Tests


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_list_models(self):
        """Test listing all registered models."""
        models = ModelRegistry.list_models()
        assert len(models) > 0
        assert "logistic_regression" in models
        assert "xgboost" in models

    def test_list_classification_models(self):
        """Test filtering classification models."""
        models = ModelRegistry.list_models(model_type="classification")
        assert "logistic_regression" in models
        assert "linear_regression" not in models

    def test_list_regression_models(self):
        """Test filtering regression models."""
        models = ModelRegistry.list_models(model_type="regression")
        assert "linear_regression" in models
        assert "logistic_regression" not in models

    def test_get_model_class(self):
        """Test retrieving model class."""
        model_class = ModelRegistry.get_model_class("logistic_regression")
        assert model_class == LogisticRegressionModel

    def test_get_model_info(self):
        """Test retrieving model information."""
        info = ModelRegistry.get_model_info("xgboost")
        assert info["class"] == XGBoostModel
        assert info["type"] == "both"
        assert "description" in info

    def test_create_model(self):
        """Test creating model instance from registry."""
        model = ModelRegistry.create_model("random_forest", model_type="classification")
        assert isinstance(model, RandomForestModel)
        assert model.model_type == "classification"

    def test_is_registered(self):
        """Test checking if model is registered."""
        assert ModelRegistry.is_registered("lightgbm")
        assert not ModelRegistry.is_registered("nonexistent_model")


# Classification Model Tests


class TestLogisticRegressionModel:
    """Tests for LogisticRegressionModel."""

    def test_init(self):
        """Test model initialization."""
        model = LogisticRegressionModel()
        assert model.name == "LogisticRegression"
        assert model.model_type == "classification"
        assert not model.is_fitted

    def test_invalid_model_type(self):
        """Test that model only supports classification."""
        with pytest.raises(ValueError):
            LogisticRegressionModel(model_type="regression")

    def test_fit_predict(self, classification_data):
        """Test fitting and prediction."""
        X, y = classification_data
        model = LogisticRegressionModel()

        # Fit model
        model.fit(X, y)
        assert model.is_fitted
        assert model.feature_names == X.columns.tolist()

        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_predict_proba(self, classification_data):
        """Test probability predictions."""
        X, y = classification_data
        model = LogisticRegressionModel()
        model.fit(X, y)

        probas = model.predict_proba(X)
        assert probas.shape[0] == len(y)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_feature_importance(self, classification_data):
        """Test feature importance extraction."""
        X, y = classification_data
        model = LogisticRegressionModel()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(X.columns)
        assert all(k in importance for k in X.columns)

    def test_save_load(self, classification_data, temp_dir):
        """Test model serialization."""
        X, y = classification_data
        model = LogisticRegressionModel()
        model.fit(X, y)

        # Save model
        save_path = Path(temp_dir) / "logistic_model"
        model.save(save_path)
        assert (Path(temp_dir) / "logistic_model.pkl").exists()
        assert (Path(temp_dir) / "logistic_model.json").exists()

        # Load model
        new_model = LogisticRegressionModel()
        new_model.load(save_path)
        assert new_model.is_fitted

        # Compare predictions
        pred1 = model.predict(X)
        pred2 = new_model.predict(X)
        np.testing.assert_array_equal(pred1, pred2)


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def test_classification(self, classification_data):
        """Test Random Forest classification."""
        X, y = classification_data
        model = RandomForestModel(model_type="classification")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

        probas = model.predict_proba(X)
        assert probas.shape[0] == len(y)

    def test_regression(self, regression_data):
        """Test Random Forest regression."""
        X, y = regression_data
        model = RandomForestModel(model_type="regression")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_feature_importance(self, classification_data):
        """Test feature importance."""
        X, y = classification_data
        model = RandomForestModel()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(X.columns)
        assert all(v >= 0 for v in importance.values())


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_classification(self, classification_data):
        """Test XGBoost classification."""
        X, y = classification_data
        model = XGBoostModel(model_type="classification")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

        probas = model.predict_proba(X)
        assert probas.shape[0] == len(y)

    def test_regression(self, regression_data):
        """Test XGBoost regression."""
        X, y = regression_data
        model = XGBoostModel(model_type="regression")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_hyperparameters(self):
        """Test custom hyperparameters."""
        params = {"n_estimators": 50, "learning_rate": 0.05}
        model = XGBoostModel(hyperparameters=params)

        assert model.hyperparameters["n_estimators"] == 50
        assert model.hyperparameters["learning_rate"] == 0.05


class TestLightGBMModel:
    """Tests for LightGBMModel."""

    def test_classification(self, classification_data):
        """Test LightGBM classification."""
        X, y = classification_data
        model = LightGBMModel(model_type="classification")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

        probas = model.predict_proba(X)
        assert probas.shape[0] == len(y)

    def test_regression(self, regression_data):
        """Test LightGBM regression."""
        X, y = regression_data
        model = LightGBMModel(model_type="regression")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)


class TestLinearRegressionModel:
    """Tests for LinearRegressionModel."""

    def test_init(self):
        """Test model initialization."""
        model = LinearRegressionModel()
        assert model.name == "LinearRegression"
        assert model.model_type == "regression"
        assert not model.is_fitted

    def test_invalid_model_type(self):
        """Test that model only supports regression."""
        with pytest.raises(ValueError):
            LinearRegressionModel(model_type="classification")

    def test_fit_predict(self, regression_data):
        """Test fitting and prediction."""
        X, y = regression_data
        model = LinearRegressionModel()

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

        # Check R² score is reasonable
        assert model.metadata["r2_score"] > 0.5


class TestGradientBoostingModel:
    """Tests for GradientBoostingModel."""

    def test_classification(self, classification_data):
        """Test Gradient Boosting classification."""
        X, y = classification_data
        model = GradientBoostingModel(model_type="classification")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_regression(self, regression_data):
        """Test Gradient Boosting regression."""
        X, y = regression_data
        model = GradientBoostingModel(model_type="regression")

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y)


# General Model Tests


class TestBaseModelFunctionality:
    """Tests for base model functionality across all models."""

    @pytest.mark.parametrize(
        "model_class,model_type",
        [
            (RandomForestModel, "classification"),
            (GradientBoostingModel, "classification"),
            (XGBoostModel, "classification"),
            (LightGBMModel, "classification"),
        ],
    )
    def test_get_set_params(self, model_class, model_type):
        """Test get_params and set_params methods."""
        model = model_class(model_type=model_type)

        params = model.get_params()
        assert isinstance(params, dict)
        assert len(params) > 0

        model.set_params(n_estimators=50)
        assert model.hyperparameters["n_estimators"] == 50

    @pytest.mark.parametrize(
        "model_class,model_type",
        [
            (LogisticRegressionModel, "classification"),
            (LinearRegressionModel, "regression"),
            (RandomForestModel, "classification"),
        ],
    )
    def test_metadata(
        self, model_class, model_type, classification_data, regression_data
    ):
        """Test metadata extraction."""
        data = (
            classification_data if model_type == "classification" else regression_data
        )
        X, y = data

        model = model_class(model_type=model_type)
        model.fit(X, y)

        metadata = model.get_metadata()
        assert metadata["name"] == model.name
        assert metadata["model_type"] == model_type
        assert metadata["is_fitted"] is True
        assert metadata["feature_names"] == X.columns.tolist()

    def test_predict_before_fit(self, classification_data):
        """Test that predict raises error before fitting."""
        X, y = classification_data
        model = LogisticRegressionModel()

        with pytest.raises(RuntimeError):
            model.predict(X)
