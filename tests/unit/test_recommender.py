"""
Unit tests for Model Recommender system
"""

import numpy as np
import pandas as pd
import pytest

from automl.models.model_metadata import MODEL_METADATA
from automl.models.recommender import (
    DatasetCharacteristics,
    ModelRecommendation,
    ModelRecommender,
)


class TestDatasetCharacteristics:
    """Test dataset characteristics extraction"""

    def test_small_dataset_analysis(self):
        """Test analysis of small dataset"""
        # Create small dataset (200 samples)
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(200, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        df["target"] = np.random.randint(0, 2, 200)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )

        assert chars.n_samples == 200
        assert chars.n_features == 5
        assert chars.problem_type == "classification"
        assert chars.size_category in ["small", "medium"]

    def test_large_dataset_analysis(self):
        """Test analysis of large dataset"""
        # Create large dataset (50K samples)
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(50000, 20), columns=[f"feature_{i}" for i in range(20)]
        )
        df["target"] = np.random.randint(0, 3, 50000)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )

        assert chars.n_samples == 50000
        assert chars.n_features == 20
        # Large dataset should be classified as medium, large, or very_large
        assert chars.size_category in ["medium", "large", "very_large"]

    def test_imbalanced_dataset_detection(self):
        """Test detection of class imbalance"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(1000, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        # Create imbalanced classes: 90% class 0, 10% class 1
        df["target"] = np.concatenate([np.zeros(900), np.ones(100)])

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )

        assert bool(chars.is_imbalanced) is True

    def test_high_cardinality_detection(self):
        """Test detection of high cardinality features"""
        np.random.seed(42)
        # Create dataset with many unique values in numerical column (high cardinality)
        df = pd.DataFrame(
            {
                "feat_1": np.random.randn(500),  # continuous numerical
                "feat_2": np.random.choice(
                    ["A", "B", "C"], 500
                ),  # low cardinality categorical
                "feat_3": [
                    f"ID_{i}" for i in range(500)
                ],  # high cardinality categorical
                "target": np.random.randint(0, 2, 500),
            }
        )

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )

        # Should detect high cardinality in feat_3
        assert chars.max_cardinality >= 100  # feat_3 has 500 unique values

    def test_missing_values_detection(self):
        """Test detection of missing values"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(200, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        # Add missing values
        df.iloc[::10, 0] = np.nan  # 10% missing in first column
        df["target"] = np.random.randint(0, 2, 200)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )

        assert bool(chars.has_missing) is True
        assert chars.missing_percentage > 0


class TestModelRecommender:
    """Test model recommendation logic"""

    def test_recommend_for_small_dataset(self):
        """Test recommendations for small dataset"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(200, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        df["target"] = np.random.randint(0, 2, 200)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )
        recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

        # Should return 3 recommendations
        assert len(recommendations) == 3

        # All should be ModelRecommendation objects
        for rec in recommendations:
            assert isinstance(rec, ModelRecommendation)
            assert 0 <= rec.score <= 100
            assert rec.model_name in MODEL_METADATA

        # Scores should be in descending order
        scores = [r.score for r in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_for_large_dataset(self):
        """Test recommendations for large dataset"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(50000, 20), columns=[f"feature_{i}" for i in range(20)]
        )
        df["target"] = np.random.randint(0, 3, 50000)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )
        recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

        assert len(recommendations) == 3

        # Should prefer gradient boosting models for large datasets
        model_names = [r.model_name for r in recommendations]

        # XGBoost, LightGBM, or ensemble models should score well
        top_model = recommendations[0].model_name
        assert top_model in [
            "xgboost",
            "lightgbm",
            "gradient_boosting",
            "random_forest",
        ]

    def test_recommend_for_regression(self):
        """Test recommendations for regression problems"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(1000, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        df["target"] = np.random.randn(1000)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="regression"
        )
        recommendations = recommender.recommend_models(dataset_chars=chars, top_k=5)

        # All recommended models should support regression
        for rec in recommendations:
            model_meta = MODEL_METADATA[rec.model_name]
            # model_meta is a ModelCharacteristics object, use attribute access
            assert model_meta.model_type in ["both", "regression"]

    def test_recommendation_justification(self):
        """Test that recommendations include justifications"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(200, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        df["target"] = np.random.randint(0, 2, 200)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )
        recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

        # Each recommendation should have justifications
        for rec in recommendations:
            assert len(rec.justification) > 0
            assert all(isinstance(j, str) for j in rec.justification)

    def test_recommendation_warnings(self):
        """Test that warnings are provided when appropriate"""
        np.random.seed(42)
        # Create very small dataset
        df = pd.DataFrame(np.random.randn(50, 2), columns=["f1", "f2"])
        df["target"] = np.random.randint(0, 2, 50)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )
        recommendations = recommender.recommend_models(dataset_chars=chars, top_k=3)

        # Should have warnings about dataset size
        all_warnings = []
        for rec in recommendations:
            all_warnings.extend(rec.warnings)

        # At least one model should warn about small dataset
        assert len(all_warnings) > 0

    def test_top_k_parameter(self):
        """Test that top_k parameter works correctly"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(500, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        df["target"] = np.random.randint(0, 2, 500)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )

        # Test different top_k values
        for k in [1, 3, 5]:
            recommendations = recommender.recommend_models(dataset_chars=chars, top_k=k)
            assert len(recommendations) == k

    def test_score_range(self):
        """Test that all scores are in valid range"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(1000, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        df["target"] = np.random.randint(0, 2, 1000)

        recommender = ModelRecommender()
        chars = recommender.analyze_dataset(
            df, target_column="target", problem_type="classification"
        )
        recommendations = recommender.recommend_models(
            dataset_chars=chars, top_k=6
        )  # Get all models

        # All scores should be between 0 and 100
        for rec in recommendations:
            assert 0 <= rec.score <= 100
            assert isinstance(rec.score, (int, float))


class TestModelRecommenderIntegration:
    """Integration tests for recommender with AutoML"""

    def test_recommender_with_different_problem_types(self):
        """Test recommender handles both classification and regression"""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(500, 10), columns=[f"feature_{i}" for i in range(10)]
        )

        recommender = ModelRecommender()

        # Test classification
        df["target_class"] = np.random.randint(0, 2, 500)
        class_chars = recommender.analyze_dataset(
            df, target_column="target_class", problem_type="classification"
        )
        class_recs = recommender.recommend_models(dataset_chars=class_chars, top_k=3)
        assert len(class_recs) == 3

        # Test regression
        df["target_reg"] = np.random.randn(500)
        reg_chars = recommender.analyze_dataset(
            df.drop(columns=["target_class"]),
            target_column="target_reg",
            problem_type="regression",
        )
        reg_recs = recommender.recommend_models(dataset_chars=reg_chars, top_k=3)
        assert len(reg_recs) == 3

        # Different recommendations based on problem type
        class_models = {r.model_name for r in class_recs}
        reg_models = {r.model_name for r in reg_recs}

        # logistic_regression should be in classification but not regression
        if "logistic_regression" in class_models:
            assert "logistic_regression" not in reg_models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
