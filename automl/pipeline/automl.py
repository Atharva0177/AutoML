"""
AutoML Orchestrator.

This module provides the main AutoML class that orchestrates
the entire machine learning pipeline from data loading to model training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from automl.data.loaders import CSVLoader
from automl.data.validators import DataValidator
from automl.eda import ProblemDetector, StatisticalProfiler
from automl.models import ModelRegistry
from automl.models.recommender import ModelRecommender
from automl.preprocessing import (
    CategoricalEncoder,
    DataSplitter,
    MissingValueHandler,
    NumericalScaler,
    PipelineBuilder,
)
from automl.training import MetricsCalculator, Trainer

try:
    from automl.tracking import MLflowTracker

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowTracker = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class AutoML:
    """
    Main AutoML orchestrator class.

    Provides end-to-end automated machine learning pipeline including:
    - Data loading and validation
    - Exploratory data analysis
    - Preprocessing
    - Model selection and training
    - Model comparison and selection

    Example:
        >>> automl = AutoML()
        >>> results = automl.fit('data.csv', target_column='target')
        >>> predictions = automl.predict(test_data)
    """

    def __init__(
        self,
        problem_type: Optional[str] = None,
        use_cross_validation: bool = False,
        cv_folds: int = 5,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
        optimize_hyperparameters: bool = False,
        n_trials: int = 50,
        optimization_timeout: Optional[int] = None,
        enable_mlflow: bool = False,
        experiment_name: str = "automl_experiments",
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize AutoML pipeline.

        Args:
            problem_type: 'classification' or 'regression' (auto-detected if None)
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of CV folds
            test_size: Test set size (0.0-1.0)
            validation_size: Validation set size (0.0-1.0)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            optimize_hyperparameters: Whether to use Bayesian hyperparameter optimization
            n_trials: Number of optimization trials per model
            optimization_timeout: Timeout for optimization in seconds
            enable_mlflow: Whether to enable MLflow experiment tracking
            experiment_name: Name for MLflow experiment
            mlflow_tracking_uri: URI for MLflow tracking server
        """
        self.problem_type = problem_type
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_trials = n_trials
        self.optimization_timeout = optimization_timeout
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE

        # MLflow setup
        self.mlflow_tracker: Optional[Any] = None
        self.mlflow_run_id: Optional[str] = None
        if self.enable_mlflow:
            self.mlflow_tracker = MLflowTracker(  # type: ignore[misc]
                experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri
            )

        # Pipeline components
        self.preprocessing_pipeline: Optional[PipelineBuilder] = None
        self.label_encoder: Optional[LabelEncoder] = None  # For encoding string labels
        self.best_model: Optional[Any] = None
        self.model_comparison_results: Optional[Dict[str, Any]] = None
        self.eda_results: Optional[Dict[str, Any]] = None
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None

        # Data storage for model recommendation
        self._current_df: Optional[pd.DataFrame] = None
        self._current_target: Optional[str] = None

        logger.info(
            f"Initialized AutoML pipeline (problem_type={problem_type}, "
            f"cv={use_cross_validation}, optimize={optimize_hyperparameters}, "
            f"MLflow={self.enable_mlflow}, verbose={verbose})"
        )

    def fit(
        self,
        data: Union[str, Path, pd.DataFrame],
        target_column: str,
        categorical_features: Optional[List[str]] = None,
        models_to_try: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fit AutoML pipeline on data.

        Args:
            data: Path to CSV file or DataFrame
            target_column: Name of target column
            categorical_features: List of categorical feature names (auto-detected if None)
            models_to_try: List of model names to try (all if None)

        Returns:
            Dictionary containing pipeline results and best model
        """
        # Start MLflow run if enabled
        if self.enable_mlflow and self.mlflow_tracker:
            run_name = f"automl_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            self.mlflow_run_id = self.mlflow_tracker.start_run(
                run_name=run_name,
                tags={
                    "automl": "True",
                    "target_column": target_column,
                    "problem_type": self.problem_type or "auto",
                },
            )

        try:
            if self.verbose:
                print("=" * 80)
                print("AutoML Pipeline - Starting")
                print("=" * 80)

            # Step 1: Load and validate data
            if self.verbose:
                print("\n[1/7] Loading and validating data...")
            df = self._load_data(data)
            self._validate_data(df, target_column)

            # Store for model recommendation
            self._current_df = df
            self._current_target = target_column

            # Step 2: EDA and problem type detection
            if self.verbose:
                print("\n[2/7] Performing exploratory data analysis...")
            self._perform_eda(df, target_column)

            # Step 3: Detect problem type if not specified
            if self.problem_type is None:
                if self.verbose:
                    print("\n[3/7] Auto-detecting problem type...")
                self._detect_problem_type(df, target_column)
            else:
                if self.verbose:
                    print(f"\n[3/7] Using specified problem type: {self.problem_type}")

            # Log problem type to MLflow
            if self.enable_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.log_param("problem_type", self.problem_type)
                self.mlflow_tracker.log_params(
                    {
                        "n_samples": len(df),
                        "n_features": len(df.columns) - 1,
                        "cv": self.use_cross_validation,
                        "optimize_hyperparameters": self.optimize_hyperparameters,
                    }
                )

            # Step 4: Preprocess data
            if self.verbose:
                print("\n[4/7] Preprocessing data...")
            X_train, X_test, y_train, y_test = self._preprocess_data(
                df, target_column, categorical_features
            )

            # Step 5: Select models
            if self.verbose:
                print("\n[5/7] Selecting models to train...")
            models = self._select_models(models_to_try)

            # Step 6: Train and compare models
            if self.verbose:
                print(f"\n[6/7] Training and comparing {len(models)} models...")
            self._train_models(models, X_train, y_train, X_test, y_test)

            # Step 7: Select best model
            if self.verbose:
                print("\n[7/7] Selecting best model...")
            self._select_best_model()

            # Compile results
            results = self._compile_results(X_test, y_test)

            # Log final results to MLflow
            if (
                self.enable_mlflow
                and self.mlflow_tracker
                and self.model_comparison_results
            ):
                best_model_name = self.model_comparison_results.get("best_model")
                if best_model_name:
                    self.mlflow_tracker.set_tag("best_model", best_model_name)
                    self.mlflow_tracker.log_metric(
                        "best_model_score",
                        self.model_comparison_results.get("best_score", 0),
                    )

            if self.verbose:
                print("\n" + "=" * 80)
                print("AutoML Pipeline - Completed")
                print("=" * 80)
                self._print_summary(results)

            return results

        finally:
            # End MLflow run
            if self.enable_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.end_run()

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the best model.

        Args:
            data: Features to predict on

        Returns:
            Predictions array

        Raises:
            RuntimeError: If pipeline hasn't been fitted yet
        """
        if self.best_model is None:
            raise RuntimeError(
                "AutoML pipeline hasn't been fitted yet. Call fit() first."
            )

        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)

        # Apply preprocessing
        if self.preprocessing_pipeline is not None:
            data = self.preprocessing_pipeline.transform(data)

        # Make predictions
        return self.best_model.predict(data)

    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            data: Features to predict on

        Returns:
            Probability predictions

        Raises:
            RuntimeError: If pipeline hasn't been fitted yet
            NotImplementedError: If best model doesn't support probabilities
        """
        if self.best_model is None:
            raise RuntimeError(
                "AutoML pipeline hasn't been fitted yet. Call fit() first."
            )

        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)

        # Apply preprocessing
        if self.preprocessing_pipeline is not None:
            data = self.preprocessing_pipeline.transform(data)

        # Make predictions
        return self.best_model.predict_proba(data)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the AutoML pipeline.

        Args:
            filepath: Path to save the pipeline
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save best model
        if self.best_model is not None:
            model_path = filepath / "best_model"
            self.best_model.save(str(model_path))

        # Save preprocessing pipeline
        if self.preprocessing_pipeline is not None:
            pipeline_path = filepath / "preprocessing_pipeline"
            self.preprocessing_pipeline.save(str(pipeline_path))

        logger.info(f"AutoML pipeline saved to {filepath}")

    def _load_data(self, data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from file or DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data

        # Load from CSV
        loader = CSVLoader()
        df = loader.load(Path(data))

        if self.verbose:
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        return df

    def _validate_data(self, df: pd.DataFrame, target_column: str) -> None:
        """Validate loaded data."""
        validator = DataValidator()

        # Check target column exists
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Validate schema (use warnings instead of errors for minor issues)
        is_valid, issues = validator.validate(df)
        if not is_valid:
            # Check for severe issues only
            severe_issues = [
                issue
                for issue in issues
                if "Insufficient" in issue
                or "empty" in issue
                or "Excessive missing" in issue
            ]
            if severe_issues:
                raise ValueError(f"Data validation failed: {', '.join(severe_issues)}")
            else:
                # Just log minor issues
                logger.warning(f"Data validation warnings: {', '.join(issues)}")

        if self.verbose:
            print("  Data validation passed")

    def _perform_eda(self, df: pd.DataFrame, target_column: str) -> None:
        """Perform exploratory data analysis."""
        profiler = StatisticalProfiler()
        profile = profiler.generate_profile(df)

        self.eda_results = profile

        if self.verbose:
            overview = profile.get("overview", {})
            missing = profile.get("missing_analysis", {})
            total_missing = missing.get("total_missing", 0)
            print(f"  Numerical features: {overview.get('n_numerical', 0)}")
            print(f"  Categorical features: {overview.get('n_categorical', 0)}")
            print(f"  Missing values: {total_missing}")

    def _detect_problem_type(self, df: pd.DataFrame, target_column: str) -> None:
        """Detect problem type from target variable."""
        detector = ProblemDetector()
        problem_info = detector.detect_problem_type(df, target_column=target_column)
        detected_type = problem_info.get("problem_type")

        # Normalize problem type to 'classification' or 'regression'
        if detected_type in ["binary_classification", "multiclass_classification"]:
            self.problem_type = "classification"
        elif detected_type == "regression":
            self.problem_type = "regression"
        else:
            self.problem_type = detected_type

        if self.verbose:
            print(f"  Detected problem type: {self.problem_type}")

    def _preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        categorical_features: Optional[List[str]],
    ) -> tuple:
        """Preprocess data and split into train/test sets."""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode target for classification if labels are non-numeric
        if self.problem_type == "classification":
            # Check if target needs encoding (non-numeric labels)
            if y.dtype == "object" or y.dtype.name == "category":
                # First try to convert to numeric (handles '0', '1' stored as strings)
                try:
                    y_numeric = pd.to_numeric(y, errors="raise")
                    # If successful and values are integers, use them directly
                    if np.all(y_numeric == y_numeric.astype(int)):
                        y = y_numeric.astype(int)
                        logger.info(
                            f"Target converted from string to numeric: {y.unique()}"
                        )
                    else:
                        # Float labels - encode them
                        logger.info(f"Encoding non-integer target labels: {y.unique()}")
                        self.label_encoder = LabelEncoder()
                        y_encoded = self.label_encoder.fit_transform(y)
                        y = pd.Series(y_encoded, index=y.index, name=y.name)  # type: ignore[arg-type]
                        logger.info(f"Encoded to: {y.unique()}")
                except (ValueError, TypeError):
                    # Not numeric - encode string labels
                    logger.info(f"Encoding categorical target labels: {y.unique()}")
                    self.label_encoder = LabelEncoder()
                    y_encoded = self.label_encoder.fit_transform(y)
                    y = pd.Series(y_encoded, index=y.index, name=y.name)  # type: ignore[arg-type]
                    logger.info(f"Encoded to: {y.unique()}")
            else:
                logger.info(f"Target labels are already numeric: {y.unique()}")

        self.feature_names = X.columns.tolist()
        self.target_name = target_column

        # Split data
        splitter = DataSplitter(
            test_size=self.test_size,
            validation_size=0.0 if self.use_cross_validation else self.validation_size,
            random_state=self.random_state,
            stratify=self.problem_type == "classification",
        )

        if self.use_cross_validation:
            X_train, X_test, y_train, y_test = splitter.split(X, y)  # type: ignore[misc]
        else:
            split_result = splitter.split(X, y)
            if len(split_result) == 6:
                X_train, _, X_test, y_train, _, y_test = split_result  # type: ignore[misc]
            else:
                X_train, X_test, y_train, y_test = split_result  # type: ignore[misc]

        # Build preprocessing pipeline
        self.preprocessing_pipeline = PipelineBuilder()

        # Handle missing values
        self.preprocessing_pipeline.add_missing_handler(strategy="mean")

        # Encode categorical features
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        if categorical_features:
            self.preprocessing_pipeline.add_encoder(method="onehot")

        # Scale numerical features
        numerical_features = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        if numerical_features:
            self.preprocessing_pipeline.add_scaler(method="standard")

        # Fit and transform
        X_train = self.preprocessing_pipeline.fit_transform(X_train)
        X_test = self.preprocessing_pipeline.transform(X_test)

        if self.verbose:
            print(f"  Training set: {len(X_train)} samples")
            print(f"  Test set: {len(X_test)} samples")
            print(f"  Features after preprocessing: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def _select_models(self, models_to_try: Optional[List[str]]) -> List[Any]:
        """Select models to train based on problem type and dataset characteristics."""
        if models_to_try is None:
            # Use intelligent model recommendation
            if self._current_df is not None and self._current_target is not None:
                if self.verbose:
                    print("  Using intelligent model recommendation...")

                recommender = ModelRecommender(verbose=self.verbose)
                # Ensure problem_type is set before analyzing dataset
                if self.problem_type is None:
                    raise RuntimeError(
                        "Problem type must be determined before model recommendation"
                    )
                recommender.analyze_dataset(
                    self._current_df, self._current_target, self.problem_type
                )

                # Get top 3 recommended models by default
                recommendations = recommender.recommend_models(top_k=3)
                model_names = [rec.model_name for rec in recommendations]

                if self.verbose and not recommender.verbose:
                    # Print brief summary if recommender didn't print
                    print(f"  Recommended models: {', '.join(model_names)}")
            else:
                # Fallback: Get all models for this problem type
                model_names = ModelRegistry.list_models(model_type=self.problem_type)
        else:
            model_names = models_to_try

        # Create model instances
        models = []
        for name in model_names:
            try:
                model = ModelRegistry.create_model(name, model_type=self.problem_type)
                models.append(model)
            except Exception as e:
                logger.warning(f"Failed to create model {name}: {e}")

        if self.verbose:
            print(f"  Selected {len(models)} models: {[m.name for m in models]}")

        return models

    def _train_models(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Train and compare all models."""
        trainer = Trainer(
            use_cross_validation=self.use_cross_validation,
            cv_folds=self.cv_folds,
            cv_stratified=self.problem_type == "classification",
            random_state=self.random_state,
            optimize_hyperparameters=self.optimize_hyperparameters,
            n_trials=self.n_trials,
            optimization_timeout=self.optimization_timeout,
            mlflow_tracker=self.mlflow_tracker,
            log_to_mlflow=self.enable_mlflow,
        )

        if self.verbose and len(models) > 1:
            # Use tqdm for progress bar
            models_iter = tqdm(
                models, desc="  Training models", disable=not self.verbose
            )
        else:
            models_iter = models

        # Train models one by one to show progress
        all_results = []
        trained_models_dict = {}  # Map model name to trained model instance
        for model in models_iter:
            try:
                if self.use_cross_validation:
                    result = trainer.train(model, X_train, y_train)
                else:
                    result = trainer.train(model, X_train, y_train, X_test, y_test)
                all_results.append(result)
                trained_models_dict[model.name] = model  # Store trained model
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_trace = traceback.format_exc()
                logger.error(f"Failed to train {model.name}: {error_msg}")
                logger.debug(f"Traceback:\n{error_trace}")
                if self.verbose:
                    print(f"\n  ❌ Failed to train {model.name}: {error_msg}")
                all_results.append(
                    {"model_name": model.name, "status": "failed", "error": error_msg}
                )

        # Create comparison results
        self.model_comparison_results = {
            "models": all_results,
            "model_names": [m.name for m in models],
            "rankings": trainer._rank_models(all_results),
            "trained_models": trained_models_dict,  # Store all trained models
        }

        # Add best score for easy access
        if self.model_comparison_results["rankings"]:
            self.model_comparison_results["best_score"] = self.model_comparison_results[
                "rankings"
            ][0]["score"]
        else:
            self.model_comparison_results["best_score"] = 0

    def _select_best_model(self) -> None:
        """Select the best performing model."""
        if self.model_comparison_results is None:
            raise RuntimeError("No models have been trained yet")

        rankings = self.model_comparison_results["rankings"]

        if not rankings:
            # Collect error messages from failed models
            all_results = self.model_comparison_results.get("models", [])
            failed_models = [r for r in all_results if r.get("status") == "failed"]
            
            if failed_models:
                error_summary = "\n\nFailed models:\n"
                for failed in failed_models[:3]:  # Show up to 3 errors
                    error_summary += f"  • {failed.get('model_name', 'Unknown')}: {failed.get('error', 'Unknown error')}\n"
                
                if len(failed_models) > 3:
                    error_summary += f"  ... and {len(failed_models) - 3} more\n"
                
                raise RuntimeError(f"No models trained successfully.{error_summary}\n"
                                 f"Please check your data and configuration.")
            else:
                raise RuntimeError("No models trained successfully")

        best_model_name = rankings[0]["model_name"]

        # Get the trained model instance from the dictionary
        trained_models = self.model_comparison_results.get("trained_models", {})
        self.best_model = trained_models.get(best_model_name)

        if self.best_model is None:
            raise RuntimeError(f"Could not retrieve trained model: {best_model_name}")

        if self.verbose:
            print(f"  Best model: {best_model_name}")
            print(f"  Score: {rankings[0]['score']:.4f}")

    def _compile_results(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Compile final results."""
        best_model_name = None
        if self.model_comparison_results is not None:
            rankings = self.model_comparison_results.get("rankings", [])
            if rankings:
                best_model_name = rankings[0]["model_name"]

        results = {
            "problem_type": self.problem_type,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "n_samples_train": len(X_test),  # Approximate
            "model_comparison": self.model_comparison_results,
            "best_model": best_model_name,
        }

        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary of AutoML pipeline results."""
        print(f"\nProblem Type: {results['problem_type']}")
        print(f"Features: {results['n_features']}")
        print(f"\nBest Model: {results['best_model']}")

        print("\nModel Rankings:")
        print(f"{'Rank':<6} {'Model':<25} {'Score':<12} {'Time (s)':<10}")
        print("-" * 60)
        for ranking in results["model_comparison"]["rankings"][:5]:
            print(
                f"{ranking['rank']:<6} "
                f"{ranking['model_name']:<25} "
                f"{ranking['score']:<12.4f} "
                f"{ranking['training_time']:<10.2f}"
            )
