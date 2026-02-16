"""
Command-line interface for AutoML.

This module provides a Click-based CLI for running AutoML tasks
from the command line.
"""

import json
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
import yaml

from automl.pipeline import AutoML


@click.group()
@click.version_option(version="0.1.0", prog_name="AutoML")
def cli():
    """
    AutoML - Automated Machine Learning Pipeline

    A powerful command-line tool for automated machine learning.
    """
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("target_column", type=str)
@click.option(
    "--problem-type",
    type=click.Choice(["classification", "regression"]),
    help="Problem type (auto-detected if not specified)",
)
@click.option(
    "--config", type=click.Path(exists=True), help="Path to YAML configuration file"
)
@click.option(
    "--models",
    multiple=True,
    help="Models to try (e.g., --models logistic_regression --models random_forest)",
)
@click.option(
    "--cv/--no-cv",
    default=False,
    help="Use cross-validation (default: holdout validation)",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    help="Number of cross-validation folds (default: 5)",
)
@click.option(
    "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
)
@click.option(
    "--validation-size",
    type=float,
    default=0.2,
    help="Validation set size for holdout (default: 0.2)",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "--output", type=click.Path(), help="Output directory for results and model"
)
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def train(
    data_path: str,
    target_column: str,
    problem_type: Optional[str],
    config: Optional[str],
    models: tuple,
    cv: bool,
    cv_folds: int,
    test_size: float,
    validation_size: float,
    random_state: int,
    output: Optional[str],
    quiet: bool,
):
    """
    Train AutoML models on a dataset.

    DATA_PATH: Path to CSV file containing training data

    TARGET_COLUMN: Name of the target column

    Examples:

      \b
      # Auto-detect problem type and train all models
      automl train data.csv target

      \b
      # Specify problem type and use cross-validation
      automl train data.csv target --problem-type classification --cv --cv-folds 10

      \b
      # Train specific models only
      automl train data.csv target --models logistic_regression --models random_forest

      \b
      # Use configuration file
      automl train data.csv target --config config.yaml
    """
    # Load config if provided
    if config:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Override with config values
        problem_type = problem_type or config_data.get("problem_type")
        cv = cv or config_data.get("use_cross_validation", False)
        cv_folds = config_data.get("cv_folds", cv_folds)
        test_size = config_data.get("test_size", test_size)
        validation_size = config_data.get("validation_size", validation_size)
        random_state = config_data.get("random_state", random_state)
        models = models or tuple(config_data.get("models", []))

    # Create AutoML instance
    automl = AutoML(
        problem_type=problem_type,
        use_cross_validation=cv,
        cv_folds=cv_folds,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        verbose=not quiet,
    )

    # Train models
    click.echo(f"Loading data from: {data_path}")
    results = automl.fit(
        data=data_path,
        target_column=target_column,
        models_to_try=list(models) if models else None,
    )

    # Save results if output specified
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        automl.save(output_path)

        # Save results as JSON
        results_file = output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        click.echo(f"\nResults saved to: {output_path}")

    click.echo("\n✓ Training completed successfully!")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--output", type=click.Path(), help="Output file for predictions (default: stdout)"
)
@click.option(
    "--probabilities",
    is_flag=True,
    help="Output class probabilities (classification only)",
)
def predict(
    model_path: str, data_path: str, output: Optional[str], probabilities: bool
):
    """
    Make predictions using a trained model.

    MODEL_PATH: Path to saved model directory

    DATA_PATH: Path to CSV file containing features

    Examples:

      \b
      # Make predictions and save to file
      automl predict models/best_model data.csv --output predictions.csv

      \b
      # Get class probabilities
      automl predict models/best_model data.csv --probabilities
    """
    click.echo(f"Loading model from: {model_path}")
    # Note: This would require implementing model loading
    # For now, this is a placeholder
    click.echo("Prediction functionality coming soon!")


@cli.command()
@click.argument("output_path", type=click.Path())
def init_config(output_path: str):
    """
    Create a sample configuration file.

    OUTPUT_PATH: Path where config file will be created

    Example:

      automl init-config config.yaml
    """
    config = {
        "problem_type": None,  # Auto-detect
        "use_cross_validation": False,
        "cv_folds": 5,
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "models": ["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        "preprocessing": {
            "missing_values": {
                "numerical_strategy": "mean",
                "categorical_strategy": "most_frequent",
            },
            "scaling": {"strategy": "standard"},
            "encoding": {"strategy": "onehot"},
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Configuration file created: {output_path}")
    click.echo("Edit this file and use it with: automl train --config " + output_path)


@cli.command()
def list_models():
    """List all available models."""
    from automl.models import ModelRegistry

    click.echo("\n=== Available Models ===\n")

    click.echo("Classification Models:")
    classification_models = ModelRegistry.list_models(model_type="classification")
    for model in classification_models:
        click.echo(f"  - {model}")

    click.echo("\nRegression Models:")
    regression_models = ModelRegistry.list_models(model_type="regression")
    for model in regression_models:
        click.echo(f"  - {model}")


if __name__ == "__main__":
    cli()
