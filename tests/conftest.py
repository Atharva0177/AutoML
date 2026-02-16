"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "numeric_1": np.random.randn(100),
        "numeric_2": np.random.randint(0, 100, 100),
        "cat_1": np.random.choice(["A", "B", "C"], 100),
        "cat_2": np.random.choice(["X", "Y"], 100),
        "target": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def sample_df_with_missing():
    """Create a sample DataFrame with missing values for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "numeric_1": np.random.randn(100),
        "numeric_2": np.random.randint(0, 100, 100),
        "cat_1": np.random.choice(["A", "B", "C"], 100),
        "cat_2": np.random.choice(["X", "Y"], 100),
    })
    
    # Introduce missing values
    df.loc[np.random.choice(100, 10, replace=False), "numeric_1"] = np.nan
    df.loc[np.random.choice(100, 10, replace=False), "numeric_2"] = np.nan
    df.loc[np.random.choice(100, 5, replace=False), "cat_1"] = np.nan
    df.loc[np.random.choice(100, 5, replace=False), "cat_2"] = np.nan
    
    return df


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    """Create a temporary CSV file."""
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_parquet(sample_df, tmp_path):
    """Create a temporary Parquet file."""
    parquet_path = tmp_path / "sample.parquet"
    sample_df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
