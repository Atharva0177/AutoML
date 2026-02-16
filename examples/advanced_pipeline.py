"""
Advanced Example: Custom Data Loading and Validation

This script shows how to use AutoML with more advanced options.
"""

from automl import AutoML
from automl.data.loaders import CSVLoader
from automl.data.validators import QualityValidator
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    """Advanced example workflow."""
    
    print("Advanced AutoML Example")
    print("=" * 60)
    print()
    
    # Create a more complex dataset with issues
    print("Creating dataset with various data quality issues...")
    np.random.seed(123)
    n_samples = 500
    
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randint(0, 100, n_samples),
        "feature_3": np.random.choice(["A", "B", "C", "D", "E"], n_samples),
        "feature_4": [1] * n_samples,  # Constant column
        "feature_5": np.random.randn(n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    df.loc[np.random.choice(df.index, 50), "feature_1"] = np.nan
    df.loc[np.random.choice(df.index, 30), "feature_3"] = np.nan
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)
    
    # Save dataset
    data_path = Path("data/advanced_example.csv")
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"  ✓ Dataset saved: {data_path}")
    print(f"  ✓ Shape: {df.shape}")
    print()
    
    # Custom CSV loader usage
    print("Using custom CSV loader options...")
    loader = CSVLoader()
    loaded_df = loader.load(data_path, encoding="utf-8")
    print(f"  ✓ Loaded {len(loaded_df)} rows")
    print(f"  ✓ Encoding: {loader.metadata.get('encoding')}")
    print()
    
    # Quality validation
    print("Running quality validation...")
    quality_validator = QualityValidator()
    quality_report = quality_validator.generate_quality_report(loaded_df)
    
    print(f"  Quality Score: {quality_report['overall_score']:.1f}/100")
    print(f"  Missing Values: {quality_report['missing_values']['percentage']:.1f}%")
    print(f"  Duplicates: {quality_report['duplicates']['count']} rows ({quality_report['duplicates']['percentage']:.1f}%)")
    print()
    
    print("Recommendations:")
    for i, rec in enumerate(quality_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    print()
    
    # Use AutoML
    print("Loading data with AutoML...")
    aml = AutoML()
    aml.load_data(data_path, target_column="target")
    
    info = aml.get_data_info()
    print(f"  ✓ Quality Score: {info['quality_score']:.1f}/100")
    print()
    
    # Save results
    output_dir = Path("results/advanced_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    aml.save_metadata(output_dir / "metadata.json")
    print(f"  ✓ Metadata saved to {output_dir}")
    print()
    
    print("=" * 60)
    print("Advanced Example Complete!")
    print()


if __name__ == "__main__":
    main()
