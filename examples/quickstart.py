"""
Quick Start Example for AutoML

This script demonstrates basic usage of the AutoML system.
"""

from automl import AutoML
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    
    n_samples = 1000
    data = {
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.normal(50000, 20000, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "employment_years": np.random.randint(0, 40, n_samples),
        "num_accounts": np.random.randint(1, 10, n_samples),
        "region": np.random.choice(["North", "South", "East", "West"], n_samples),
        "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n_samples),
        "loan_approved": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main function demonstrating AutoML usage."""
    
    print("=" * 60)
    print("AutoML Quick Start Example")
    print("=" * 60)
    print()
    
    # Step 1: Create sample data
    print("Step 1: Creating sample dataset...")
    df = create_sample_data()
    
    # Save to CSV
    data_path = Path("data/sample_loan_data.csv")
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"  ✓ Sample data saved to {data_path}")
    print(f"  ✓ Dataset shape: {df.shape}")
    print()
    
    # Step 2: Initialize AutoML
    print("Step 2: Initializing AutoML...")
    aml = AutoML()
    print("  ✓ AutoML initialized")
    print()
    
    # Step 3: Load data
    print("Step 3: Loading data...")
    aml.load_data(data_path, target_column="loan_approved")
    print("  ✓ Data loaded successfully")
    print()
    
    # Step 4: Get data information
    print("Step 4: Data Summary")
    print("-" * 60)
    info = aml.get_data_info()
    print(f"Shape: {info['shape']}")
    print(f"Target Column: {info['target_column']}")
    print(f"Quality Score: {info['quality_score']:.1f}/100")
    print(f"Missing Values: {info['missing_percentage']:.1f}%")
    print()
    print("Columns:")
    for col, dtype in info['dtypes'].items():
        print(f"  - {col}: {dtype}")
    print()
    
    # Step 5: Save metadata
    print("Step 5: Saving metadata...")
    output_dir = Path("results/quickstart")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    aml.save_metadata(metadata_path)
    print(f"  ✓ Metadata saved to {metadata_path}")
    print()
    
    # Step 6: Future steps (placeholder)
    print("Step 6: Model Training")
    print("-" * 60)
    print("  Note: Model training will be implemented in Phase 1, Month 3")
    print("  For now, data loading and validation are complete!")
    print()
    
    print("=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Explore the metadata.json file")
    print("  2. Try loading your own dataset")
    print("  3. Use the CLI: automl train -i data.csv -t target")
    print()


if __name__ == "__main__":
    main()
