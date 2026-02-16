"""
EDA Example - Demonstrating Exploratory Data Analysis features.

This example shows how to use the AutoML EDA module to:
- Generate statistical profiles
- Detect problem types
- Analyze correlations
- Generate visualizations
"""

from pathlib import Path

import numpy as np
import pandas as pd

from automl import AutoML

print("=" * 60)
print("AutoML EDA Example")
print("=" * 60)

# Step 1: Create sample dataset
print("\nStep 1: Creating sample dataset...")
np.random.seed(42)

n_samples = 500
data = {
    "age": np.random.randint(18, 80, n_samples),
    "income": np.random.exponential(50000, n_samples),
    "credit_score": np.random.randint(300, 850, n_samples),
    "employment_years": np.random.randint(0, 40, n_samples),
    "num_accounts": np.random.randint(0, 10, n_samples),
    "region": np.random.choice(["North", "South", "East", "West"], n_samples),
    "education": np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], n_samples
    ),
}

# Create target with some relationship to features
loan_probability = (
    (data["credit_score"] - 300) / 550 * 0.5
    + (data["income"] / 100000) * 0.3
    + (data["employment_years"] / 40) * 0.2
)
data["loan_approved"] = (
    loan_probability + np.random.normal(0, 0.2, n_samples) > 0.5
).astype(int)

df = pd.DataFrame(data)

# Add some missing values
np.random.seed(42)
missing_mask = np.random.random((n_samples, len(df.columns))) < 0.05
df = df.mask(missing_mask)

# Add a duplicate row
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Save dataset
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
data_file = data_dir / "eda_example.csv"
df.to_csv(data_file, index=False)
print(f"  ✓ Dataset saved: {data_file}")
print(f"  ✓ Shape: {df.shape}")

# Step 2: Initialize AutoML
print("\nStep 2: Initializing AutoML...")
aml = AutoML()
print("  ✓ AutoML initialized")

# Step 3: Load data
print("\nStep 3: Loading data...")
aml.load_data(data_file, target_column="loan_approved")
print("  ✓ Data loaded")

# Step 4: Generate Statistical Profile
print("\nStep 4: Generating Statistical Profile...")
print("-" * 60)
profile = aml.generate_profile()

print("\nDataset Overview:")
print(f"  Rows: {profile['overview']['n_rows']}")
print(f"  Columns: {profile['overview']['n_columns']}")
print(f"  Numerical columns: {profile['overview']['n_numerical']}")
print(f"  Categorical columns: {profile['overview']['n_categorical']}")
print(f"  Memory usage: {profile['overview']['memory_usage_mb']:.2f} MB")
print(f"  Duplicate rows: {profile['overview']['duplicate_rows']}")

print("\nMissing Values:")
missing_info = profile["missing_analysis"]
print(f"  Total missing: {missing_info['total_missing']}")
print(f"  Missing percentage: {missing_info['missing_percentage']:.2f}%")
print(f"  Columns with missing values: {missing_info['columns_with_missing']}")

print("\nData Quality Insights:")
for insight in profile["summary"]["insights"]:
    print(f"  - {insight}")

# Step 5: Detect Problem Type
print("\nStep 5: Detecting Problem Type...")
print("-" * 60)
problem_info = aml.detect_problem_type()

print(f"\nProblem Type: {problem_info['problem_type'].upper()}")
print(f"Target Column: {problem_info['target_column']}")
print(f"Number of Classes: {problem_info.get('n_classes', 'N/A')}")

if "is_balanced" in problem_info:
    balance_status = "Balanced" if problem_info["is_balanced"] else "Imbalanced"
    print(f"Class Balance: {balance_status}")
    if not problem_info["is_balanced"]:
        print(f"Imbalance Ratio: {problem_info.get('imbalance_ratio', 0):.2f}:1")

print("\nClass Distribution:")
class_dist = problem_info.get("class_distribution", {})
for class_name, count in class_dist.items():
    percentage = (count / problem_info["n_samples"]) * 100
    print(f"  {class_name}: {count} ({percentage:.1f}%)")

print("\nRecommendations:")
for rec in problem_info.get("recommendations", []):
    print(f"  - {rec}")

print("\nSuggested Metrics:", ", ".join(aml.problem_detector.get_suggested_metrics()))
print("Suggested Models:", ", ".join(aml.problem_detector.get_suggested_models()))

# Step 6: Analyze Correlations
print("\nStep 6: Analyzing Correlations...")
print("-" * 60)
corr_analysis = aml.analyze_correlations(threshold=0.5)

print(f"\nAnalyzed {corr_analysis['n_features']} numerical features")
print(
    f"Found {len(corr_analysis['high_correlations'])} high correlations (threshold: 0.5)"
)

if corr_analysis["high_correlations"]:
    print("\nTop High Correlations:")
    for corr in corr_analysis["high_correlations"][:5]:
        print(
            f"  {corr['feature_1']} <-> {corr['feature_2']}: {corr['correlation']:.3f}"
        )

if "target_correlations" in corr_analysis and corr_analysis["target_correlations"]:
    print("\nTop Features Correlated with Target:")
    target_corrs = corr_analysis["target_correlations"].get("top_absolute", {})
    for feature, corr_value in list(target_corrs.items())[:5]:
        print(f"  {feature}: {corr_value:.3f}")

if corr_analysis["multicollinearity"]["detected"]:
    print(
        f"\nMulticollinearity Detected: {corr_analysis['multicollinearity']['severity'].upper()}"
    )
    print("Recommendations:")
    for rec in corr_analysis["recommendations"]:
        print(f"  - {rec}")

# Step 7: Generate Visualizations
print("\nStep 7: Generating Visualizations...")
print("-" * 60)
viz_dir = Path("visualizations") / "eda_example"
plots = aml.generate_visualizations(output_dir=viz_dir)

print("\nGenerated Visualizations:")
for plot_type, plot_path in plots.items():
    if plot_path:
        print(f"  ✓ {plot_type}: {plot_path}")

# Step 8: Run Complete EDA
print("\nStep 8: Running Complete EDA (all-in-one)...")
print("-" * 60)
output_dir = Path("results") / "eda_example"
output_dir.mkdir(parents=True, exist_ok=True)

# Note: This would normally be used instead of individual steps
# eda_results = aml.run_eda(generate_visualizations=True, output_dir=output_dir)
print("  ✓ Complete EDA functionality available via aml.run_eda()")

# Step 9: Get Field-Specific Statistics
print("\nStep 9: Accessing Specific Statistics...")
print("-" * 60)

# Get stats for a specific column
income_stats = aml.profiler.get_column_stats("income")
if income_stats:
    print("\nIncome Statistics:")
    print(f"  Mean: ${income_stats['mean']:.2f}")
    print(f"  Median: ${income_stats['50%']:.2f}")
    print(f"  Std Dev: ${income_stats['std']:.2f}")
    print(f"  Min: ${income_stats['min']:.2f}")
    print(f"  Max: ${income_stats['max']:.2f}")
    print(f"  Outliers: {income_stats['outliers']}")

print("\n" + "=" * 60)
print("EDA Example Complete!")
print("=" * 60)

print("\nNext Steps:")
print("  1. Review generated visualizations in:", viz_dir)
print("  2. Examine statistical profiles for data quality")
print("  3. Use problem type info to select appropriate models")
print("  4. Consider correlation insights for feature engineering")
print("  5. Proceed to preprocessing and model training")
