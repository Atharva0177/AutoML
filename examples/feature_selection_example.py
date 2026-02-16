"""
Feature Selection Example

This example demonstrates how to use the FeatureSelector class to select
the most relevant features using various methods.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from automl.preprocessing.feature_selection import FeatureSelector, SelectionMethod


def create_classification_data():
    """Create classification dataset with redundant features."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=15,
        n_redundant=10,
        n_repeated=0,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    return X_df, y_series


def create_regression_data():
    """Create regression dataset."""
    np.random.seed(42)
    result = make_regression(
        n_samples=400,
        n_features=25,
        n_informative=12,
        random_state=42
    )
    # Handle both 2-tuple and 3-tuple returns
    X = result[0]
    y = result[1]
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    return X_df, y_series


def example_1_correlation_selection():
    """Example 1: Correlation-based selection."""
    print("=" * 80)
    print("Example 1: Correlation-Based Feature Selection")
    print("=" * 80)
    
    X, y = create_regression_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select features based on correlation with target
    fs = FeatureSelector(
        method='correlation',
        correlation_threshold=0.15
    )
    X_selected = fs.fit_transform(X, y)
    
    print(f"\nSelected {len(fs.selected_features_)} features with correlation >= 0.15")
    print(f"Selected features: {fs.selected_features_[:5]}...")
    
    # Show top correlated features
    ranking = fs.get_feature_ranking()
    print("\nTop 10 features by correlation:")
    print(ranking.head(10))


def example_2_chi2_selection():
    """Example 2: Chi-square selection."""
    print("\n" + "=" * 80)
    print("Example 2: Chi-Square Feature Selection")
    print("=" * 80)
    
    X, y = create_classification_data()
    # Make data non-negative for chi2
    X_positive = X - X.min() + 1
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select top 15 features using chi-square test
    fs = FeatureSelector(
        method='chi2',
        k_features=15
    )
    X_selected = fs.fit_transform(X_positive, y)
    
    print(f"\nSelected top {X_selected.shape[1]} features using chi-square test")
    
    summary = fs.get_selection_summary()
    print(f"Selection rate: {summary['selection_rate']:.1%}")
    selected_names = summary['selected_feature_names']
    if isinstance(selected_names, list):
        print(f"Selected features: {selected_names[:5]}...")


def example_3_mutual_information():
    """Example 3: Mutual information selection."""
    print("\n" + "=" * 80)
    print("Example 3: Mutual Information Feature Selection")
    print("=" * 80)
    
    X, y = create_classification_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select features using mutual information
    fs = FeatureSelector(
        method='mutual_info',
        k_features=12,
        task_type='classification'
    )
    X_selected = fs.fit_transform(X, y)
    
    print(f"\nSelected {X_selected.shape[1]} features using mutual information")
    
    # Show top features
    ranking = fs.get_feature_ranking()
    print("\nTop 10 features by mutual information:")
    print(ranking[['feature', 'score', 'selected']].head(10))


def example_4_anova_f_test():
    """Example 4: ANOVA F-test selection."""
    print("\n" + "=" * 80)
    print("Example 4: ANOVA F-Test Feature Selection")
    print("=" * 80)
    
    X, y = create_regression_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select features using ANOVA F-test
    fs = FeatureSelector(
        method='anova_f',
        k_features=0.5,  # Select top 50% of features
        task_type='regression'
    )
    X_selected = fs.fit_transform(X, y)
    
    print(f"\nSelected top 50% features: {X_selected.shape[1]} features")
    print(f"Selected features: {fs.selected_features_[:8]}")
    
    summary = fs.get_selection_summary()
    print(f"\nSelection summary:")
    print(f"  Total features: {summary['total_features']}")
    print(f"  Selected: {summary['selected_features']}")
    print(f"  Selection rate: {summary['selection_rate']:.1%}")


def example_5_rfe():
    """Example 5: Recursive Feature Elimination."""
    print("\n" + "=" * 80)
    print("Example 5: Recursive Feature Elimination (RFE)")
    print("=" * 80)
    
    X, y = create_classification_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select features using RFE with Random Forest
    fs = FeatureSelector(
        method='rfe',
        k_features=10,
        task_type='classification'
    )
    X_selected = fs.fit_transform(X, y)
    
    print(f"\nRFE selected {X_selected.shape[1]} features")
    print(f"Selected features: {fs.selected_features_}")
    
    # Show ranking
    ranking = fs.get_feature_ranking()
    print("\nFeature ranking (inverse of elimination order):")
    print(ranking[['feature', 'score', 'selected']].head(15))


def example_6_l1_regularization():
    """Example 6: L1 (LASSO) regularization selection."""
    print("\n" + "=" * 80)
    print("Example 6: L1 Regularization Feature Selection")
    print("=" * 80)
    
    X, y = create_regression_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select features using L1 regularization
    fs = FeatureSelector(
        method='l1',
        task_type='regression'
    )
    X_selected = fs.fit_transform(X, y)
    
    print(f"\nL1 regularization selected {X_selected.shape[1]} features (non-zero coefficients)")
    print(f"Selected features: {fs.selected_features_}")
    
    # Show coefficients
    ranking = fs.get_feature_ranking()
    print("\nTop features by coefficient magnitude:")
    print(ranking[ranking['selected']][['feature', 'score']].head(10))


def example_7_tree_importance():
    """Example 7: Tree-based importance selection."""
    print("\n" + "=" * 80)
    print("Example 7: Tree-Based Feature Importance Selection")
    print("=" * 80)
    
    X, y = create_classification_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Select features using tree importance
    fs = FeatureSelector(
        method='tree_importance',
        k_features=12,
        task_type='classification'
    )
    X_selected = fs.fit_transform(X, y)
    
    print(f"\nTree importance selected top {X_selected.shape[1]} features")
    print(f"Selected features: {fs.selected_features_}")
    
    # Show importance scores
    ranking = fs.get_feature_ranking()
    print("\nFeature importance ranking:")
    print(ranking[['feature', 'score', 'selected']].head(15))


def example_8_method_comparison():
    """Example 8: Comparing different selection methods."""
    print("\n" + "=" * 80)
    print("Example 8: Comparing Feature Selection Methods")
    print("=" * 80)
    
    X, y = create_classification_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    methods = ['correlation', 'mutual_info', 'anova_f', 'rfe', 'tree_importance']
    k_features = 10
    
    results = {}
    for method in methods:
        fs = FeatureSelector(
            method=method,
            k_features=k_features,
            task_type='classification'
        )
        if method == 'correlation':
            # Convert to regression-like problem for correlation
            continue
        X_selected = fs.fit_transform(X, y)
        results[method] = set(fs.selected_features_)
    
    print(f"\nComparison of methods (selecting top {k_features} features):")
    print("-" * 80)
    
    for method, features in results.items():
        print(f"{method:20s}: {sorted(list(features))[:5]}...")
    
    # Find common features selected by all methods
    if results:
        common_features = set.intersection(*results.values())
        print(f"\nFeatures selected by all methods ({len(common_features)}):")
        print(f"  {sorted(list(common_features))}")
        
        # Find unique features per method
        print(f"\nMethod-specific selections:")
        for method, features in results.items():
            unique = features - set.union(*[f for m, f in results.items() if m != method])
            if unique:
                print(f"  {method}: {sorted(list(unique))[:3]}")


def example_9_train_test_consistency():
    """Example 9: Train/test consistency."""
    print("\n" + "=" * 80)
    print("Example 9: Ensuring Train/Test Consistency")
    print("=" * 80)
    
    X, y = create_classification_data()
    
    # Split data
    train_size = 400
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
    
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    
    # Fit on training data
    fs = FeatureSelector(
        method='anova_f',
        k_features=15,
        task_type='classification'
    )
    X_train_selected = fs.fit_transform(X_train, y_train)
    
    print(f"\nSelected {X_train_selected.shape[1]} features on training data")
    print(f"Features: {fs.selected_features_[:5]}...")
    
    # Apply same selection to test data
    X_test_selected = fs.transform(X_test)
    
    print(f"\nTest data after selection: {X_test_selected.shape}")
    print(f"✓ Same features applied to test data")
    
    # Verify columns match
    assert list(X_train_selected.columns) == list(X_test_selected.columns)
    print(f"✓ Train and test have identical feature sets")


def example_10_real_world_pipeline():
    """Example 10: Real-world feature selection pipeline."""
    print("\n" + "=" * 80)
    print("Example 10: Real-World Feature Selection Pipeline")
    print("=" * 80)
    
    # Create realistic dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'num_accounts': np.random.poisson(3, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'num_inquiries': np.random.poisson(2, n_samples),
        'payment_history': np.random.randint(0, 100, n_samples),
        'utilization_rate': np.random.uniform(0, 1, n_samples),
        'late_payments': np.random.poisson(1, n_samples)
    })
    
    # Add some redundant features
    data['age_squared'] = data['age'] ** 2
    data['income_log'] = np.log1p(data['income'])
    data['credit_income_ratio'] = data['credit_score'] / (data['income'] + 1)
    data['noise_1'] = np.random.randn(n_samples)
    data['noise_2'] = np.random.randn(n_samples)
    
    # Create target (loan default)
    y = (
        (data['credit_score'] < 600) * 0.3 +
        (data['debt_ratio'] > 0.5) * 0.3 +
        (data['late_payments'] > 2) * 0.4 +
        np.random.randn(n_samples) * 0.1
    )
    y = (y > 0.5).astype(int)
    y = pd.Series(y, name='default')
    
    print(f"Dataset: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"Target variable: loan default (0/1)")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Step 1: Quick filter with correlation
    print("\n" + "-" * 80)
    print("Step 1: Quick correlation filter")
    fs_corr = FeatureSelector(method='correlation', correlation_threshold=0.05)
    data_filtered = fs_corr.fit_transform(data, y)
    print(f"  Features after correlation filter: {data_filtered.shape[1]}")
    
    # Step 2: Mutual information for non-linear relationships
    print("\nStep 2: Mutual information selection")
    fs_mi = FeatureSelector(
        method='mutual_info',
        k_features=10,
        task_type='classification'
    )
    data_selected = fs_mi.fit_transform(data_filtered, y)
    print(f"  Features after mutual information: {data_selected.shape[1]}")
    print(f"  Selected: {fs_mi.selected_features_}")
    
    # Step 3: RFE for final selection
    print("\nStep 3: RFE for final feature set")
    fs_rfe = FeatureSelector(
        method='rfe',
        k_features=7,
        task_type='classification'
    )
    data_final = fs_rfe.fit_transform(data_selected, y)
    print(f"  Final features: {data_final.shape[1]}")
    print(f"  Selected: {fs_rfe.selected_features_}")
    
    # Get final ranking
    ranking = fs_rfe.get_feature_ranking()
    print("\n" + "-" * 80)
    print("Final Feature Ranking:")
    print(ranking[['feature', 'score', 'selected']])
    
    print("\n" + "=" * 80)
    print("Pipeline Summary:")
    print(f"  Original features: {data.shape[1]}")
    print(f"  After correlation filter: {data_filtered.shape[1]}")
    print(f"  After mutual information: {data_selected.shape[1]}")
    print(f"  Final features: {data_final.shape[1]}")
    print(f"  Reduction: {(1 - data_final.shape[1]/data.shape[1]):.1%}")
    print("=" * 80)


if __name__ == "__main__":
    example_1_correlation_selection()
    example_2_chi2_selection()
    example_3_mutual_information()
    example_4_anova_f_test()
    example_5_rfe()
    example_6_l1_regularization()
    example_7_tree_importance()
    example_8_method_comparison()
    example_9_train_test_consistency()
    example_10_real_world_pipeline()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
