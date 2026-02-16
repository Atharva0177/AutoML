"""
Example demonstrating outlier detection and handling.

This example shows how to use different outlier detection methods:
- IQR (Interquartile Range)
- Isolation Forest
- Z-Score
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl.preprocessing.cleaners.outlier_detector import OutlierDetector


def create_sample_data_with_outliers():
    """Create sample data with intentional outliers."""
    np.random.seed(42)

    # Normal distribution
    normal_data = np.random.randn(200) * 10 + 50

    # Add extreme outliers
    outliers = [0, 5, 95, 100, 110, 120]

    data = np.concatenate([normal_data, outliers])

    df = pd.DataFrame(
        {
            "value": data,
            "feature1": np.random.randn(len(data)) * 5 + 25,
            "feature2": np.random.randn(len(data)) * 3 + 15,
            "category": np.random.choice(["A", "B", "C"], len(data)),
        }
    )

    return df


def example_iqr_cap():
    """Example 1: IQR method with capping (Winsorization)."""
    print("=" * 70)
    print("Example 1: IQR Method with Capping")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_outliers()
    print(f"\nOriginal data shape: {data.shape}")
    print(f"\nValue column statistics:")
    print(data["value"].describe())

    # Detect and cap outliers
    detector = OutlierDetector(
        strategy="iqr", action="cap", threshold=1.5  # Standard IQR multiplier
    )

    result = detector.fit_transform(data)

    print(f"\nAfter capping outliers:")
    print(result["value"].describe())

    # Show bounds
    summary = detector.get_outlier_summary()
    print(f"\nOutlier detection summary:")
    print(f"  Strategy: {summary['strategy']}")
    print(f"  Action: {summary['action']}")
    print(
        f"  Outliers detected in 'value': {summary['outlier_counts'].get('value', 0)}"
    )

    if "value" in summary["bounds"]:
        lower, upper = summary["bounds"]["value"]
        print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")

    print(f"\nShape after processing: {result.shape}")


def example_iqr_remove():
    """Example 2: IQR method with removal."""
    print("\n" + "=" * 70)
    print("Example 2: IQR Method with Removal")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_outliers()
    rows_before = len(data)

    print(f"\nOriginal data: {rows_before} rows")

    # Detect and remove outliers
    detector = OutlierDetector(strategy="iqr", action="remove", threshold=1.5)

    result = detector.fit_transform(data)
    rows_after = len(result)
    rows_removed = rows_before - rows_after

    print(f"After removing outliers: {rows_after} rows")
    print(f"Rows removed: {rows_removed} ({rows_removed/rows_before*100:.2f}%)")

    # Show report
    report = detector.get_outlier_report(data)
    print(f"\nOutlier Report:")
    print(report.to_string(index=False))


def example_iqr_flag():
    """Example 3: IQR method with flagging."""
    print("\n" + "=" * 70)
    print("Example 3: IQR Method with Flagging")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_outliers()

    # Detect and flag outliers
    detector = OutlierDetector(strategy="iqr", action="flag", threshold=1.5)

    result = detector.fit_transform(data)

    print(f"\nData shape: {result.shape}")
    print(f"\nOutlier flag column added: 'is_outlier'")
    print(f"Total outliers flagged: {result['is_outlier'].sum()}")
    print(f"Percentage: {result['is_outlier'].mean()*100:.2f}%")

    # Show some outlier rows
    print(f"\nSample of flagged outliers:")
    outlier_rows = result[result["is_outlier"] == 1].head()
    print(outlier_rows[["value", "feature1", "feature2", "is_outlier"]])


def example_zscore():
    """Example 4: Z-Score method."""
    print("\n" + "=" * 70)
    print("Example 4: Z-Score Method")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_outliers()

    print(f"\nOriginal data statistics:")
    print(data["value"].describe())

    # Detect outliers using Z-score
    detector = OutlierDetector(
        strategy="zscore", action="cap", zscore_threshold=3.0  # 3 standard deviations
    )

    result = detector.fit_transform(data)

    print(f"\nAfter Z-score capping (threshold=3.0):")
    print(result["value"].describe())

    summary = detector.get_outlier_summary()
    if "value" in summary["bounds"]:
        lower, upper = summary["bounds"]["value"]
        print(f"\nBounds: [{lower:.2f}, {upper:.2f}]")
        print(f"Outliers detected: {summary['outlier_counts'].get('value', 0)}")


def example_isolation_forest():
    """Example 5: Isolation Forest method."""
    print("\n" + "=" * 70)
    print("Example 5: Isolation Forest Method")
    print("=" * 70)

    # Create multivariate data with outliers
    np.random.seed(42)

    # Normal cluster
    n_normal = 200
    X1 = np.random.randn(n_normal, 2) * 2 + [5, 5]

    # Add outliers
    outliers = np.array([[15, 15], [-5, -5], [15, -5], [-5, 15], [20, 0], [0, 20]])

    X = np.vstack([X1, outliers])

    data = pd.DataFrame(X, columns=["feature1", "feature2"])

    print(f"\nData shape: {data.shape}")

    # Apply Isolation Forest
    detector = OutlierDetector(
        strategy="isolation_forest",
        action="flag",
        contamination=0.05,  # Expect 5% outliers
    )

    result = detector.fit_transform(data)

    print(f"\nOutliers detected: {result['is_outlier'].sum()}")
    print(f"Percentage: {result['is_outlier'].mean()*100:.2f}%")

    # Show some detected outliers
    print(f"\nSample of detected outliers:")
    outlier_rows = result[result["is_outlier"] == 1].head(10)
    print(outlier_rows)


def example_comparison():
    """Example 6: Compare different methods."""
    print("\n" + "=" * 70)
    print("Example 6: Comparing Different Methods")
    print("=" * 70)

    # Create same data for all methods
    data = create_sample_data_with_outliers()

    print(f"\nOriginal data: {len(data)} rows")

    # IQR method
    iqr_detector = OutlierDetector(strategy="iqr", action="flag", threshold=1.5)
    iqr_result = iqr_detector.fit_transform(data.copy())
    iqr_outliers = iqr_result["is_outlier"].sum()

    # Z-Score method
    zscore_detector = OutlierDetector(
        strategy="zscore", action="flag", zscore_threshold=3.0
    )
    zscore_result = zscore_detector.fit_transform(data.copy())
    zscore_outliers = zscore_result["is_outlier"].sum()

    # Isolation Forest
    iforest_detector = OutlierDetector(
        strategy="isolation_forest", action="flag", contamination=0.05
    )
    iforest_result = iforest_detector.fit_transform(data.copy())
    iforest_outliers = iforest_result["is_outlier"].sum()

    # Compare results
    print(f"\nOutliers detected by each method:")
    print(
        f"  IQR (threshold=1.5):           {iqr_outliers} ({iqr_outliers/len(data)*100:.2f}%)"
    )
    print(
        f"  Z-Score (threshold=3.0):       {zscore_outliers} ({zscore_outliers/len(data)*100:.2f}%)"
    )
    print(
        f"  Isolation Forest (cont=0.05):  {iforest_outliers} ({iforest_outliers/len(data)*100:.2f}%)"
    )


def example_real_world():
    """Example 7: Real-world scenario with sensor data."""
    print("\n" + "=" * 70)
    print("Example 7: Real-World Scenario - Sensor Data")
    print("=" * 70)

    # Simulate sensor data with occasional malfunctions
    np.random.seed(123)
    n_samples = 500

    # Normal readings
    temperature = np.random.randn(n_samples) * 2 + 22  # ~22°C
    humidity = np.random.randn(n_samples) * 5 + 60  # ~60%
    pressure = np.random.randn(n_samples) * 10 + 1013  # ~1013 hPa

    # Add sensor malfunctions (outliers)
    malfunction_indices = np.random.choice(n_samples, 15, replace=False)
    temperature[malfunction_indices[:5]] = [50, -10, 100, -20, 60]
    humidity[malfunction_indices[5:10]] = [150, 200, -50, 120, 180]
    pressure[malfunction_indices[10:]] = [800, 1200, 1300, 700, 1400]

    data = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1min"),
        }
    )

    print(f"\nSensor data: {len(data)} readings")
    print(f"\nOriginal statistics:")
    print(data[["temperature", "humidity", "pressure"]].describe())

    # Detect outliers using IQR
    detector = OutlierDetector(strategy="iqr", action="cap", threshold=1.5)

    cleaned_data = detector.fit_transform(data)

    print(f"\nAfter outlier detection and capping:")
    print(cleaned_data[["temperature", "humidity", "pressure"]].describe())

    # Show report
    report = detector.get_outlier_report(data)
    print(f"\nOutlier Detection Report:")
    print(
        report[
            [
                "column",
                "outlier_count",
                "outlier_percentage",
                "lower_bound",
                "upper_bound",
            ]
        ].to_string(index=False)
    )


def example_pipeline():
    """Example 8: Using outlier detection in a preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("Example 8: Outlier Detection in Preprocessing Pipeline")
    print("=" * 70)

    # Create data
    data = create_sample_data_with_outliers()

    print(f"\n1. Original data: {data.shape}")
    print(f"   Value range: [{data['value'].min():.2f}, {data['value'].max():.2f}]")

    # Step 1: Flag outliers for analysis
    print(f"\n2. Step 1: Flag outliers for analysis")
    detector_flag = OutlierDetector(strategy="iqr", action="flag", threshold=1.5)
    flagged_data = detector_flag.fit_transform(data.copy())
    print(f"   Outliers flagged: {flagged_data['is_outlier'].sum()}")

    # Step 2: Cap extreme values
    print(f"\n3. Step 2: Cap extreme outliers")
    detector_cap = OutlierDetector(strategy="iqr", action="cap", threshold=1.5)
    capped_data = detector_cap.fit_transform(data.copy())
    print(
        f"   Value range after capping: [{capped_data['value'].min():.2f}, {capped_data['value'].max():.2f}]"
    )

    # Step 3: For modeling, might want to remove extreme cases
    print(f"\n4. Step 3: Remove extreme cases for modeling")
    detector_remove = OutlierDetector(
        strategy="iqr", action="remove", threshold=2.0
    )  # More lenient
    modeling_data = detector_remove.fit_transform(data.copy())
    print(f"   Final modeling data: {modeling_data.shape}")
    print(f"   Rows removed: {len(data) - len(modeling_data)}")


if __name__ == "__main__":
    # Run examples
    example_iqr_cap()
    example_iqr_remove()
    example_iqr_flag()
    example_zscore()
    example_isolation_forest()
    example_comparison()
    example_real_world()
    example_pipeline()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
