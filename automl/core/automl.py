"""Core AutoML class."""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import pandas as pd

from automl.data.loaders import CSVLoader, ParquetLoader
from automl.data.validators import DataValidator, SchemaValidator, QualityValidator
from automl.data.metadata import DatasetMetadata
from automl.eda import StatisticalProfiler, ProblemDetector, CorrelationAnalyzer, VisualizationGenerator
from automl.utils.logger import get_logger
from automl.utils.exceptions import DataLoadError, UnsupportedFormatError
from automl.config.config import config

logger = get_logger(__name__)


class AutoML:
    """
    Main AutoML class for automated machine learning.
    
    This class orchestrates the entire AutoML pipeline from data loading
    to model training and evaluation.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize AutoML system.
        
        Args:
            config_path: Optional path to custom configuration file
        """
        logger.info("Initializing AutoML system")
        
        self.data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.metadata = DatasetMetadata()
        
        # Initialize validators
        self.data_validator = DataValidator()
        self.schema_validator = SchemaValidator()
        self.quality_validator = QualityValidator()
        
        # Initialize EDA components
        self.profiler = StatisticalProfiler()
        self.problem_detector = ProblemDetector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.viz_generator: Optional[VisualizationGenerator] = None
        
        # Initialize loaders
        self._loaders = {
            "csv": CSVLoader(),
            "parquet": ParquetLoader(),
        }
        
        logger.info("AutoML system initialized")

    def load_data(
        self,
        filepath: Union[str, Path],
        target_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to data file
            target_column: Name of target column (optional, can be set later)
            **kwargs: Additional arguments for loader
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataLoadError: If loading fails
            UnsupportedFormatError: If file format not supported
        """
        filepath = Path(filepath)
        logger.info(f"Loading data from {filepath}")
        
        # Determine file format
        file_ext = filepath.suffix.lower().lstrip(".")
        
        # Map extensions to loaders
        ext_mapping = {
            "csv": "csv",
            "txt": "csv",
            "parquet": "parquet",
            "pq": "parquet",
        }
        
        loader_type = ext_mapping.get(file_ext)
        
        if loader_type is None:
            raise UnsupportedFormatError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {list(ext_mapping.keys())}"
            )
        
        # Load data
        loader = self._loaders[loader_type]
        self.data = loader.load(filepath, **kwargs)
        
        # Store file metadata
        self.metadata.add_file_metadata(loader.metadata)
        
        # Set target column if provided
        if target_column:
            self.target_column = target_column
        
        # Validate data
        self._validate_data()
        
        # Generate quality report
        self._generate_quality_report()
        
        if self.data is None:
            raise DataLoadError("Data loading failed - data is None")
            
        logger.info(f"Data loaded successfully: {self.data.shape}")
        return self.data

    def _validate_data(self) -> None:
        """Validate loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Validating data...")
        
        # Basic validation
        is_valid, issues = self.data_validator.validate(self.data)
        
        validation_results = {
            "is_valid": is_valid,
            "issues": issues,
        }
        
        # Target validation if specified
        if self.target_column:
            target_valid, target_issues = self.data_validator.validate_target(
                self.data, self.target_column
            )
            validation_results["target_valid"] = target_valid
            validation_results["target_issues"] = target_issues
        
        self.metadata.add_validation_results(validation_results)
        
        # Infer schema
        schema = self.schema_validator.infer_schema(self.data)
        self.metadata.add_schema(schema)

    def _generate_quality_report(self) -> None:
        """Generate data quality report."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Generating data quality report...")
        
        quality_report = self.quality_validator.generate_quality_report(self.data)
        self.metadata.add_quality_report(quality_report)
        
        # Log recommendations
        if quality_report["recommendations"]:
            logger.info("Data quality recommendations:")
            for rec in quality_report["recommendations"]:
                logger.info(f"  - {rec}")

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "target_column": self.target_column,
            "quality_score": self.metadata.get("quality.overall_score"),
            "missing_percentage": self.metadata.get("quality.missing_values.percentage"),
        }

    def set_target(self, target_column: str) -> None:
        """
        Set target column for supervised learning.
        
        Args:
            target_column: Name of target column
        """
        if self.data is None:
            raise ValueError("No data loaded. Load data first.")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Column '{target_column}' not found in data")
        
        self.target_column = target_column
        logger.info(f"Target column set to: {target_column}")
        
        # Validate target
        target_valid, target_issues = self.data_validator.validate_target(
            self.data, target_column
        )
        
        if not target_valid:
            logger.warning(f"Target validation issues: {target_issues}")

    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Train models on loaded data.
        
        Args:
            **kwargs: Training configuration
            
        Returns:
            Training results
        """
        # Placeholder for Phase 1, Month 3
        logger.warning("Training functionality not yet implemented (Phase 1, Month 3)")
        return {"status": "not_implemented"}

    def get_best_model(self):
        """
        Get best trained model.
        
        Returns:
            Best model
        """
        # Placeholder for Phase 1, Month 3
        logger.warning("Model training not yet implemented (Phase 1, Month 3)")
        return None

    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical profile of the dataset.
        
        Returns:
            Statistical profile dictionary
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        profile = self.profiler.generate_profile(self.data)
        return profile

    def detect_problem_type(self) -> Dict[str, Any]:
        """
        Detect the machine learning problem type.
        
        Returns:
            Problem type information
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        problem_info = self.problem_detector.detect_problem_type(
            self.data, self.target_column
        )
        
        # Update target column if inferred
        if problem_info.get("target_column") and not self.target_column:
            self.target_column = problem_info["target_column"]
            logger.info(f"Auto-detected target column: {self.target_column}")
        
        return problem_info

    def analyze_correlations(
        self,
        method: str = "pearson",
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Analyze feature correlations.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Threshold for high correlations
            
        Returns:
            Correlation analysis results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        correlation_analysis = self.correlation_analyzer.analyze_correlations(
            self.data,
            target_column=self.target_column,
            method=method,
            threshold=threshold,
        )
        
        return correlation_analysis

    def generate_visualizations(
        self,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        """
        Generate EDA visualizations.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if output_dir is None:
            output_dir = Path("visualizations")
        
        self.viz_generator = VisualizationGenerator(output_dir)
        
        # Get correlation matrix if available
        correlation_matrix = None
        if self.correlation_analyzer.correlation_matrix is not None:
            correlation_matrix = self.correlation_analyzer.correlation_matrix
        
        plots = self.viz_generator.generate_all_visualizations(
            self.data,
            target_column=self.target_column,
            correlation_matrix=correlation_matrix,
        )
        
        return plots

    def run_eda(
        self,
        generate_visualizations: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run complete exploratory data analysis.
        
        Args:
            generate_visualizations: Whether to generate visualizations
            output_dir: Directory for outputs
            
        Returns:
            Complete EDA results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Running comprehensive EDA...")
        
        eda_results = {}
        
        # Statistical profiling
        logger.info("Generating statistical profile...")
        eda_results["profile"] = self.generate_profile()
        
        # Problem detection
        logger.info("Detecting problem type...")
        eda_results["problem_type"] = self.detect_problem_type()
        
        # Correlation analysis
        logger.info("Analyzing correlations...")
        eda_results["correlations"] = self.analyze_correlations()
        
        # Visualizations
        if generate_visualizations:
            logger.info("Generating visualizations...")
            eda_results["visualizations"] = self.generate_visualizations(output_dir)
        
        logger.info("EDA complete")
        return eda_results

    def save_metadata(self, filepath: Path) -> None:
        """
        Save dataset metadata to file.
        
        Args:
            filepath: Path to save metadata
        """
        self.metadata.save(filepath)

    def __repr__(self) -> str:
        """String representation."""
        if self.data is None:
            return "AutoML(no data loaded)"
        return f"AutoML(data shape={self.data.shape}, target={self.target_column})"
