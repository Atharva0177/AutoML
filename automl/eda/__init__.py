"""EDA (Exploratory Data Analysis) module for AutoML."""

from automl.eda.profiler import StatisticalProfiler
from automl.eda.problem_detector import ProblemDetector
from automl.eda.correlation import CorrelationAnalyzer
from automl.eda.visualizations import VisualizationGenerator

__all__ = [
    "StatisticalProfiler",
    "ProblemDetector",
    "CorrelationAnalyzer",
    "VisualizationGenerator",
]
