"""Custom exceptions for AutoML system."""


class AutoMLException(Exception):
    """Base exception for AutoML system."""

    pass


class DataLoadError(AutoMLException):
    """Raised when data loading fails."""

    pass


class DataValidationError(AutoMLException):
    """Raised when data validation fails."""

    pass


class PreprocessingError(AutoMLException):
    """Raised when preprocessing fails."""

    pass


class ModelTrainingError(AutoMLException):
    """Raised when model training fails."""

    pass


class ConfigurationError(AutoMLException):
    """Raised when configuration is invalid."""

    pass


class UnsupportedFormatError(AutoMLException):
    """Raised when file format is not supported."""

    pass


class InsufficientDataError(AutoMLException):
    """Raised when dataset is too small."""

    pass


class FeatureEngineeringError(AutoMLException):
    """Raised when feature engineering fails."""

    pass


class ModelNotFoundError(AutoMLException):
    """Raised when requested model is not found."""

    pass


class HyperparameterOptimizationError(AutoMLException):
    """Raised when HPO fails."""

    pass


# Alias for cleaner code
ValidationError = DataValidationError
