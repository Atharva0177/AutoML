"""Configuration management for AutoML system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from automl.utils.exceptions import ConfigurationError
from automl.utils.logger import get_logger

logger = get_logger(__name__)


class Config:
    """Configuration manager for AutoML system."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to custom config file. If None, uses default.
        """
        if not self._config:  # Only load once
            self._load_config(config_path)
            self._load_environment()

    def _load_config(self, config_path: Optional[Path] = None) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file
        """
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent / "settings.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Environment variables loaded from .env")

        # Override config with environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        # Logging level
        if os.getenv("AUTOML_LOG_LEVEL"):
            self._config.setdefault("logging", {})
            self._config["logging"]["level"] = os.getenv("AUTOML_LOG_LEVEL")

        # Data directory
        if os.getenv("AUTOML_DATA_DIR"):
            self._config.setdefault("data", {})
            self._config["data"]["data_dir"] = os.getenv("AUTOML_DATA_DIR")

        # Results directory
        if os.getenv("AUTOML_RESULTS_DIR"):
            self._config.setdefault("output", {})
            self._config["output"]["results_dir"] = os.getenv("AUTOML_RESULTS_DIR")

        # MLflow tracking URI
        if os.getenv("MLFLOW_TRACKING_URI"):
            self._config.setdefault("tracking", {})
            self._config["tracking"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI")

        # Number of jobs
        n_jobs_env = os.getenv("AUTOML_N_JOBS")
        if n_jobs_env:
            self._config.setdefault("general", {})
            self._config["general"]["n_jobs"] = int(n_jobs_env)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.

        Args:
            key: Dot-separated key path (e.g., "data.test_size")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path.

        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def save(self, filepath: Path) -> None:
        """
        Save current configuration to file.

        Args:
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, "w") as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self._config, updates)

    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """
        Recursively update nested dictionary.

        Args:
            base: Base dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get config value using dictionary syntax."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set config value using dictionary syntax."""
        self.set(key, value)

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({len(self._config)} sections)"


# Global config instance
config = Config()
