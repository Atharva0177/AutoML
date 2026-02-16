"""Metadata management for datasets."""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from automl.utils.logger import get_logger
from automl.utils.helpers import save_json, load_json

logger = get_logger(__name__)


class DatasetMetadata:
    """Manages metadata for datasets."""

    def __init__(self):
        """Initialize metadata."""
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }

    def add_file_metadata(self, file_metadata: Dict) -> None:
        """Add file-level metadata."""
        self.metadata["file"] = file_metadata

    def add_validation_results(self, validation_results: Dict) -> None:
        """Add validation results."""
        self.metadata["validation"] = validation_results

    def add_quality_report(self, quality_report: Dict) -> None:
        """Add quality report."""
        self.metadata["quality"] = quality_report

    def add_schema(self, schema: Dict) -> None:
        """Add schema information."""
        self.metadata["schema"] = schema

    def add_statistics(self, statistics: Dict) -> None:
        """Add statistical summary."""
        self.metadata["statistics"] = statistics

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        keys = key.split(".")
        value = self.metadata
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """Set metadata value by key."""
        keys = key.split(".")
        metadata = self.metadata
        
        for k in keys[:-1]:
            if k not in metadata:
                metadata[k] = {}
            metadata = metadata[k]
        
        metadata[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get all metadata as dictionary."""
        return self.metadata.copy()

    def save(self, filepath: Path) -> None:
        """
        Save metadata to JSON file.
        
        Args:
            filepath: Path to save metadata
        """
        save_json(self.metadata, filepath)
        logger.info(f"Metadata saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "DatasetMetadata":
        """
        Load metadata from JSON file.
        
        Args:
            filepath: Path to metadata file
            
        Returns:
            DatasetMetadata instance
        """
        metadata_obj = cls()
        metadata_obj.metadata = load_json(filepath)
        logger.info(f"Metadata loaded from {filepath}")
        return metadata_obj

    def __repr__(self) -> str:
        """String representation."""
        return f"DatasetMetadata({len(self.metadata)} keys)"
