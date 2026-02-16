"""Utility functions for AutoML system."""

from typing import Any, Dict, Optional
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate_hash(data: Any) -> str:
    """
    Generate hash for any data object.
    
    Args:
        data: Data to hash
        
    Returns:
        Hash string
    """
    if isinstance(data, pd.DataFrame):
        # Hash DataFrame by converting to string
        data_bytes = data.to_json().encode('utf-8')
    elif isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    else:
        data_bytes = str(data).encode('utf-8')
    
    return hashlib.md5(data_bytes).hexdigest()


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, filepath: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, "r") as f:
        result: Dict = json.load(f)
        return result


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes to human readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    size_float = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024.0:
            return f"{size_float:.2f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.2f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{int(hours)}h {int(mins)}m"


def get_memory_usage(obj: Any) -> int:
    """
    Get memory usage of object in bytes.
    
    Args:
        obj: Object to measure
        
    Returns:
        Memory usage in bytes
    """
    if isinstance(obj, pd.DataFrame):
        return int(obj.memory_usage(deep=True).sum())
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    else:
        import sys
        return sys.getsizeof(obj)
