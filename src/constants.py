"""Centralized constants for RoadVision - single source of truth for classes and palette."""

import numpy as np
from src.config import get_config

def _load_constants():
    """Load constants from config.yaml."""
    config = get_config()

    # Load classes from config
    classes_config = config.get("classes", {})
    classes = {int(k): v for k, v in classes_config.items()}

    # Load palette from config
    palette_config = config.get("palette", {})
    palette = {int(k): tuple(v) for k, v in palette_config.items()}

    # Get num_classes from evaluation config or compute from classes
    evaluation_config = config.get("evaluation", {})
    num_classes = evaluation_config.get("num_classes", max(classes.keys()) + 1 if classes else 19)

    # Get ignore_index from evaluation config
    ignore_index = evaluation_config.get("ignore_index", 255)

    # Pre-computed palette array for vectorized colorization
    # Shape: (num_classes, 3) - one RGB color per class ID
    palette_array = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        if class_id < num_classes:
            palette_array[class_id] = color

    # Get display settings from config
    display_config = config.get("display", {})
    max_display_width = display_config.get("max_width", 1200)
    max_display_height = display_config.get("max_height", 800)

    return {
        "classes": classes,
        "palette": palette,
        "palette_array": palette_array,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "max_display_width": max_display_width,
        "max_display_height": max_display_height,
    }

# Lazy load constants
_constants = None

def get_constants():
    """Get constants dict (lazy loaded)."""
    global _constants
    if _constants is None:
        _constants = _load_constants()
    return _constants

def get_classes():
    """Get class names mapping."""
    return get_constants()["classes"]

def get_palette():
    """Get color palette mapping."""
    return get_constants()["palette"]

def get_palette_array():
    """Get pre-computed palette array for vectorized colorization."""
    return get_constants()["palette_array"]

def get_num_classes():
    """Get number of classes."""
    return get_constants()["num_classes"]

def get_ignore_index():
    """Get ignore index for segmentation evaluation."""
    return get_constants()["ignore_index"]

def get_max_display_width():
    """Get maximum display width for images."""
    return get_constants()["max_display_width"]

def get_max_display_height():
    """Get maximum display height for images."""
    return get_constants()["max_display_height"]