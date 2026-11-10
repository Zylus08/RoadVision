"""RoadVision: Autonomous Road-Scene Semantic Understanding and Robustness Evaluation System."""

__version__ = "1.0.0"

# Public API exports
from src.config import get_config, get_model_config, get_corruption_config, get_evaluation_config, get_video_config
from src.inference import RoadVisionSegmentor, PredictionResult
from src.video_processor import VideoProcessor, processed_video_path
from src.corruptions import apply_corruption, CORRUPTION_REGISTRY
from src.visualization import (
    CITYSCAPES_CLASSES,
    CITYSCAPES_PALETTE,
    CITYSCAPES_PALETTE_ARRAY,
    color_mask,
    overlay_mask,
    image_to_data_uri,
    create_diff_map,
    create_legend_image,
)
from src.metrics import SegmentationEvaluator, compute_degradation
from src.constants import (
    get_classes,
    get_palette,
    get_palette_array,
    get_num_classes,
    get_ignore_index,
    get_max_display_width,
    get_max_display_height,
)

__all__ = [
    # Config
    "get_config",
    "get_model_config",
    "get_corruption_config",
    "get_evaluation_config",
    "get_video_config",
    # Inference
    "RoadVisionSegmentor",
    "PredictionResult",
    # Video processing
    "VideoProcessor",
    "processed_video_path",
    # Corruptions
    "apply_corruption",
    "CORRUPTION_REGISTRY",
    # Visualization
    "CITYSCAPES_CLASSES",
    "CITYSCAPES_PALETTE",
    "CITYSCAPES_PALETTE_ARRAY",
    "color_mask",
    "overlay_mask",
    "image_to_data_uri",
    "create_diff_map",
    "create_legend_image",
    # Metrics
    "SegmentationEvaluator",
    "compute_degradation",
    # Constants
    "get_classes",
    "get_palette",
    "get_palette_array",
    "get_num_classes",
    "get_ignore_index",
    "get_max_display_width",
    "get_max_display_height",
]
