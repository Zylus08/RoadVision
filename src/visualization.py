"""Visualization utilities for RoadVision semantic segmentation."""

import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from src.constants import (
    get_classes,
    get_palette,
    get_palette_array,
)

# Re-export constants for backward compatibility
CITYSCAPES_CLASSES = get_classes()
CITYSCAPES_PALETTE = get_palette()
CITYSCAPES_PALETTE_ARRAY = get_palette_array()


def image_to_data_uri(image: Image.Image) -> str:
    """Encode an image as a PNG data URI for inline display."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def color_mask(mask: np.ndarray, palette: dict = None, palette_array: np.ndarray = None) -> np.ndarray:
    """Convert a 2D integer class mask into an RGB colorized mask.

    Args:
        mask: 2D numpy array of shape (H, W) with integer class IDs.
        palette: Dictionary mapping class ID to (R, G, B) tuple (legacy).
        palette_array: Pre-computed array of shape (num_classes, 3) for vectorized lookup.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.
    """
    if palette_array is None:
        if palette is None:
            palette_array = CITYSCAPES_PALETTE_ARRAY
        else:
            # Build array from dict (for custom palettes)
            max_class = max(palette.keys()) if palette else 18
            palette_array = np.zeros((max_class + 1, 3), dtype=np.uint8)
            for class_id, color in palette.items():
                palette_array[class_id] = color

    # Clip mask to valid range for palette array
    mask_clipped = np.clip(mask, 0, palette_array.shape[0] - 1)

    # Vectorized lookup: palette_array[mask] broadcasts mask indices to colors
    # Result shape: (H, W, 3)
    return palette_array[mask_clipped]

def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, palette: dict = None) -> np.ndarray:
    """Blend original RGB image and colorized semantic segmentation mask.
    
    Args:
        image: RGB image (H, W, 3), uint8.
        mask: 2D integer mask (H, W) or RGB colored mask (H, W, 3).
        alpha: Transparency weight for segmentation mask (0.0 to 1.0).
        palette: Class color palette.
        
    Returns:
        Blended RGB image (H, W, 3), uint8.
    """
    if mask.ndim == 2:
        colored = color_mask(mask, palette=palette)
    else:
        colored = mask.astype(np.uint8)
        
    if image.shape[:2] != colored.shape[:2]:
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    blended = cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, colored, alpha, 0)
    return blended

def create_diff_map(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Generate a difference heatmap showing where predictions diverge.
    
    Args:
        mask1: First 2D class mask (e.g., clean prediction).
        mask2: Second 2D class mask (e.g., corrupted prediction).
        
    Returns:
        RGB image highlighting disagreeing pixels in bright red/orange.
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    diff = (mask1 != mask2).astype(np.uint8) * 255
    diff_rgb = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    # Bright orange/red for corrupted disagreement
    diff_rgb[diff > 0] = [255, 69, 0]
    return diff_rgb

def create_legend_image(classes_present: list, palette: dict = None, class_names: dict = None, width: int = 240) -> np.ndarray:
    """Generate a clean visual legend showing detected classes with color chips.
    
    Args:
        classes_present: List of class IDs present in the image.
        palette: Mapping of class ID to RGB.
        class_names: Mapping of class ID to name.
        width: Width of legend image in pixels.
        
    Returns:
        RGB image containing the legend.
    """
    if palette is None:
        palette = CITYSCAPES_PALETTE
    if class_names is None:
        class_names = CITYSCAPES_CLASSES
        
    if not classes_present:
        classes_present = list(range(min(5, len(class_names))))
        
    row_height = 28
    height = max(100, len(classes_present) * row_height + 20)
    
    img = Image.new("RGB", (width, height), color=(25, 27, 34))
    draw = ImageDraw.Draw(img)
    
    y = 12
    for cid in classes_present:
        color = palette.get(cid, (180, 180, 180))
        name = class_names.get(cid, f"Class {cid}")
        
        # Draw color chip
        draw.rectangle([12, y + 2, 32, y + 20], fill=color, outline=(255, 255, 255))
        # Draw label
        draw.text((42, y + 3), f"{name.capitalize()}", fill=(240, 240, 240))
        y += row_height
        
    return np.array(img)
