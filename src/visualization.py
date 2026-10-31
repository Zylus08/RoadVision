"""Visualization utilities for RoadVision semantic segmentation."""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CITYSCAPES_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

CITYSCAPES_PALETTE = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32)     # bicycle
}

def color_mask(mask: np.ndarray, palette: dict = None) -> np.ndarray:
    """Convert a 2D integer class mask into an RGB colorized mask.
    
    Args:
        mask: 2D numpy array of shape (H, W) with integer class IDs.
        palette: Dictionary mapping class ID to (R, G, B) tuple.
        
    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.
    """
    if palette is None:
        palette = CITYSCAPES_PALETTE
        
    H, W = mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id, color in palette.items():
        colored[mask == class_id] = color
        
    return colored

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
