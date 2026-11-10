"""Inference engine for RoadVision semantic segmentation models."""

import os
import time
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Tuple, Dict, Optional, Union, List, TypedDict

from src.config import get_model_config, get_cache_dir
from src.visualization import color_mask, overlay_mask
from src.constants import get_classes, get_palette, get_num_classes, get_ignore_index

# Resource limits to prevent DoS (loaded from config lazily)
def _get_resource_limits():
    """Get resource limits from config with fallback defaults."""
    resources = get_model_config().get("resources", {})
    return {
        "max_image_dimension": resources.get("max_image_dimension", 4096),
        "max_image_pixels": resources.get("max_image_pixels", 8_000_000),
        "max_video_duration_seconds": resources.get("max_video_duration_seconds", 60),
        "max_video_file_size_mb": resources.get("max_video_file_size_mb", 500),
        "max_image_file_size_mb": resources.get("max_image_file_size_mb", 100),
    }

_RESOURCE_LIMITS = _get_resource_limits()
MAX_IMAGE_DIMENSION = _RESOURCE_LIMITS["max_image_dimension"]
MAX_IMAGE_PIXELS = _RESOURCE_LIMITS["max_image_pixels"]
MAX_VIDEO_DURATION_SECONDS = _RESOURCE_LIMITS["max_video_duration_seconds"]
MAX_VIDEO_FILE_SIZE_MB = _RESOURCE_LIMITS["max_video_file_size_mb"]


class PredictionResult(TypedDict):
    """Type definition for prediction result dictionary."""
    pred_mask: np.ndarray
    colored_mask: np.ndarray
    overlay: np.ndarray
    class_proportions: Dict[str, float]
    present_classes: List[int]
    latency_ms: float
    fps: float
    orig_shape: Tuple[int, int]
    device: str


def validate_image_input(image: Union[np.ndarray, Image.Image, str]) -> None:
    """Validate image input for size and type constraints.

    Args:
        image: Input image as path, PIL Image, or numpy array.

    Raises:
        ValueError: If image exceeds resource limits.
        TypeError: If image type is not supported.
    """
    if isinstance(image, str):
        # Check file size
        file_size_mb = os.path.getsize(image) / (1024 * 1024)
        if file_size_mb > 100:  # 100 MB max for images
            raise ValueError(f"Image file too large: {file_size_mb:.1f} MB (max 100 MB)")

        # Check dimensions without loading full image
        with Image.open(image) as img:
            width, height = img.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ValueError(f"Image dimensions too large: {width}x{height} (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image has too many pixels: {width * height:,} (max {MAX_IMAGE_PIXELS:,})")

    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(f"Image dimensions too large: {width}x{height} (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image has too many pixels: {width * height:,} (max {MAX_IMAGE_PIXELS:,})")
        if image.dtype != np.uint8:
            raise TypeError(f"Image array must be uint8, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be RGB (H, W, 3), got shape {image.shape}")

    elif isinstance(image, Image.Image):
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(f"Image dimensions too large: {width}x{height} (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image has too many pixels: {width * height:,} (max {MAX_IMAGE_PIXELS:,})")

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def validate_video_input(video_path: str) -> None:
    """Validate video input for size and duration constraints.

    Args:
        video_path: Path to video file.

    Raises:
        ValueError: If video exceeds resource limits.
    """
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_FILE_SIZE_MB:
        raise ValueError(f"Video file too large: {file_size_mb:.1f} MB (max {MAX_VIDEO_FILE_SIZE_MB} MB)")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(f"Video too long: {duration:.1f} seconds (max {MAX_VIDEO_DURATION_SECONDS} seconds)")
    finally:
        cap.release()

class RoadVisionSegmentor:
    """End-to-end segmentation predictor supporting SegFormer and DeepLabV3."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        architecture: Optional[str] = None,
        device: str = "auto",
        input_size: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize RoadVision segmentation model.

        Args:
            model_name: HuggingFace model repo ID or torchvision model name.
            architecture: 'segformer' or 'deeplabv3'.
            device: 'auto', 'cuda', or 'cpu'.
            input_size: (height, width) for model forward pass before upsampling.
            cache_dir: Local directory for caching weights.
        """
        from src.config import get_model_config
        model_config = get_model_config()

        self.model_name = model_name or model_config.get("name", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.architecture = (architecture or model_config.get("architecture", "segformer")).lower()
        self.input_size = input_size or tuple(model_config.get("input_size", [512, 512]))
        from src.config import get_cache_dir
        self.cache_dir = cache_dir or get_cache_dir()
        # Ensure path is expanded (fixes ~ expansion issues)
        self.cache_dir = os.path.expanduser(self.cache_dir)

        # Device resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.class_names = get_classes()
        self.palette = get_palette()
        
        self._load_model()
        
    def _load_model(self):
        """Load pretrained model and processor/transforms."""
        if self.architecture == "segformer":
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            self.processor = SegformerImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Extract id2label if available
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                self.class_names = {int(k): v for k, v in self.model.config.id2label.items()}
                
        elif self.architecture == "deeplabv3":
            import torchvision.models.segmentation as seg_models
            
            weights = seg_models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self.model = seg_models.deeplabv3_mobilenet_v3_large(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.transforms = weights.transforms()
        else:
            raise ValueError(f"Unsupported architecture '{self.architecture}'. Use 'segformer' or 'deeplabv3'.")
            
    @torch.inference_mode()
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str],
        return_probabilities: bool = False
    ) -> PredictionResult:
        """Run semantic segmentation inference on a single image.

        Args:
            image: Path to image file, PIL Image, or RGB uint8 numpy array.
            return_probabilities: Whether to return soft probability tensor.

        Returns:
            Dict containing:
                'pred_mask': 2D numpy array of class indices (H, W).
                'colored_mask': RGB numpy array (H, W, 3).
                'overlay': RGB blended numpy array (H, W, 3).
                'class_proportions': Dict mapping class name to % of scene.
                'present_classes': List of present class IDs.
                'latency_ms': Model forward pass time in milliseconds.
                'fps': Effective model throughput (1000 / latency_ms).
                'orig_shape': (H, W) of input image.
        """
        # Validate input
        validate_image_input(image)

        # Load and convert image to RGB PIL and numpy
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at path: {image}")
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)
        
        # Timing start
        start_time = time.perf_counter()
        
        if self.architecture == "segformer":
            h_in, w_in = self.input_size
            inputs = self.processor(
                images=pil_img,
                size={"height": h_in, "width": w_in},
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Bilinear upsample logits to exact original image size (orig_h, orig_w)
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            pred_tensor = upsampled_logits.argmax(dim=1)[0]
            pred_mask = pred_tensor.cpu().numpy().astype(np.uint8)
            
        elif self.architecture == "deeplabv3":
            # Apply torchvision transforms
            tensor_input = self.transforms(pil_img).unsqueeze(0).to(self.device)
            outputs = self.model(tensor_input)["out"]
            upsampled_logits = torch.nn.functional.interpolate(
                outputs,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            pred_tensor = upsampled_logits.argmax(dim=1)[0]
            pred_mask = pred_tensor.cpu().numpy().astype(np.uint8)
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        # Colorize and blend
        colored = color_mask(pred_mask, palette=self.palette)
        overlay = overlay_mask(img_np, colored, alpha=0.45)
        
        # Compute class proportions
        unique_classes, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        class_proportions = {}
        present_classes = []
        
        for cid, count in zip(unique_classes, counts):
            cid = int(cid)
            name = self.class_names.get(cid, f"class_{cid}")
            pct = float(count / total_pixels * 100.0)
            class_proportions[name] = round(pct, 2)
            present_classes.append(cid)
            
        result = {
            "pred_mask": pred_mask,
            "colored_mask": colored,
            "overlay": overlay,
            "class_proportions": class_proportions,
            "present_classes": present_classes,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1),
            "orig_shape": (orig_h, orig_w),
            "device": str(self.device)
        }
        
        return result

    @torch.inference_mode()
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        return_probabilities: bool = False
    ) -> List[PredictionResult]:
        """Run semantic segmentation inference on a batch of images.

        Args:
            images: List of image paths, PIL Images, or RGB uint8 numpy arrays.
            return_probabilities: Whether to return soft probability tensor.

        Returns:
            List of Dicts containing the same keys as predict().
        """
        if not images:
            return []

        # Convert all images to PIL and numpy
        pil_images = []
        orig_sizes = []
        img_arrays = []

        for image in images:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found at path: {image}")
                pil_img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_img = image.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

            orig_w, orig_h = pil_img.size
            pil_images.append(pil_img)
            orig_sizes.append((orig_h, orig_w))
            img_arrays.append(np.array(pil_img))

        # Timing start
        start_time = time.perf_counter()

        if self.architecture == "segformer":
            h_in, w_in = self.input_size
            # Batch processing with processor
            inputs = self.processor(
                images=pil_images,
                size={"height": h_in, "width": w_in},
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (batch, num_classes, H_out, W_out)

            # Process each image in batch
            results = []
            for i, (orig_h, orig_w) in enumerate(orig_sizes):
                # Bilinear upsample logits to exact original image size
                upsampled_logits = torch.nn.functional.interpolate(
                    logits[i:i+1],
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False
                )
                pred_tensor = upsampled_logits.argmax(dim=1)[0]
                pred_mask = pred_tensor.cpu().numpy().astype(np.uint8)

                # Colorize and blend
                colored = color_mask(pred_mask, palette=self.palette)
                overlay = overlay_mask(img_arrays[i], colored, alpha=0.45)

                # Compute class proportions
                unique_classes, counts = np.unique(pred_mask, return_counts=True)
                total_pixels = pred_mask.size
                class_proportions = {}
                present_classes = []

                for cid, count in zip(unique_classes, counts):
                    cid = int(cid)
                    name = self.class_names.get(cid, f"class_{cid}")
                    pct = float(count / total_pixels * 100.0)
                    class_proportions[name] = round(pct, 2)
                    present_classes.append(cid)

                results.append({
                    "pred_mask": pred_mask,
                    "colored_mask": colored,
                    "overlay": overlay,
                    "class_proportions": class_proportions,
                    "present_classes": present_classes,
                    "orig_shape": (orig_h, orig_w),
                })

        elif self.architecture == "deeplabv3":
            # For DeepLabV3, batch process by stacking transformed tensors
            # Apply transforms to all images and stack
            tensor_inputs = torch.stack([self.transforms(pil_img) for pil_img in pil_images]).to(self.device)

            # Batch forward pass
            outputs = self.model(tensor_inputs)["out"]  # Shape: (batch, num_classes, H_out, W_out)

            # Process each image in batch
            results = []
            for i, (orig_h, orig_w) in enumerate(orig_sizes):
                # Bilinear upsample logits to exact original image size
                upsampled_logits = torch.nn.functional.interpolate(
                    outputs[i:i+1],
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False
                )
                pred_tensor = upsampled_logits.argmax(dim=1)[0]
                pred_mask = pred_tensor.cpu().numpy().astype(np.uint8)

                colored = color_mask(pred_mask, palette=self.palette)
                overlay = overlay_mask(img_arrays[i], colored, alpha=0.45)

                unique_classes, counts = np.unique(pred_mask, return_counts=True)
                total_pixels = pred_mask.size
                class_proportions = {}
                present_classes = []

                for cid, count in zip(unique_classes, counts):
                    cid = int(cid)
                    name = self.class_names.get(cid, f"class_{cid}")
                    pct = float(count / total_pixels * 100.0)
                    class_proportions[name] = round(pct, 2)
                    present_classes.append(cid)

                results.append({
                    "pred_mask": pred_mask,
                    "colored_mask": colored,
                    "overlay": overlay,
                    "class_proportions": class_proportions,
                    "present_classes": present_classes,
                    "orig_shape": (orig_h, orig_w),
                })

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = total_latency_ms / len(images) if images else 0.0
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        # Add timing info to all results
        for r in results:
            r["latency_ms"] = round(avg_latency_ms, 2)
            r["fps"] = round(fps, 1)
            r["device"] = str(self.device)

        return results
