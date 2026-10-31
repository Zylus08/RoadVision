"""Inference engine for RoadVision semantic segmentation models."""

import os
import time
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Tuple, Dict, Optional, Union

# Point huggingface and torch caches to local project directory on E: drive
os.environ["HF_HOME"] = os.path.abspath("E:/RoadVision/.cache/huggingface")
os.environ["TORCH_HOME"] = os.path.abspath("E:/RoadVision/.cache/torch")

from src.visualization import color_mask, overlay_mask, CITYSCAPES_CLASSES, CITYSCAPES_PALETTE

class RoadVisionSegmentor:
    """End-to-end segmentation predictor supporting SegFormer and DeepLabV3."""
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        architecture: str = "segformer",
        device: str = "auto",
        input_size: Tuple[int, int] = (512, 512),
        cache_dir: Optional[str] = "E:/RoadVision/.cache/huggingface"
    ):
        """Initialize RoadVision segmentation model.
        
        Args:
            model_name: HuggingFace model repo ID or torchvision model name.
            architecture: 'segformer' or 'deeplabv3'.
            device: 'auto', 'cuda', or 'cpu'.
            input_size: (height, width) for model forward pass before upsampling.
            cache_dir: Local directory for caching weights.
        """
        self.model_name = model_name
        self.architecture = architecture.lower()
        self.input_size = input_size
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")
        
        # Device resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.class_names = CITYSCAPES_CLASSES
        self.palette = CITYSCAPES_PALETTE
        
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
    ) -> Dict:
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
