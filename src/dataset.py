"""Dataset management and synthetic demonstration asset generation for RoadVision."""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict

class RoadSceneDataset:
    """Manages road scene images and paired segmentation ground truth labels."""
    
    def __init__(self, data_dir: str = "data/samples", label_dir: Optional[str] = "data/labels"):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.samples = self._discover_samples()
        
    def _discover_samples(self) -> List[Dict]:
        """Discover valid images and paired labels in the dataset directories."""
        if not os.path.exists(self.data_dir):
            return []
            
        exts = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            
        image_files = sorted(image_files)
        dataset_items = []
        
        for img_path in image_files:
            basename = os.path.basename(img_path)
            stem = os.path.splitext(basename)[0]
            
            # Check for corresponding label
            label_path = None
            if self.label_dir and os.path.exists(self.label_dir):
                candidate_names = [
                    f"{stem}_label.png",
                    f"{stem}_mask.png",
                    f"{stem}.png",
                    basename
                ]
                for cname in candidate_names:
                    cpath = os.path.join(self.label_dir, cname)
                    if os.path.exists(cpath):
                        label_path = cpath
                        break
                        
            dataset_items.append({
                "image_path": img_path,
                "label_path": label_path,
                "name": stem
            })
            
        return dataset_items
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        mask = None
        if item["label_path"]:
            mask_img = Image.open(item["label_path"])
            mask = np.array(mask_img)
            # If 3-channel RGB label, take first channel or convert to 2D
            if mask.ndim == 3:
                mask = mask[:, :, 0]
                
        return {
            "image": image,
            "mask": mask,
            "image_path": item["image_path"],
            "label_path": item["label_path"],
            "name": item["name"]
        }

def generate_sample_dashcam_video(
    output_path: str = "data/samples/sample_dashcam.mp4",
    num_frames: int = 60,
    fps: int = 20,
    source_images_dir: str = "data/samples"
) -> str:
    """Generate a realistic test dashcam video sequence by panning across road scene images.
    
    Args:
        output_path: Target video file path.
        num_frames: Number of video frames to generate.
        fps: Target frame rate.
        source_images_dir: Directory containing high-resolution road scene images.
        
    Returns:
        Path to generated video file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    images = sorted(glob.glob(os.path.join(source_images_dir, "*.png")))
    
    out_w, out_h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    if not writer.isOpened():
        output_path = output_path.rsplit(".", 1)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
    if not images:
        # Generate synthetic road scene frame if no images
        for f in range(num_frames):
            frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            frame[:out_h//2, :] = [180, 130, 70] # sky
            frame[out_h//2:, :] = [128, 64, 128] # road
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return output_path
        
    # Read first available road image
    src_img = cv2.imread(images[0])
    src_h, src_w, _ = src_img.shape
    
    # Animate gentle vehicle driving motion: slight zoom and vertical pitch oscillation
    for i in range(num_frames):
        t = i / float(num_frames)
        # Gentle forward drive crop animation
        crop_factor = 1.0 - (0.08 * np.sin(t * np.pi))
        crop_w = int(src_w * crop_factor)
        crop_h = int(src_h * crop_factor)
        
        x_offset = int((src_w - crop_w) * 0.5)
        y_offset = int((src_h - crop_h) * 0.6 + 5.0 * np.sin(i * 0.3)) # Road vibration
        
        y_offset = max(0, min(src_h - crop_h, y_offset))
        x_offset = max(0, min(src_w - crop_w, x_offset))
        
        cropped = src_img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(resized)
        
    writer.release()
    return output_path
