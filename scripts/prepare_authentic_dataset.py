"""Script to download authentic, paired Cityscapes validation samples with genuine labelTrainIds."""

import os
import urllib.request
import numpy as np
from PIL import Image

VAL_SAMPLES = [
    ("frankfurt_000000_000294", "frankfurt"),
    ("frankfurt_000000_000576", "frankfurt"),
    ("frankfurt_000000_001016", "frankfurt"),
    ("frankfurt_000000_001236", "frankfurt"),
    ("frankfurt_000000_001751", "frankfurt"),
]

BASE_URL = "https://huggingface.co/datasets/tackhwa/cityscapes/resolve/main/cityscapes"

def download_authentic_subset():
    os.makedirs("data/samples/authentic", exist_ok=True)
    os.makedirs("data/labels/authentic", exist_ok=True)
    
    print("Downloading 5 authentic Cityscapes validation pairs...")
    for stem, city in VAL_SAMPLES:
        img_url = f"{BASE_URL}/leftImg8bit/val/{city}/{stem}_leftImg8bit.png"
        gt_url = f"{BASE_URL}/gtFine/val/{city}/{stem}_gtFine_labelTrainIds.png"
        
        img_dst = f"data/samples/authentic/{stem}.png"
        gt_dst = f"data/labels/authentic/{stem}_label.png"
        
        if not os.path.exists(img_dst):
            print(f"Fetching image: {stem}...")
            urllib.request.urlretrieve(img_url, img_dst)
            
        if not os.path.exists(gt_dst):
            print(f"Fetching ground truth: {stem}...")
            urllib.request.urlretrieve(gt_url, gt_dst)
            
        gt_arr = np.array(Image.open(gt_dst))
        u_labels = np.unique(gt_arr)
        print(f"  {stem}: shape={gt_arr.shape}, unique labels={len(u_labels)} classes: {u_labels.tolist()}")
        
    print("Download and verification complete!")

if __name__ == "__main__":
    download_authentic_subset()

