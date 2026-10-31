import os
import io
import pandas as pd
import numpy as np
from PIL import Image

def prepare_subset():
    url = "https://huggingface.co/datasets/Chris1/cityscapes_segmentation/resolve/main/data/test-00000-of-00007.parquet"
    print("Reading parquet from:", url)
    df = pd.read_parquet(url)
    
    os.makedirs("data/samples", exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)
    
    num_samples = 20
    print(f"Extracting {num_samples} authentic Cityscapes road-scene evaluation samples...")
    
    unique_label_vals = set()
    for idx in range(num_samples):
        row = df.iloc[idx]
        img_bytes = row['image']['bytes']
        mask_bytes = row['semantic_segmentation']['bytes']
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes))
        
        mask_arr = np.array(mask)
        unique_label_vals.update(np.unique(mask_arr).tolist())
        
        img_path = f"data/samples/road_scene_{idx+1:03d}.png"
        mask_path = f"data/labels/road_scene_{idx+1:03d}_label.png"
        
        img.save(img_path)
        mask.save(mask_path)
        print(f"Saved {img_path} ({img.size}) and {mask_path} (unique classes: {len(np.unique(mask_arr))})")
        
    print("Preparation complete!")
    print("Unique label values found in extracted masks:", sorted(list(unique_label_vals)))

if __name__ == "__main__":
    prepare_subset()
