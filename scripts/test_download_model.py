import os
import sys

# Ensure caches stay on E: drive
os.environ["HF_HOME"] = os.path.abspath("E:/RoadVision/.cache/huggingface")
os.environ["TORCH_HOME"] = os.path.abspath("E:/RoadVision/.cache/torch")

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

model_id = "nvidia/segformer-b0-finetuned-cityscapes-512-512"
print(f"Loading processor for {model_id}...")
processor = SegformerImageProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])

print(f"Loading model weights for {model_id}...")
model = SegformerForSemanticSegmentation.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
model.eval()

print("Model successfully loaded!")
print(f"Number of classes: {len(model.config.id2label)}")
print(f"Class labels: {model.config.id2label}")
