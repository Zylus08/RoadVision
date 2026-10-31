import os
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

os.environ["TORCH_HOME"] = os.path.abspath("E:/RoadVision/.cache/torch")

print("Testing Torchvision DeepLabV3-MobileNetV3...")
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()

print("DeepLabV3 loaded successfully!")
print("Transforms:", weights.transforms())
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
