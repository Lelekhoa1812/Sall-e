# Run installation on terminal while setting up
# pip install transformers accelerate torchvision matplotlib--quiet

# Clone LoveDA finetuning dataset
# git clone https://github.com/Junjue-Wang/LoveDA.git
# cd LoveDA

import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# Upload an image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path).convert("RGB")

# Load feature extractor and model
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Post-process segmentation
logits = outputs.logits  # Shape: [1, 150, H, W]
segmentation = logits.argmax(dim=1)[0].cpu().numpy()

# Decode to color image
def decode_segmentation(mask):
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        # ... (extend if needed up to 150 classes)
    ])
    # Safe fallback for missing colors
    decoded = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in np.unique(mask):
        decoded[mask == label] = colors[label % len(colors)]
    return decoded

decoded_mask = decode_segmentation(segmentation)

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(decoded_mask)
ax[1].set_title("Segmented Image (Riverbank / Obstacles Approx.)")
ax[1].axis("off")
plt.show()