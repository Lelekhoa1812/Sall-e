import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# Upload image
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

# Get predicted class per pixel
logits = outputs.logits  # Shape: [1, 150, H, W]
segmentation = logits.argmax(dim=1)[0].cpu().numpy()

# ADE20K palette
ade_palette = np.array([
    [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
    [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70],
    [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7],
    [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
    [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71],
    [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
    [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
    [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0],
    [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255],
    [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0],
    [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41],
    [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0],
    [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204],
    [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255],
    [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10],
    [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31],
    [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255],
    [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0],
    [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212],
    [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255],
    [0, 41, 255], [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255],
    [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0],
    [92, 0, 255]
], dtype=np.uint8)

# Custom color-to-class-name mapping (grouped)
custom_class_map = {
    "Garbage":                 [(150, 5, 61)],
    "Water":                   [(0, 102, 200), (11, 102, 255), (31, 0, 255)],
    "Grass / Vegetation":      [(10, 255, 71), (143, 255, 140)],
    "Tree / Natural Obstacle": [(4, 200, 3), (235, 12, 255), (255, 6, 82), (255, 163, 0)],
    "Sand / Soil / Ground":    [(80, 50, 50), (230, 230, 230)],
    "Buildings / Structures":  [(255, 0, 255), (184, 0, 255), (120, 120, 120), (7, 255, 224)],
    "Sky / Background":        [(180, 120, 120)],
    "Undetecable":             [(0, 0, 0)],
    "Unknown Class": []
}

# Identify ADE20K class IDs for water and garbage
id2label = model.config.id2label
water_class_ids = [int(idx) for idx, name in id2label.items() if "water" in name.lower()]
garbage_class_ids = [int(idx) for idx, name in id2label.items() if "garbage" in name.lower()]

# Build RGB class reverse lookup: (r,g,b) -> class name
rgb_to_name = {}
for class_name, color_list in custom_class_map.items():
    for rgb in color_list:
        rgb_to_name[tuple(rgb)] = class_name

# Decode segmentation mask to color image
decoded_mask = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
for label in np.unique(segmentation):
    decoded_mask[segmentation == label] = ade_palette[label]

# Approximate RGB class matching with ±30 tolerance
TOLERANCE = 30
def match_rgb_to_class(rgb, rgb_class_map, tolerance=TOLERANCE):
    for class_name, rgb_list in rgb_class_map.items():
        for ref_rgb in rgb_list:
            if all(abs(c1 - c2) <= tolerance for c1, c2 in zip(rgb, ref_rgb)):
                return class_name
    return "Unknown Class"

# Get pixel coordinates by mapping from segmentation
class_mask = np.empty(segmentation.shape, dtype=object)
for y in range(segmentation.shape[0]):
    for x in range(segmentation.shape[1]):
        rgb = tuple(decoded_mask[y, x])
        class_mask[y, x] = match_rgb_to_class(rgb, custom_class_map)
water_coords = np.argwhere(class_mask == "Water")
garbage_coords = np.argwhere(class_mask == "Garbage")
if len(water_coords) > 0:
    print("Total water coordination (pixels) detected ", len(water_coords))
if len(garbage_coords) > 0:
    print("Total garbage coordination (pixels) detected ", len(garbage_coords))

# Show original image and mask
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(decoded_mask)
ax[1].set_title("Segmented Image (Overlay Garbage / Water Coordinations)")
if len(water_coords) > 0:
    print("Total water coordination (pixels) detected")
    ax[1].scatter(water_coords[:, 1], water_coords[:, 0], s=1, c='blue', label='Water')
if len(garbage_coords) > 0:
    ax[1].scatter(garbage_coords[:, 1], garbage_coords[:, 0], s=1, c='red', label='Garbage')
ax[1].legend(loc='upper right')
ax[1].axis("off")
plt.tight_layout()
plt.show()

# Create a grouped legend: Class → all RGB values
from collections import defaultdict
legend = defaultdict(list)
used_labels = np.unique(segmentation)
# Append label as color + class name (unknown exception)
for label in used_labels:
    rgb = tuple(ade_palette[label])
    class_name = match_rgb_to_class(rgb, custom_class_map)
    legend[class_name].append(rgb)

# Plot legend as table
fig, ax = plt.subplots(figsize=(10, len(legend) * 0.6))
# Dimension
row_height = 0.6
square_size = 0.3
for i, (class_name, rgb_list) in enumerate(legend.items()):
    y = i * row_height
    # First column: draw color patches
    for j, rgb in enumerate(rgb_list[:3]):  # up to 3 variants
        patch_color = np.array(rgb) / 255
        ax.add_patch(plt.Rectangle((0.1 + j * (square_size + 0.05), y), square_size, square_size, color=patch_color, edgecolor='black'))
    # Second column: class name
    ax.text(1.2, y + square_size / 2, class_name, fontsize=10, va='center')
    # Third column: RGB(s)
    rgb_str = ", ".join([f"({r},{g},{b})" for (r, g, b) in rgb_list])
    ax.text(4, y + square_size / 2, rgb_str, fontsize=10, va='center')
# Layouts
ax.set_xlim(0, 8)
ax.set_ylim(0, len(legend) * row_height)
ax.axis("off")
plt.title("Legend of Semantic Classes", fontsize=13)
plt.tight_layout()
plt.show()
